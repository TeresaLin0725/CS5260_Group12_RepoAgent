from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from api.agent.events import AgentEvent, AgentEventType
from api.agent.planner import RuleBasedPlanner
from api.agent.intent_classifier import (
    EmbeddingIntentClassifier,
    LLMIntentClassifier,
    IntentResult,
)
from api.agent.tools.export_tools import build_export_tool_registry

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentScheduleResult:
    handled: bool
    content: Optional[str] = None
    events: List[AgentEvent] = field(default_factory=list)
    intent_result: Optional[IntentResult] = None


class AgentScheduler:
    """Three-tier intent scheduler.

    The scheduler classifies user intent across three tiers (rule-based,
    embedding, LLM) but **never short-circuits to export directly**.
    The LLM always responds first; export actions are appended via
    stage-2 inference (``infer_second_stage_action``) after the LLM has
    had a chance to answer the user.  This ensures the user always sees
    the model's reasoning and explanation before any export is triggered.
    """

    def __init__(
        self,
        planner: RuleBasedPlanner,
        embedding_classifier: Optional[EmbeddingIntentClassifier] = None,
        llm_classifier: Optional[LLMIntentClassifier] = None,
    ):
        self.planner = planner
        self.embedding_classifier = embedding_classifier
        self.llm_classifier = llm_classifier

    @classmethod
    def default(cls) -> "AgentScheduler":
        registry = build_export_tool_registry()
        return cls(planner=RuleBasedPlanner(registry))

    def with_classifiers(
        self,
        embed_fn: Optional[Callable] = None,
        llm_fn: Optional[Callable] = None,
    ) -> "AgentScheduler":
        """Return a copy of this scheduler with tier-2/3 classifiers attached.

        Call this once per request when the embedding / LLM callables are
        available (they depend on the provider chosen by the user).
        """
        emb_cls = EmbeddingIntentClassifier(embed_fn) if embed_fn else self.embedding_classifier
        llm_cls = LLMIntentClassifier(llm_fn) if llm_fn else self.llm_classifier
        return AgentScheduler(
            planner=self.planner,
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        )

    # ------------------------------------------------------------------
    # Tier 1 — synchronous rule-based scheduling (unchanged API)
    # ------------------------------------------------------------------

    def schedule(self, query: str, language: str = "en") -> AgentScheduleResult:
        """Tier-1 rule-based classification.  Always returns handled=False
        so the LLM responds first; detected export intent is logged for
        stage-2 to act on after the LLM answer."""
        events = [AgentEvent(AgentEventType.PLAN_CREATED, "Created plan from user query.")]
        plan = self.planner.plan(query=query)

        if plan.tool:
            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    f"Detected tool {plan.tool.name} (invoke={plan.should_invoke}). "
                    f"Deferring to LLM; export will be appended via stage-2.",
                    tool_name=plan.tool.name,
                )
            )
        elif plan.ambiguous:
            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    "Multiple tools matched. LLM will respond first; "
                    "stage-2 will infer the best export from the response.",
                )
            )
        else:
            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    f"No export tool detected. reason={plan.reason or 'no_tool_intent_match'}. "
                    f"Continue with normal model response.",
                )
            )

        return AgentScheduleResult(handled=False, events=events)

    # ------------------------------------------------------------------
    # Full 3-tier scheduling (async)
    # ------------------------------------------------------------------

    async def schedule_with_intent(
        self,
        query: str,
        language: str = "en",
    ) -> AgentScheduleResult:
        """Run all three classification tiers for intent metadata.

        Always returns ``handled=False`` so the LLM responds first.
        Export intent is carried in ``intent_result`` for stage-2 to act on.
        """
        # ── Tier 1: Rule-based ─────────────────────────────────────
        tier1 = self.schedule(query=query, language=language)
        # tier1 is always handled=False now

        # ── Tier 2: Embedding similarity ───────────────────────────
        intent: Optional[IntentResult] = None
        if self.embedding_classifier is not None:
            try:
                intent = await self.embedding_classifier.classify(query)
                if intent:
                    logger.info(
                        "Intent tier-2 (embedding): %s conf=%.3f",
                        intent.intent, intent.confidence,
                    )
            except Exception as exc:
                logger.warning("Embedding classifier error: %s", exc)

        # ── Tier 3: LLM fallback ──────────────────────────────────
        if intent is None and self.llm_classifier is not None:
            try:
                intent = await self.llm_classifier.classify(query)
                if intent:
                    logger.info(
                        "Intent tier-3 (LLM): %s conf=%.3f",
                        intent.intent, intent.confidence,
                    )
            except Exception as exc:
                logger.warning("LLM classifier error: %s", exc)

        if intent is None:
            return AgentScheduleResult(
                handled=False,
                events=tier1.events + [AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    "All 3 intent tiers returned no confident match. Deferring to ReAct.",
                )],
            )

        # Carry intent as metadata — never invoke export directly.
        # Stage-2 will append the export action after the LLM responds.
        msg = (
            f"Intent classified as {intent.intent} (conf={intent.confidence:.2f}, "
            f"source={intent.source}). LLM responds first; stage-2 handles export."
        )
        return AgentScheduleResult(
            handled=False,
            events=tier1.events + [AgentEvent(
                AgentEventType.TOOL_SKIPPED, msg,
            )],
            intent_result=intent,
        )

    def infer_second_stage_action(self, query: str, assistant_response: str) -> Optional[str]:
        """Infer a tool action tag after the LLM has responded (stage-2).

        Two independent paths:
        1. **Query-driven**: the user's query has generation/export intent
           → infer the best tool from ``query + response`` text.
        2. **Response-driven**: the LLM's response explicitly recommends a
           format (e.g. "I recommend PDF") → detect and honour that,
           regardless of whether the query had explicit generation intent.
        """
        text = assistant_response or ""
        normalized = text.lower()
        if "[action:" in normalized:
            return None
        if "error:" in normalized:
            return None

        # Path 1: query has generation/export intent (explicit or implicit)
        if self.planner.has_generation_intent(query):
            inferred = self.planner.infer_best_tool(f"{query}\n{text}")
            if inferred:
                return inferred.action_tag

        # Path 2: LLM response recommends a format (works even when the
        # query has no explicit generation verb, e.g. "给我最合适的文件")
        inferred_from_response = self._infer_from_recommendation(text)
        if inferred_from_response:
            return inferred_from_response

        return None

    def _pre_infer_from_query(self, query: str) -> Optional[str]:
        """Pre-planning: infer tool action tag from broad format hints in the
        query itself, before deferring to the LLM.  Only called when standard
        planner scoring found no matching tool but generation intent was detected."""
        normalized = (query or "").lower()
        _FORMAT_HINTS = {
            "[ACTION:GENERATE_PDF]": (
                "文档", "document", "documentation", "报告", "report",
                "文件", "手册", "manual", "白皮书", "whitepaper",
                "说明书", "技术资料",
            ),
            "[ACTION:GENERATE_PPT]": (
                "演示", "展示", "幻灯片", "汇报", "presentation",
                "slides", "keynote", "讲解",
            ),
            "[ACTION:GENERATE_VIDEO]": (
                "视频", "录像", "动画", "video", "walkthrough",
            ),
            "[ACTION:GENERATE_POSTER]": (
                "画报", "海报", "宣传", "infographic", "poster",
                "图文",
            ),
        }
        scores = {}
        for tag, hints in _FORMAT_HINTS.items():
            scores[tag] = sum(1 for h in hints if h in normalized)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if ranked[0][1] <= 0:
            return None
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def _infer_from_recommendation(self, text: str) -> Optional[str]:
        """Detect format recommendation patterns in the LLM response."""
        normalized = (text or "").lower()
        # Map recommendation keywords to action tags
        pdf_signals = ("pdf", "报告", "技术报告", "technical report")
        ppt_signals = ("ppt", "演示", "幻灯片", "presentation", "slides")
        video_signals = ("视频", "video", "walkthrough")
        poster_signals = ("poster", "画报", "海报", "infographic", "图文海报", "pictorial")

        recommendation_contexts = (
            "recommend", "suggest", "best", "most suitable", "choose",
            "推荐", "建议", "最合适", "最适合", "选择",
        )
        has_recommendation = any(ctx in normalized for ctx in recommendation_contexts)
        if not has_recommendation:
            return None

        pdf_score = sum(1 for s in pdf_signals if s in normalized)
        ppt_score = sum(1 for s in ppt_signals if s in normalized)
        video_score = sum(1 for s in video_signals if s in normalized)
        poster_score = sum(1 for s in poster_signals if s in normalized)

        scores = {
            "[ACTION:GENERATE_PDF]": pdf_score,
            "[ACTION:GENERATE_PPT]": ppt_score,
            "[ACTION:GENERATE_VIDEO]": video_score,
            "[ACTION:GENERATE_POSTER]": poster_score,
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if ranked[0][1] <= 0:
            return None
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    @staticmethod
    def _clarify_message(language: str) -> str:
        if (language or "").lower().startswith("zh"):
            return "你的请求可能同时需要多种输出格式。请明确告诉我你要 `PDF`、`PPT`、`Video` 还是 `Poster`（画报）。"
        return "Your request matches multiple outputs. Please specify one: `PDF`, `PPT`, `Video`, or `Poster`."

    @staticmethod
    def _tool_preamble(tool_name: str, language: str) -> str:
        if (language or "").lower().startswith("zh"):
            mapping = {
                "GENERATE_PDF": "我将为该仓库生成一份 PDF 技术报告。",
                "GENERATE_PPT": "我将通过 Gamma 为该仓库生成一份精美 PPT 演示文稿。",
                "GENERATE_VIDEO": "我将为该仓库生成一个视频概览。",
                "GENERATE_POSTER": "我将通过 NanoBanana 为该仓库生成一份图文画报。",
            }
            return mapping.get(tool_name, "我将开始处理你的导出请求。")

        mapping = {
            "GENERATE_PDF": "I will generate a PDF technical report for this repository.",
            "GENERATE_PPT": "I will generate a professionally designed PPT presentation via Gamma.",
            "GENERATE_VIDEO": "I will generate a video overview for this repository.",
            "GENERATE_POSTER": "I will generate an illustrated poster for this repository via NanoBanana.",
        }
        return mapping.get(tool_name, "I will process your export request.")

    def build_export_hint(self, query: str) -> str:
        """Build a system-prompt hint controlling response verbosity based on
        detected export intent.

        - **Clear export intent** (e.g. "生成pdf报告"): tell the LLM to
          respond briefly — a 1-3 sentence acknowledgement is enough.
        - **Open-ended / recommendation** queries: tell the LLM to respond
          thoroughly and recommend a format.
        - **No export intent**: default length guidance.
        """
        plan = self.planner.plan(query=query)
        has_gen = self.planner.has_generation_intent(query)
        has_reasoning = any(
            m in (query or "").lower()
            for m in self.planner.reasoning_markers
        )

        # Clear, unambiguous export intent (no reasoning needed)
        if has_gen and not has_reasoning:
            return (
                "- The user has a clear export intent. Keep your response BRIEF "
                "(1-3 sentences: acknowledge what you will generate and for which repository). "
                "Then include the corresponding [ACTION:...] tag. Do NOT write a long analysis."
            )

        # Export intent + reasoning markers (e.g. "推荐最合适的格式导出")
        if has_gen and has_reasoning:
            return (
                "- The user wants you to analyze/recommend AND export. "
                "Provide a thorough analysis of the repository first, then recommend ONE of the "
                "four export formats (PDF/PPT/Video/Poster) with clear reasoning, and include "
                "the corresponding [ACTION:...] tag on the last line. "
                "Target 400-1500 characters of substantive content."
            )

        # No explicit export intent — default thorough response
        return (
            "- Target 400-1500 characters of substantive content "
            "(or equivalent in other languages), adjusting based on question complexity."
        )
