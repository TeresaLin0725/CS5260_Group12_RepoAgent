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

    Tier 1 — **Rule-based planner**: keyword / synonym / fuzzy scoring.
             Fastest (~0 ms), highest precision for export intents.
    Tier 2 — **Embedding classifier**: cosine similarity against cached
             intent examples.  ~50-100 ms, no LLM cost.
    Tier 3 — **LLM classifier**: structured prompt to a lightweight model.
             ~100-200 ms, highest recall.

    The tiers cascade: if tier-N produces a confident result, later tiers
    are skipped.  Non-export intents (SEARCH_CODE, EXPLAIN_CODE, etc.) are
    surfaced as metadata on ``AgentScheduleResult.intent_result`` so that
    downstream code (ReAct, DR) can use the hint.
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
        """Tier-1 only (synchronous). Used when classifiers are unavailable."""
        events = [AgentEvent(AgentEventType.PLAN_CREATED, "Created plan from user query.")]
        plan = self.planner.plan(query=query)

        if plan.ambiguous:
            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    "Multiple tools matched. Asking for clarification.",
                )
            )
            return AgentScheduleResult(
                handled=True,
                content=self._clarify_message(language),
                events=events,
            )

        if plan.needs_reasoning:
            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    "Tool intent detected but query needs reasoning first. Falling back to LLM response.",
                    tool_name=plan.tool.name if plan.tool else None,
                )
            )
            return AgentScheduleResult(handled=False, events=events)

        if not plan.should_invoke or plan.tool is None:
            if plan.tool is None and self.planner.has_generation_intent(query):
                pre_action = self._pre_infer_from_query(query)
                if pre_action:
                    tool_name = pre_action.replace("[ACTION:", "").replace("]", "")
                    events.append(
                        AgentEvent(
                            AgentEventType.TOOL_SELECTED,
                            f"Pre-inferred tool {tool_name} from query format hints.",
                            tool_name=tool_name,
                        )
                    )
                    return AgentScheduleResult(
                        handled=True,
                        content=f"{self._tool_preamble(tool_name, language)}\n{pre_action}",
                        events=events,
                    )

            events.append(
                AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    f"No direct tool execution. reason={plan.reason or 'no_tool_intent_match'}. Continue with normal model response.",
                )
            )
            return AgentScheduleResult(handled=False, events=events)

        events.append(
            AgentEvent(
                AgentEventType.TOOL_SELECTED,
                f"Selected tool {plan.tool.name}.",
                tool_name=plan.tool.name,
            )
        )
        return AgentScheduleResult(
            handled=True,
            content=f"{self._tool_preamble(plan.tool.name, language)}\n{plan.tool.action_tag}",
            events=events,
        )

    # ------------------------------------------------------------------
    # Full 3-tier scheduling (async)
    # ------------------------------------------------------------------

    async def schedule_with_intent(
        self,
        query: str,
        language: str = "en",
    ) -> AgentScheduleResult:
        """Run all three classification tiers, cascading on failure.

        Returns an ``AgentScheduleResult`` whose ``intent_result`` field
        carries the winning classification (or ``None``).
        """
        # ── Tier 1: Rule-based ─────────────────────────────────────
        tier1 = self.schedule(query=query, language=language)
        if tier1.handled:
            logger.info("Intent tier-1 (rule-based) handled query.")
            return tier1

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
            # All tiers failed to classify → fall through to ReAct
            return AgentScheduleResult(
                handled=False,
                events=tier1.events + [AgentEvent(
                    AgentEventType.TOOL_SKIPPED,
                    "All 3 intent tiers returned no confident match. Deferring to ReAct.",
                )],
            )

        # ── Act on the intent ──────────────────────────────────────
        if intent.is_export:
            action_tag = intent.action_tag()
            tool_name = intent.tool_name()
            if action_tag and tool_name:
                events = tier1.events + [AgentEvent(
                    AgentEventType.TOOL_SELECTED,
                    f"Tier-{2 if intent.source == 'embedding' else 3} classified as {intent.intent} "
                    f"(conf={intent.confidence:.2f}). Invoking {tool_name}.",
                    tool_name=tool_name,
                )]
                return AgentScheduleResult(
                    handled=True,
                    content=f"{self._tool_preamble(tool_name, language)}\n{action_tag}",
                    events=events,
                    intent_result=intent,
                )

        # Non-export intent → attach as metadata but don't handle
        return AgentScheduleResult(
            handled=False,
            events=tier1.events + [AgentEvent(
                AgentEventType.TOOL_SKIPPED,
                f"Intent classified as {intent.intent} (conf={intent.confidence:.2f}, "
                f"source={intent.source}). Not an export — deferring to ReAct.",
            )],
            intent_result=intent,
        )

    def infer_second_stage_action(self, query: str, assistant_response: str) -> Optional[str]:
        """Infer a tool action tag after LLM explanation (stage-2)."""
        text = assistant_response or ""
        normalized = text.lower()
        if not self.planner.has_generation_intent(query, strict=True):
            return None
        if "[action:" in normalized:
            return None
        if "error:" in normalized:
            return None

        inferred = self.planner.infer_best_tool(f"{query}\n{text}")
        if inferred:
            return inferred.action_tag

        # Fallback: detect recommendation language in LLM response
        # When the LLM explicitly recommends a format, infer the tool
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
                "GENERATE_PPT": "我将为该仓库生成一份 PPT 演示文稿。",
                "GENERATE_VIDEO": "我将为该仓库生成一个视频概览。",
                "GENERATE_POSTER": "我将通过 NanoBanana 为该仓库生成一份图文画报。",
            }
            return mapping.get(tool_name, "我将开始处理你的导出请求。")

        mapping = {
            "GENERATE_PDF": "I will generate a PDF technical report for this repository.",
            "GENERATE_PPT": "I will generate a PPT presentation for this repository.",
            "GENERATE_VIDEO": "I will generate a video overview for this repository.",
            "GENERATE_POSTER": "I will generate an illustrated poster for this repository via NanoBanana.",
        }
        return mapping.get(tool_name, "I will process your export request.")
