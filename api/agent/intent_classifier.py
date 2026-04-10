"""
Intent classification backends for the agent planner.

Provides two complementary classifiers that run **only** when the
rule-based planner cannot determine the user's intent:

* **EmbeddingIntentClassifier** (Plan B) — compares the query embedding
  against pre-defined intent examples via cosine similarity.  Latency is
  dominated by a single embedding call (~50-100 ms).  No LLM token cost.

* **LLMIntentClassifier** (Plan A) — sends a structured classification
  prompt to a lightweight model (haiku / flash / nano).  Higher accuracy
  but incurs token cost and ~100-200 ms latency.

Both expose an ``async classify(query) -> IntentResult`` interface so
the planner can call them interchangeably.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Shared data structures
# ============================================================================


@dataclass
class IntentResult:
    """Outcome of an intent classification attempt."""

    intent: str  # e.g. "EXPORT_PDF", "GENERAL_CHAT", …
    confidence: float  # 0.0 – 1.0
    export_format: Optional[str] = None  # "pdf" | "ppt" | "video" | "poster" | None
    source: str = "unknown"  # "embedding" | "llm"

    @property
    def is_export(self) -> bool:
        return self.intent.startswith("EXPORT_")

    def action_tag(self) -> Optional[str]:
        _MAP = {
            "EXPORT_PDF": "[ACTION:GENERATE_PDF]",
            "EXPORT_PPT": "[ACTION:GENERATE_PPT]",
            "EXPORT_VIDEO": "[ACTION:GENERATE_VIDEO]",
            "EXPORT_POSTER": "[ACTION:GENERATE_POSTER]",
        }
        return _MAP.get(self.intent)

    def tool_name(self) -> Optional[str]:
        _MAP = {
            "EXPORT_PDF": "GENERATE_PDF",
            "EXPORT_PPT": "GENERATE_PPT",
            "EXPORT_VIDEO": "GENERATE_VIDEO",
            "EXPORT_POSTER": "GENERATE_POSTER",
        }
        return _MAP.get(self.intent)


# ============================================================================
# Plan B – Embedding-based intent classification
# ============================================================================


# Pre-defined intent → example sentences.
# The classifier embeds these once (lazily) and caches the vectors.
INTENT_EXAMPLES: Dict[str, Tuple[str, ...]] = {
    "EXPORT_PDF": (
        "generate a technical report",
        "export as pdf",
        "write a document",
        "create a pdf report",
        "make a documentation file",
        "produce a project report",
        "帮我出一份项目介绍的文档",
        "生成技术报告",
        "导出PDF文件",
        "做一份项目文档",
        "写一份报告",
        "create documentation for this repo",
        "I need a pdf report",
        "draft a technical manual",
    ),
    "EXPORT_PPT": (
        "make slides",
        "create presentation",
        "team demo",
        "build a slide deck",
        "generate ppt",
        "make a powerpoint",
        "prepare slides for a meeting",
        "做一份演示文稿",
        "生成幻灯片",
        "制作PPT",
        "给我做一份汇报材料",
    ),
    "EXPORT_VIDEO": (
        "create a video walkthrough",
        "generate a video overview",
        "make a video introduction",
        "produce a project video",
        "制作一个视频",
        "生成视频概览",
        "做一个项目介绍视频",
    ),
    "EXPORT_POSTER": (
        "create a poster",
        "generate an infographic",
        "make an illustrated poster",
        "design a visual summary",
        "海报制作",
        "生成画报",
        "做一份宣传海报",
        "create a pictorial overview",
    ),
    "SEARCH_CODE": (
        "find the authentication function",
        "where is the login handler",
        "search for database connection",
        "look up the API endpoint",
        "找到认证函数",
        "搜索数据库连接代码",
        "where is the WebSocket handler defined",
        "find where RAG is initialized",
        "locate the main entry point file",
        "show me where the config is loaded",
        "grep for the class definition",
        "which file contains the scheduler",
        "哪个文件里有认证逻辑",
        "找一下数据库初始化的代码",
    ),
    "EXPLAIN_CODE": (
        "explain how this module works",
        "what does this function do",
        "describe the architecture",
        "how does the auth flow work",
        "解释这个模块的作用",
        "这段代码是什么意思",
        "how does the HybridRetriever combine FAISS and BM25",
        "explain the ReAct loop implementation",
        "what is BM25 and how is it used here",
        "describe the architecture of this project",
        "how does the RRF algorithm work",
        "what is the purpose of the planner module",
        "how does memory consolidation work",
        "explain how the scheduler routes requests",
        "这个模块的工作原理是什么",
        "intent_classifier 的分类逻辑是什么",
        "调度器的工作流程是怎样的",
        "describe how the agent system processes queries",
    ),
    "COMPARE": (
        "compare FAISS and BM25",
        "what is the difference between embedding and LLM classifier",
        "which approach is better for retrieval",
        "compare the two retrieval methods",
        "对比一下这两种方法",
        "FAISS和BM25哪个效果更好",
        "compare ReAct vs direct answer",
        "what are the trade-offs between PDF and PPT",
    ),
    "GENERAL_CHAT": (
        "hello",
        "hi there",
        "what is this project about",
        "tell me about yourself",
        "how are you",
        "你好",
        "这个项目是做什么的",
        "thanks",
        "thank you",
        "what can you do",
        "what environment variables do I need",
        "how do I deploy this project",
        "what testing framework does this project use",
        "what database does this project use",
        "how do I get started",
        "怎么部署这个项目",
        "这个项目支持哪些大模型Provider",
        "can you help me understand this repo",
        "what programming language is used",
        "is there a quickstart guide",
    ),
}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingIntentClassifier:
    """Classify user intent via cosine similarity against cached example
    embeddings.

    Parameters
    ----------
    embed_fn : async (texts: list[str]) -> list[list[float]]
        Async function that returns embedding vectors for a batch of texts.
    confidence_threshold : float
        Minimum cosine similarity to accept a classification (default 0.75).
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], Awaitable[List[List[float]]]],
        *,
        confidence_threshold: float = 0.75,
    ):
        self._embed_fn = embed_fn
        self._confidence_threshold = confidence_threshold
        # Lazy-initialised caches
        self._example_embeddings: Optional[Dict[str, List[List[float]]]] = None
        self._example_texts: Optional[Dict[str, Tuple[str, ...]]] = None

    async def _ensure_cached(self) -> None:
        """Embed all intent examples once and cache the vectors."""
        if self._example_embeddings is not None:
            return

        self._example_embeddings = {}
        self._example_texts = dict(INTENT_EXAMPLES)

        # Flatten all examples into one batch for efficiency
        all_texts: List[str] = []
        intent_ranges: List[Tuple[str, int, int]] = []
        for intent, examples in INTENT_EXAMPLES.items():
            start = len(all_texts)
            all_texts.extend(examples)
            intent_ranges.append((intent, start, len(all_texts)))

        try:
            all_embeddings = await self._embed_fn(all_texts)
        except Exception:
            logger.warning("Failed to embed intent examples; embedding classifier disabled.", exc_info=True)
            self._example_embeddings = {}
            return

        for intent, start, end in intent_ranges:
            self._example_embeddings[intent] = all_embeddings[start:end]

    async def classify(self, query: str) -> Optional[IntentResult]:
        """Classify *query* by comparing its embedding against cached examples.

        Returns ``None`` if confidence is below threshold or if the
        embedding backend is unavailable.
        """
        await self._ensure_cached()
        if not self._example_embeddings:
            return None

        try:
            query_embeddings = await self._embed_fn([query])
            query_vec = query_embeddings[0]
        except Exception:
            logger.warning("Failed to embed query for intent classification.", exc_info=True)
            return None

        best_intent = "GENERAL_CHAT"
        best_score = -1.0
        second_score = -1.0

        for intent, example_vecs in self._example_embeddings.items():
            # Use maximum similarity across examples for this intent
            if not example_vecs:
                continue
            max_sim = max(_cosine_similarity(query_vec, ev) for ev in example_vecs)
            if max_sim > best_score:
                second_score = best_score
                best_score = max_sim
                best_intent = intent
            elif max_sim > second_score:
                second_score = max_sim

        if best_score < self._confidence_threshold:
            return None

        # Ambiguity guard: if top-2 are too close, decline
        if best_score - second_score < 0.05:
            logger.debug(
                "Embedding classifier: ambiguous (%.3f vs %.3f), declining.",
                best_score,
                second_score,
            )
            return None

        _EXPORT_FORMAT_MAP = {
            "EXPORT_PDF": "pdf",
            "EXPORT_PPT": "ppt",
            "EXPORT_VIDEO": "video",
            "EXPORT_POSTER": "poster",
        }

        return IntentResult(
            intent=best_intent,
            confidence=round(best_score, 4),
            export_format=_EXPORT_FORMAT_MAP.get(best_intent),
            source="embedding",
        )


# ============================================================================
# Plan A – LLM-based intent classification
# ============================================================================


_INTENT_CLASSIFIER_PROMPT = """\
You are an intent classifier for a code-repository assistant.
Classify the user's query into exactly ONE of the following intents:

  EXPORT_PDF     – user wants a PDF / document / technical report generated
  EXPORT_PPT     – user wants a PPT / slides / presentation generated
  EXPORT_VIDEO   – user wants a video overview generated
  EXPORT_POSTER  – user wants a poster / infographic / 画报 generated
  SEARCH_CODE    – user wants to find or look up code in the repository
  EXPLAIN_CODE   – user wants an explanation of code or architecture
  COMPARE        – user wants to compare options or approaches
  GENERAL_CHAT   – general greeting, off-topic, or informational question

User query: {query}

Respond with ONLY a JSON object (no markdown fences):
{{"intent": "<INTENT>", "confidence": <0.0-1.0>, "export_format": <null|"pdf"|"ppt"|"video"|"poster">}}
"""


class LLMIntentClassifier:
    """Classify user intent via a lightweight LLM call.

    Parameters
    ----------
    llm_fn : async (prompt: str) -> str
        Async callable that sends a prompt to a lightweight model
        (e.g. claude-3-haiku, gpt-5-nano, gemini-2.5-flash-lite)
        and returns the full response text.
    confidence_threshold : float
        Minimum confidence to accept (default 0.6).
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
        *,
        confidence_threshold: float = 0.6,
    ):
        self._llm_fn = llm_fn
        self._confidence_threshold = confidence_threshold

    _VALID_INTENTS = frozenset({
        "EXPORT_PDF",
        "EXPORT_PPT",
        "EXPORT_VIDEO",
        "EXPORT_POSTER",
        "SEARCH_CODE",
        "EXPLAIN_CODE",
        "COMPARE",
        "GENERAL_CHAT",
    })

    async def classify(self, query: str) -> Optional[IntentResult]:
        """Classify *query* by prompting the LLM with a structured prompt.

        Returns ``None`` if the LLM response is unparseable or confidence
        is below threshold.
        """
        prompt = _INTENT_CLASSIFIER_PROMPT.format(query=query)

        try:
            raw = await self._llm_fn(prompt)
        except Exception:
            logger.warning("LLM intent classifier call failed.", exc_info=True)
            return None

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> Optional[IntentResult]:
        """Parse the LLM response into an IntentResult."""
        text = (raw or "").strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Attempt brace extraction
            match = re.search(r"\{[^}]+\}", text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.debug("LLM classifier: unparseable response: %s", text[:200])
                    return None
            else:
                logger.debug("LLM classifier: no JSON found in: %s", text[:200])
                return None

        intent = (data.get("intent") or "").upper().strip()
        confidence = data.get("confidence", 0.0)
        export_format = data.get("export_format")

        if intent not in self._VALID_INTENTS:
            logger.debug("LLM classifier: unknown intent %r", intent)
            return None

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        if confidence < self._confidence_threshold:
            return None

        return IntentResult(
            intent=intent,
            confidence=round(confidence, 4),
            export_format=export_format if isinstance(export_format, str) else None,
            source="llm",
        )
