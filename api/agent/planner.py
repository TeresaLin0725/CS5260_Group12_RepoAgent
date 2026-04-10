import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from api.agent.tools.base import AgentTool, ToolRegistry

logger = logging.getLogger(__name__)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Synonym map: canonical tool keyword → tuple of synonyms.
# Checked when the keyword itself is absent from the query.
# ---------------------------------------------------------------------------
_TOOL_KEYWORD_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    # --- PDF ---
    "pdf": ("文档", "document", "documentation", "文件", "手册", "manual"),
    "report": ("报告书", "报表"),
    "technical report": ("技术文档", "tech report", "技术报告"),
    "报告": ("报告书", "报表"),
    "pdf报告": ("pdf文档", "pdf文件"),
    # --- PPT ---
    "ppt": ("pptx", "powerpoint"),
    "slides": ("slide deck",),
    "presentation": ("演示文稿",),
    "deck": ("slide deck",),
    "演示": ("展示", "演示文稿", "汇报"),
    "幻灯片": (),
    # --- Video ---
    "video": ("录像", "clip"),
    "walkthrough": ("walk-through",),
    "overview video": ("视频概览",),
    "视频": ("录像", "影片"),
    # --- Poster ---
    "poster": ("宣传图", "宣传画"),
    "illustrated poster": (),
    "pictorial": ("图文画报",),
    "infographic": ("信息图",),
    "画报": ("宣传画", "图文画报"),
    "海报": ("宣传图", "宣传海报"),
    "图文海报": (),
    "画报制作": ("海报制作",),
}

# ---------------------------------------------------------------------------
# Regex patterns that indicate *implicit* generation intent — i.e. the user
# wants something produced but does not use an explicit verb like "generate".
# ---------------------------------------------------------------------------
_IMPLICIT_INTENT_PATTERNS: Tuple[re.Pattern, ...] = (
    # Chinese
    re.compile(r"帮我(?:出|做|弄|搞|写|制作|准备)"),
    re.compile(r"给我(?:做|写|出|弄|搞|制作|准备)"),
    re.compile(r"(?:来一份|来个|搞一[个份]|弄一[个份]|做一[个份]|出一[个份]|写一[个份])"),
    re.compile(
        r"我(?:想要|需要|要)(?:一[个份])?.{0,10}"
        r"(?:文档|报告|演示|视频|海报|画报|幻灯片)"
    ),
    re.compile(r"(?:能否|能不能|可以|请).{0,15}(?:做|制作|出|写|生成|导出|创建)"),
    # English
    re.compile(
        r"\b(?:give me|get me)\b.{0,20}"
        r"\b(?:report|document|pdf|ppt|presentation|video|poster|slides)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i need|i want|i'd like)\b.{0,20}"
        r"\b(?:report|document|pdf|ppt|presentation|video|poster|slides)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prepare|draft|put together|come up with|write)\b"
        r".{0,30}\b(?:report|document|presentation|video|poster)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:can you|could you|please)\b.{0,30}"
        r"\b(?:make|create|generate|produce|build)\b",
        re.IGNORECASE,
    ),
)


@dataclass
class ToolPlan:
    should_invoke: bool
    tool: Optional[AgentTool] = None
    ambiguous: bool = False
    needs_reasoning: bool = False
    reason: Optional[str] = None


class RuleBasedPlanner:
    # Scoring weights
    _EXACT_WEIGHT = 3
    _SYNONYM_WEIGHT = 2
    _FUZZY_WEIGHT = 1
    # A tool must reach at least this score to be considered matched.
    # Prevents fuzzy-only false positives (e.g. "export" ↔ "report").
    _MIN_MATCH_SCORE = 2

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.generation_markers = (
            "generate",
            "create",
            "make",
            "export",
            "build",
            "produce",
            "生成",
            "导出",
            "创建",
            "产出",
            "做一份",
            "进行生成",
            "来生成",
            # --- expanded synonyms ---
            "制作",
            "编写",
            "输出",
            "转换",
            "准备",
            "draft",
            "write up",
            "put together",
        )
        # Queries containing these hints should go through LLM reasoning first,
        # instead of being short-circuited by deterministic scheduler output.
        self.reasoning_markers = (
            "introduce",
            "overview",
            "analyze",
            "analysis",
            "compare",
            "why",
            "which",
            "best",
            "recommend",
            "choose",
            "select",
            "先介绍",
            "介绍",
            "分析",
            "比较",
            "为什么",
            "哪个",
            "最合适",
            "推荐",
            "选择",
        )

    # ------------------------------------------------------------------
    # Generation intent detection
    # ------------------------------------------------------------------

    def has_generation_intent(self, query: str, *, strict: bool = False) -> bool:
        """Check whether *query* expresses intent to produce / export output.

        Parameters
        ----------
        strict : bool
            When *True*, only explicit keyword markers are considered (safe
            for stage-2 gating where false positives are costly).  When
            *False* (default), implicit regex patterns are also checked.
        """
        normalized = (query or "").lower()
        if any(marker in normalized for marker in self.generation_markers):
            return True
        if strict:
            return False
        return any(pat.search(normalized) for pat in _IMPLICIT_INTENT_PATTERNS)

    # ------------------------------------------------------------------
    # Weighted tool scoring
    # ------------------------------------------------------------------

    def _score_tool(self, tool: AgentTool, normalized_query: str) -> int:
        """Score *tool* against *normalized_query* using exact, synonym, and
        fuzzy matching layers."""
        score = 0
        for keyword in tool.keywords:
            kw = keyword.lower()
            # 1. Exact substring match
            if kw in normalized_query:
                score += self._EXACT_WEIGHT
                continue
            # 2. Synonym match
            synonyms = _TOOL_KEYWORD_SYNONYMS.get(kw, ())
            if any(syn.lower() in normalized_query for syn in synonyms):
                score += self._SYNONYM_WEIGHT
                continue
            # 3. Fuzzy match (ASCII keywords ≥ 3 chars only, avoids CJK noise)
            if kw.isascii() and len(kw) >= 3:
                threshold = 1 if len(kw) <= 4 else 2
                for word in normalized_query.split():
                    if (
                        abs(len(word) - len(kw)) <= threshold
                        and _levenshtein_distance(word, kw) <= threshold
                    ):
                        score += self._FUZZY_WEIGHT
                        break
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_best_tool(self, text: str) -> Optional[AgentTool]:
        normalized = (text or "").lower()
        scores = {}
        for tool in self.registry.all():
            scores[tool.name] = self._score_tool(tool, normalized)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < self._MIN_MATCH_SCORE:
            return None
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return self.registry.get(ranked[0][0])

    def plan(self, query: str) -> ToolPlan:
        normalized = (query or "").lower()
        has_generation_intent = self.has_generation_intent(query)

        # Score all tools
        scored: List[Tuple[AgentTool, int]] = []
        for tool in self.registry.all():
            s = self._score_tool(tool, normalized)
            if s >= self._MIN_MATCH_SCORE:
                scored.append((tool, s))

        if not scored:
            return ToolPlan(should_invoke=False)

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Disambiguation: if top two have equal scores, declare ambiguous
        if len(scored) > 1 and scored[0][1] == scored[1][1]:
            return ToolPlan(
                should_invoke=False,
                ambiguous=True,
                reason="matched_multiple_tools",
            )

        best_tool = scored[0][0]

        if not has_generation_intent:
            # Reasoning markers only matter when there's no explicit generation intent.
            if any(marker in normalized for marker in self.reasoning_markers):
                return ToolPlan(
                    should_invoke=False,
                    tool=best_tool,
                    needs_reasoning=True,
                    reason="query_requires_reasoning_before_tool_execution",
                )
            return ToolPlan(
                should_invoke=False,
                tool=best_tool,
                reason="format_mentioned_without_explicit_generation_intent",
            )

        # Explicit generation intent + single tool match → invoke directly.
        return ToolPlan(should_invoke=True, tool=best_tool)

    # ------------------------------------------------------------------
    # plan_with_fallback removed — was dead code (never called in
    # production; only used by the also-removed
    # schedule_with_classifiers).  See git history if needed.
    # ------------------------------------------------------------------
