from dataclasses import dataclass
from typing import List, Optional

from api.agent.tools.base import AgentTool, ToolRegistry


@dataclass
class ToolPlan:
    should_invoke: bool
    tool: Optional[AgentTool] = None
    ambiguous: bool = False
    needs_reasoning: bool = False
    reason: Optional[str] = None


class RuleBasedPlanner:
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

    def has_generation_intent(self, query: str) -> bool:
        normalized = (query or "").lower()
        return any(marker in normalized for marker in self.generation_markers)

    def infer_best_tool(self, text: str) -> Optional[AgentTool]:
        normalized = (text or "").lower()
        scores = {}
        for tool in self.registry.all():
            score = 0
            for keyword in tool.keywords:
                if keyword.lower() in normalized:
                    score += 1
            scores[tool.name] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] <= 0:
            return None
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return self.registry.get(ranked[0][0])

    def plan(self, query: str) -> ToolPlan:
        normalized = (query or "").lower()
        has_generation_intent = self.has_generation_intent(query)
        matched_tools: List[AgentTool] = []
        for tool in self.registry.all():
            for keyword in tool.keywords:
                if keyword.lower() in normalized:
                    matched_tools.append(tool)
                    break

        if not matched_tools:
            return ToolPlan(should_invoke=False)
        if len(matched_tools) > 1:
            return ToolPlan(
                should_invoke=False,
                ambiguous=True,
                reason="matched_multiple_tools",
            )

        if any(marker in normalized for marker in self.reasoning_markers):
            return ToolPlan(
                should_invoke=False,
                tool=matched_tools[0],
                needs_reasoning=True,
                reason="query_requires_reasoning_before_tool_execution",
            )

        if not has_generation_intent:
            return ToolPlan(
                should_invoke=False,
                tool=matched_tools[0],
                reason="format_mentioned_without_explicit_generation_intent",
            )

        return ToolPlan(should_invoke=True, tool=matched_tools[0])
