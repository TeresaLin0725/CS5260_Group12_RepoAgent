from dataclasses import dataclass, field
from typing import List, Optional

from api.agent.events import AgentEvent, AgentEventType
from api.agent.planner import RuleBasedPlanner
from api.agent.tools.export_tools import build_export_tool_registry


@dataclass
class AgentScheduleResult:
    handled: bool
    content: Optional[str] = None
    events: List[AgentEvent] = field(default_factory=list)


class AgentScheduler:
    def __init__(self, planner: RuleBasedPlanner):
        self.planner = planner

    @classmethod
    def default(cls) -> "AgentScheduler":
        registry = build_export_tool_registry()
        return cls(planner=RuleBasedPlanner(registry))

    def schedule(self, query: str, language: str = "en") -> AgentScheduleResult:
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

    def infer_second_stage_action(self, query: str, assistant_response: str) -> Optional[str]:
        """Infer a tool action tag after LLM explanation (stage-2)."""
        text = assistant_response or ""
        normalized = text.lower()
        if not self.planner.has_generation_intent(query):
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

    def _infer_from_recommendation(self, text: str) -> Optional[str]:
        """Detect format recommendation patterns in the LLM response."""
        normalized = (text or "").lower()
        # Map recommendation keywords to action tags
        pdf_signals = ("pdf", "报告", "技术报告", "technical report")
        ppt_signals = ("ppt", "演示", "幻灯片", "presentation", "slides")
        video_signals = ("视频", "video", "walkthrough")

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

        scores = {
            "[ACTION:GENERATE_PDF]": pdf_score,
            "[ACTION:GENERATE_PPT]": ppt_score,
            "[ACTION:GENERATE_VIDEO]": video_score,
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
            return "你的请求可能同时需要多种输出格式。请明确告诉我你要 `PDF`、`PPT` 还是 `Video`。"
        return "Your request matches multiple outputs. Please specify one: `PDF`, `PPT`, or `Video`."

    @staticmethod
    def _tool_preamble(tool_name: str, language: str) -> str:
        if (language or "").lower().startswith("zh"):
            mapping = {
                "GENERATE_PDF": "我将为该仓库生成一份 PDF 技术报告。",
                "GENERATE_PPT": "我将为该仓库生成一份 PPT 演示文稿。",
                "GENERATE_VIDEO": "我将为该仓库生成一个视频概览。",
            }
            return mapping.get(tool_name, "我将开始处理你的导出请求。")

        mapping = {
            "GENERATE_PDF": "I will generate a PDF technical report for this repository.",
            "GENERATE_PPT": "I will generate a PPT presentation for this repository.",
            "GENERATE_VIDEO": "I will generate a video overview for this repository.",
        }
        return mapping.get(tool_name, "I will process your export request.")
