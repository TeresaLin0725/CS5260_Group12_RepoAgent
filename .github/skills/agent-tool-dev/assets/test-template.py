"""
Template: Unit tests for a new agent tool.

INSTRUCTIONS:
  1. Copy this file to tests/unit/test_agent_tools.py (or rename as needed)
  2. Replace YOURNAME / your_tool / keyword placeholders
  3. Run: pytest tests/unit/test_agent_tools.py -v

This template covers both Export Tools and ReAct Tools.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Project root on sys.path for direct imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.agent.tools.base import AgentTool, ToolRegistry
from api.agent.tools.export_tools import build_export_tool_registry
from api.agent.planner import RuleBasedPlanner, ToolPlan
from api.agent.scheduler import AgentScheduler, AgentScheduleResult
from api.agent.events import AgentEventType


# ============================================================================
# SECTION A: Export Tool — Registration
# ============================================================================


class TestExportToolRegistration:
    """Verify the new tool is properly registered."""

    def test_tool_exists_in_registry(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_YOURNAME")
        assert tool is not None, "Tool GENERATE_YOURNAME not found in registry"

    def test_tool_action_tag(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_YOURNAME")
        assert tool.action_tag == "[ACTION:GENERATE_YOURNAME]"

    def test_tool_has_english_keywords(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_YOURNAME")
        assert any(
            kw for kw in tool.keywords if kw.isascii()
        ), "Tool must have at least one English keyword"

    def test_tool_has_chinese_keywords(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_YOURNAME")
        assert any(
            kw for kw in tool.keywords if not kw.isascii()
        ), "Tool must have at least one Chinese keyword"

    def test_no_keyword_overlap_with_existing_tools(self):
        """New tool keywords must not collide with existing tools."""
        registry = build_export_tool_registry()
        tools = list(registry.all())
        new_tool = registry.get("GENERATE_YOURNAME")
        assert new_tool is not None

        new_keywords = set(kw.lower() for kw in new_tool.keywords)
        for tool in tools:
            if tool.name == new_tool.name:
                continue
            existing_keywords = set(kw.lower() for kw in tool.keywords)
            overlap = new_keywords & existing_keywords
            assert not overlap, (
                f"Keyword overlap between {new_tool.name} and {tool.name}: {overlap}"
            )


# ============================================================================
# SECTION B: Export Tool — Planner
# ============================================================================


class TestPlannerWithNewTool:
    """Verify planner correctly handles queries for the new tool."""

    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_direct_generation_intent(self):
        """'generate + keyword' → should_invoke=True"""
        plan = self.planner.plan("generate english_keyword1 for this repo")
        assert plan.should_invoke is True
        assert plan.tool is not None
        assert plan.tool.name == "GENERATE_YOURNAME"

    def test_chinese_generation_intent(self):
        """Chinese generation marker + keyword → should_invoke=True"""
        plan = self.planner.plan("生成这个仓库的中文关键词1")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_YOURNAME"

    def test_reasoning_marker_blocks_direct_invoke(self):
        """'analyze + keyword' → needs_reasoning, not direct invoke"""
        plan = self.planner.plan("analyze and recommend english_keyword1")
        assert plan.should_invoke is False
        assert plan.needs_reasoning is True

    def test_keyword_without_generation_intent(self):
        """Keyword mentioned without 'generate/create' → no invoke"""
        plan = self.planner.plan("what is english_keyword1?")
        assert plan.should_invoke is False

    def test_no_match_for_unrelated_query(self):
        plan = self.planner.plan("explain the project architecture")
        assert plan.tool is None


# ============================================================================
# SECTION C: Export Tool — Scheduler
# ============================================================================


class TestSchedulerWithNewTool:
    """Verify scheduler dispatches the new tool correctly."""

    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_schedule_returns_action_tag(self):
        result = self.scheduler.schedule("generate english_keyword1", "en")
        assert result.handled is True
        assert "[ACTION:GENERATE_YOURNAME]" in result.content

    def test_schedule_chinese(self):
        result = self.scheduler.schedule("生成中文关键词1", "zh")
        assert result.handled is True
        assert "[ACTION:GENERATE_YOURNAME]" in result.content

    def test_events_contain_tool_selected(self):
        result = self.scheduler.schedule("generate english_keyword1", "en")
        event_types = [e.event_type for e in result.events]
        assert AgentEventType.TOOL_SELECTED in event_types

    def test_second_stage_inference(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate documentation for this repo",
            assistant_response="I recommend creating english_keyword1 for detailed coverage.",
        )
        assert action == "[ACTION:GENERATE_YOURNAME]"


# ============================================================================
# SECTION D: ReAct Tool — Async Callable
# Uncomment and adapt if adding a ReAct tool
# ============================================================================


# class MockRAG:
#     """Minimal mock for RAG instance."""
#     def __init__(self, documents=None, raise_error=False):
#         self._documents = documents or []
#         self._raise_error = raise_error
#
#     def __call__(self, query, language="en"):
#         if self._raise_error:
#             raise RuntimeError("Mock RAG error")
#         docs = [
#             SimpleNamespace(text=d["text"], meta_data=d.get("meta", {}))
#             for d in self._documents
#         ]
#         return [SimpleNamespace(documents=docs)]
#
#
# class TestReActToolCallable:
#     """Test the async tool in isolation."""
#
#     @pytest.mark.asyncio
#     async def test_your_tool_returns_result(self):
#         from api.agent.tools.search_tools import build_react_tools
#         mock_rag = MockRAG(documents=[
#             {"text": "sample code content", "meta": {"file_path": "src/main.py"}},
#         ])
#         tools = build_react_tools(mock_rag, language="en")
#         result = await tools["your_tool"]("test query")
#         assert isinstance(result, str)
#         assert len(result) > 0
#
#     @pytest.mark.asyncio
#     async def test_your_tool_handles_error(self):
#         from api.agent.tools.search_tools import build_react_tools
#         mock_rag = MockRAG(raise_error=True)
#         tools = build_react_tools(mock_rag, language="en")
#         result = await tools["your_tool"]("test query")
#         assert "error" in result.lower()
#         # Must return string, never raise
#         assert isinstance(result, str)


# ============================================================================
# SECTION E: ReAct Integration (optional)
# ============================================================================


# class TestReActRunnerWithNewTool:
#     @pytest.mark.asyncio
#     async def test_runner_invokes_new_tool(self):
#         from api.agent.react import ReActRunner
#
#         call_log = []
#
#         async def mock_tool(input_str: str) -> str:
#             call_log.append(input_str)
#             return "mock tool observation data"
#
#         async def mock_llm(prompt: str) -> str:
#             if "Observation" not in prompt:
#                 return (
#                     "Thought: I need to gather data.\n"
#                     "Action: your_tool\n"
#                     "Action Input: test input"
#                 )
#             return (
#                 "Thought: I have the data I need.\n"
#                 "Final Answer: Based on the observation, here is the answer."
#             )
#
#         runner = ReActRunner(
#             tools={"your_tool": mock_tool},
#             max_iterations=3,
#         )
#         chunks = []
#         async for chunk in runner.run(
#             query="test query",
#             system_prompt="You are a test agent.",
#             initial_context="",
#             llm_fn=mock_llm,
#             language="en",
#         ):
#             chunks.append(chunk)
#
#         assert len(call_log) == 1
#         assert call_log[0] == "test input"
