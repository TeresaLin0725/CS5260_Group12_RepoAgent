"""
Template: Unit tests for the poster export tool.

INSTRUCTIONS:
  These tests are already integrated into tests/unit/test_agent_tools.py.
  Use this as a reference for what to test when extending the poster tool.

Run: pytest tests/unit/test_agent_tools.py -k Poster -v
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.agent.tools.base import AgentTool, ToolRegistry
from api.agent.tools.export_tools import build_export_tool_registry
from api.agent.planner import RuleBasedPlanner, ToolPlan
from api.agent.scheduler import AgentScheduler, AgentScheduleResult
from api.agent.events import AgentEventType


# ============================================================================
# Registration Tests
# ============================================================================

class TestPosterToolRegistration:
    """Verify the poster tool is registered with correct properties."""

    def test_poster_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_POSTER")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_POSTER]"
        assert "poster" in tool.keywords
        assert "画报" in tool.keywords

    def test_poster_no_keyword_overlap(self):
        """Poster keywords must not overlap with PDF/PPT/Video."""
        registry = build_export_tool_registry()
        poster = registry.get("GENERATE_POSTER")
        for tool in registry.all():
            if tool.name == "GENERATE_POSTER":
                continue
            overlap = set(poster.keywords) & set(tool.keywords)
            assert not overlap, f"Overlap with {tool.name}: {overlap}"


# ============================================================================
# Planner Tests
# ============================================================================

class TestPosterPlanner:
    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_direct_poster_english(self):
        plan = self.planner.plan("generate a poster for this repo")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_direct_poster_chinese(self):
        plan = self.planner.plan("生成画报")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_infographic_keyword(self):
        plan = self.planner.plan("create an infographic")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_reasoning_marker_blocks_invoke(self):
        plan = self.planner.plan("analyze and recommend a poster")
        assert plan.should_invoke is False
        assert plan.needs_reasoning is True


# ============================================================================
# Scheduler Tests
# ============================================================================

class TestPosterScheduler:
    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_schedule_poster_english(self):
        result = self.scheduler.schedule("generate poster", "en")
        assert result.handled is True
        assert "[ACTION:GENERATE_POSTER]" in result.content

    def test_schedule_poster_chinese(self):
        result = self.scheduler.schedule("生成画报", "zh")
        assert result.handled is True
        assert "[ACTION:GENERATE_POSTER]" in result.content

    def test_second_stage_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate a visual summary",
            assistant_response="I recommend creating an infographic poster.",
        )
        assert action == "[ACTION:GENERATE_POSTER]"
