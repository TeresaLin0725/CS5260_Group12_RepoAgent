"""
Template: Unit tests for the Gamma PPTX export tool.

INSTRUCTIONS:
  These tests should be integrated into tests/unit/test_agent_tools.py.
  Use this as a reference for what to test when extending the Gamma PPT tool.

Run: pytest tests/unit/test_agent_tools.py -k GammaPpt -v
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

class TestGammaPptToolRegistration:
    """Verify the Gamma PPT tool is registered with correct properties."""

    def test_gamma_ppt_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_GAMMA_PPT")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_GAMMA_PPT]"
        assert "gamma ppt" in tool.keywords
        assert "精美ppt" in tool.keywords

    def test_gamma_ppt_no_keyword_overlap(self):
        """Gamma PPT keywords must not overlap with PDF/PPT/Video/Poster."""
        registry = build_export_tool_registry()
        gamma = registry.get("GENERATE_GAMMA_PPT")
        for tool in registry.all():
            if tool.name == "GENERATE_GAMMA_PPT":
                continue
            overlap = set(gamma.keywords) & set(tool.keywords)
            assert not overlap, f"Overlap with {tool.name}: {overlap}"


# ============================================================================
# Planner Tests
# ============================================================================

class TestGammaPptPlanner:
    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_direct_gamma_ppt_english(self):
        plan = self.planner.plan("generate gamma ppt for this repo")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_GAMMA_PPT"

    def test_direct_gamma_ppt_chinese(self):
        plan = self.planner.plan("生成精美ppt")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_GAMMA_PPT"

    def test_ai_ppt_keyword(self):
        plan = self.planner.plan("create ai ppt")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_GAMMA_PPT"


# ============================================================================
# Scheduler Tests
# ============================================================================

class TestGammaPptScheduler:
    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_schedule_gamma_ppt_english(self):
        result = self.scheduler.schedule("generate gamma ppt", "en")
        assert result.handled is True
        assert "[ACTION:GENERATE_GAMMA_PPT]" in result.content

    def test_schedule_gamma_ppt_chinese(self):
        result = self.scheduler.schedule("生成精美ppt", "zh")
        assert result.handled is True
        assert "[ACTION:GENERATE_GAMMA_PPT]" in result.content

    def test_second_stage_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate a visual presentation",
            assistant_response="I recommend creating a gamma ppt for polished design.",
        )
        assert action == "[ACTION:GENERATE_GAMMA_PPT]"
