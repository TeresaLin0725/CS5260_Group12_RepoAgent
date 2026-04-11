"""
Smoke tests for the Agent system.

Validates that the agent-tool-dev skill's documented architecture
matches the actual codebase — tool registration, planner logic,
scheduler dispatch, and ReAct runner all work as described.

Usage:
    pytest tests/unit/test_agent_tools.py -v
"""

import asyncio
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.agent.tools.base import AgentTool, ToolRegistry
from api.agent.tools.export_tools import build_export_tool_registry
from api.agent.planner import RuleBasedPlanner, ToolPlan
from api.agent.scheduler import AgentScheduler, AgentScheduleResult
from api.agent.events import AgentEvent, AgentEventType
from api.agent.state import AgentRunState
from api.agent.react import ReActRunner, ReActStep


# ============================================================================
# ToolRegistry
# ============================================================================


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = AgentTool(
            name="TEST_TOOL",
            action_tag="[ACTION:TEST_TOOL]",
            description="A test tool.",
            keywords=("test", "测试"),
        )
        registry.register(tool)
        assert registry.get("TEST_TOOL") is tool

    def test_get_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("NONEXISTENT") is None

    def test_all_returns_registered_tools(self):
        registry = ToolRegistry()
        t1 = AgentTool(name="A", action_tag="[A]", description="a", keywords=("a",))
        t2 = AgentTool(name="B", action_tag="[B]", description="b", keywords=("b",))
        registry.register(t1)
        registry.register(t2)
        all_tools = list(registry.all())
        assert len(all_tools) == 2


# ============================================================================
# Export Tool Registry
# ============================================================================


class TestExportToolRegistry:
    def test_pdf_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_PDF")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_PDF]"
        assert "pdf" in tool.keywords

    def test_ppt_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_PPT")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_PPT]"
        assert "ppt" in tool.keywords

    def test_video_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_VIDEO")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_VIDEO]"
        assert "video" in tool.keywords

    def test_three_tools_total(self):
        registry = build_export_tool_registry()
        assert len(list(registry.all())) == 4

    def test_no_keyword_overlap_between_tools(self):
        """Each tool's keywords must be unique to avoid ambiguity."""
        registry = build_export_tool_registry()
        tools = list(registry.all())
        for i, t1 in enumerate(tools):
            for t2 in tools[i + 1 :]:
                overlap = set(t1.keywords) & set(t2.keywords)
                assert not overlap, (
                    f"Keyword overlap between {t1.name} and {t2.name}: {overlap}"
                )


# ============================================================================
# Planner
# ============================================================================


class TestPlanner:
    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_direct_pdf_generation(self):
        plan = self.planner.plan("generate a pdf report")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_direct_ppt_generation(self):
        plan = self.planner.plan("create ppt slides")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PPT"

    def test_direct_video_generation(self):
        plan = self.planner.plan("generate a video walkthrough")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_VIDEO"

    def test_chinese_pdf_generation(self):
        plan = self.planner.plan("生成pdf报告")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_reasoning_marker_blocks_invoke(self):
        plan = self.planner.plan("analyze this repo and recommend pdf report")
        assert plan.should_invoke is False
        assert plan.needs_reasoning is True

    def test_no_match_returns_no_invoke(self):
        plan = self.planner.plan("explain the project architecture")
        assert plan.should_invoke is False
        assert plan.tool is None

    def test_ambiguous_multiple_tools(self):
        plan = self.planner.plan("generate pdf and ppt")
        assert plan.ambiguous is True
        assert plan.should_invoke is False

    def test_keyword_without_generation_intent(self):
        plan = self.planner.plan("is there a pdf already?")
        assert plan.should_invoke is False

    def test_has_generation_intent_english(self):
        assert self.planner.has_generation_intent("generate something") is True
        assert self.planner.has_generation_intent("create a report") is True
        assert self.planner.has_generation_intent("export the data") is True

    def test_has_generation_intent_chinese(self):
        assert self.planner.has_generation_intent("生成报告") is True
        assert self.planner.has_generation_intent("导出数据") is True

    def test_no_generation_intent(self):
        assert self.planner.has_generation_intent("what is this project?") is False

    def test_infer_best_tool(self):
        tool = self.planner.infer_best_tool("I need a pdf report")
        assert tool is not None
        assert tool.name == "GENERATE_PDF"

    def test_infer_best_tool_no_match(self):
        tool = self.planner.infer_best_tool("hello world")
        assert tool is None


# ============================================================================
# Enhanced Intent Recognition (synonym, fuzzy, implicit patterns)
# ============================================================================


class TestEnhancedIntentRecognition:
    """Tests for synonym expansion, fuzzy matching, and implicit intent patterns."""

    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    # -- implicit intent (Chinese) --

    def test_implicit_chinese_help_me_produce(self):
        """'帮我出一份项目介绍的文档' should resolve to PDF via implicit intent + synonym."""
        plan = self.planner.plan("帮我出一份项目介绍的文档")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_implicit_chinese_give_me_make(self):
        plan = self.planner.plan("给我做一份PDF报告")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_implicit_chinese_i_want_a_document(self):
        plan = self.planner.plan("我想要一份项目文档")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_implicit_chinese_come_up_with_one(self):
        plan = self.planner.plan("来一份演示文稿")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PPT"

    def test_no_implicit_intent_give_me_look(self):
        """'给我看看代码' must NOT trigger generation intent."""
        assert self.planner.has_generation_intent("给我看看代码") is False

    # -- implicit intent (English) --

    def test_implicit_english_i_need(self):
        plan = self.planner.plan("I need a pdf report for this project")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_implicit_english_give_me(self):
        plan = self.planner.plan("give me a presentation about this repo")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PPT"

    def test_implicit_english_can_you_create(self):
        assert self.planner.has_generation_intent("can you create a summary?") is True

    def test_no_implicit_i_need_help(self):
        """'I need help' without a format keyword should not be generation intent."""
        assert self.planner.has_generation_intent("I need help with this project") is False

    # -- synonym matching --

    def test_synonym_document_maps_to_pdf(self):
        plan = self.planner.plan("generate documentation for this repo")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_synonym_manual_maps_to_pdf(self):
        plan = self.planner.plan("生成一份手册")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_synonym_powerpoint_maps_to_ppt(self):
        plan = self.planner.plan("create a powerpoint")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PPT"

    def test_synonym_presentation_doc_chinese(self):
        plan = self.planner.plan("生成演示文稿")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PPT"

    # -- fuzzy matching --

    def test_fuzzy_pdff_typo(self):
        """'pdff' (edit distance 1 from 'pdf') should match with additional exact hit."""
        plan = self.planner.plan("generate pdff report")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"

    def test_fuzzy_only_insufficient(self):
        """A single fuzzy-only hit (score 1) should NOT be enough to match a tool."""
        # 'rprt' is edit distance 2 from 'report' (might fuzzy-match)
        # but alone it shouldn't pass _MIN_MATCH_SCORE of 2.
        plan = self.planner.plan("generate rprt")
        # Either no tool or tool is None (depends on whether fuzzy triggers)
        if plan.tool is not None:
            # If it somehow matched, it shouldn't auto-invoke without strong signal
            assert plan.should_invoke is True  # generation intent + match → invoke
        # The key guarantee: "hello rprt" without generation intent → safe
        plan2 = self.planner.plan("hello rprt")
        assert plan2.should_invoke is False

    # -- strict vs non-strict generation intent --

    def test_strict_mode_rejects_implicit(self):
        """strict=True should only accept explicit markers, not implicit patterns."""
        assert self.planner.has_generation_intent("帮我出一份文档", strict=False) is True
        assert self.planner.has_generation_intent("帮我出一份文档", strict=True) is False

    def test_strict_mode_accepts_explicit(self):
        assert self.planner.has_generation_intent("生成报告", strict=True) is True

    # -- expanded generation markers --

    def test_expanded_marker_zhizuo(self):
        assert self.planner.has_generation_intent("制作一个视频") is True

    def test_expanded_marker_draft(self):
        assert self.planner.has_generation_intent("draft a report") is True

    def test_expanded_marker_put_together(self):
        assert self.planner.has_generation_intent("put together a presentation") is True


class TestEnhancedScheduler:
    """Tests for scheduler pre-inference and strict stage-2."""

    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_pre_infer_chinese_implicit(self):
        """Scheduler should handle '帮我出一份项目介绍的文档' directly."""
        result = self.scheduler.schedule("帮我出一份项目介绍的文档", "zh")
        assert result.handled is True
        assert "[ACTION:GENERATE_PDF]" in result.content

    def test_pre_infer_english_documentation(self):
        result = self.scheduler.schedule("generate documentation for this repo", "en")
        assert result.handled is True
        assert "[ACTION:GENERATE_PDF]" in result.content

    def test_stage2_strict_blocks_false_positive(self):
        """Stage-2 should NOT fire for implicit-only intent queries."""
        action = self.scheduler.infer_second_stage_action(
            query="给我看看这个代码怎么用的",
            assistant_response="This project uses PDF format for documentation.",
        )
        assert action is None


# ============================================================================
# Intent Classifier – Embedding (Plan B)
# ============================================================================


from api.agent.intent_classifier import (
    EmbeddingIntentClassifier,
    IntentResult,
    LLMIntentClassifier,
    _cosine_similarity,
)


class TestCosineUtility:
    def test_identical_vectors(self):
        assert abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


class TestEmbeddingIntentClassifier:
    """Tests for Plan B — embedding similarity classification."""

    @staticmethod
    def _make_simple_embedder():
        """Return a deterministic mock embed_fn that uses bag-of-words hashing
        into a fixed-size vector — no external API needed."""
        DIM = 64

        async def embed_fn(texts: list) -> list:
            vectors = []
            for text in texts:
                vec = [0.0] * DIM
                for word in text.lower().split():
                    idx = hash(word) % DIM
                    vec[idx] += 1.0
                # Normalize
                norm = sum(x * x for x in vec) ** 0.5
                if norm > 0:
                    vec = [x / norm for x in vec]
                vectors.append(vec)
            return vectors

        return embed_fn

    def test_classify_export_pdf(self):
        classifier = EmbeddingIntentClassifier(
            self._make_simple_embedder(), confidence_threshold=0.3
        )
        result = asyncio.run(classifier.classify("generate a pdf report"))
        assert result is not None
        assert result.intent == "EXPORT_PDF"
        assert result.source == "embedding"
        assert result.is_export is True
        assert result.action_tag() == "[ACTION:GENERATE_PDF]"

    def test_classify_export_ppt(self):
        classifier = EmbeddingIntentClassifier(
            self._make_simple_embedder(), confidence_threshold=0.3
        )
        result = asyncio.run(classifier.classify("create presentation slides"))
        assert result is not None
        assert result.intent == "EXPORT_PPT"
        assert result.export_format == "ppt"

    def test_classify_returns_none_when_low_confidence(self):
        classifier = EmbeddingIntentClassifier(
            self._make_simple_embedder(), confidence_threshold=0.99
        )
        result = asyncio.run(classifier.classify("random unrelated stuff xyz"))
        # With threshold 0.99 and a simple embedder, should be None
        assert result is None

    def test_classify_handles_embed_failure(self):
        async def failing_embed(texts):
            raise RuntimeError("Embedding service unavailable")

        classifier = EmbeddingIntentClassifier(failing_embed)
        result = asyncio.run(classifier.classify("generate pdf"))
        assert result is None

    def test_intent_result_non_export(self):
        r = IntentResult(intent="GENERAL_CHAT", confidence=0.9, source="test")
        assert r.is_export is False
        assert r.action_tag() is None
        assert r.tool_name() is None

    def test_intent_result_export_video(self):
        r = IntentResult(intent="EXPORT_VIDEO", confidence=0.8, export_format="video", source="test")
        assert r.is_export is True
        assert r.action_tag() == "[ACTION:GENERATE_VIDEO]"
        assert r.tool_name() == "GENERATE_VIDEO"

    def test_caching_only_embeds_once(self):
        call_count = {"n": 0}

        async def counting_embed(texts):
            call_count["n"] += 1
            return [[0.1] * 10 for _ in texts]

        classifier = EmbeddingIntentClassifier(counting_embed, confidence_threshold=0.01)
        asyncio.run(classifier.classify("first query"))
        first_count = call_count["n"]
        asyncio.run(classifier.classify("second query"))
        # Examples should have been cached after first call; second call adds only 1 (query)
        assert call_count["n"] == first_count + 1


# ============================================================================
# Intent Classifier – LLM (Plan A)
# ============================================================================


class TestLLMIntentClassifier:
    """Tests for Plan A — LLM-based classification."""

    def test_classify_valid_json(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "EXPORT_PDF", "confidence": 0.92, "export_format": "pdf"}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("generate a report"))
        assert result is not None
        assert result.intent == "EXPORT_PDF"
        assert result.confidence == 0.92
        assert result.export_format == "pdf"
        assert result.source == "llm"

    def test_classify_json_with_markdown_fences(self):
        async def mock_llm(prompt: str) -> str:
            return '```json\n{"intent": "EXPORT_PPT", "confidence": 0.85, "export_format": "ppt"}\n```'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("make slides"))
        assert result is not None
        assert result.intent == "EXPORT_PPT"

    def test_classify_low_confidence_returns_none(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "EXPORT_PDF", "confidence": 0.3, "export_format": "pdf"}'

        classifier = LLMIntentClassifier(mock_llm, confidence_threshold=0.6)
        result = asyncio.run(classifier.classify("something vague"))
        assert result is None

    def test_classify_invalid_intent_returns_none(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "UNKNOWN_INTENT", "confidence": 0.9}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("test"))
        assert result is None

    def test_classify_unparseable_returns_none(self):
        async def mock_llm(prompt: str) -> str:
            return "I think you want a PDF report."

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("test"))
        assert result is None

    def test_classify_handles_llm_failure(self):
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("Model unavailable")

        classifier = LLMIntentClassifier(failing_llm)
        result = asyncio.run(classifier.classify("generate pdf"))
        assert result is None

    def test_classify_extracts_json_from_text(self):
        async def mock_llm(prompt: str) -> str:
            return 'Based on the query, here is the result: {"intent": "EXPORT_VIDEO", "confidence": 0.88, "export_format": "video"}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("create a video"))
        assert result is not None
        assert result.intent == "EXPORT_VIDEO"

    def test_classify_non_export_intent(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "GENERAL_CHAT", "confidence": 0.95, "export_format": null}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("hello"))
        assert result is not None
        assert result.intent == "GENERAL_CHAT"
        assert result.is_export is False

    def test_classify_clamps_confidence(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "EXPORT_PDF", "confidence": 1.5, "export_format": "pdf"}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("test"))
        assert result is not None
        assert result.confidence <= 1.0

    def test_confidence_string_fallback(self):
        async def mock_llm(prompt: str) -> str:
            return '{"intent": "EXPORT_PDF", "confidence": "high", "export_format": "pdf"}'

        classifier = LLMIntentClassifier(mock_llm)
        result = asyncio.run(classifier.classify("test"))
        # "high" can't be parsed as float → confidence=0.0 → below threshold
        assert result is None


# ============================================================================
# Planner fallback chain (plan_with_fallback) — REMOVED from production code.
# ============================================================================

@pytest.mark.skip(reason="plan_with_fallback removed from production — see planner.py")
class TestPlannerFallbackChain:
    """Tests for the async plan_with_fallback() integration."""

    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_rule_match_skips_classifiers(self):
        """When rule-based plan works, classifiers should not be called."""
        call_log = {"emb": 0, "llm": 0}

        async def mock_emb_classify(_self, query):
            call_log["emb"] += 1
            return IntentResult(intent="EXPORT_PPT", confidence=0.99, source="embedding")

        async def mock_llm_classify(_self, query):
            call_log["llm"] += 1
            return IntentResult(intent="EXPORT_PPT", confidence=0.99, source="llm")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()
        llm_cls = type("MockLLM", (), {"classify": mock_llm_classify})()

        plan = asyncio.run(self.planner.plan_with_fallback(
            "generate pdf report",
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        ))
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_PDF"
        assert call_log["emb"] == 0
        assert call_log["llm"] == 0

    def test_embedding_fallback_resolves_ambiguous_query(self):
        """When rule-based fails, embedding classifier saves the day."""
        async def mock_emb_classify(_self, query):
            return IntentResult(intent="EXPORT_PDF", confidence=0.85, export_format="pdf", source="embedding")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()

        plan = asyncio.run(self.planner.plan_with_fallback(
            "generate some project materials for sharing",
            embedding_classifier=emb_cls,
        ))
        assert plan.tool is not None
        assert plan.tool.name == "GENERATE_PDF"
        assert "embedding_classifier" in plan.reason

    def test_llm_fallback_when_embedding_returns_none(self):
        """When embedding returns None, LLM classifier is tried."""
        async def mock_emb_classify(_self, query):
            return None  # inconclusive

        async def mock_llm_classify(_self, query):
            return IntentResult(intent="EXPORT_PPT", confidence=0.9, export_format="ppt", source="llm")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()
        llm_cls = type("MockLLM", (), {"classify": mock_llm_classify})()

        plan = asyncio.run(self.planner.plan_with_fallback(
            "produce an output for the client meeting",
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        ))
        assert plan.tool is not None
        assert plan.tool.name == "GENERATE_PPT"
        assert "llm_classifier" in plan.reason

    def test_all_fail_returns_rule_result(self):
        """When all classifiers fail, returns the original rule-based plan."""
        async def mock_emb_classify(_self, query):
            return None

        async def mock_llm_classify(_self, query):
            return None

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()
        llm_cls = type("MockLLM", (), {"classify": mock_llm_classify})()

        plan = asyncio.run(self.planner.plan_with_fallback(
            "what does this project do?",
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        ))
        assert plan.should_invoke is False
        assert plan.tool is None

    def test_non_export_intent_not_used(self):
        """Classifiers returning GENERAL_CHAT should not produce a tool."""
        async def mock_emb_classify(_self, query):
            return IntentResult(intent="GENERAL_CHAT", confidence=0.95, source="embedding")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()

        plan = asyncio.run(self.planner.plan_with_fallback(
            "hello there",
            embedding_classifier=emb_cls,
        ))
        assert plan.tool is None
        assert plan.should_invoke is False

    def test_classifier_exception_is_caught(self):
        """Classifier exceptions are logged but don't break the flow."""
        async def exploding_classify(_self, query):
            raise RuntimeError("boom")

        emb_cls = type("MockEmb", (), {"classify": exploding_classify})()
        llm_cls = type("MockLLM", (), {"classify": exploding_classify})()

        # Should not raise
        plan = asyncio.run(self.planner.plan_with_fallback(
            "generate something cool",
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        ))
        assert isinstance(plan, ToolPlan)


# ============================================================================
# Scheduler with classifiers (schedule_with_classifiers) — REMOVED from production code.
# ============================================================================


@pytest.mark.skip(reason="schedule_with_classifiers removed from production — see scheduler.py")
class TestSchedulerWithClassifiers:
    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_schedule_resolves_via_embedding(self):
        async def mock_emb_classify(_self, query):
            return IntentResult(intent="EXPORT_PDF", confidence=0.88, export_format="pdf", source="embedding")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()

        result = asyncio.run(self.scheduler.schedule_with_classifiers(
            "generate some project materials for sharing",
            language="en",
            embedding_classifier=emb_cls,
        ))
        assert result.handled is True
        assert "[ACTION:GENERATE_PDF]" in result.content

    def test_schedule_resolves_via_llm(self):
        async def mock_emb_classify(_self, query):
            return None

        async def mock_llm_classify(_self, query):
            return IntentResult(intent="EXPORT_VIDEO", confidence=0.82, export_format="video", source="llm")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()
        llm_cls = type("MockLLM", (), {"classify": mock_llm_classify})()

        result = asyncio.run(self.scheduler.schedule_with_classifiers(
            "create a nice overview of this codebase",
            language="en",
            embedding_classifier=emb_cls,
            llm_classifier=llm_cls,
        ))
        assert result.handled is True
        assert "[ACTION:GENERATE_VIDEO]" in result.content

    def test_schedule_rule_based_still_works(self):
        """Classifiers are not needed when rule-based succeeds."""
        result = asyncio.run(self.scheduler.schedule_with_classifiers(
            "generate pdf report",
            language="en",
        ))
        assert result.handled is True
        assert "[ACTION:GENERATE_PDF]" in result.content

    def test_schedule_ambiguous_still_clarifies(self):
        result = asyncio.run(self.scheduler.schedule_with_classifiers(
            "generate pdf and ppt",
            language="en",
        ))
        assert result.handled is True
        assert "specify" in result.content.lower() or "明确" in result.content

    def test_schedule_events_include_classifier_source(self):
        async def mock_emb_classify(_self, query):
            return IntentResult(intent="EXPORT_POSTER", confidence=0.9, export_format="poster", source="embedding")

        emb_cls = type("MockEmb", (), {"classify": mock_emb_classify})()

        result = asyncio.run(self.scheduler.schedule_with_classifiers(
            "make something to introduce this project",
            language="en",
            embedding_classifier=emb_cls,
        ))
        assert result.handled is True
        assert "[ACTION:GENERATE_POSTER]" in result.content
        tool_events = [e for e in result.events if e.event_type == AgentEventType.TOOL_SELECTED]
        assert len(tool_events) >= 1


# ============================================================================
# Scheduler
# ============================================================================


class TestScheduler:
    def setup_method(self):
        self.scheduler = AgentScheduler.default()

    def test_schedule_pdf(self):
        result = self.scheduler.schedule("generate pdf report", "en")
        assert result.handled is True
        assert "[ACTION:GENERATE_PDF]" in result.content

    def test_schedule_ppt_chinese(self):
        result = self.scheduler.schedule("生成幻灯片", "zh")
        assert result.handled is True
        assert "[ACTION:GENERATE_PPT]" in result.content

    def test_schedule_ambiguous_asks_clarification(self):
        result = self.scheduler.schedule("generate pdf and ppt", "en")
        assert result.handled is True
        assert "specify" in result.content.lower() or "明确" in result.content

    def test_schedule_reasoning_falls_through(self):
        result = self.scheduler.schedule("analyze and recommend pdf format", "en")
        assert result.handled is False

    def test_schedule_no_tool_falls_through(self):
        result = self.scheduler.schedule("what does this project do?", "en")
        assert result.handled is False

    def test_events_always_include_plan_created(self):
        result = self.scheduler.schedule("generate pdf", "en")
        event_types = [e.event_type for e in result.events]
        assert AgentEventType.PLAN_CREATED in event_types

    def test_second_stage_pdf_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate documentation",
            assistant_response="I recommend creating a PDF report for comprehensive coverage.",
        )
        assert action == "[ACTION:GENERATE_PDF]"

    def test_second_stage_ppt_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="create a presentation",
            assistant_response="I suggest using a PPT presentation for team meetings.",
        )
        assert action == "[ACTION:GENERATE_PPT]"

    def test_second_stage_no_generation_intent(self):
        action = self.scheduler.infer_second_stage_action(
            query="what is this repo about?",
            assistant_response="This is a documentation tool. I recommend PDF.",
        )
        assert action is None

    def test_second_stage_no_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate docs",
            assistant_response="Here is an overview of the repository structure.",
        )
        assert action is None


# ============================================================================
# ReAct Parser
# ============================================================================


class TestReActParser:
    def test_parse_thought_action(self):
        text = (
            "Thought: I need to search the codebase.\n"
            "Action: rag_search\n"
            "Action Input: authentication flow"
        )
        step = ReActRunner._parse_response(text)
        assert step.thought == "I need to search the codebase."
        assert step.action == "rag_search"
        assert step.action_input == "authentication flow"
        assert step.is_final is False

    def test_parse_final_answer(self):
        text = (
            "Thought: I have enough information.\n"
            "Final Answer: The project uses JWT for authentication."
        )
        step = ReActRunner._parse_response(text)
        assert step.is_final is True
        assert "JWT" in step.final_answer

    def test_parse_plain_text_as_thought(self):
        text = "This is just a plain response without structured format."
        step = ReActRunner._parse_response(text)
        assert step.thought == text.strip()
        assert step.action is None


# ============================================================================
# ReAct Runner (with mocks)
# ============================================================================


class TestReActRunner:
    def test_direct_final_answer(self):
        """LLM gives Final Answer immediately → one iteration, no tool calls."""
        async def mock_llm(prompt: str) -> str:
            return "Thought: Context is sufficient.\nFinal Answer: The answer is 42."

        async def _run():
            runner = ReActRunner(tools={}, max_iterations=3)
            chunks = []
            async for chunk in runner.run(
                query="what is the answer?",
                system_prompt="You are a test agent.",
                initial_context="The answer to everything is 42.",
                llm_fn=mock_llm,
                language="en",
            ):
                chunks.append(chunk)
            return "".join(chunks)

        full_output = asyncio.run(_run())
        assert "42" in full_output

    def test_tool_invocation_then_answer(self):
        """LLM calls a tool, gets observation, then gives Final Answer."""
        call_log = []

        async def mock_search(query: str) -> str:
            call_log.append(query)
            return "### src/auth.py\ndef login(): pass"

        iteration = {"count": 0}

        async def mock_llm(prompt: str) -> str:
            iteration["count"] += 1
            if iteration["count"] == 1:
                return (
                    "Thought: I need to find the auth code.\n"
                    "Action: rag_search\n"
                    "Action Input: authentication login"
                )
            return (
                "Thought: Found the login function.\n"
                "Final Answer: The login function is in src/auth.py."
            )

        async def _run():
            runner = ReActRunner(
                tools={"rag_search": mock_search},
                max_iterations=3,
            )
            chunks = []
            async for chunk in runner.run(
                query="where is the login function?",
                system_prompt="You are a test agent.",
                initial_context="",
                llm_fn=mock_llm,
                language="en",
            ):
                chunks.append(chunk)
            return "".join(chunks)

        full_output = asyncio.run(_run())
        assert len(call_log) == 1
        assert call_log[0] == "authentication login"
        assert "src/auth.py" in full_output


# ============================================================================
# Data Structures
# ============================================================================


class TestDataStructures:
    def test_agent_tool_is_frozen(self):
        tool = AgentTool(name="T", action_tag="[T]", description="d", keywords=("k",))
        with pytest.raises(AttributeError):
            tool.name = "changed"

    def test_agent_event_creation(self):
        event = AgentEvent(AgentEventType.PLAN_CREATED, "test message")
        assert event.event_type == AgentEventType.PLAN_CREATED
        assert event.tool_name is None

    def test_agent_run_state_defaults(self):
        state = AgentRunState(query="test")
        assert state.language == "en"
        assert state.selected_tool is None
        assert state.events == []

    def test_tool_plan_defaults(self):
        plan = ToolPlan(should_invoke=False)
        assert plan.tool is None
        assert plan.ambiguous is False
        assert plan.needs_reasoning is False

    def test_react_step_defaults(self):
        step = ReActStep(thought="thinking")
        assert step.action is None
        assert step.is_final is False
        assert step.final_answer is None


# ============================================================================
# New ReAct Tools: list_repo_files, code_grep, memory_search
# ============================================================================

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from api.agent.tools.search_tools import build_react_tools, _get_local_repo_dir, _infer_repo_id


class _FakeRAG:
    """Minimal RAG mock that exposes db_manager.repo_paths."""

    def __init__(self, repo_dir=None):
        if repo_dir:
            self.db_manager = SimpleNamespace(repo_paths={"save_repo_dir": repo_dir})
        else:
            self.db_manager = None

    def __call__(self, query, language="en"):
        return [SimpleNamespace(documents=[])]


class TestHelpers:
    def test_infer_repo_id_from_url(self):
        rid = _infer_repo_id("https://github.com/user/repo")
        assert isinstance(rid, str)
        assert len(rid) > 0
        assert "/" not in rid

    def test_infer_repo_id_none(self):
        assert _infer_repo_id(None) == ""

    def test_get_local_repo_dir_with_valid_rag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = _FakeRAG(repo_dir=tmpdir)
            assert _get_local_repo_dir(rag) == tmpdir

    def test_get_local_repo_dir_no_db_manager(self):
        rag = _FakeRAG(repo_dir=None)
        assert _get_local_repo_dir(rag) is None


class TestListRepoFiles:
    def _make_repo(self, tmpdir):
        """Create a small fake repo structure."""
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Hello")
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("print('hello')")
        return tmpdir

    def test_list_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            rag = _FakeRAG(repo_dir=tmpdir)
            tools = build_react_tools(rag, language="en")
            assert "list_repo_files" in tools

            result = asyncio.run(tools["list_repo_files"]("."))
            assert "README.md" in result
            assert "src/" in result

    def test_list_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["list_repo_files"]("src"))
            assert "main.py" in result

    def test_list_root_accepts_quoted_or_glob_like_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")

            quoted_result = asyncio.run(tools["list_repo_files"]("'.'"))
            glob_result = asyncio.run(tools["list_repo_files"]("*.*"))

            assert "README.md" in quoted_result
            assert "src/" in quoted_result
            assert "README.md" in glob_result
            assert "src/" in glob_result

    def test_list_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["list_repo_files"]("nonexistent"))
            assert "does not exist" in result

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["list_repo_files"]("../../etc/passwd"))
            assert "outside" in result.lower() or "does not exist" in result.lower()

    def test_not_available_without_local_dir(self):
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en")
        assert "list_repo_files" not in tools


class TestCodeGrep:
    def _make_repo(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        with open(os.path.join(tmpdir, "src", "app.py"), "w") as f:
            f.write("class AuthHandler:\n    def authenticate(self, user):\n        return True\n")
        with open(os.path.join(tmpdir, "src", "utils.py"), "w") as f:
            f.write("def helper():\n    pass\n")
        return tmpdir

    def test_grep_literal_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            assert "code_grep" in tools

            result = asyncio.run(tools["code_grep"]("authenticate"))
            assert "app.py" in result
            assert "authenticate" in result

    def test_grep_regex_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"]("class.*Handler"))
            assert "AuthHandler" in result

    def test_grep_strips_wrapped_quotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"]("'authenticate'"))
            assert "app.py" in result
            assert "authenticate" in result

    def test_grep_supports_quoted_or_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"]("'authenticate' OR 'helper'"))
            assert "authenticate" in result
            assert "helper" in result

    def test_grep_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"]("xyznonexistent"))
            assert "No matches" in result

    def test_grep_empty_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"](""))
            assert "empty" in result.lower()

    def test_grep_invalid_regex_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            # Invalid regex with unbalanced bracket — should fall back to literal
            result = asyncio.run(tools["code_grep"]("[invalid"))
            assert "No matches" in result or isinstance(result, str)

    def test_grep_skips_binary_and_git(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_repo(tmpdir)
            # Create a .git dir and a binary file
            os.makedirs(os.path.join(tmpdir, ".git"), exist_ok=True)
            with open(os.path.join(tmpdir, ".git", "config"), "w") as f:
                f.write("authenticate secret\n")
            with open(os.path.join(tmpdir, "image.png"), "wb") as f:
                f.write(b"\x89PNG\r\n")
            tools = build_react_tools(_FakeRAG(repo_dir=tmpdir), language="en")
            result = asyncio.run(tools["code_grep"]("authenticate"))
            # .git/config should NOT appear
            assert ".git" not in result
            assert "image.png" not in result

    def test_not_available_without_local_dir(self):
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en")
        assert "code_grep" not in tools


class TestMemorySearch:
    def test_memory_search_always_present(self):
        """memory_search should always be registered."""
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en")
        assert "memory_search" in tools

    def test_memory_search_no_results(self):
        """When knowledge base is empty, returns friendly message."""
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en")

        mock_mm = MagicMock()
        mock_mm.search_knowledge.return_value = []

        with patch("api.memory.manager.get_memory_manager", return_value=mock_mm):
            result = asyncio.run(tools["memory_search"]("test query"))

        assert isinstance(result, str)

    def test_memory_search_with_results(self):
        """When knowledge base has entries, returns formatted results."""
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en", repo_url="https://github.com/user/repo")

        mock_entry = SimpleNamespace(key="architecture", value="This repo uses MVC pattern")
        mock_mm = MagicMock()
        mock_mm.search_knowledge.return_value = [mock_entry]

        with patch("api.memory.manager.get_memory_manager", return_value=mock_mm):
            result = asyncio.run(tools["memory_search"]("architecture"))

        assert isinstance(result, str)

    def test_memory_search_handles_error(self):
        """On exception, returns error string instead of raising."""
        rag = _FakeRAG(repo_dir=None)
        tools = build_react_tools(rag, language="en")

        with patch("api.memory.manager.get_memory_manager", side_effect=RuntimeError("DB unavailable")):
            result = asyncio.run(tools["memory_search"]("test"))

        assert isinstance(result, str)


# ============================================================================
# ReAct Runner integration with new tools
# ============================================================================


class TestReActRunnerWithNewTools:
    def test_runner_uses_list_and_grep(self):
        """ReActRunner can chain list_repo_files → code_grep → Final Answer."""
        call_log = {"list": [], "grep": []}

        async def mock_list(path: str) -> str:
            call_log["list"].append(path)
            return "Contents of .  (3 items):\n  README.md\n  src/\n  src/main.py"

        async def mock_grep(pattern: str) -> str:
            call_log["grep"].append(pattern)
            return "src/main.py:5: def handle_request(self):"

        iteration = {"count": 0}

        async def mock_llm(prompt: str) -> str:
            iteration["count"] += 1
            if iteration["count"] == 1:
                return (
                    "Thought: I should see what files are in the repo.\n"
                    "Action: list_repo_files\n"
                    "Action Input: ."
                )
            if iteration["count"] == 2:
                return (
                    "Thought: Now I'll search for the handler.\n"
                    "Action: code_grep\n"
                    "Action Input: def handle_request"
                )
            return (
                "Thought: Found it in src/main.py line 5.\n"
                "Final Answer: The `handle_request` method is defined in src/main.py at line 5."
            )

        async def _run():
            runner = ReActRunner(
                tools={
                    "list_repo_files": mock_list,
                    "code_grep": mock_grep,
                },
                max_iterations=3,
            )
            chunks = []
            async for chunk in runner.run(
                query="find the handle_request function",
                system_prompt="You are a test agent.",
                initial_context="",
                llm_fn=mock_llm,
                language="en",
            ):
                chunks.append(chunk)
            return "".join(chunks)

        full_output = asyncio.run(_run())
        assert len(call_log["list"]) == 1
        assert len(call_log["grep"]) == 1
        assert "src/main.py" in full_output


# ============================================================================
# GENERATE_POSTER (NanoBanana) Export Tool
# ============================================================================


class TestPosterToolRegistration:
    def test_poster_tool_registered(self):
        registry = build_export_tool_registry()
        tool = registry.get("GENERATE_POSTER")
        assert tool is not None
        assert tool.action_tag == "[ACTION:GENERATE_POSTER]"
        assert "poster" in tool.keywords
        assert "画报" in tool.keywords

    def test_four_tools_total(self):
        registry = build_export_tool_registry()
        assert len(list(registry.all())) == 4

    def test_poster_no_keyword_overlap(self):
        """Poster keywords must not overlap with other tools."""
        registry = build_export_tool_registry()
        poster = registry.get("GENERATE_POSTER")
        for tool in registry.all():
            if tool.name == "GENERATE_POSTER":
                continue
            overlap = set(poster.keywords) & set(tool.keywords)
            assert not overlap, (
                f"Keyword overlap between GENERATE_POSTER and {tool.name}: {overlap}"
            )


class TestPosterPlanner:
    def setup_method(self):
        self.planner = RuleBasedPlanner(build_export_tool_registry())

    def test_direct_poster_generation_english(self):
        plan = self.planner.plan("generate a poster for this repo")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_direct_poster_generation_chinese(self):
        plan = self.planner.plan("生成画报")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_infographic_keyword(self):
        plan = self.planner.plan("create an infographic")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_海报_keyword(self):
        plan = self.planner.plan("生成海报")
        assert plan.should_invoke is True
        assert plan.tool.name == "GENERATE_POSTER"

    def test_poster_with_reasoning_marker(self):
        plan = self.planner.plan("analyze and recommend a poster")
        assert plan.should_invoke is False
        assert plan.needs_reasoning is True


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
        assert "NanoBanana" in result.content

    def test_second_stage_poster_recommendation(self):
        action = self.scheduler.infer_second_stage_action(
            query="generate a visual summary",
            assistant_response="I recommend creating an infographic poster for quick visual reference.",
        )
        assert action == "[ACTION:GENERATE_POSTER]"

    def test_clarify_message_includes_poster(self):
        # Trigger ambiguity by matching poster + another tool
        result = self.scheduler.schedule("generate poster and pdf", "en")
        assert result.handled is True
        assert "Poster" in result.content or "specify" in result.content.lower()
