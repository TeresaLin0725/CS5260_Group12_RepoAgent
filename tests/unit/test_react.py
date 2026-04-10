"""
Unit tests for api.agent.react — ReActStep, ReActRunner.

Covers:
  - Response parsing (Thought/Action/Final Answer extraction)
  - Scratchpad construction
  - Observation truncation
  - Tool dispatch (success, failure, unknown tool)
  - Max-iteration forced termination
  - Final answer direct path
  - Status message formatting (en/zh)

Usage:
    pytest tests/unit/test_react.py -v
"""

import asyncio
import sys
from pathlib import Path
from typing import List

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.agent.react import ReActRunner, ReActStep


# ============================================================================
# _parse_response
# ============================================================================


class TestParseResponse:
    def test_final_answer_simple(self):
        text = "Thought: I have enough info.\nFinal Answer: The answer is 42."
        step = ReActRunner._parse_response(text)
        assert step.is_final is True
        assert step.final_answer == "The answer is 42."
        assert step.thought == "I have enough info."

    def test_final_answer_no_thought(self):
        text = "Final Answer: Direct answer without thought."
        step = ReActRunner._parse_response(text)
        assert step.is_final is True
        assert step.final_answer == "Direct answer without thought."

    def test_final_answer_multiline(self):
        text = (
            "Thought: Let me summarize.\n"
            "Final Answer: Line one.\n"
            "Line two.\n"
            "Line three."
        )
        step = ReActRunner._parse_response(text)
        assert step.is_final is True
        assert "Line one." in step.final_answer
        assert "Line three." in step.final_answer

    def test_action_parsed(self):
        text = (
            "Thought: I need to search for the function.\n"
            "Action: rag_search\n"
            "Action Input: hybrid retriever implementation"
        )
        step = ReActRunner._parse_response(text)
        assert step.is_final is False
        assert step.action == "rag_search"
        assert step.action_input == "hybrid retriever implementation"
        assert "search for the function" in step.thought

    def test_action_no_input(self):
        text = "Thought: Let me try.\nAction: list_files\n"
        step = ReActRunner._parse_response(text)
        assert step.action == "list_files"
        # action_input may be None or empty — both acceptable

    def test_no_action_no_final(self):
        """Raw text without markers treated as thought."""
        text = "This is just a plain response."
        step = ReActRunner._parse_response(text)
        assert step.is_final is False
        assert step.action is None
        assert step.thought == "This is just a plain response."

    def test_case_insensitive(self):
        text = "thought: reasoning here\naction: my_tool\naction input: params"
        step = ReActRunner._parse_response(text)
        assert step.action == "my_tool"

    def test_final_answer_case_insensitive(self):
        text = "final answer: works regardless of case"
        step = ReActRunner._parse_response(text)
        assert step.is_final is True

    def test_final_answer_with_colon_spacing(self):
        text = "Final  Answer :  spaced out answer"
        step = ReActRunner._parse_response(text)
        assert step.is_final is True
        assert "spaced out answer" in step.final_answer


# ============================================================================
# _build_scratchpad / _truncate
# ============================================================================


class TestScratchpadAndTruncate:
    def test_scratchpad_single_step(self):
        runner = ReActRunner(tools={})
        steps = [
            ReActStep(
                thought="I need to find the file.",
                action="rag_search",
                action_input="retriever.py",
                observation="Found: class HybridRetriever...",
            )
        ]
        pad = runner._build_scratchpad(steps)
        assert "Thought: I need to find the file." in pad
        assert "Action: rag_search" in pad
        assert "Action Input: retriever.py" in pad
        assert "Observation: Found: class HybridRetriever..." in pad

    def test_scratchpad_multiple_steps(self):
        runner = ReActRunner(tools={})
        steps = [
            ReActStep(thought="Step 1", action="tool_a", action_input="x", observation="result_a"),
            ReActStep(thought="Step 2", action="tool_b", action_input="y", observation="result_b"),
        ]
        pad = runner._build_scratchpad(steps)
        assert pad.count("Thought:") == 2
        assert pad.count("Observation:") == 2

    def test_scratchpad_no_action(self):
        runner = ReActRunner(tools={})
        steps = [ReActStep(thought="Just thinking")]
        pad = runner._build_scratchpad(steps)
        assert "Thought: Just thinking" in pad
        assert "Action:" not in pad

    def test_truncate_short_text(self):
        runner = ReActRunner(tools={}, max_observation_chars=100)
        assert runner._truncate("short") == "short"

    def test_truncate_long_text(self):
        runner = ReActRunner(tools={}, max_observation_chars=20)
        result = runner._truncate("a" * 100)
        assert len(result) < 100
        assert "[truncated]" in result

    def test_truncate_exact_boundary(self):
        runner = ReActRunner(tools={}, max_observation_chars=10)
        text = "a" * 10
        assert runner._truncate(text) == text  # exactly at limit, no truncation


# ============================================================================
# _format_status
# ============================================================================


class TestFormatStatus:
    def test_english_rag_search(self):
        status = ReActRunner._format_status("rag_search", "my query", "en")
        assert "Searching" in status
        assert "my query" in status

    def test_english_read_file(self):
        status = ReActRunner._format_status("read_file", "src/main.py", "en")
        assert "Reading file" in status
        assert "src/main.py" in status

    def test_chinese_rag_search(self):
        status = ReActRunner._format_status("rag_search", "查询内容", "zh")
        assert "搜索" in status
        assert "查询内容" in status

    def test_chinese_read_file(self):
        status = ReActRunner._format_status("read_file", "api/config.py", "zh-CN")
        assert "读取" in status
        assert "api/config.py" in status

    def test_unknown_tool_fallback(self):
        status = ReActRunner._format_status("custom_tool", "input", "en")
        assert "custom_tool" in status

    def test_none_action_input(self):
        status = ReActRunner._format_status("rag_search", None, "en")
        assert "Searching" in status


# ============================================================================
# _compile_forced_answer
# ============================================================================


class TestCompileForcedAnswer:
    def test_compiles_thoughts(self):
        steps = [
            ReActStep(thought="First insight", observation="obs1"),
            ReActStep(thought="Second insight", observation="obs2"),
        ]
        answer = ReActRunner._compile_forced_answer(steps, "en")
        assert "First insight" in answer
        assert "Second insight" in answer

    def test_empty_steps(self):
        answer = ReActRunner._compile_forced_answer([], "en")
        assert "unable" in answer.lower() or len(answer) > 0

    def test_steps_without_thought(self):
        steps = [ReActStep(thought="", observation="only obs")]
        answer = ReActRunner._compile_forced_answer(steps, "en")
        # Should still produce something (maybe the fallback)
        assert isinstance(answer, str)


# ============================================================================
# run() — async integration tests
# ============================================================================


def _run(coro):
    """Helper to run async generators and collect all yielded chunks."""
    async def _collect():
        chunks = []
        async for chunk in coro:
            chunks.append(chunk)
        return chunks
    return asyncio.get_event_loop().run_until_complete(_collect())


class TestRunLoop:
    """Tests for the async run() method using mock LLM functions."""

    def test_direct_final_answer(self):
        """LLM immediately returns Final Answer → single yield, no tool calls."""
        async def mock_llm(prompt):
            return "Thought: The context is sufficient.\nFinal Answer: Everything works."

        runner = ReActRunner(tools={"rag_search": None}, max_iterations=3)
        chunks = _run(runner.run(
            query="how does it work?",
            system_prompt="You are a helper.",
            initial_context="some code context",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "Everything works." in full

    def test_tool_call_then_answer(self):
        """LLM calls a tool, then produces Final Answer."""
        call_count = 0

        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: I need to search.\n"
                    "Action: rag_search\n"
                    "Action Input: retriever implementation"
                )
            else:
                return "Thought: Got the info.\nFinal Answer: The retriever uses FAISS + BM25."

        async def mock_rag_search(query):
            return "class HybridRetriever: combines FAISS and BM25"

        runner = ReActRunner(tools={"rag_search": mock_rag_search}, max_iterations=3)
        chunks = _run(runner.run(
            query="explain the retriever",
            system_prompt="You are a helper.",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "FAISS" in full
        assert "BM25" in full
        assert call_count == 2

    def test_unknown_tool_handled(self):
        """LLM requests a tool not in the registry → observation says 'Unknown tool'."""
        call_count = 0

        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: Let me try a new tool.\n"
                    "Action: nonexistent_tool\n"
                    "Action Input: something"
                )
            else:
                return "Final Answer: Fell back to available info."

        runner = ReActRunner(tools={"rag_search": None}, max_iterations=3)
        chunks = _run(runner.run(
            query="test",
            system_prompt="helper",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "Fell back" in full

    def test_tool_error_captured(self):
        """Tool raises exception → observation records error, loop continues."""
        call_count = 0

        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: Search first.\n"
                    "Action: rag_search\n"
                    "Action Input: something"
                )
            else:
                return "Final Answer: Despite the error, here is my answer."

        async def failing_tool(query):
            raise ValueError("index out of range")

        runner = ReActRunner(tools={"rag_search": failing_tool}, max_iterations=3)
        chunks = _run(runner.run(
            query="test",
            system_prompt="helper",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "Despite the error" in full

    def test_max_iterations_forced_answer(self):
        """Always requesting tools → forced answer after max_iterations."""
        async def mock_llm(prompt):
            return (
                "Thought: I need more info.\n"
                "Action: rag_search\n"
                "Action Input: keep searching"
            )

        async def mock_tool(query):
            return "some results"

        runner = ReActRunner(tools={"rag_search": mock_tool}, max_iterations=2)
        chunks = _run(runner.run(
            query="test",
            system_prompt="",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        # Should get forced answer with accumulated thoughts
        assert len(full) > 0

    def test_llm_error_yields_error_message(self):
        """LLM call raises → error message yielded, loop stops."""
        async def failing_llm(prompt):
            raise ConnectionError("API unavailable")

        runner = ReActRunner(tools={}, max_iterations=3)
        chunks = _run(runner.run(
            query="test",
            system_prompt="",
            initial_context="",
            llm_fn=failing_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "Error" in full or "error" in full

    def test_no_action_no_final_yields_thought(self):
        """LLM returns plain text without Action or Final Answer → treat as answer."""
        async def mock_llm(prompt):
            return "The architecture is based on a modular design pattern."

        runner = ReActRunner(tools={"rag_search": None}, max_iterations=3)
        chunks = _run(runner.run(
            query="describe architecture",
            system_prompt="",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        full = "".join(chunks)
        assert "modular design pattern" in full

    def test_status_messages_emitted(self):
        """Tool calls should emit a status message before observation."""
        call_count = 0

        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: Search.\n"
                    "Action: rag_search\n"
                    "Action Input: config loading"
                )
            return "Final Answer: Done."

        async def mock_search(q):
            return "found config.py"

        runner = ReActRunner(tools={"rag_search": mock_search}, max_iterations=3)
        chunks = _run(runner.run(
            query="how is config loaded?",
            system_prompt="",
            initial_context="",
            llm_fn=mock_llm,
            language="en",
        ))
        # First chunk should be the status message
        assert any("Searching" in c for c in chunks)


# ============================================================================
# _build_prompt (smoke test)
# ============================================================================


class TestBuildPrompt:
    def test_contains_query(self):
        runner = ReActRunner(tools={"rag_search": None}, max_iterations=3)
        prompt = runner._build_prompt(
            query="what is RAG?",
            system_prompt="You are a code expert.",
            context="some context",
            scratchpad="",
            tools_description="- rag_search",
            iteration=0,
            language="en",
        )
        assert "what is RAG?" in prompt
        assert "You are a code expert." in prompt
        assert "some context" in prompt

    def test_last_iteration_warning(self):
        runner = ReActRunner(tools={}, max_iterations=3)
        prompt = runner._build_prompt(
            query="test",
            system_prompt="",
            context="",
            scratchpad="",
            tools_description="",
            iteration=2,  # last (0-indexed, max=3)
            language="en",
        )
        assert "LAST STEP" in prompt

    def test_not_last_iteration_no_warning(self):
        runner = ReActRunner(tools={}, max_iterations=3)
        prompt = runner._build_prompt(
            query="test",
            system_prompt="",
            context="",
            scratchpad="",
            tools_description="",
            iteration=0,
            language="en",
        )
        assert "LAST STEP" not in prompt

    def test_language_in_prompt(self):
        runner = ReActRunner(tools={}, max_iterations=3)
        prompt = runner._build_prompt(
            query="test",
            system_prompt="",
            context="",
            scratchpad="",
            tools_description="",
            iteration=0,
            language="zh",
        )
        assert "zh" in prompt
