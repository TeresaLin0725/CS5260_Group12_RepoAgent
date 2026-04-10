"""
Server-side Deep Research orchestrator.

Implements a multi-phase research loop:
  Phase 1 — Query Decomposition: break complex question into sub-questions
  Phase 2 — Iterative Research: for each sub-question, RAG + ReAct tool calls
  Phase 3 — Synthesis: merge all findings into a final comprehensive answer

The orchestrator streams structured ``ResearchEvent`` objects so that both
HTTP (SSE) and WebSocket transports can relay progress to the frontend.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class ResearchEventType(str, Enum):
    """Types of events emitted during deep research."""
    PLAN = "plan"                      # sub-question plan created
    ITERATION_START = "iteration_start"  # beginning of a research iteration
    TOOL_CALL = "tool_call"            # a ReAct tool is being invoked
    FINDING = "finding"                # partial finding text (streamable)
    ITERATION_END = "iteration_end"    # one iteration completed
    GAP_ANALYSIS = "gap_analysis"      # new gaps / follow-up questions
    SYNTHESIS_START = "synthesis_start" # starting final synthesis
    CONCLUSION = "conclusion"          # final answer text (streamable)
    ERROR = "error"                    # non-fatal error
    COMPLETE = "complete"              # research finished


@dataclass
class ResearchEvent:
    """A single event emitted by the orchestrator."""
    event_type: ResearchEventType
    data: str = ""
    iteration: int = 0
    sub_question: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Serialize to an SSE-compatible JSON line for streaming."""
        return json.dumps({
            "type": self.event_type.value,
            "data": self.data,
            "iteration": self.iteration,
            "sub_question": self.sub_question,
            "metadata": self.metadata,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sub-question model
# ---------------------------------------------------------------------------

class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    SKIPPED = "skipped"


@dataclass
class SubQuestion:
    id: int
    text: str
    priority: int = 1          # higher = more important
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    finding: str = ""
    dependency: Optional[int] = None  # id of prerequisite sub-question


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """You are a research planner. Given the user's question about a code repository, break it into 3-6 focused sub-questions that together fully answer the original question.

DECOMPOSITION STRATEGY — cover MULTIPLE DIMENSIONS, not just features:
1. **Mechanism**: HOW is it built? (entry points, core algorithms, key data structures)
2. **Data Flow**: HOW does data move through the system? (input → transformation → output, parameter passing)
3. **Design Rationale**: WHY was it designed this way? (trade-offs, constraints, patterns chosen over alternatives)
4. **Error Handling & Edge Cases**: WHAT happens when things go wrong? (fallbacks, validation, retry logic)
5. **Dependencies & Integration**: HOW does it connect to other components? (interfaces, contracts, shared state)

Rules:
- Each sub-question MUST require reading actual source code files (not README or docs alone) to answer
- Sub-questions should target DIFFERENT dimensions — avoid asking 3 "how does X work" questions
- At least one sub-question must be a "trace the execution path" question (e.g. "When a user does X, what is the exact call chain from A → B → C, including parameters passed and return values?")
- At least one sub-question must be a "design analysis" question (e.g. "WHY does module X use pattern Y instead of Z? What trade-offs does this create?")
- Good example: "How does the WebSocket handler in api/websocket_wiki.py process incoming chat requests — what is the function call chain from connection to response, including how errors are propagated?"
- Bad example: "What is the project's architecture?" (too vague, answerable from README)
- Order from most fundamental (understanding prerequisites) to most specific (deep analysis)
- If the question is simple enough, output just 1-2 sub-questions
- Respond ONLY with a valid JSON array of strings, no commentary

User question: {query}

Repository context (first 4000 chars):
{context_preview}

JSON array of sub-questions:"""

_GAP_ANALYSIS_PROMPT = """Given the original question and the findings so far, identify 0-2 remaining gaps that need investigation. If the question is fully answered, return an empty array.

Original question: {query}

Findings so far:
{findings_summary}

Respond ONLY with a JSON array of strings (gap descriptions). Empty array [] if no gaps remain:"""

_PRE_SYNTHESIS_PROMPT = """You are organizing research findings before writing a final analysis.

Original question: {query}

Research findings (organized by sub-question):
{all_findings}

Before writing the final answer, organize your thinking by answering these questions concisely:

1. CORE INSIGHT: What is the single most important discovery from all the findings? (1-2 sentences)
2. CONNECTIONS: Which findings reinforce each other? Which contradict or reveal tension? (2-3 sentences)
3. SURPRISES: What was unexpected or non-obvious in the code? (1-2 sentences)
4. EVIDENCE GAPS: What couldn't be fully determined despite investigation? (1 sentence)
5. NARRATIVE ARC: What's the best order to present these findings for maximum clarity? (numbered list of 3-5 themes)

Respond with a brief structured outline — NOT the final answer. Keep it under 800 characters."""

_SYNTHESIS_PROMPT = """You are an expert code analyst writing an in-depth technical analysis.

{base_prompt}

Original question: {query}

Pre-analysis outline (use this to structure your answer):
{outline}

Research findings (organized by sub-question):
{all_findings}

Key source code evidence collected during investigation:
{raw_evidence}

Write a thorough, ANALYTICAL answer that reads like a senior engineer explaining the codebase to a colleague.

Writing style:
- Write in a natural, flowing narrative — NOT a rigid template with formulaic headings
- Start by directly answering the core question in 2-3 sentences, then naturally deepen the analysis
- Use headings (## or ###) ONLY when transitioning to a genuinely different topic, not for every paragraph
- Between sections, write transition sentences that connect ideas (e.g. "This design choice directly affects how X handles...")
- Vary your paragraph structure — mix explanatory prose, code examples, and analytical observations
- Do NOT use headings like "Executive Summary", "结论", "总结" — these are filler. Just write the actual content.

Analytical depth:
- Go BEYOND describing what exists — analyze HOW components interact and WHY they are designed this way
- When showing code from the evidence section, explain what it REVEALS about the design, don't just paste it
- Trace concrete execution paths: "When a user does X, the request flows through A.method() which calls B.process(param), and the result is stored in C.field"
- Point out interesting patterns, trade-offs, or potential issues you noticed in the code
- Compare different parts of the codebase when they use different approaches for similar problems
- If the codebase has limitations or technical debt, mention them with evidence
- Highlight CONNECTIONS between different findings — show how design decisions in one area affect another
- If the pre-analysis identified contradictions or surprises, address them explicitly
- You MUST directly quote and analyze code snippets from the raw evidence section — they are verified source code

Evidence requirements:
- Every claim must be backed by a specific file path and function/class name in `inline code`
- Include 6-15 code snippets total (3-15 lines each) from the raw evidence — more is better for depth
- After each code snippet, explain what it reveals that isn't obvious from the code alone
- Use the raw evidence to show actual implementation details, not just summaries

Formatting:
- Aim for 8000-15000 characters of substantive analysis (this is a DEEP research answer, be thorough)
- Use markdown naturally: `inline code`, ```code blocks```, **bold** for emphasis, bullet lists where they help
- Do NOT include raw JSON, tool output, debugging artifacts, or literal \\n
- Do NOT mention the research process, sub-questions, iterations, or the pre-analysis outline"""

_SUBQ_RESEARCH_PROMPT = """You are researching a specific aspect of a code repository.

{base_prompt}

Focus question: {sub_question}
(Part of the broader question: {original_query})

{previous_findings}

CRITICAL: You MUST use tools (rag_search, read_file, read_function, find_references, list_repo_files, code_grep) to read ACTUAL SOURCE CODE before answering.
Do NOT answer based solely on README, documentation files, or initial context. Dig into .py, .ts, .js, .java (etc.) files.

INVESTIGATION STRATEGY — adapt based on the type of question:
- For "How does X work?" → Read the entry-point file, then trace the call chain 2-3 levels deep. Show data transformations at each step. Use `read_function` to extract specific function bodies, and `find_references` to understand callers.
- For "Why is X designed this way?" → Compare with alternative approaches visible in the code, look for comments explaining rationale, analyze what constraints the design addresses (concurrency? extensibility? performance?).
- For "What happens when X fails/errors?" → Find error handlers, fallback logic, validation checks, retry mechanisms. Trace the unhappy path through the code.
- For "How do X and Y interact?" → Find the interface/contract between them, show BOTH sides of the integration point (the caller AND the callee), identify shared state or data structures.

What a GOOD answer looks like:
- "In `api/websocket_wiki.py`, the function `handle_websocket_chat()` (line ~140) accepts a WebSocket connection and parses the incoming JSON into a `ChatCompletionRequest`. It then calls `request_rag()` to retrieve context, passing the query through `_infer_language_code_from_query()` first. The key data structure is..."

What a BAD answer looks like:
- "The project uses WebSocket for real-time communication. The architecture includes..." (vague, no code evidence)

DEPTH REQUIREMENTS:
- Read at least 2 actual source code files using tools before writing your answer
- For every claim, cite the exact file path and function/class/variable name
- Include 2-4 short code snippets (3-10 lines) showing the most revealing parts
- After each snippet, explain what it REVEALS about the design — not just what it does, but WHY it matters
- Trace at least 2 levels of call chain: A() → B(params) → C(result) with actual parameter names
- Point out at least one interesting pattern, trade-off, or potential issue in the code
- Aim for 2000-4000 characters of evidence-backed analysis"""


class DeepResearchOrchestrator:
    """
    Server-side orchestrator for multi-phase deep research.

    Usage::

        orchestrator = DeepResearchOrchestrator(tools=mcp_tools, max_iterations=5)
        async for event in orchestrator.run(query, rag_context, llm_fn, ...):
            await send_to_client(event)
    """

    def __init__(
        self,
        tools: Dict[str, Callable[[str], Awaitable[str]]],
        max_iterations: int = 5,
        max_sub_questions: int = 7,
    ):
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_sub_questions = max_sub_questions

    # ------------------------------------------------------------------ run
    async def run(
        self,
        query: str,
        base_system_prompt: str,
        initial_context: str,
        llm_fn: Callable[[str], Awaitable[str]],
        language: str = "en",
        trace: Any = None,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Execute the full deep research pipeline, yielding events.

        Args:
            query: The user's original question.
            base_system_prompt: Repository-aware system prompt (role, repo info).
            initial_context: Pre-retrieved RAG context text.
            llm_fn: ``async (prompt) -> str`` — provider-agnostic LLM call.
            language: Target response language code.
            trace: Optional Langfuse trace/span for observability.
        """
        from api.tracing import get_tracer
        tracer = get_tracer()

        _start = time.time()
        context_preview = (initial_context or "")[:4000]

        # ============================================================
        # Phase 1: Query Decomposition
        # ============================================================
        logger.info("DeepResearch Phase 1: decomposing query")
        sub_questions = await self._decompose(query, context_preview, llm_fn)

        plan_text = "\n".join(f"{i+1}. {sq.text}" for i, sq in enumerate(sub_questions))
        yield ResearchEvent(
            event_type=ResearchEventType.PLAN,
            data=plan_text,
            metadata={"sub_question_count": len(sub_questions)},
        )

        # ============================================================
        # Phase 2: Iterative Research
        # ============================================================
        all_findings: Dict[int, str] = {}  # sq.id → finding
        all_observations: Dict[int, List[Dict[str, str]]] = {}  # sq.id → raw tool outputs
        accumulated_context = initial_context or ""

        for iteration in range(self.max_iterations):
            pending = [sq for sq in sub_questions if sq.status == SubQuestionStatus.PENDING]
            if not pending:
                logger.info("DeepResearch: all sub-questions resolved at iteration %d", iteration)
                break

            # Pick the highest-priority pending sub-question
            current_sq = max(pending, key=lambda sq: sq.priority)
            current_sq.status = SubQuestionStatus.IN_PROGRESS

            yield ResearchEvent(
                event_type=ResearchEventType.ITERATION_START,
                iteration=iteration + 1,
                sub_question=current_sq.text,
                metadata={"pending_count": len(pending), "sub_question_id": current_sq.id},
            )

            # Build previous findings summary for context
            prev_findings = ""
            if all_findings:
                parts = []
                for sq in sub_questions:
                    if sq.id in all_findings:
                        parts.append(f"### {sq.text}\n{all_findings[sq.id]}")
                prev_findings = "\n\n".join(parts)

            # Run ReAct-style research for this sub-question
            # Wrap tools to record raw observations (actual file contents, search results)
            sq_observations: List[Dict[str, str]] = []
            all_observations[current_sq.id] = sq_observations
            recording_tools = self._make_recording_tools(sq_observations)

            finding_parts: List[str] = []
            try:
                async for chunk in self._research_sub_question(
                    sub_question=current_sq.text,
                    original_query=query,
                    base_prompt=base_system_prompt,
                    context=accumulated_context,
                    previous_findings=prev_findings,
                    llm_fn=llm_fn,
                    language=language,
                    iteration=iteration,
                    trace=trace,
                    tools_override=recording_tools,
                ):
                    if chunk.event_type == ResearchEventType.TOOL_CALL:
                        yield chunk
                    elif chunk.event_type == ResearchEventType.FINDING:
                        finding_parts.append(chunk.data)
                        yield chunk
                    elif chunk.event_type == ResearchEventType.ERROR:
                        yield chunk
            except Exception as exc:
                logger.error("DeepResearch error on sub-question '%s': %s", current_sq.text, exc)
                yield ResearchEvent(
                    event_type=ResearchEventType.ERROR,
                    data=str(exc),
                    iteration=iteration + 1,
                    sub_question=current_sq.text,
                )

            finding_text = "".join(finding_parts)
            current_sq.finding = finding_text
            current_sq.status = SubQuestionStatus.RESOLVED
            all_findings[current_sq.id] = finding_text

            # Compress accumulated context to avoid token explosion
            accumulated_context = self._compress_context(
                initial_context, all_findings, sub_questions
            )

            yield ResearchEvent(
                event_type=ResearchEventType.ITERATION_END,
                iteration=iteration + 1,
                sub_question=current_sq.text,
                data=finding_text[:500] + "..." if len(finding_text) > 500 else finding_text,
            )

            # Gap analysis: should we add new sub-questions?
            if iteration < self.max_iterations - 1 and len(sub_questions) < self.max_sub_questions:
                new_gaps = await self._analyze_gaps(
                    query, all_findings, sub_questions, llm_fn
                )
                if new_gaps:
                    gap_texts = [g.text for g in new_gaps]
                    sub_questions.extend(new_gaps)
                    yield ResearchEvent(
                        event_type=ResearchEventType.GAP_ANALYSIS,
                        iteration=iteration + 1,
                        data="\n".join(gap_texts),
                        metadata={"new_gaps": len(new_gaps)},
                    )
                elif not any(sq.status == SubQuestionStatus.PENDING for sq in sub_questions):
                    # No gaps and nothing pending → done
                    break

        # ============================================================
        # Phase 3: Synthesis
        # ============================================================
        yield ResearchEvent(event_type=ResearchEventType.SYNTHESIS_START)

        logger.info("DeepResearch Phase 3: synthesizing final answer")
        final_answer = await self._synthesize(
            query=query,
            base_prompt=base_system_prompt,
            all_findings=all_findings,
            all_observations=all_observations,
            sub_questions=sub_questions,
            llm_fn=llm_fn,
            language=language,
            trace=trace,
        )

        yield ResearchEvent(
            event_type=ResearchEventType.CONCLUSION,
            data=final_answer,
        )

        _elapsed = round(time.time() - _start, 1)
        yield ResearchEvent(
            event_type=ResearchEventType.COMPLETE,
            metadata={
                "total_iterations": sum(1 for sq in sub_questions if sq.status == SubQuestionStatus.RESOLVED),
                "total_sub_questions": len(sub_questions),
                "elapsed_seconds": _elapsed,
            },
        )
        logger.info("DeepResearch complete in %.1fs, %d sub-questions resolved", _elapsed, len(all_findings))

    # ------------------------------------------------------------------ Phase 1
    async def _decompose(
        self,
        query: str,
        context_preview: str,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> List[SubQuestion]:
        """Use LLM to decompose the query into sub-questions."""
        prompt = _DECOMPOSE_PROMPT.format(
            query=query,
            context_preview=context_preview,
        )
        try:
            raw = await llm_fn(prompt)
            questions = self._parse_json_array(raw)
            if not questions:
                # Fallback: treat the whole query as a single sub-question
                return [SubQuestion(id=0, text=query, priority=3)]
            return [
                SubQuestion(id=i, text=q.strip(), priority=max(1, 4 - i))
                for i, q in enumerate(questions[:6])
            ]
        except Exception as exc:
            logger.warning("Decomposition failed: %s, using original query", exc)
            return [SubQuestion(id=0, text=query, priority=3)]

    # ------------------------------------------------------------------ Phase 2 per sub-question
    async def _research_sub_question(
        self,
        sub_question: str,
        original_query: str,
        base_prompt: str,
        context: str,
        previous_findings: str,
        llm_fn: Callable[[str], Awaitable[str]],
        language: str,
        iteration: int,
        trace: Any = None,
        tools_override: Optional[Dict[str, Callable[[str], Awaitable[str]]]] = None,
    ) -> AsyncIterator[ResearchEvent]:
        """
        Research one sub-question using a mini ReAct loop (up to 5 tool calls).
        Yields TOOL_CALL and FINDING events.
        """
        from api.agent.react import ReActRunner

        # Build a focused system prompt for this sub-question
        prev_section = ""
        if previous_findings:
            prev_section = f"\n<previous_findings>\n{previous_findings[:12000]}\n</previous_findings>"

        system_prompt = _SUBQ_RESEARCH_PROMPT.format(
            base_prompt=base_prompt,
            sub_question=sub_question,
            original_query=original_query,
            previous_findings=prev_section,
        )

        # Use a ReAct runner with enough steps for thorough investigation
        tools = tools_override or self.tools
        runner = ReActRunner(
            tools=tools,
            max_iterations=5,
            max_observation_chars=20000,
        )

        finding_parts = []
        async for chunk in runner.run(
            query=sub_question,
            system_prompt=system_prompt,
            initial_context=context[:18000],  # generous context per sub-question
            llm_fn=llm_fn,
            language=language,
            trace=trace,
        ):
            # ReActRunner yields status lines ("> 🔍 ...") and the final answer
            if chunk.startswith("> "):
                # This is a tool-call status line
                yield ResearchEvent(
                    event_type=ResearchEventType.TOOL_CALL,
                    data=chunk.strip(),
                    iteration=iteration + 1,
                    sub_question=sub_question,
                )
            else:
                cleaned = self._clean_output(chunk)
                if cleaned:
                    finding_parts.append(cleaned)
                    yield ResearchEvent(
                        event_type=ResearchEventType.FINDING,
                        data=cleaned,
                        iteration=iteration + 1,
                        sub_question=sub_question,
                    )

    # ------------------------------------------------------------------ Gap Analysis
    async def _analyze_gaps(
        self,
        query: str,
        all_findings: Dict[int, str],
        sub_questions: List[SubQuestion],
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> List[SubQuestion]:
        """Identify remaining gaps and create new sub-questions."""
        findings_summary = "\n\n".join(
            f"**{sq.text}**: {all_findings.get(sq.id, '(not yet researched)')[:2500]}"
            for sq in sub_questions
        )

        prompt = _GAP_ANALYSIS_PROMPT.format(
            query=query,
            findings_summary=findings_summary,
        )

        try:
            raw = await llm_fn(prompt)
            gaps = self._parse_json_array(raw)
            if not gaps:
                return []

            next_id = max(sq.id for sq in sub_questions) + 1
            new_sqs = []
            for i, gap_text in enumerate(gaps[:2]):  # max 2 new gaps per iteration
                if gap_text.strip():
                    new_sqs.append(SubQuestion(
                        id=next_id + i,
                        text=gap_text.strip(),
                        priority=1,  # lower priority than original questions
                    ))
            return new_sqs
        except Exception as exc:
            logger.warning("Gap analysis failed: %s", exc)
            return []

    # ------------------------------------------------------------------ Phase 3
    async def _synthesize(
        self,
        query: str,
        base_prompt: str,
        all_findings: Dict[int, str],
        all_observations: Dict[int, List[Dict[str, str]]],
        sub_questions: List[SubQuestion],
        llm_fn: Callable[[str], Awaitable[str]],
        language: str,
        trace: Any = None,
    ) -> str:
        """Synthesize all findings into a final comprehensive answer.

        Runs a two-step process:
        1. Pre-synthesis: organize findings into a coherent outline
        2. Final synthesis: write the full analysis guided by the outline

        The raw observations (actual file contents, search results) are
        included as a separate evidence section so the LLM can reference
        verified source code directly, not just ReAct summaries.
        """
        # Clean findings before synthesis to prevent JSON artifacts leaking into the prompt
        cleaned_findings = {
            sq_id: self._clean_output(finding)
            for sq_id, finding in all_findings.items()
        }
        findings_text = "\n\n---\n\n".join(
            f"### Sub-question: {sq.text}\n\n{cleaned_findings.get(sq.id, '(no findings)')}"
            for sq in sub_questions
            if sq.status == SubQuestionStatus.RESOLVED
        )

        # Build raw evidence section from tool observations
        raw_evidence = self._format_raw_evidence(all_observations, sub_questions)

        # Step 1: Pre-synthesis — organize thinking before writing
        outline = ""
        try:
            pre_prompt = _PRE_SYNTHESIS_PROMPT.format(
                query=query,
                all_findings=findings_text,
            )
            outline = await llm_fn(pre_prompt)
            outline = self._clean_output(outline)
            logger.info("Pre-synthesis outline: %s", outline[:300])
        except Exception as exc:
            logger.warning("Pre-synthesis failed (non-fatal): %s", exc)
            outline = "(outline unavailable)"

        # Step 2: Final synthesis guided by the outline + raw evidence
        prompt = _SYNTHESIS_PROMPT.format(
            base_prompt=base_prompt,
            query=query,
            outline=outline,
            all_findings=findings_text,
            raw_evidence=raw_evidence,
        )

        try:
            result = await llm_fn(prompt)
            return self._clean_output(result)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            # Fallback: concatenate findings
            return f"## Research Findings\n\n{findings_text}"

    # ------------------------------------------------------------------ Helpers

    def _make_recording_tools(
        self, observations: List[Dict[str, str]],
    ) -> Dict[str, Callable[[str], Awaitable[str]]]:
        """Wrap self.tools so that every call records its raw output.

        The observations list is mutated in-place — the caller keeps a
        reference and can inspect recorded data after the ReAct run.
        """
        def _wrap(name: str, fn: Callable[[str], Awaitable[str]]):
            async def wrapped(input_str: str) -> str:
                result = await fn(input_str)
                # Record truncated observation for later synthesis
                observations.append({
                    "tool": name,
                    "input": (input_str or "")[:300],
                    "output": (result or "")[:6000],
                })
                return result
            return wrapped

        return {name: _wrap(name, fn) for name, fn in self.tools.items()}

    def _format_raw_evidence(
        self,
        all_observations: Dict[int, List[Dict[str, str]]],
        sub_questions: List[SubQuestion],
    ) -> str:
        """Format raw tool observations into a structured evidence section.

        Prioritizes read_file/read_function/code_grep outputs (actual code)
        over rag_search results, and caps the total budget.
        """
        if not all_observations:
            return "(no raw evidence collected)"

        # Categorize observations by value: code-reading tools are higher value
        _CODE_TOOLS = {"read_file", "read_function", "code_grep", "find_references"}
        evidence_parts: List[str] = []
        total_budget = 25000  # generous budget for raw evidence

        for sq in sub_questions:
            obs_list = all_observations.get(sq.id, [])
            if not obs_list:
                continue

            # Sort: code tools first, then search tools
            code_obs = [o for o in obs_list if o["tool"] in _CODE_TOOLS]
            other_obs = [o for o in obs_list if o["tool"] not in _CODE_TOOLS]
            sorted_obs = code_obs + other_obs

            sq_parts: List[str] = []
            sq_budget = min(6000, total_budget // max(1, len(all_observations)))

            for obs in sorted_obs:
                if sq_budget <= 0:
                    break
                output = obs["output"]
                if not output or output.startswith("No ") or output.startswith("Error"):
                    continue
                header = f"[{obs['tool']}({obs['input'][:100]})]"
                # Truncate individual observation
                snippet = output[:sq_budget]
                entry = f"{header}:\n{snippet}"
                sq_parts.append(entry)
                sq_budget -= len(entry)

            if sq_parts:
                section = f"#### Evidence for: {sq.text}\n" + "\n\n".join(sq_parts)
                evidence_parts.append(section)
                total_budget -= len(section)

            if total_budget <= 0:
                break

        return "\n\n---\n\n".join(evidence_parts) if evidence_parts else "(no raw evidence collected)"

    def _compress_context(
        self,
        initial_context: str,
        all_findings: Dict[int, str],
        sub_questions: List[SubQuestion],
    ) -> str:
        """Keep context within token budget by summarizing findings.

        Prioritizes paragraphs containing code snippets (``` or `) to
        preserve concrete evidence across iterations.
        """
        base = (initial_context or "")[:6000]
        findings_parts = []
        for sq in sub_questions:
            if sq.id in all_findings:
                preserved = self._prioritize_code_paragraphs(
                    all_findings[sq.id], max_chars=4000
                )
                findings_parts.append(f"[{sq.text}]: {preserved}")
        findings_str = "\n".join(findings_parts)
        if len(findings_str) > 15000:
            findings_str = findings_str[-15000:]
        return f"{base}\n\n<accumulated_findings>\n{findings_str}\n</accumulated_findings>"

    @staticmethod
    def _prioritize_code_paragraphs(text: str, max_chars: int) -> str:
        """Compress text while keeping code-bearing paragraphs first.

        Paragraphs that contain inline code (`) or code fences (```) are
        more likely to carry concrete evidence, so they are preserved with
        higher priority than pure-prose paragraphs.
        """
        if len(text) <= max_chars:
            return text
        paragraphs = text.split("\n\n")
        code_paras = [p for p in paragraphs if "```" in p or "`" in p]
        text_paras = [p for p in paragraphs if p not in code_paras]
        # Reorder: code-bearing paragraphs first, then prose
        reordered = code_paras + text_paras
        result_parts: List[str] = []
        budget = max_chars
        for p in reordered:
            if budget <= 0:
                break
            result_parts.append(p[:budget])
            budget -= len(p) + 2  # +2 for the "\n\n" joiner
        return "\n\n".join(result_parts)

    @staticmethod
    def _clean_output(text: str) -> str:
        """Clean LLM output: remove JSON artifacts, literal \\n, duplicate paragraphs."""
        if not text:
            return text
        # Remove any [RESEARCH_EVENT] markers that leaked into LLM output
        text = re.sub(r'\[RESEARCH_EVENT\]\{.*?\}', '', text)
        # Remove standalone JSON objects that look like research event payloads
        # e.g. {"type": "tool_call", "data": "...", "iteration": 1, ...}
        text = re.sub(r'\{"type"\s*:\s*"[^"]*"\s*,\s*"data"\s*:\s*"[^"]*"[^}]*\}', '', text)
        # Remove tool status lines ("> ⚙️ Running ...", "> 🔍 Searching ...")
        text = re.sub(r'^>\s*[⚙️🔍📄]\s*.+$', '', text, flags=re.MULTILINE)
        # Remove ReAct reasoning artifacts (Thought:, Action:, Action Input:, Observation:)
        text = re.sub(r'^(Thought|Action|Action Input|Observation)\s*:.*$', '', text, flags=re.MULTILINE)
        # Replace literal \n with actual newlines
        text = text.replace('\\n', '\n')
        # Collapse 3+ consecutive blank lines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove duplicate consecutive paragraphs
        paragraphs = text.split('\n\n')
        deduped = []
        for p in paragraphs:
            if not deduped or p.strip() != deduped[-1].strip():
                deduped.append(p)
        text = '\n\n'.join(deduped)
        return text.strip()

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        """Extract a JSON array of strings from LLM output."""
        # Try to find JSON array in the text
        text = text.strip()
        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Find the array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    return [str(item) for item in arr if item]
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse numbered list
        lines = text.split("\n")
        items = []
        for line in lines:
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                items.append(m.group(1).strip())
        return items
