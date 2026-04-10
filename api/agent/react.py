"""
ReAct (Reasoning + Acting) loop runner.

Implements a multi-step reasoning loop where the LLM can:
1. Think about what information it needs
2. Call tools (rag_search, read_file) to gather context
3. Observe the results
4. Repeat until it has enough info for a high-quality answer
5. Produce a Final Answer
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """One iteration of the ReAct loop."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


class ReActRunner:
    """
    Executes a ReAct loop: Thought → Action → Observation → ... → Final Answer.

    The runner is provider-agnostic — it accepts an ``llm_fn`` async callable
    that performs the actual LLM invocation and returns the full response text.
    """

    def __init__(
        self,
        tools: Dict[str, Callable[[str], Awaitable[str]]],
        max_iterations: int = 3,
        max_observation_chars: int = 12000,
    ):
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_observation_chars = max_observation_chars

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _parse_response(text: str) -> ReActStep:
        """Parse LLM output into a ReActStep."""
        # Check for Final Answer
        final_match = re.search(r"Final\s*Answer\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if final_match:
            thought_match = re.search(
                r"Thought\s*:\s*(.*?)(?=Final\s*Answer\s*:)", text, re.DOTALL | re.IGNORECASE
            )
            return ReActStep(
                thought=thought_match.group(1).strip() if thought_match else "",
                is_final=True,
                final_answer=final_match.group(1).strip(),
            )

        # Parse Thought / Action / Action Input
        thought_match = re.search(
            r"Thought\s*:\s*(.*?)(?=Action\s*:|$)", text, re.DOTALL | re.IGNORECASE
        )
        action_match = re.search(r"Action\s*:\s*(\S+)", text, re.IGNORECASE)
        input_match = re.search(
            r"Action\s*Input\s*:\s*(.*?)(?=\n\n|Thought\s*:|$)", text, re.DOTALL | re.IGNORECASE
        )

        return ReActStep(
            thought=thought_match.group(1).strip() if thought_match else text.strip(),
            action=action_match.group(1).strip() if action_match else None,
            action_input=input_match.group(1).strip() if input_match else None,
        )

    # ------------------------------------------------------------- scratchpad
    def _build_scratchpad(self, steps: List[ReActStep]) -> str:
        parts: List[str] = []
        for step in steps:
            parts.append(f"Thought: {step.thought}")
            if step.action:
                parts.append(f"Action: {step.action}")
                parts.append(f"Action Input: {step.action_input or ''}")
            if step.observation is not None:
                obs = self._truncate(step.observation)
                parts.append(f"Observation: {obs}")
        return "\n".join(parts)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_observation_chars:
            return text
        # Smart truncation: keep head 60% + tail 40% to preserve both
        # file headers/imports AND key logic at the end of files.
        head_budget = int(self.max_observation_chars * 0.6)
        tail_budget = self.max_observation_chars - head_budget - 60  # 60 chars for separator
        return (
            text[:head_budget]
            + "\n\n... [middle truncated — showing head & tail] ...\n\n"
            + text[-tail_budget:]
        )

    # --------------------------------------------------------------- prompts
    def _build_prompt(
        self,
        query: str,
        system_prompt: str,
        context: str,
        scratchpad: str,
        tools_description: str,
        iteration: int,
        language: str,
    ) -> str:
        remaining = self.max_iterations - iteration

        react_block = (
            f"{system_prompt}\n\n"
            f"You have access to the following tools:\n{tools_description}\n\n"
            "Use the following format STRICTLY:\n\n"
            "Thought: <your reasoning about what to do next>\n"
            "Action: <tool name>\n"
            "Action Input: <input for the tool>\n\n"
            "When you have gathered enough information, use:\n\n"
            "Thought: <your final reasoning>\n"
            "Final Answer: <your comprehensive answer in markdown format>\n\n"
            f"RULES:\n"
            f"- You have {remaining} step(s) remaining. Use them wisely.\n"
            "- You MUST call at least one tool before giving a Final Answer. Do NOT answer from initial context alone.\n"
            "- READ ACTUAL SOURCE CODE files (.py, .ts, .js, etc.), not just README or docs.\n"
            "- Each Action must be one of the listed tools.\n"
            f"- Your Final Answer MUST cite specific file paths, function names, and include code excerpts from your observations. Respond in **{language}**.\n"
            "- Do NOT fabricate code that was not found in observations.\n"
            "- After showing code in your Final Answer, explain what it reveals about the design — don't just paste code without analysis.\n"
        )

        if iteration == self.max_iterations - 1:
            react_block += "- THIS IS YOUR LAST STEP. You MUST provide a Final Answer NOW.\n"

        parts = [react_block]

        if context and context.strip():
            parts.append(f"\n<initial_context>\n{context}\n</initial_context>")

        if scratchpad:
            parts.append(f"\n<previous_steps>\n{scratchpad}\n</previous_steps>")

        parts.append(f"\n<query>\n{query}\n</query>\n")

        return "\n".join(parts)

    # -------------------------------------------------------------- run loop
    async def run(
        self,
        query: str,
        system_prompt: str,
        initial_context: str,
        llm_fn: Callable[[str], Awaitable[str]],
        language: str = "en",
        trace: Any = None,
    ) -> AsyncIterator[str]:
        """
        Run the ReAct loop, yielding status updates and the final answer.

        Args:
            query: User's original question.
            system_prompt: Base system prompt (repo info, role).
            initial_context: Pre-retrieved RAG context (may be empty).
            llm_fn: ``async (prompt) -> str`` — calls the LLM, returns full text.
            language: Target response language code.
            trace: Optional Langfuse trace/span to nest generations and tool spans under.
        """
        from api.tracing import get_tracer
        tracer = get_tracer()

        steps: List[ReActStep] = []
        _tool_help = {
            "rag_search": "Search the codebase with a semantic query to find relevant code, functions, or documentation. Input: a natural language query string.",
            "read_file": "Read the full content of a specific file in the repository. Input: the file path (e.g. src/main.py).",
            "read_function": "Extract a specific function or class definition from a file WITHOUT reading the entire file. Much more efficient than read_file for targeted investigation. Input: 'file_path::function_or_class_name' (e.g. 'api/rag.py::RAG' or 'src/utils.py::process_request').",
            "find_references": "Find all files that import, call, or reference a specific identifier. Use this to understand who calls a function, where a class is used, or how components are connected. Input: an identifier name (e.g. 'ReActRunner', 'handle_websocket_chat').",
            "list_repo_files": "List files and directories under a given path in the repository. Use '.' or '' to list the root. Input: a relative directory path (e.g. src/components).",
            "code_grep": "Search for an exact string or regex pattern across all source files in the repository. Use this to find function definitions, class names, imports, or specific code patterns. Input: a search string or regex pattern (e.g. 'def authenticate' or 'class.*Handler').",
            "memory_search": "Search the long-term knowledge base for previously learned facts, user preferences, and insights from past conversations about this repository. Input: a natural language query.",
        }
        tools_desc = "\n".join(
            f"- **{name}**: {_tool_help.get(name, name)}" for name in self.tools.keys()
        )

        # Parent span for the react loop
        react_span_parent = trace  # may be None / _NoOpObj — tracer handles it

        for iteration in range(self.max_iterations):
            scratchpad = self._build_scratchpad(steps)
            prompt = self._build_prompt(
                query=query,
                system_prompt=system_prompt,
                context=initial_context,
                scratchpad=scratchpad,
                tools_description=tools_desc,
                iteration=iteration,
                language=language,
            )

            # ---- LLM Generation (traced) ----
            response_text = None
            with tracer.generation(
                react_span_parent,
                f"react_iteration_{iteration}",
                input=prompt[:2000],
                metadata={"iteration": iteration, "max_iterations": self.max_iterations},
            ) as gen_ctx:
                try:
                    response_text = await llm_fn(prompt)
                    gen_ctx.end(
                        output=response_text[:2000] if response_text else "",
                        metadata={"iteration": iteration},
                    )
                except Exception as exc:
                    logger.error("ReAct LLM call failed at iteration %d: %s", iteration, exc)
                    gen_ctx.end(metadata={"error": str(exc), "iteration": iteration})
                    yield f"\nError during reasoning: {exc}"
                    return

            if response_text is None:
                yield "\nError during reasoning: empty response"
                return

            logger.info(
                "ReAct iteration %d/%d response (first 300 chars): %s",
                iteration + 1,
                self.max_iterations,
                response_text[:300],
            )

            step = self._parse_response(response_text)

            # ---- Final Answer
            if step.is_final:
                yield step.final_answer or ""
                return

            # ---- Execute tool (traced) ----
            if step.action and step.action in self.tools:
                status = self._format_status(step.action, step.action_input, language)
                yield status + "\n\n"

                with tracer.span(
                    react_span_parent,
                    f"tool_call_{step.action}",
                    input={"query": step.action_input or ""},
                    metadata={"iteration": iteration},
                ) as tool_ctx:
                    try:
                        observation = await self.tools[step.action](step.action_input or "")
                        step.observation = observation
                        tool_ctx.end(
                            output={"result_length": len(observation) if observation else 0},
                        )
                    except Exception as exc:
                        step.observation = f"Tool error: {exc}"
                        logger.error("ReAct tool %s error: %s", step.action, exc)
                        tool_ctx.end(metadata={"error": str(exc)})
            elif step.action:
                step.observation = (
                    f"Unknown tool '{step.action}'. Available tools: {', '.join(self.tools.keys())}"
                )
            else:
                # No action and no Final Answer — the model didn't follow ReAct format.
                # If it looks like the model tried to give a direct answer (no "Thought:"
                # prefix, has substantive content), yield it as the answer.
                raw = step.thought.strip()
                # Check if it's just internal reasoning (mentions tool names) vs real answer
                tool_mention_count = sum(1 for t in self.tools if t in raw.lower())
                is_reasoning = (
                    tool_mention_count > 0
                    or raw.startswith("I need to")
                    or raw.startswith("我需要")
                    or raw.startswith("Let me")
                    or "I will use" in raw
                    or "我将使用" in raw
                )
                if not is_reasoning and len(raw) > 50:
                    yield raw
                    return
                # Otherwise, treat it as a failed step and continue to next iteration
                step.observation = "No tool called. Please specify an Action or provide a Final Answer."

            steps.append(step)

        # Max iterations exhausted — synthesize answer via LLM fallback
        logger.warning("ReAct reached max iterations (%d). Synthesizing final answer via LLM.", self.max_iterations)
        try:
            synthesis = await self._synthesize_forced_answer(
                steps, query, system_prompt, llm_fn, language,
            )
            yield synthesis
        except Exception as synth_exc:
            logger.error("LLM synthesis fallback failed: %s. Using static compilation.", synth_exc)
            yield self._compile_forced_answer(steps, language)

    # --------------------------------------------------------- helper methods
    async def _synthesize_forced_answer(
        self,
        steps: List[ReActStep],
        query: str,
        system_prompt: str,
        llm_fn: Callable[[str], Awaitable[str]],
        language: str,
    ) -> str:
        """Call LLM one final time to synthesize collected observations into a
        coherent answer when max iterations are exhausted."""
        # Collect all non-error observations
        evidence_parts: List[str] = []
        for step in steps:
            if step.observation and not step.observation.startswith("Tool error:") \
               and not step.observation.startswith("No tool called"):
                label = ""
                if step.action and step.action_input:
                    label = f"[{step.action}({step.action_input})]:\n"
                truncated = step.observation[:3000]
                evidence_parts.append(f"{label}{truncated}")

        if not evidence_parts:
            return self._compile_forced_answer(steps, language)

        evidence_text = "\n\n---\n\n".join(evidence_parts)

        lang_instruction = f"Respond in **{language}** language." if language else ""
        synthesis_prompt = (
            f"{system_prompt}\n\n"
            "You have already investigated the codebase. Below is the evidence you collected.\n"
            "Based ONLY on this evidence, provide a comprehensive, well-structured answer "
            f"to the user's question. {lang_instruction}\n"
            "Do NOT mention your investigation process. Just give the final answer directly.\n\n"
            f"<evidence>\n{evidence_text}\n</evidence>\n\n"
            f"<query>\n{query}\n</query>\n\n"
            "Answer:"
        )

        result = await llm_fn(synthesis_prompt)
        if result and result.strip():
            return result.strip()
        # If LLM returned empty, fall back to static compilation
        return self._compile_forced_answer(steps, language)

    @staticmethod
    def _format_status(action: str, action_input: Optional[str], language: str) -> str:
        inp = action_input or ""
        if (language or "").lower().startswith("zh"):
            templates = {
                "rag_search": f"> 🔍 正在搜索代码库: *{inp}*",
                "read_file": f"> 📄 正在读取文件: *{inp}*",
                "read_function": f"> 📄 正在读取函数定义: *{inp}*",
                "list_repo_files": f"> 📄 正在浏览文件目录: *{inp}*",
                "code_grep": f"> 🔍 正在搜索代码: *{inp}*",
                "find_references": f"> 🔍 正在查找引用: *{inp}*",
                "memory_search": f"> 🔍 正在搜索知识库: *{inp}*",
            }
        else:
            templates = {
                "rag_search": f"> 🔍 Searching codebase: *{inp}*",
                "read_file": f"> 📄 Reading file: *{inp}*",
                "read_function": f"> 📄 Reading function: *{inp}*",
                "list_repo_files": f"> 📄 Browsing directory: *{inp}*",
                "code_grep": f"> 🔍 Grepping code: *{inp}*",
                "find_references": f"> 🔍 Finding references: *{inp}*",
                "memory_search": f"> 🔍 Searching knowledge base: *{inp}*",
            }
        return templates.get(action, f"> ⚙️ Running {action}: {inp}")

    @staticmethod
    def _compile_forced_answer(steps: List[ReActStep], language: str) -> str:
        """Compile accumulated observations into a best-effort answer.

        Extracts evidence (code snippets from observations) and synthesizes
        a user-friendly answer. Internal reasoning (thoughts) is hidden.
        """
        evidence_parts: List[str] = []
        observations_text: List[str] = []
        for step in steps:
            if step.observation:
                # Extract code-block snippets and file-path headers from observations
                obs = step.observation
                # Skip error/empty observations
                if obs.startswith("Tool error:") or obs.startswith("Search error:") or obs.startswith("No "):
                    continue
                code_blocks = re.findall(r"(```[\s\S]*?```)", obs)
                file_headers = re.findall(r"(### .+)", obs)
                if file_headers:
                    evidence_parts.extend(file_headers[:3])
                if code_blocks:
                    evidence_parts.extend(code_blocks[:2])
                elif step.action and step.action_input:
                    # No code blocks — include a short excerpt of the raw observation
                    snippet = obs[:600].strip()
                    if snippet:
                        evidence_parts.append(f"From `{step.action}({step.action_input})`:\n{snippet}")
                # Collect all non-error observations for summary
                observations_text.append(obs)

        if evidence_parts:
            return "\n\n".join(evidence_parts)
        elif observations_text:
            # No structured evidence; give a summary of what was found
            combined = "\n\n".join(observations_text)
            return combined[:3000]
        else:
            if (language or "").lower().startswith("zh"):
                return "抱歉，未能从代码库中检索到足够的信息来完整回答这个问题。请尝试更具体的问题，或者使用 Deep Research 模式获取更详细的分析。"
            return "I was unable to gather enough information to provide a complete answer. Please try a more specific question, or use Deep Research mode for a more thorough analysis."
