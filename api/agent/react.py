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
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional

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
        return text[: self.max_observation_chars] + "\n... [truncated]"

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
            "- If the initial context is already sufficient, go directly to Final Answer.\n"
            "- Each Action must be one of the listed tools.\n"
            f"- Your Final Answer should be comprehensive, well-structured, and in **{language}**.\n"
            "- Do NOT fabricate code that was not found in observations.\n"
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
    ) -> AsyncIterator[str]:
        """
        Run the ReAct loop, yielding status updates and the final answer.

        Args:
            query: User's original question.
            system_prompt: Base system prompt (repo info, role).
            initial_context: Pre-retrieved RAG context (may be empty).
            llm_fn: ``async (prompt) -> str`` — calls the LLM, returns full text.
            language: Target response language code.
        """
        steps: List[ReActStep] = []
        tools_desc = "\n".join(
            f"- **{name}**: {name}" for name in self.tools.keys()
        )
        # Enrich tool descriptions
        _tool_help = {
            "rag_search": "Search the codebase with a semantic query to find relevant code, functions, or documentation. Input: a natural language query string.",
            "read_file": "Read the full content of a specific file in the repository. Input: the file path (e.g. src/main.py).",
        }
        tools_desc = "\n".join(
            f"- **{name}**: {_tool_help.get(name, name)}" for name in self.tools.keys()
        )

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

            try:
                response_text = await llm_fn(prompt)
            except Exception as exc:
                logger.error("ReAct LLM call failed at iteration %d: %s", iteration, exc)
                yield f"\nError during reasoning: {exc}"
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

            # ---- Execute tool
            if step.action and step.action in self.tools:
                status = self._format_status(step.action, step.action_input, language)
                yield status + "\n\n"

                try:
                    observation = await self.tools[step.action](step.action_input or "")
                    step.observation = observation
                except Exception as exc:
                    step.observation = f"Tool error: {exc}"
                    logger.error("ReAct tool %s error: %s", step.action, exc)
            elif step.action:
                step.observation = (
                    f"Unknown tool '{step.action}'. Available tools: {', '.join(self.tools.keys())}"
                )
            else:
                # No action and no Final Answer — treat thought as the answer
                yield step.thought
                return

            steps.append(step)

        # Max iterations exhausted — compile best-effort answer
        logger.warning("ReAct reached max iterations (%d). Forcing answer.", self.max_iterations)
        yield self._compile_forced_answer(steps, language)

    # --------------------------------------------------------- helper methods
    @staticmethod
    def _format_status(action: str, action_input: Optional[str], language: str) -> str:
        inp = action_input or ""
        if (language or "").lower().startswith("zh"):
            templates = {
                "rag_search": f"> 🔍 正在搜索代码库: *{inp}*",
                "read_file": f"> 📄 正在读取文件: *{inp}*",
            }
        else:
            templates = {
                "rag_search": f"> 🔍 Searching codebase: *{inp}*",
                "read_file": f"> 📄 Reading file: *{inp}*",
            }
        return templates.get(action, f"> ⚙️ Running {action}: {inp}")

    @staticmethod
    def _compile_forced_answer(steps: List[ReActStep], language: str) -> str:
        """Compile accumulated observations into a best-effort answer."""
        parts: List[str] = []
        for step in steps:
            if step.thought:
                parts.append(step.thought)
        return "\n\n".join(parts) if parts else "I was unable to gather enough information to provide a complete answer."
