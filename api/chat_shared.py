"""
Shared utilities for the chat pipeline.

Contains helpers previously duplicated across ``websocket_wiki.py`` and
``simple_chat.py``:

- ``ThinkBlockFilter`` — streaming <think> tag suppression
- ``infer_language_code`` — CJK detection
- ``build_one_shot_deep_research_prompt`` — DR one-shot prompt builder
- ``build_mcp_tools`` — MCP bridge builder for ReAct tools
- ``retrieve_rag_context`` — unified RAG retrieval + formatting
- ``build_prompt`` — final prompt assembly with memory/context injection
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from pydantic import BaseModel, Field

from api.config import configs
from api.data_pipeline import get_file_content
from api.mcp.registry import MCPToolRegistry, MCPTool, ToolCategory, ToolSchema, ToolParameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models (shared across WebSocket + HTTP transports)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None)
    token: Optional[str] = Field(None)
    type: Optional[str] = Field("github")
    user_id: Optional[str] = Field(None)
    provider: str = Field("openai")
    model: Optional[str] = Field(None)
    language: Optional[str] = Field("en")
    excluded_dirs: Optional[str] = Field(None)
    excluded_files: Optional[str] = Field(None)
    included_dirs: Optional[str] = Field(None)
    included_files: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Think-block filter (streaming)
# ---------------------------------------------------------------------------

class ThinkBlockFilter:
    """Streaming filter that suppresses content inside <think>...</think> blocks."""

    def __init__(self):
        self._in_think = False
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        output_parts: list[str] = []

        while self._buffer:
            if self._in_think:
                end_idx = self._buffer.find("</think>")
                if end_idx == -1:
                    self._buffer = ""
                    break
                self._buffer = self._buffer[end_idx + len("</think>"):]
                self._in_think = False
            else:
                start_idx = self._buffer.find("<think>")
                if start_idx == -1:
                    safe_end = len(self._buffer)
                    for i in range(1, min(len("<think>"), len(self._buffer) + 1)):
                        if self._buffer.endswith("<think>"[:i]):
                            safe_end = len(self._buffer) - i
                            break
                    output_parts.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                    break
                else:
                    output_parts.append(self._buffer[:start_idx])
                    self._buffer = self._buffer[start_idx + len("<think>"):]
                    self._in_think = True

        return "".join(output_parts)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def infer_language_code(query: str, requested_language: Optional[str] = None) -> str:
    """Prefer Chinese response when the user query contains CJK characters."""
    if re.search(r"[\u4e00-\u9fff]", query or ""):
        return "zh"
    return requested_language or "en"


def get_language_name(language_code: str) -> str:
    """Map a language code to its display name using the configured lang map."""
    supported = configs["lang_config"]["supported_languages"]
    return supported.get(language_code, "English")


# ---------------------------------------------------------------------------
# Deep Research one-shot prompt
# ---------------------------------------------------------------------------

def build_one_shot_deep_research_prompt(
    repo_type: str,
    repo_url: str,
    repo_name: str,
    language_name: str,
) -> str:
    return f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are performing a Deep Research response for the user's latest question.
IMPORTANT: You MUST respond in {language_name} language.
</role>

<guidelines>
- Deliver one complete, high-quality answer that is MORE DETAILED and THOROUGH than a normal chat response
- Focus entirely on answering the user's specific question — do NOT force a fixed template
- Do NOT use prescribed section headings like "Key Findings", "Architecture (Detailed)", "Practical Recommendations", etc. Instead, choose headings and structure that naturally fit the user's question
- Start with a brief opening paragraph that directly addresses the question, then expand into details
- Organize your response around the TOPICS that matter for the user's question, not around a predetermined framework
- Use ## headings only when they genuinely help organize distinct topics; for simpler questions, flowing prose with occasional bold emphasis is better than many headings
- Mix paragraphs, bullet points and code examples naturally — avoid walls of bullets without connecting prose
- When explaining features or capabilities, give CONCRETE examples of how they work and what results the user can expect
- Reference specific files, functions, or code paths from the repository when they support your explanation
- Use simple analogies where they help clarify complex ideas, but don't force them
- Write in a conversational yet knowledgeable tone — as if explaining to a colleague, not writing a formal report
- Target roughly 800-2000 characters of substantive content (or equivalent in other languages), but let the question's complexity determine the actual length
- Do NOT pad the response with generic advice, boilerplate recommendations, or filler sections
- Every sentence should provide information the user actually asked about or needs to understand
- Do NOT output iterative placeholders like "Research Update", "Next Steps", or "Continue research"
</guidelines>

<style>
- Natural, readable, and engaging — prioritize flow over rigid structure
- Smooth transitions between ideas; avoid fragmented bullet-only output
- Concrete and specific rather than generic and abstract
- When technical terms are needed, briefly explain them in context
- Avoid filler, repetition, and overly formal tone
- If evidence is insufficient, state uncertainty explicitly
</style>"""


# ---------------------------------------------------------------------------
# MCP Tool bridge builder
# ---------------------------------------------------------------------------

# Tool-level call statistics (survives across requests within a process)
_mcp_call_stats: Dict[str, Dict[str, Any]] = {}


def get_mcp_call_stats() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of tool call statistics."""
    return dict(_mcp_call_stats)


def _record_tool_call(tool_name: str, latency_ms: float, success: bool) -> None:
    """Record a single tool invocation for statistics."""
    if tool_name not in _mcp_call_stats:
        _mcp_call_stats[tool_name] = {
            "total_calls": 0,
            "success_count": 0,
            "error_count": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
        }
    stats = _mcp_call_stats[tool_name]
    stats["total_calls"] += 1
    if success:
        stats["success_count"] += 1
    else:
        stats["error_count"] += 1
    stats["total_latency_ms"] += latency_ms
    stats["avg_latency_ms"] = round(stats["total_latency_ms"] / stats["total_calls"], 2)


# ---------------------------------------------------------------------------
# Embedding callable factory for Tier-2 intent classification
# ---------------------------------------------------------------------------

def create_embed_fn(rag_instance: Any):
    """Create an async embedding function from a RAG instance's embedder.

    Returns an ``async (texts: list[str]) -> list[list[float]]`` callable
    compatible with ``EmbeddingIntentClassifier``.  Returns ``None`` if the
    embedder is unavailable.
    """
    embedder = getattr(rag_instance, "embedder", None)
    if embedder is None:
        return None

    import asyncio

    async def embed_fn(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # The adalflow embedder is synchronous — offload to a thread
        output = await asyncio.to_thread(embedder, input=texts)
        return [e.embedding for e in output.data]

    return embed_fn


# Comprehensive parameter metadata for all known tools
_TOOL_PARAM_MAP: Dict[str, tuple] = {
    "rag_search": ("query", "string", "Semantic search query for the codebase"),
    "read_file": ("file_path", "string", "Path to the file to read"),
    "read_function": ("query", "string", "file_path::function_or_class_name"),
    "find_references": ("identifier", "string", "Identifier name to find references for"),
    "list_repo_files": ("path", "string", "Relative directory path to list"),
    "code_grep": ("pattern", "string", "Exact string or regex pattern to search for"),
    "memory_search": ("query", "string", "Natural language query for the knowledge base"),
}


def build_mcp_tools(
    rag_tools: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap per-request RAG tools in a fresh MCPToolRegistry.

    The MCP layer provides:
      * **Parameter validation** — required param presence + type check
      * **Call statistics** — per-tool call count, latency, error rate
      * **Unified schema** — every tool has a machine-readable ToolSchema

    Returns a dict of bridge callables compatible with ReActRunner
    (i.e. ``async (str) -> str``).
    """
    import time as _time

    registry = MCPToolRegistry()

    for tool_name, tool_fn in rag_tools.items():
        meta = _TOOL_PARAM_MAP.get(tool_name, ("input", "string", "Tool input"))
        param_name, param_type, param_desc = meta

        def _make_handler(fn, pname):
            async def handler(input_data: dict) -> str:
                return await fn(input_data.get(pname, ""))
            return handler

        registry.register(MCPTool(
            name=tool_name,
            description=f"Codebase tool: {tool_name}",
            category=ToolCategory.SEARCH,
            schema=ToolSchema(parameters=[
                ToolParameter(name=param_name, type=param_type,
                              description=param_desc, required=True)
            ]),
            handler=_make_handler(tool_fn, param_name),
        ))

    def _make_bridge(reg, name, pname):
        async def bridge(input_str: str) -> str:
            t0 = _time.monotonic()
            success = True
            try:
                result = await reg.execute(name, {pname: input_str})
                return result
            except Exception as exc:
                success = False
                raise
            finally:
                elapsed = round((_time.monotonic() - t0) * 1000, 2)
                _record_tool_call(name, elapsed, success)
        return bridge

    return {
        name: _make_bridge(
            registry, name,
            _TOOL_PARAM_MAP.get(name, ("input", "string", ""))[0],
        )
        for name in rag_tools
    }


# ---------------------------------------------------------------------------
# RAG context retrieval
# ---------------------------------------------------------------------------

def parse_filter_params(
    excluded_dirs: Optional[str],
    excluded_files: Optional[str],
    included_dirs: Optional[str],
    included_files: Optional[str],
) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
    """Parse comma/newline-separated filter strings into lists."""
    def _split(val: Optional[str]) -> Optional[List[str]]:
        if not val:
            return None
        return [unquote(v) for v in val.split("\n") if v.strip()]

    return _split(excluded_dirs), _split(excluded_files), _split(included_dirs), _split(included_files)


def retrieve_rag_context(
    rag_instance: Any,
    query: str,
    language: Optional[str],
    file_path: Optional[str] = None,
) -> str:
    """Run RAG retrieval and format the result as context text.

    Returns the formatted context string (may be empty).
    """
    try:
        rag_query = query
        if file_path:
            rag_query = f"Contexts related to {file_path}"

        results = rag_instance(rag_query, language=language)
        if not results or not results[0].documents:
            return ""

        documents = results[0].documents
        docs_by_file: Dict[str, list] = {}
        for doc in documents:
            fp = doc.meta_data.get("file_path", "unknown")
            docs_by_file.setdefault(fp, []).append(doc)

        parts = []
        for fp, docs in docs_by_file.items():
            header = f"## File Path: {fp}\n\n"
            content = "\n\n".join(doc.text for doc in docs)
            parts.append(f"{header}{content}")

        return "\n\n" + "-" * 10 + "\n\n".join(parts)
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return ""


def fetch_file_content(
    repo_url: str,
    file_path: Optional[str],
    repo_type: Optional[str],
    token: Optional[str],
) -> str:
    """Fetch file content from the repository. Returns empty string on failure."""
    if not file_path:
        return ""
    try:
        return get_file_content(repo_url, file_path, repo_type or "github", token) or ""
    except Exception as exc:
        logger.error("File content fetch error for '%s': %s", file_path, exc)
        return ""


def build_conversation_history(rag_instance: Any) -> str:
    """Format the RAG memory into an XML conversation history string."""
    parts = []
    for turn_id, turn in rag_instance.memory().items():
        if not isinstance(turn_id, int) and hasattr(turn, "user_query") and hasattr(turn, "assistant_response"):
            parts.append(
                f"<turn>\n<user>{turn.user_query.query_str}</user>\n"
                f"<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>"
            )
    return "\n".join(parts)


def assemble_prompt(
    system_prompt: str,
    conversation_history: str,
    context_text: str,
    query: str,
    file_path: Optional[str] = None,
    file_content: str = "",
    memory_context: str = "",
    provider: str = "openai",
) -> str:
    """Assemble the final prompt from all components."""
    prompt = f"/no_think {system_prompt}\n\n"

    if conversation_history:
        prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

    if file_content:
        prompt += f'<currentFileContent path="{file_path}">\n{file_content}\n</currentFileContent>\n\n'

    if context_text.strip():
        prompt += f"<START_OF_CONTEXT>\n{context_text}\n<END_OF_CONTEXT>\n\n"
    else:
        prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

    if memory_context:
        prompt += f"<memory_context>\n{memory_context}\n</memory_context>\n\n"

    prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

    if provider == "ollama":
        prompt += " /no_think"

    return prompt


def build_memory_context(
    episodic_ctx: Optional[dict],
    knowledge_context: str = "",
) -> str:
    """Build the <memory_context> inner content from episodic + knowledge data."""
    parts = []
    if episodic_ctx:
        if episodic_ctx.get("session_summary"):
            parts.append(f"Session summary: {episodic_ctx['session_summary'][:500]}")
        if episodic_ctx.get("session_topics"):
            parts.append(f"Topics discussed: {', '.join(episodic_ctx['session_topics'][:5])}")
    if knowledge_context:
        parts.append(f"Related knowledge:\n{knowledge_context}")
    return "\n".join(parts)


def detect_modes(messages: list) -> Tuple[bool, bool]:
    """Detect [DEEP RESEARCH] and [AGENT] tags in messages.

    Strips the tags from the **last** message content.
    Returns ``(is_deep_research, is_agent_mode)``.
    """
    is_deep_research = False
    is_agent_mode = False

    # Determine mode from the LAST user message only.
    # Previous messages may carry stale tags from earlier mode selections;
    # the current mode is what the user chose for this turn.
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None) or ""
        if "[DEEP RESEARCH]" in content:
            is_deep_research = True
        if "[AGENT]" in content:
            is_agent_mode = True

        # Strip tags from the last message
        if is_deep_research:
            last.content = content.replace("[DEEP RESEARCH]", "").strip()
            content = last.content
        if is_agent_mode:
            last.content = content.replace("[AGENT]", "").strip()
    return is_deep_research, is_agent_mode
