"""
Template: Adding a new tool to the RepoHelper Agent system.

INSTRUCTIONS:
  1. Copy the section you need (Export Tool OR ReAct Tool)
  2. Replace all YOURNAME / your_tool / YOUR_DESCRIPTION placeholders
  3. Follow the SKILL.md procedure for which files to modify

This file is a reference — do NOT import it directly.
"""

# ============================================================================
# SECTION A: Export Tool Definition
# Add to: api/agent/tools/export_tools.py → build_export_tool_registry()
# ============================================================================

"""
registry.register(
    AgentTool(
        name="GENERATE_YOURNAME",
        action_tag="[ACTION:GENERATE_YOURNAME]",
        description="Generate a repository YOUR_DESCRIPTION.",
        keywords=(
            # English keywords (lowercase)
            "english_keyword1",
            "english_keyword2",
            # Chinese keywords
            "中文关键词1",
            "中文关键词2",
        ),
    )
)
"""


# ============================================================================
# SECTION B: Scheduler Preamble Messages
# Add to: api/agent/scheduler.py → _tool_preamble() both mappings
# ============================================================================

"""
# In the Chinese mapping dict:
"GENERATE_YOURNAME": "我将为该仓库生成一份……。",

# In the English mapping dict:
"GENERATE_YOURNAME": "I will generate a ... for this repository.",
"""


# ============================================================================
# SECTION C: Stage 2 Recommendation Signals
# Add to: api/agent/scheduler.py → _infer_from_recommendation()
# ============================================================================

"""
# 1. Define signal keywords:
yourname_signals = ("keyword1", "keyword2", "中文关键词1")

# 2. Add scoring line alongside existing pdf_score / ppt_score / video_score:
yourname_score = sum(1 for s in yourname_signals if s in normalized)

# 3. Add to scores dict:
scores["[ACTION:GENERATE_YOURNAME]"] = yourname_score
"""


# ============================================================================
# SECTION D: ReAct Tool (async callable)
# Add to: api/agent/tools/search_tools.py → build_react_tools()
# ============================================================================

"""
import logging
logger = logging.getLogger(__name__)

async def your_tool(input_str: str) -> str:
    \"\"\"Brief description of what this tool does. Input: what the input represents.\"\"\"
    try:
        # --- Your implementation here ---
        # Example: call an analysis function, query a database, etc.
        result = await some_async_operation(input_str)

        # Format the output as a readable string
        if not result:
            return "No results found."

        # Truncate if needed to avoid token bloat
        output = str(result)
        if len(output) > 10000:
            output = output[:10000] + "\\n... [truncated]"
        return output

    except Exception as exc:
        logger.error("your_tool error: %s", exc)
        return f"Tool error: {exc}"

# Register in tools dict:
tools["your_tool"] = your_tool
"""


# ============================================================================
# SECTION E: ReAct Tool Help Text
# Add to: api/agent/react.py → _tool_help dict in run()
# ============================================================================

"""
_tool_help = {
    "rag_search": "Search the codebase with a semantic query...",
    "read_file": "Read the full content of a specific file...",
    # Add your tool:
    "your_tool": "Description of what the tool does and what input it expects.",
}
"""


# ============================================================================
# SECTION F: MCP Registration (Optional)
# Add to: wherever _build_mcp_tools() is called, or api/mcp/registry.py
# ============================================================================

"""
from api.mcp.registry import MCPTool, ToolCategory, ToolSchema, ToolParameter

registry.register(MCPTool(
    name="your_tool",
    description="What this tool does.",
    category=ToolCategory.ANALYSIS,  # SEARCH, READ, WRITE, MEMORY, ANALYSIS, EXPORT, CUSTOM
    schema=ToolSchema(parameters=[
        ToolParameter(
            name="input",
            type="string",
            description="What the input represents.",
            required=True,
        ),
    ]),
    handler=your_tool_handler,
))
"""
