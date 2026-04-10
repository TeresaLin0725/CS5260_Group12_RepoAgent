---
name: agent-tool-dev
description: 'Add new tools/capabilities to the RepoHelper Agent system. Use when: adding agent tools, extending export tools, creating ReAct tools, modifying planner keywords, updating scheduler actions, registering MCP tools, or expanding agent capabilities. Covers the full workflow from tool definition to test.'
argument-hint: 'Describe the tool you want to add (e.g., "add a code complexity analysis tool")'
---

# Agent Tool Development

## When to Use

- Adding a new **export tool** (like PDF/PPT/Video) to the deterministic scheduler
- Adding a new **ReAct tool** (like rag_search/read_file) for the reasoning loop
- Modifying planner keyword matching or scheduler action flow
- Registering tools in the MCP layer
- Writing tests for any agent component

## Architecture Overview

Load [architecture reference](./references/architecture.md) for the full system map.

```
api/agent/
├── tools/
│   ├── base.py            # AgentTool dataclass + ToolRegistry
│   ├── export_tools.py    # Deterministic export tools (PDF/PPT/Video)
│   └── search_tools.py    # ReAct async tools (rag_search, read_file)
├── planner.py             # RuleBasedPlanner: keyword matching → ToolPlan
├── scheduler.py           # AgentScheduler: 2-stage action dispatch
├── react.py               # ReActRunner: Think→Act→Observe loop
├── events.py              # AgentEventType enum + AgentEvent
├── state.py               # AgentRunState container
├── llm_utils.py           # Provider-agnostic LLM callable factory
└── __init__.py            # Public API exports
```

## Two Types of Tools

### Type 1: Export Tool (Deterministic)

Triggered by keyword matching in `RuleBasedPlanner`. Produces an `[ACTION:XXX]` tag that the frontend handles. No LLM reasoning needed.

**Examples**: GENERATE_PDF, GENERATE_PPT, GENERATE_VIDEO

**Files to modify**: `tools/export_tools.py`, `planner.py` (keywords), `scheduler.py` (preamble messages + Stage 2 signals)

### Type 2: ReAct Tool (Reasoning)

Called by the `ReActRunner` during its Think→Act→Observe loop. The LLM decides when and how to call these tools.

**Examples**: rag_search, read_file

**Files to modify**: `tools/search_tools.py`, `react.py` (tool descriptions in `_tool_help`)

## Procedure: Adding an Export Tool

Use [tool template](./assets/tool-template.py) as starting code.

### Step 1 — Define the tool in `api/agent/tools/export_tools.py`

```python
registry.register(
    AgentTool(
        name="GENERATE_YOURNAME",
        action_tag="[ACTION:GENERATE_YOURNAME]",
        description="Generate a repository YOUR_DESCRIPTION.",
        keywords=(
            "keyword1", "keyword2",   # English
            "关键词1", "关键词2",       # Chinese
        ),
    )
)
```

**Rules**:
- `name` must be `GENERATE_` prefixed and UPPER_SNAKE_CASE
- `action_tag` must be `[ACTION:{name}]`
- Keywords must include both English and Chinese variants
- Keywords should be specific enough to avoid ambiguity with existing tools

### Step 2 — Add preamble messages in `api/agent/scheduler.py`

Add entries to both language mappings in `_tool_preamble()`:

```python
# Chinese mapping
"GENERATE_YOURNAME": "我将为该仓库生成……。",
# English mapping
"GENERATE_YOURNAME": "I will generate ... for this repository.",
```

### Step 3 — Add Stage 2 signals in `api/agent/scheduler.py`

In `_infer_from_recommendation()`, add signal keywords:

```python
yourname_signals = ("keyword1", "keyword2", "关键词1")
```

And include in the scoring block:

```python
yourname_score = sum(1 for s in yourname_signals if s in normalized)
scores["[ACTION:GENERATE_YOURNAME]"] = yourname_score
```

### Step 4 — Update `api/agent/planner.py` keywords (if needed)

If your tool introduces new generation-intent words, add them to `self.generation_markers`.

### Step 5 — Implement the backend handler

Create the actual export logic (e.g., `api/yourname_export.py`) and wire it to the route in `api/simple_chat.py` or `api/routes/endpoints.py`.

### Step 6 — Write tests

Use [test template](./assets/test-template.py). Must cover:
- Tool registration and keyword matching
- Planner plan() returns correct ToolPlan
- Scheduler schedule() returns correct action tag
- Stage 2 inference detects recommendation

## Procedure: Adding a ReAct Tool

### Step 1 — Add the async callable in `api/agent/tools/search_tools.py`

```python
async def your_tool(input_str: str) -> str:
    """One-line description of what this tool does."""
    try:
        result = await some_operation(input_str)
        return format_result(result)
    except Exception as exc:
        logger.error("your_tool error: %s", exc)
        return f"Tool error: {exc}"

tools["your_tool"] = your_tool
```

**Rules**:
- Signature must be `async def name(input: str) -> str`
- Always return a string (never raise)
- Log errors and return user-friendly error string
- Truncate large outputs to avoid token bloat

### Step 2 — Register the tool description in `api/agent/react.py`

Add to the `_tool_help` dict inside `ReActRunner.run()`:

```python
_tool_help = {
    ...
    "your_tool": "Description of what the tool does and what input it expects.",
}
```

### Step 3 — (Optional) Register in MCP layer

If using MCP integration, register in `api/mcp/registry.py`:

```python
registry.register(MCPTool(
    name="your_tool",
    description="...",
    category=ToolCategory.ANALYSIS,  # or SEARCH, READ, etc.
    schema=ToolSchema(parameters=[
        ToolParameter(name="input", type="string", description="...", required=True),
    ]),
    handler=async_handler,
))
```

### Step 4 — Write tests

Test the tool callable in isolation (mock external dependencies).

## Checklist

Before submitting, verify:

- [ ] Tool defined with correct `name`, `action_tag`, `keywords`
- [ ] No keyword overlap with existing tools (check `export_tools.py`)
- [ ] Preamble messages added for both `en` and `zh` (export tools)
- [ ] Stage 2 signals added in `_infer_from_recommendation()` (export tools)
- [ ] Tool help text registered in `react.py` `_tool_help` (ReAct tools)
- [ ] Async callable follows `(str) -> str` signature (ReAct tools)
- [ ] Error handling returns string, never raises
- [ ] Unit tests pass: `pytest tests/unit/test_agent_tools.py -v`
- [ ] `__init__.py` exports updated if adding new public classes

## References

- [Architecture deep-dive](./references/architecture.md)
- [Testing patterns](./references/testing-patterns.md)
- [Tool template](./assets/tool-template.py)
- [Test template](./assets/test-template.py)
