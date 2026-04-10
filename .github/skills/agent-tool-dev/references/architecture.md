# Agent Architecture Reference

## System Overview

RepoHelper's Agent system is a two-stage action scheduler with optional multi-step ReAct reasoning. It handles user queries about repositories, deciding whether to invoke deterministic tools (exports) or engage in LLM-powered reasoning with tool access.

## Execution Flow

```
User Query (may contain [AGENT] or [DEEP RESEARCH] tags)
    │
    ├─ Tag stripping & mode detection
    │
    ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: AgentScheduler.schedule()         │
│  ├─ RuleBasedPlanner.plan(query)            │
│  │   ├─ Keyword match → matched_tools[]     │
│  │   ├─ Check generation_markers            │
│  │   └─ Check reasoning_markers             │
│  │                                          │
│  ├─ Ambiguous → clarify message             │
│  ├─ Needs reasoning → fall through to LLM   │
│  ├─ No match → fall through to LLM          │
│  └─ Single match + gen intent → ACTION tag  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  LLM RESPONSE (standard chat or ReAct)      │
│  ├─ Standard: single LLM call with context  │
│  └─ ReAct: multi-step Think→Act→Observe     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: infer_second_stage_action()       │
│  ├─ Check generation intent in query        │
│  ├─ Score format mentions in LLM response   │
│  └─ Return ACTION tag if confident          │
└─────────────────────────────────────────────┘
```

## Core Data Structures

### AgentTool (frozen dataclass) — `api/agent/tools/base.py`

```python
@dataclass(frozen=True)
class AgentTool:
    name: str            # "GENERATE_PDF" — UPPER_SNAKE_CASE
    action_tag: str      # "[ACTION:GENERATE_PDF]" — used by frontend
    description: str     # Human-readable purpose
    keywords: Iterable[str]  # Trigger words (en + zh)
```

### ToolRegistry — `api/agent/tools/base.py`

```python
class ToolRegistry:
    register(tool: AgentTool) -> None
    get(name: str) -> Optional[AgentTool]
    all() -> Iterable[AgentTool]
```

### ToolPlan — `api/agent/planner.py`

```python
@dataclass
class ToolPlan:
    should_invoke: bool        # True = execute tool immediately
    tool: Optional[AgentTool]
    ambiguous: bool            # Multiple tools matched equally
    needs_reasoning: bool      # Query needs LLM analysis first
    reason: Optional[str]      # Why this decision was made
```

### AgentScheduleResult — `api/agent/scheduler.py`

```python
@dataclass
class AgentScheduleResult:
    handled: bool              # True = scheduler produced a response
    content: Optional[str]     # Action tag or clarification message
    events: List[AgentEvent]   # Planning trace
```

### AgentEvent — `api/agent/events.py`

```python
class AgentEventType(str, Enum):
    PLAN_CREATED    # Plan generated
    TOOL_SELECTED   # Tool chosen for execution
    TOOL_SKIPPED    # Tool skipped (ambiguous/reasoning)
    REACT_STEP      # ReAct iteration
    REACT_TOOL_CALL # ReAct tool invocation
    REACT_FINISHED  # ReAct completed

@dataclass
class AgentEvent:
    event_type: AgentEventType
    message: str
    tool_name: Optional[str] = None
```

### ReActStep — `api/agent/react.py`

```python
@dataclass
class ReActStep:
    thought: str                   # LLM's reasoning
    action: Optional[str]          # Tool to call
    action_input: Optional[str]    # Tool input
    observation: Optional[str]     # Tool result
    is_final: bool                 # Has Final Answer
    final_answer: Optional[str]
```

## Planner Keywords

### generation_markers (triggers direct tool invocation)

```
generate, create, make, export, build, produce,
生成, 导出, 创建, 产出, 做一份, 进行生成, 来生成
```

### reasoning_markers (forces LLM-first flow)

```
introduce, overview, analyze, analysis, compare, why, which,
best, recommend, choose, select,
先介绍, 介绍, 分析, 比较, 为什么, 哪个, 最合适, 推荐, 选择
```

## Current Tool Inventory

### Export Tools (Deterministic)

| Name | Action Tag | Keywords |
|------|-----------|----------|
| GENERATE_PDF | `[ACTION:GENERATE_PDF]` | pdf, report, technical report, 报告, pdf报告 |
| GENERATE_PPT | `[ACTION:GENERATE_PPT]` | ppt, slides, presentation, deck, 演示, 幻灯片 |
| GENERATE_VIDEO | `[ACTION:GENERATE_VIDEO]` | video, walkthrough, overview video, 视频 |

### ReAct Tools (Reasoning Loop)

| Name | Input | Description |
|------|-------|-------------|
| rag_search | semantic query string | Search codebase via RAG embeddings, returns top 6 docs |
| read_file | file path string | Read file content from repository (requires repo_url) |

## MCP Integration

Tools can be wrapped in the MCP (Model Context Protocol) layer for standardized parameter validation and logging:

- `ToolCategory`: SEARCH, READ, WRITE, MEMORY, ANALYSIS, EXPORT, CUSTOM
- `ToolSchema`: defines parameters with type, description, required flag
- `MCPToolRegistry`: unified register/execute interface

## LLM Provider Support

`llm_utils.py` provides `create_llm_callable()` that returns `async (prompt: str) -> str` for:

- OpenAI / Azure OpenAI
- OpenRouter
- Ollama (strips `<think>` tags)
- AWS Bedrock
- DashScope (Alibaba)
- Google Gemini (sync → async via `asyncio.to_thread`)

## Entry Points

- **HTTP**: `api/simple_chat.py` → `POST /chat/completions/stream`
- **WebSocket**: `api/websocket_wiki.py` → delegates agent mode to HTTP handler
- **Public API**: `api/agent/__init__.py` exports `AgentScheduler`, `AgentScheduleResult`, `ReActRunner`
