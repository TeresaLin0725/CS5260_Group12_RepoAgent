# Agent Testing Patterns

## Project Test Setup

- Framework: **pytest** with `pytest-asyncio` for async tests
- Config: `pytest.ini` at project root
- Structure: `tests/unit/` for unit tests, `tests/integration/` for integration
- Run: `pytest tests/unit/test_agent_tools.py -v`

## Conventions

1. **Test classes** grouped by component: `TestToolRegistry`, `TestPlanner`, `TestScheduler`, `TestReActRunner`
2. **No external dependencies** in unit tests — mock LLM calls, RAG instances, file reads
3. **sys.path injection** at top of test file (project uses relative imports):
   ```python
   from pathlib import Path
   import sys
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root))
   ```
4. **Direct imports** from `api.agent.*` modules

## What to Test for Export Tools

### ToolRegistry

```python
def test_tool_registered():
    registry = build_export_tool_registry()
    tool = registry.get("GENERATE_YOURNAME")
    assert tool is not None
    assert tool.action_tag == "[ACTION:GENERATE_YOURNAME]"

def test_tool_keywords():
    registry = build_export_tool_registry()
    tool = registry.get("GENERATE_YOURNAME")
    assert "keyword1" in tool.keywords
```

### Planner

```python
def test_plan_direct_invocation():
    """Keyword match + generation intent → should_invoke=True"""
    registry = build_export_tool_registry()
    planner = RuleBasedPlanner(registry)
    plan = planner.plan("generate a keyword1 for this repo")
    assert plan.should_invoke is True
    assert plan.tool.name == "GENERATE_YOURNAME"

def test_plan_needs_reasoning():
    """Keyword match + reasoning marker → needs_reasoning=True"""
    planner = RuleBasedPlanner(build_export_tool_registry())
    plan = planner.plan("analyze this repo and recommend keyword1")
    assert plan.needs_reasoning is True
    assert plan.should_invoke is False

def test_plan_no_match():
    """No keywords match → should_invoke=False"""
    planner = RuleBasedPlanner(build_export_tool_registry())
    plan = planner.plan("what is this project about?")
    assert plan.should_invoke is False
    assert plan.tool is None

def test_plan_ambiguous():
    """Multiple tools match → ambiguous=True"""
    planner = RuleBasedPlanner(build_export_tool_registry())
    plan = planner.plan("generate a pdf and ppt")
    assert plan.ambiguous is True
```

### Scheduler

```python
def test_schedule_action():
    scheduler = AgentScheduler.default()
    result = scheduler.schedule("generate pdf report", "en")
    assert result.handled is True
    assert "[ACTION:GENERATE_PDF]" in result.content

def test_schedule_clarify():
    scheduler = AgentScheduler.default()
    result = scheduler.schedule("generate pdf and ppt", "en")
    assert result.handled is True
    assert "specify" in result.content.lower()

def test_second_stage_inference():
    scheduler = AgentScheduler.default()
    action = scheduler.infer_second_stage_action(
        query="generate documentation",
        assistant_response="I recommend creating a PDF report for comprehensive coverage.",
    )
    assert action == "[ACTION:GENERATE_PDF]"
```

## What to Test for ReAct Tools

### Async Tool Callable

```python
import pytest

@pytest.mark.asyncio
async def test_your_tool_success():
    """Tool returns formatted result on success."""
    mock_rag = MockRAG(documents=[...])
    tools = build_react_tools(mock_rag, language="en")
    result = await tools["your_tool"]("test input")
    assert "expected content" in result

@pytest.mark.asyncio
async def test_your_tool_error():
    """Tool returns error string, never raises."""
    mock_rag = MockRAG(raise_error=True)
    tools = build_react_tools(mock_rag, language="en")
    result = await tools["your_tool"]("test input")
    assert "error" in result.lower()
```

### ReAct Integration

```python
@pytest.mark.asyncio
async def test_react_runner_calls_tool():
    """ReActRunner invokes tool and uses observation."""
    call_log = []

    async def mock_tool(input_str: str) -> str:
        call_log.append(input_str)
        return "mock observation"

    async def mock_llm(prompt: str) -> str:
        if "Observation" not in prompt:
            return "Thought: I need info\nAction: mock_tool\nAction Input: test query"
        return "Thought: Got it\nFinal Answer: The answer based on mock observation."

    runner = ReActRunner(tools={"mock_tool": mock_tool}, max_iterations=3)
    chunks = []
    async for chunk in runner.run(
        query="test",
        system_prompt="You are a test agent.",
        initial_context="",
        llm_fn=mock_llm,
        language="en",
    ):
        chunks.append(chunk)

    assert len(call_log) == 1
    full_output = "".join(chunks)
    assert "mock observation" in full_output.lower() or "answer" in full_output.lower()
```

## Mock Patterns

### Mock RAG Instance

```python
class MockRAG:
    def __init__(self, documents=None, raise_error=False):
        self._documents = documents or []
        self._raise_error = raise_error

    def __call__(self, query, language="en"):
        if self._raise_error:
            raise RuntimeError("RAG error")
        from types import SimpleNamespace
        docs = [
            SimpleNamespace(text=d["text"], meta_data=d.get("meta", {}))
            for d in self._documents
        ]
        return [SimpleNamespace(documents=docs)]
```

### Mock LLM Callable

```python
async def mock_llm_fn(prompt: str) -> str:
    """Returns a canned response for testing."""
    return "Thought: I have enough context.\nFinal Answer: This is a test response."
```

## Common Pitfalls

- **Forgetting `@pytest.mark.asyncio`** on async test functions
- **Not mocking external calls** — never call real LLM/RAG in unit tests
- **Testing keyword overlap** — always verify new keywords don't collide with existing tools
- **Missing Chinese keyword tests** — test both English and Chinese variants
