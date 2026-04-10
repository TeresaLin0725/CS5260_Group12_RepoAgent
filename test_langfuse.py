"""Quick smoke test: send a test trace to Langfuse v4 and verify connectivity."""
import os, sys, logging, time

# Enable debug logging to see Langfuse errors
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env manually (no python-dotenv dependency needed)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"')

print("LANGFUSE_ENABLED:", os.environ.get("LANGFUSE_ENABLED"))
print("LANGFUSE_PUBLIC_KEY:", (os.environ.get("LANGFUSE_PUBLIC_KEY") or "")[:12] + "...")
print("LANGFUSE_BASE_URL:", os.environ.get("LANGFUSE_BASE_URL"))
print("LANGFUSE_HOST:", os.environ.get("LANGFUSE_HOST"))

from api.tracing.tracer import _langfuse_available, _is_enabled

print("langfuse SDK available:", _langfuse_available)
print("_is_enabled():", _is_enabled())

# Force a fresh tracer (bypass singleton)
from api.tracing.tracer import AgentTracer
t = AgentTracer()
print("Tracer enabled:", t.enabled)

if not t.enabled:
    print("ERROR: Tracer not enabled — check LANGFUSE_ENABLED and keys in .env")
    sys.exit(1)

# --- Verify auth ---
try:
    ok = t._client.auth_check()
    print("auth_check():", ok)
except Exception as e:
    print("auth_check() failed:", e)
    sys.exit(1)

# --- Create a full test trace mimicking a real agent request ---
trace = t.start_trace(
    "test_trace",
    user_id="test_user",
    session_id="test_session",
    metadata={"repo_id": "test/repo", "language": "zh"},
    tags=["test", "smoke"],
    input={"query": "Hello Langfuse v4! 这是一个测试追踪。"},
)
print("Trace created:", type(trace).__name__, "trace_id:", trace.trace_id)

# Simulate intent_scheduling span
with t.span(trace, "intent_scheduling", input={"query": "test query"}) as sched:
    time.sleep(0.05)
    sched.end(output={"handled": False, "events_count": 1})

# Simulate react_loop span with a generation and tool call
with t.span(trace, "react_loop", metadata={"max_iterations": 3, "tools": ["rag_search"]}) as react:
    with t.generation(react.span, "react_iteration_0", model="test-model", input="test prompt") as gen:
        time.sleep(0.05)
        gen.end(output="Thought: test\nFinal Answer: test answer", usage={"input_tokens": 100, "output_tokens": 50})

    with t.span(react.span, "tool_call_rag_search", input={"query": "test search"}) as tool:
        time.sleep(0.02)
        tool.end(output={"result_length": 500})

# Simulate memory_operations span
with t.span(trace, "memory_operations", metadata={"user_id": "test_user"}) as mem:
    time.sleep(0.02)
    mem.end(output={"episodic_turns": 2, "knowledge_entries": 1})

# Add scores
t.score(trace, "iteration_count", 1)
t.score(trace, "tools_called_count", 1)
t.score(trace, "forced_termination", 0)
t.score(trace, "latency_ms", 1234.5)
t.score(trace, "total_tokens", 150)

t.end_trace(trace, output={"response_length": 42, "status": "success"})

print()
print("=" * 50)
print("Test trace sent to Langfuse successfully!")
print(f"trace_id: {trace.trace_id}")
print("Open https://cloud.langfuse.com -> Traces to see it.")
print("=" * 50)
