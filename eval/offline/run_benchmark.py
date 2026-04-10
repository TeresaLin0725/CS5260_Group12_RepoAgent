"""
Offline benchmark runner.

Loads test fixtures, executes each case against the target subsystem
(ReAct / Intent / RAG), collects raw results, computes metrics, and
generates a timestamped JSON report.

Usage:
    python -m eval.offline.run_benchmark [--subsystem react|intent|rag|all]
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.offline.metrics.react_metrics import (
    ReActCaseResult,
    ReActMetrics,
    aggregate_react_metrics,
    compute_react_metrics,
    judge_answer_relevance,
    judge_faithfulness,
)
from eval.offline.metrics.intent_metrics import (
    IntentCaseResult,
    compute_intent_metrics,
)
from eval.offline.metrics.rag_metrics import (
    RAGCaseResult,
    aggregate_rag_metrics,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPORTS_DIR = Path(__file__).parent / "reports"
BASELINES_DIR = Path(__file__).parent / "baselines"


# -----------------------------------------------------------------------
# Fixture loading
# -----------------------------------------------------------------------

def load_fixture(filename: str) -> List[Dict[str, Any]]:
    """Load a JSON fixture file from the fixtures directory."""
    path = FIXTURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Mock LLM for deterministic testing
# -----------------------------------------------------------------------

class MockLLM:
    """Deterministic LLM mock that replays pre-recorded golden responses.

    Supports two modes:
    1. **Exact match**: keyed by the exact prompt string.
    2. **Sequential**: returns responses in order regardless of prompt.

    Golden responses are stored in ``eval/offline/fixtures/golden_responses/``.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None, sequential: Optional[List[str]] = None):
        self._responses = responses or {}
        self._sequential = sequential or []
        self._call_count = 0

    async def __call__(self, prompt: str) -> str:
        self._call_count += 1
        if prompt in self._responses:
            return self._responses[prompt]
        if self._sequential:
            idx = min(self._call_count - 1, len(self._sequential) - 1)
            return self._sequential[idx]
        # Default: produce a Final Answer to prevent infinite loops
        return "Thought: I have enough context.\nFinal Answer: Based on the available information, I cannot provide a specific answer to this query."


class SequentialMockLLM:
    """Per-case mock LLM that replays a scripted sequence of responses.

    Each call returns the next response in the list. Once all responses
    are exhausted, repeats the last one (which should be a Final Answer).

    Used to simulate realistic ReAct loops so tool-tracking metrics
    (precision, recall, F1) reflect actual tool-call behaviour:
    - Iteration 0..N-1: Action blocks that call the expected tools
    - Last iteration:   Final Answer with the golden response text
    """

    def __init__(self, responses: List[str]):
        self._responses = responses
        self._call_idx = 0

    async def __call__(self, prompt: str) -> str:
        if self._call_idx < len(self._responses):
            response = self._responses[self._call_idx]
        else:
            response = self._responses[-1] if self._responses else (
                "Thought: I have enough context.\n"
                "Final Answer: Based on the available information, I cannot provide a specific answer."
            )
        self._call_idx += 1
        return response


def _build_case_sequential_responses(case: Dict[str, Any]) -> List[str]:
    """Build an ordered list of mock LLM responses for a single test case.

    Strategy:
    - Cases with no expected_tools → single Final Answer response.
    - Cases with N expected_tools → N Action responses then Final Answer.

    This ensures the mock ReAct loop actually calls the expected tools so that
    tool-tracking metrics (precision, recall, F1) are meaningful.
    """
    expected_tools: List[str] = case.get("expected_tools", [])
    golden_answer: str = case.get("golden_answer", "No information available.")
    query: str = case.get("query", "")

    if not expected_tools:
        return [
            f"Thought: I can answer this directly from context.\n"
            f"Final Answer: {golden_answer}"
        ]

    responses: List[str] = []
    for i, tool in enumerate(expected_tools):
        thought = (
            "I need to gather information to answer this question."
            if i == 0
            else f"I need additional context. Let me also use {tool}."
        )
        responses.append(
            f"Thought: {thought}\n"
            f"Action: {tool}\n"
            f"Action Input: {query}"
        )

    responses.append(
        f"Thought: I now have sufficient information from the tool results.\n"
        f"Final Answer: {golden_answer}"
    )
    return responses


# -----------------------------------------------------------------------
# ReAct benchmark
# -----------------------------------------------------------------------

async def run_react_benchmark(
    cases: Optional[List[Dict[str, Any]]] = None,
    llm_fn: Optional[Callable] = None,
    judge_llm_fn: Optional[Callable] = None,
    tools: Optional[Dict] = None,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Run ReAct benchmark.

    Parameters
    ----------
    cases : list of test case dicts (loaded from react_testcases.json)
    llm_fn : async LLM callable for ReAct loop (defaults to SequentialMockLLM per case)
    judge_llm_fn : async LLM callable for LLM-as-Judge (optional)
    tools : dict of tool_name -> async callable (defaults to mock tools)
    max_iterations : max ReAct iterations per case
    """
    from api.agent.react import ReActRunner

    if cases is None:
        cases = load_fixture("react_testcases.json")

    # llm_fn=None means "use per-case SequentialMockLLM" (mock mode)
    mock_mode = llm_fn is None

    if tools is None:
        tools = _build_mock_tools()

    all_results: List[ReActCaseResult] = []
    all_metrics: List[ReActMetrics] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        language = case.get("language", "en")
        expected_tools: List[str] = case.get("expected_tools", [])
        golden_answer: Optional[str] = case.get("golden_answer")
        # Use case-level max_iterations when available for correct efficiency score
        case_max_iter = case.get("expected_max_iterations", max_iterations)

        logger.info("Running ReAct case: %s", case_id)

        # In mock mode: build a per-case sequential mock that actually calls expected tools
        active_llm_fn = (
            SequentialMockLLM(_build_case_sequential_responses(case))
            if mock_mode
            else llm_fn
        )

        # Track tool calls AND observations for faithfulness scoring
        tools_called: List[str] = []
        steps: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        forced = False

        tracked_tools = _wrap_tools_for_tracking(tools, tools_called, steps)
        tracked_runner = ReActRunner(
            tools=tracked_tools, max_iterations=case_max_iter
        )

        try:
            collected_output: List[str] = []
            async for chunk in tracked_runner.run(
                query=query,
                system_prompt="You are a helpful code assistant.",
                initial_context="",
                llm_fn=active_llm_fn,
                language=language,
            ):
                collected_output.append(chunk)

            if collected_output:
                final_answer = collected_output[-1]

            forced = (
                bool(final_answer)
                and "unable to gather" in final_answer.lower()
            )

        except Exception as exc:
            logger.error("ReAct case %s failed: %s", case_id, exc)
            final_answer = None

        # Iteration count: one tool call = one iteration, plus the final answer call
        actual_iterations = max(1, len(tools_called) + (1 if final_answer else 0))

        result = ReActCaseResult(
            case_id=case_id,
            category=case.get("category", "unknown"),
            query=query,
            final_answer=final_answer,
            tools_called=tools_called,
            iterations=actual_iterations,
            max_iterations=case_max_iter,
            forced_termination=forced,
            steps=steps,
        )
        all_results.append(result)

        metrics = compute_react_metrics(result, expected_tools, golden_answer)

        # LLM-as-Judge (requires real judge LLM; skipped in pure mock mode)
        if judge_llm_fn and final_answer:
            metrics.answer_relevance = await judge_answer_relevance(
                query, final_answer, judge_llm_fn
            )
            observations = [s["observation"] for s in steps if s.get("observation")]
            if observations:
                metrics.faithfulness = await judge_faithfulness(
                    observations, final_answer, judge_llm_fn
                )

        all_metrics.append(metrics)

    aggregated = aggregate_react_metrics(all_metrics)

    return {
        "subsystem": "react",
        "mock_mode": mock_mode,
        "total_cases": len(cases),
        "aggregated_metrics": aggregated,
        "per_case_metrics": [asdict(m) for m in all_metrics],
    }


# -----------------------------------------------------------------------
# Intent benchmark
# -----------------------------------------------------------------------

async def run_intent_benchmark(
    cases: Optional[List[Dict[str, Any]]] = None,
    classifier_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run intent classification benchmark.

    Parameters
    ----------
    cases : list of test case dicts (loaded from intent_testcases.json)
    classifier_fn : async (query) -> IntentResult or dict with intent/confidence
    """
    if cases is None:
        cases = load_fixture("intent_testcases.json")

    mock_mode = classifier_fn is None
    if classifier_fn is None:
        classifier_fn = _build_mock_intent_classifier(cases)

    all_results: List[IntentCaseResult] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        expected_intent = case["expected_intent"]

        logger.info("Running intent case: %s", case_id)

        try:
            result = await classifier_fn(query)
            if result is None:
                # Classifier declined (low confidence / ambiguous)
                predicted_intent = None
                predicted_confidence = 0.0
                source = "declined"
                logger.debug(
                    "Intent case %s: classifier returned None (declined to classify)",
                    case_id,
                )
            elif hasattr(result, "intent"):
                predicted_intent = result.intent
                predicted_confidence = result.confidence
                source = getattr(result, "source", "unknown")
            elif isinstance(result, dict):
                predicted_intent = result.get("intent")
                predicted_confidence = result.get("confidence", 0.0)
                source = result.get("source", "unknown")
            else:
                predicted_intent = None
                predicted_confidence = 0.0
                source = "unknown"
                logger.warning(
                    "Intent case %s: unexpected result type %s",
                    case_id, type(result).__name__,
                )
        except Exception as exc:
            logger.error("Intent case %s failed: %s", case_id, exc)
            predicted_intent = None
            predicted_confidence = 0.0
            source = "error"

        all_results.append(
            IntentCaseResult(
                case_id=case_id,
                query=query,
                expected_intent=expected_intent,
                predicted_intent=predicted_intent,
                predicted_confidence=predicted_confidence,
                expected_confidence_min=case.get("expected_confidence_min", 0.0),
                is_ambiguous=case.get("is_ambiguous", False),
                handled=predicted_intent is not None,
                source=source,
                category=case.get("category", ""),
            )
        )

    metrics = compute_intent_metrics(all_results)

    # Log summary diagnostics
    none_count = sum(1 for r in all_results if r.predicted_intent is None)
    error_count = sum(1 for r in all_results if r.source == "error")
    declined_count = sum(1 for r in all_results if r.source == "declined")
    if none_count > 0:
        logger.warning(
            "Intent benchmark: %d/%d cases got no prediction "
            "(declined=%d, error=%d, unknown=%d). "
            "Check classifier connection / confidence threshold.",
            none_count, len(all_results), declined_count, error_count,
            none_count - declined_count - error_count,
        )

    return {
        "subsystem": "intent",
        "mock_mode": mock_mode,
        "total_cases": len(cases),
        "aggregated_metrics": asdict(metrics),
        "per_case_results": [asdict(r) for r in all_results],
    }


# -----------------------------------------------------------------------
# RAG benchmark
# -----------------------------------------------------------------------

_REPOS_DIR = Path(__file__).parent / ".repos"


async def run_rag_benchmark(
    cases: Optional[List[Dict[str, Any]]] = None,
    retriever_fn: Optional[Callable] = None,
    repo_filter: Optional[str] = None,
    use_hybrid: bool = False,
    embedder_type: str = "openai",
) -> Dict[str, Any]:
    """Run RAG retrieval benchmark.

    Parameters
    ----------
    cases : list of test case dicts (loaded from rag_testcases.json)
    retriever_fn : async (query) -> list of file paths (ranked)
        When ``None`` and a test case has a ``repo_name`` field that matches a
        directory under ``eval/offline/.repos/``, a local BM25 retriever is
        built automatically (no embedding API required).  Cases without
        ``repo_name`` fall back to the mock retriever.
    repo_filter : str or None
        When set, only run test cases where ``repo_name == repo_filter``.
        Use this to evaluate one repo at a time for clean per-repo comparison.
    use_hybrid : bool
        When ``True``, build FAISS + BM25 hybrid retrievers for local repos
        instead of BM25-only.  Requires an embedding API key.
    embedder_type : str
        Embedder provider for hybrid mode ("openai", "google", "bedrock").

    Notes
    -----
    In pure mock mode (no ``repo_name`` on any case) the retriever returns the
    golden files directly — Hit@K / MRR / NDCG are all 1.0 and latency is 0 ms.
    Cases with ``repo_name`` use a real BM25 index built from the local repo,
    producing genuine retrieval quality and latency figures.
    """
    if cases is None:
        cases = load_fixture("rag_testcases.json")

    # Filter to a single repo when requested (per-repo evaluation mode)
    if repo_filter:
        cases = [c for c in cases if c.get("repo_name") == repo_filter]
        if not cases:
            logger.warning(
                "No test cases found for repo_filter=%r. "
                "Check that rag_testcases.json has cases with repo_name=%r.",
                repo_filter, repo_filter,
            )
            return {
                "subsystem": "rag",
                "mock_mode": False,
                "repo_filter": repo_filter,
                "total_cases": 0,
                "aggregated_metrics": {},
                "per_repo_metrics": {},
                "per_case_results": [],
            }
        logger.info("repo_filter=%r: running %d cases", repo_filter, len(cases))

    # Determine overall mode
    repo_cases = [c for c in cases if c.get("repo_name") and
                  (_REPOS_DIR / c["repo_name"]).exists()]
    plain_cases = [c for c in cases if c not in repo_cases]

    # If a global retriever_fn is given, use it for everything
    global_mock = retriever_fn is None and not repo_cases
    if retriever_fn is None and not repo_cases:
        retriever_fn = _build_mock_retriever(cases)

    # Build per-repo retrievers (lazy, cached)
    # Uses hybrid (FAISS+BM25) when use_hybrid=True, otherwise BM25-only.
    repo_retrievers: Dict[str, Callable] = {}

    def _get_repo_retriever(repo_name: str) -> Callable:
        if repo_name not in repo_retrievers:
            repo_dir = _REPOS_DIR / repo_name
            if use_hybrid:
                from eval.offline.real_backends import create_local_hybrid_retriever
                repo_retrievers[repo_name] = create_local_hybrid_retriever(
                    repo_dir, embedder_type=embedder_type,
                )
                logger.info("Built local HYBRID (FAISS+BM25) index for repo: %s", repo_name)
            else:
                from eval.offline.real_backends import create_local_bm25_retriever
                repo_retrievers[repo_name] = create_local_bm25_retriever(repo_dir)
                logger.info("Built local BM25 index for repo: %s", repo_name)
        return repo_retrievers[repo_name]

    all_results: List[RAGCaseResult] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        golden_files = case.get("golden_files", [])
        repo_name: Optional[str] = case.get("repo_name")

        # Choose retriever for this case
        if retriever_fn is not None:
            active_retriever = retriever_fn
        elif repo_name and (_REPOS_DIR / repo_name).exists():
            active_retriever = _get_repo_retriever(repo_name)
        else:
            # Per-case mock fallback (returns golden files)
            active_retriever = _build_mock_retriever([case])

        logger.info("Running RAG case: %s (repo=%s)", case_id, repo_name or "mock")

        t0 = time.perf_counter()
        try:
            retrieved = await active_retriever(query)
            if isinstance(retrieved, list) and retrieved and hasattr(retrieved[0], "documents"):
                retrieved_files = []
                for ro in retrieved:
                    for doc in ro.documents:
                        path = getattr(doc, "metadata", {}).get("file_path", "")
                        if path:
                            retrieved_files.append(path)
                retrieved = retrieved_files
            elif isinstance(retrieved, list) and retrieved and isinstance(retrieved[0], dict):
                retrieved = [r.get("file_path", r.get("path", "")) for r in retrieved]
        except Exception as exc:
            logger.error("RAG case %s failed: %s", case_id, exc)
            retrieved = []

        latency = (time.perf_counter() - t0) * 1000

        all_results.append(
            RAGCaseResult(
                case_id=case_id,
                query=query,
                golden_files=golden_files,
                retrieved_files=retrieved if isinstance(retrieved, list) else [],
                retrieval_latency_ms=round(latency, 2),
                tags=case.get("tags", []) + ([f"repo:{repo_name}"] if repo_name else []),
            )
        )

    metrics = aggregate_rag_metrics(all_results)

    # Per-repo breakdown
    repo_breakdown: Dict[str, Any] = {}
    from eval.offline.metrics.rag_metrics import aggregate_rag_metrics as _agg
    for repo_name in {c.get("repo_name") for c in cases if c.get("repo_name")}:
        repo_results = [r for r in all_results if f"repo:{repo_name}" in r.tags]
        if repo_results:
            rm = _agg(repo_results)
            repo_breakdown[repo_name] = {
                "total_cases": rm.total_cases,
                "hit_at_1": rm.hit_at_1,
                "hit_at_3": rm.hit_at_3,
                "mrr": rm.mrr,
                "latency_p50_ms": rm.latency_p50_ms,
                "latency_p95_ms": rm.latency_p95_ms,
            }

    result: Dict[str, Any] = {
        "subsystem": "rag",
        "mock_mode": global_mock,
        "repo_filter": repo_filter,
        "total_cases": len(cases),
        "aggregated_metrics": asdict(metrics),
        "per_repo_metrics": repo_breakdown,
        "per_case_results": [asdict(r) for r in all_results],
    }
    if global_mock:
        result["mock_mode_note"] = (
            "Scores are 1.0 and latency is 0 ms because the mock retriever "
            "returns golden files directly. Add repo_name to test cases or "
            "run with --backend real for meaningful retrieval quality."
        )
    elif repo_cases:
        if use_hybrid:
            result["hybrid_note"] = (
                f"Cases with repo_name use a HYBRID (FAISS + BM25) index "
                f"built from local .repos/ directories with {embedder_type} embeddings."
            )
        else:
            result["local_bm25_note"] = (
                f"Cases with repo_name use a real BM25 index built from local "
                f".repos/ directories. FAISS (semantic) retrieval requires "
                f"--use-hybrid with an embedding provider."
            )
    return result


# -----------------------------------------------------------------------
# Mock helpers
# -----------------------------------------------------------------------

def _build_mock_tools() -> Dict[str, Callable]:
    """Build minimal mock tools for ReAct testing."""

    async def mock_rag_search(query: str) -> str:
        return f"Found relevant code for: {query}\n\n```python\n# Mock search result\nclass Example:\n    pass\n```"

    async def mock_read_file(path: str) -> str:
        return f"# Contents of {path}\n\nclass MockModule:\n    '''Mock file content'''\n    pass"

    async def mock_list_repo_files(path: str) -> str:
        return "api/\n  agent/\n    react.py\n    scheduler.py\n  retriever.py\n  rag.py"

    async def mock_code_grep(pattern: str) -> str:
        return f"# Search results for: {pattern}\napi/agent/react.py:42: class ReActRunner:"

    async def mock_memory_search(query: str) -> str:
        return "No relevant memories found."

    return {
        "rag_search": mock_rag_search,
        "read_file": mock_read_file,
        "list_repo_files": mock_list_repo_files,
        "code_grep": mock_code_grep,
        "memory_search": mock_memory_search,
    }


def _wrap_tools_for_tracking(
    tools: Dict[str, Callable],
    calls_tracker: List[str],
    steps_tracker: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Callable]:
    """Wrap each tool to track calls and capture observations for faithfulness scoring."""
    wrapped = {}
    for name, fn in tools.items():

        async def _tracked(
            input_str: str,
            _name=name,
            _fn=fn,
        ) -> str:
            calls_tracker.append(_name)
            result = await _fn(input_str)
            if steps_tracker is not None:
                steps_tracker.append({
                    "action": _name,
                    "action_input": input_str,
                    "observation": result,
                })
            return result

        wrapped[name] = _tracked
    return wrapped


def _build_mock_intent_classifier(cases: List[Dict[str, Any]]) -> Callable:
    """Build a mock classifier that returns perfect predictions (for structural testing)."""
    case_map = {c["query"]: c for c in cases}

    async def mock_classify(query: str):
        if query in case_map:
            case = case_map[query]
            return {
                "intent": case["expected_intent"],
                # Perfect mock always has confidence 1.0 for correct predictions
                "confidence": 1.0,
                "source": "mock",
            }
        return {"intent": "GENERAL_CHAT", "confidence": 1.0, "source": "mock"}

    return mock_classify


def _build_mock_retriever(cases: List[Dict[str, Any]]) -> Callable:
    """Build a mock retriever that returns golden files (for structural testing)."""
    case_map = {c["query"]: c for c in cases}

    async def mock_retrieve(query: str) -> List[str]:
        if query in case_map:
            golden = case_map[query].get("golden_files", [])
            # Simulate retrieval by putting golden files first + some noise
            noise = ["api/__init__.py", "api/config.py", "api/main.py"]
            return golden + [f for f in noise if f not in golden]
        return ["api/__init__.py", "api/config.py"]

    return mock_retrieve


# -----------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------

def generate_report(
    results: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate evaluation report with optional baseline comparison."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    if baseline:
        report["baseline_comparison"] = _compare_with_baseline(
            results, baseline
        )

    return report


def _compare_with_baseline(
    current: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare current results with baseline, flag regressions."""
    comparison = {}
    current_agg = current.get("aggregated_metrics", {})
    baseline_agg = baseline.get("aggregated_metrics", {})

    for key in current_agg:
        if key in baseline_agg and isinstance(current_agg[key], (int, float)):
            current_val = current_agg[key]
            baseline_val = baseline_agg[key]
            if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                delta = current_val - baseline_val
                comparison[key] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "delta": round(delta, 4),
                    "regression": delta < -0.05,  # 5% regression threshold
                }

    return comparison


def save_report(report: Dict[str, Any], subsystem: str = "all") -> Path:
    """Save report as JSON file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{subsystem}_{ts}.json"
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Report saved: %s", path)
    return path


def load_baseline(subsystem: str) -> Optional[Dict[str, Any]]:
    """Load the latest baseline for a subsystem."""
    if not BASELINES_DIR.exists():
        return None
    # Find latest baseline file
    pattern = f"baseline_{subsystem}_v*.json"
    candidates = sorted(BASELINES_DIR.glob(pattern), reverse=True)
    if not candidates:
        return None
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

async def run_all_benchmarks(
    subsystems: Optional[List[str]] = None,
    judge_llm_fn: Optional[Callable] = None,
    backend_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    repo_filter: Optional[str] = None,
    use_hybrid: bool = False,
    embedder_type: str = "openai",
) -> Dict[str, Any]:
    """Run benchmarks for specified subsystems and generate combined report.

    Parameters
    ----------
    subsystems : list of str
        Which subsystems to run ("react", "intent", "rag").
    judge_llm_fn : callable or None
        LLM callable for LLM-as-Judge (legacy param).
    backend_overrides : dict or None
        Per-subsystem override kwargs, e.g.:
        ``{"react": {"llm_fn": ..., "tools": ...}, "intent": {"classifier_fn": ...}}``
    repo_filter : str or None
        When set, only run RAG test cases for this repo_name.  Use with
        ``--repo-filter`` CLI flag for per-repo evaluation.
    use_hybrid : bool
        When ``True``, use FAISS + BM25 hybrid for local repo retrieval
        instead of BM25-only.  Requires embedding API.
    embedder_type : str
        Embedder provider for hybrid mode.
    """
    if subsystems is None:
        subsystems = ["react", "intent", "rag"]

    if backend_overrides is None:
        backend_overrides = {}

    combined: Dict[str, Any] = {}

    for sub in subsystems:
        logger.info("=" * 60)
        logger.info("Running %s benchmark", sub)
        logger.info("=" * 60)

        overrides = backend_overrides.get(sub, {})

        if sub == "react":
            result = await run_react_benchmark(
                llm_fn=overrides.get("llm_fn"),
                judge_llm_fn=overrides.get("judge_llm_fn", judge_llm_fn),
                tools=overrides.get("tools"),
            )
        elif sub == "intent":
            result = await run_intent_benchmark(
                classifier_fn=overrides.get("classifier_fn"),
            )
        elif sub == "rag":
            result = await run_rag_benchmark(
                retriever_fn=overrides.get("retriever_fn"),
                repo_filter=repo_filter,
                use_hybrid=use_hybrid,
                embedder_type=embedder_type,
            )
        else:
            logger.warning("Unknown subsystem: %s", sub)
            continue

        # Compare with baseline
        baseline = load_baseline(sub)
        combined[sub] = generate_report(result, baseline)

    # Save combined report — include repo_filter in filename when set
    subsystem_tag = "_".join(subsystems)
    if repo_filter:
        subsystem_tag = f"{subsystem_tag}_{repo_filter}"
    report_path = save_report(combined, subsystem=subsystem_tag)
    combined["report_path"] = str(report_path)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Run offline evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode (default, no API keys needed):
  python -m eval.offline.run_benchmark --subsystem all

  # Mock mode — per-repo BM25 (no API keys, uses local .repos/):
  python -m eval.offline.run_benchmark --subsystem rag

  # Hybrid mode — per-repo FAISS+BM25 (needs embedding API key):
  python -m eval.offline.run_benchmark --subsystem rag --use-hybrid
  python -m eval.offline.run_benchmark --subsystem rag --use-hybrid --embedder-type google

  # Real backend — Intent only (fastest, no repo needed):
  python -m eval.offline.run_benchmark --backend real --provider openai --model gpt-4o --subsystem intent

  # Real backend — RAG for one repo (FAISS+BM25, needs embedding API):
  python -m eval.offline.run_benchmark --backend real --provider openai --model gpt-4o --subsystem rag --repo-filter pallets_flask

  # Real backend — explicit repo URL:
  python -m eval.offline.run_benchmark --backend real --provider openai --model gpt-4o --repo https://github.com/user/repo

  # Real backend — Ollama (local, no cloud API):
  python -m eval.offline.run_benchmark --backend real --provider ollama --model qwen3:4b --subsystem intent
""",
    )
    parser.add_argument(
        "--subsystem",
        choices=["react", "intent", "rag", "all"],
        default="all",
        help="Which subsystem to evaluate (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "real"],
        default="mock",
        help="Backend mode: 'mock' for deterministic testing, 'real' for actual LLM/retriever (default: mock)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "ollama", "openrouter", "bedrock", "azure", "dashscope"],
        default="openai",
        help="LLM provider (used when --backend=real, default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (used when --backend=real, default: provider's default model)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository URL for ReAct/RAG evaluation (used when --backend=real)",
    )
    parser.add_argument(
        "--repo-type",
        choices=["github", "gitlab", "bitbucket"],
        default="github",
        help="Repository type (default: github)",
    )
    parser.add_argument(
        "--repo-token",
        type=str,
        default=None,
        help="Access token for private repositories",
    )
    parser.add_argument(
        "--intent-mode",
        choices=["llm", "embedding"],
        default="llm",
        help="Intent classifier mode (default: llm)",
    )
    parser.add_argument(
        "--repo-filter",
        type=str,
        default=None,
        metavar="REPO_NAME",
        help=(
            "Evaluate only test cases for this repo (e.g. 'pallets_flask'). "
            "For --backend real, auto-resolves to eval/offline/.repos/<REPO_NAME>. "
            "Available: pallets_flask, psf_requests, encode_httpx, "
            "pydantic_pydantic, tiangolo_fastapi"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help=(
            "Use FAISS + BM25 hybrid retrieval for local .repos/ evaluation "
            "instead of BM25-only. Requires an embedding API key "
            "(OPENAI_API_KEY, GOOGLE_API_KEY, etc.)."
        ),
    )
    parser.add_argument(
        "--embedder-type",
        choices=["openai", "google", "bedrock"],
        default="openai",
        help="Embedder provider for --use-hybrid mode (default: openai)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    subsystems = ["react", "intent", "rag"] if args.subsystem == "all" else [args.subsystem]

    # Build backend kwargs based on mode
    backend_kwargs: Dict[str, Any] = {}

    if args.backend == "real":
        from eval.offline.real_backends import (
            create_real_llm,
            create_real_intent_classifier,
            create_real_retriever,
            create_real_tools,
        )

        logger.info("=" * 60)
        logger.info("REAL BACKEND MODE: provider=%s model=%s", args.provider, args.model)
        logger.info("=" * 60)

        # LLM for ReAct + LLM-as-Judge
        if "react" in subsystems:
            if not args.repo:
                logger.warning(
                    "ReAct benchmark requires --repo for real tools. "
                    "Will use mock tools but real LLM."
                )
            backend_kwargs["react"] = {
                "llm_fn": create_real_llm(args.provider, args.model),
                "judge_llm_fn": create_real_llm(args.provider, args.model),
            }
            if args.repo:
                backend_kwargs["react"]["tools"] = create_real_tools(
                    repo_url=args.repo,
                    repo_type=args.repo_type,
                    access_token=args.repo_token,
                    provider=args.provider,
                    model=args.model,
                )

        # Intent classifier
        if "intent" in subsystems:
            backend_kwargs["intent"] = {
                "classifier_fn": create_real_intent_classifier(
                    mode=args.intent_mode,
                    provider=args.provider,
                    model=args.model,
                ),
            }

        # RAG retriever
        if "rag" in subsystems:
            # Determine repo path: --repo takes priority; --repo-filter auto-resolves local path
            repo_path = args.repo
            if not repo_path and args.repo_filter:
                local_path = _REPOS_DIR / args.repo_filter
                if local_path.exists():
                    repo_path = str(local_path)
                    logger.info(
                        "Auto-resolved --repo-filter=%r to local path: %s",
                        args.repo_filter, repo_path,
                    )
                else:
                    logger.warning(
                        "--repo-filter=%r: local path not found at %s. "
                        "Will use BM25-only mock mode.",
                        args.repo_filter, local_path,
                    )

            if not repo_path:
                logger.warning(
                    "RAG benchmark requires --repo or --repo-filter for real retriever. "
                    "Will use mock retriever."
                )
            else:
                backend_kwargs["rag"] = {
                    "retriever_fn": create_real_retriever(
                        repo_url=repo_path,
                        repo_type=args.repo_type,
                        access_token=args.repo_token,
                        provider=args.provider,
                        model=args.model,
                    ),
                }

    results = asyncio.run(
        run_all_benchmarks(
            subsystems=subsystems,
            backend_overrides=backend_kwargs,
            repo_filter=args.repo_filter,
            use_hybrid=args.use_hybrid,
            embedder_type=args.embedder_type,
        )
    )

    # Print summary via show_report
    try:
        from eval.offline.show_report import print_report
        report_path = results.get("report_path")
        if report_path:
            print_report(Path(report_path))
    except Exception:
        # Fallback: simple summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for sub in subsystems:
            if sub in results:
                sub_report = results[sub]
                sub_results = sub_report.get("results", {})
                agg = sub_results.get("aggregated_metrics", {})
                print(f"\n[{sub.upper()}]")
                for key, val in agg.items():
                    if isinstance(val, (int, float)):
                        print(f"  {key}: {val}")
                    elif isinstance(val, dict) and key != "by_category":
                        print(f"  {key}: ...")

        print(f"\nReport saved: {results.get('report_path', 'N/A')}")


if __name__ == "__main__":
    main()
