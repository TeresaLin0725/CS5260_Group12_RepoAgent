"""
Pytest integration for offline evaluation.

Runs benchmark suites as pytest tests so they integrate with CI pipelines.
Each subsystem gets its own test class with gate-check assertions.

Usage:
    pytest tests/eval/ -v
    pytest tests/eval/test_offline_eval.py -v -k react
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.offline.run_benchmark import (
    run_react_benchmark,
    run_intent_benchmark,
    run_rag_benchmark,
    generate_report,
    save_report,
    load_baseline,
)
from eval.offline.ci_gates import (
    REACT_GATES,
    INTENT_GATES,
    RAG_GATES,
    check_gates,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# -----------------------------------------------------------------------
# ReAct evaluation tests
# -----------------------------------------------------------------------

class TestReActEvaluation:
    """Offline evaluation tests for the ReAct reasoning loop."""

    @pytest.fixture(autouse=True, scope="class")
    def react_results(self, request):
        """Run ReAct benchmark once and share results across tests."""
        results = _run_async(run_react_benchmark())
        request.cls._results = results
        request.cls._metrics = results["aggregated_metrics"]

    def test_benchmark_completes(self):
        """Benchmark should complete without errors."""
        assert self._results is not None
        assert self._results["total_cases"] > 0

    @pytest.mark.xfail(reason="MockLLM always returns Final Answer without calling tools. Real gate check requires actual LLM.")
    def test_tool_f1_gate(self):
        """Tool F1 score should meet the CI gate threshold."""
        gate = REACT_GATES["avg_tool_f1"]
        value = self._metrics.get("avg_tool_f1", 0)
        assert value >= gate["min"], (
            f"Tool F1 {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_forced_termination_rate_gate(self):
        """Forced termination rate should not exceed the CI gate threshold."""
        gate = REACT_GATES["forced_termination_rate"]
        value = self._metrics.get("forced_termination_rate", 1.0)
        assert value <= gate["max"], (
            f"Forced termination rate {value:.4f} > threshold {gate['max']} ({gate['description']})"
        )

    def test_tool_error_rate_gate(self):
        """Tool error rate should not exceed the CI gate threshold."""
        gate = REACT_GATES["avg_tool_error_rate"]
        value = self._metrics.get("avg_tool_error_rate", 1.0)
        assert value <= gate["max"], (
            f"Tool error rate {value:.4f} > threshold {gate['max']} ({gate['description']})"
        )

    def test_iteration_efficiency_reasonable(self):
        """Average iteration efficiency should be positive."""
        value = self._metrics.get("avg_iteration_efficiency", 0)
        assert value > 0, "Iteration efficiency should be positive"

    def test_direct_answer_category(self):
        """Direct answer cases should have high tool precision (no unnecessary tools)."""
        by_category = self._metrics.get("by_category", {})
        direct = by_category.get("direct_answer", {})
        if direct:
            assert direct.get("avg_tool_f1", 0) >= 0.5, (
                "Direct answer cases should have high tool selection quality"
            )
    @pytest.mark.xfail(reason="MockLLM doesn't trigger real tool calls; gate violations expected with mock.")
    def test_all_react_gates(self):
        """Run all ReAct gates and report violations."""
        violations = check_gates(self._metrics, REACT_GATES, "react")
        if violations:
            msg_parts = ["ReAct CI gate violations:"]
            for v in violations:
                msg_parts.append(
                    f"  - {v['metric']}: {v['value']:.4f} "
                    f"({'<' if v['direction'] == 'min' else '>'} {v['threshold']}) "
                    f"[{v['description']}]"
                )
            # Only fail on non-LLM-judge gates (since those need a real LLM)
            hard_violations = [
                v for v in violations
                if v["metric"] not in ("avg_answer_relevance", "avg_faithfulness")
            ]
            if hard_violations:
                pytest.fail("\n".join(msg_parts))


# -----------------------------------------------------------------------
# Intent classification evaluation tests
# -----------------------------------------------------------------------

class TestIntentEvaluation:
    """Offline evaluation tests for intent classification."""

    @pytest.fixture(autouse=True, scope="class")
    def intent_results(self, request):
        """Run intent benchmark once and share results."""
        results = _run_async(run_intent_benchmark())
        request.cls._results = results
        request.cls._metrics = results["aggregated_metrics"]

    def test_benchmark_completes(self):
        """Benchmark should complete without errors."""
        assert self._results is not None
        assert self._results["total_cases"] > 0

    def test_accuracy_gate(self):
        """Overall accuracy should meet the CI gate threshold."""
        gate = INTENT_GATES["accuracy"]
        value = self._metrics.get("accuracy", 0)
        assert value >= gate["min"], (
            f"Intent accuracy {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_export_recall_gate(self):
        """Export intent recall should meet the CI gate threshold."""
        gate = INTENT_GATES["export_recall"]
        value = self._metrics.get("export_recall", 0)
        assert value >= gate["min"], (
            f"Export recall {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_export_false_positive_gate(self):
        """Export false positive rate (GENERAL_CHAT → EXPORT_*) should be low."""
        gate = INTENT_GATES["export_false_positive_rate"]
        value = self._metrics.get("export_false_positive_rate", 1.0)
        assert value <= gate["max"], (
            f"Export FP rate {value:.4f} > threshold {gate['max']} ({gate['description']})"
        )

    def test_calibration_error_gate(self):
        """Expected Calibration Error should be within bounds."""
        gate = INTENT_GATES["ece"]
        value = self._metrics.get("ece", 1.0)
        assert value <= gate["max"], (
            f"ECE {value:.4f} > threshold {gate['max']} ({gate['description']})"
        )

    def test_confusion_matrix_populated(self):
        """Confusion matrix should be populated."""
        cm = self._metrics.get("confusion_matrix", {})
        assert len(cm) > 0, "Confusion matrix should not be empty"

    def test_per_class_metrics_complete(self):
        """Per-class metrics should cover all expected intents."""
        per_class = self._metrics.get("per_class", {})
        expected_intents = {"GENERAL_CHAT", "EXPORT_PDF", "EXPORT_PPT", "EXPORT_VIDEO", "EXPORT_POSTER"}
        for intent in expected_intents:
            assert intent in per_class, f"Missing per-class metrics for {intent}"

    def test_all_intent_gates(self):
        """Run all intent gates and report violations."""
        violations = check_gates(self._metrics, INTENT_GATES, "intent")
        if violations:
            msg_parts = ["Intent CI gate violations:"]
            for v in violations:
                msg_parts.append(
                    f"  - {v['metric']}: {v['value']:.4f} "
                    f"({'<' if v['direction'] == 'min' else '>'} {v['threshold']}) "
                    f"[{v['description']}]"
                )
            pytest.fail("\n".join(msg_parts))


# -----------------------------------------------------------------------
# RAG retrieval evaluation tests
# -----------------------------------------------------------------------

class TestRAGEvaluation:
    """Offline evaluation tests for RAG retrieval."""

    @pytest.fixture(autouse=True, scope="class")
    def rag_results(self, request):
        """Run RAG benchmark once and share results."""
        results = _run_async(run_rag_benchmark())
        request.cls._results = results
        request.cls._metrics = results["aggregated_metrics"]

    def test_benchmark_completes(self):
        """Benchmark should complete without errors."""
        assert self._results is not None
        assert self._results["total_cases"] > 0

    def test_hit_at_3_gate(self):
        """Hit@3 should meet the CI gate threshold."""
        gate = RAG_GATES["hit_at_3"]
        value = self._metrics.get("hit_at_3", 0)
        assert value >= gate["min"], (
            f"Hit@3 {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_hit_at_6_gate(self):
        """Hit@6 should meet the CI gate threshold."""
        gate = RAG_GATES["hit_at_6"]
        value = self._metrics.get("hit_at_6", 0)
        assert value >= gate["min"], (
            f"Hit@6 {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_mrr_gate(self):
        """MRR should meet the CI gate threshold."""
        gate = RAG_GATES["mrr"]
        value = self._metrics.get("mrr", 0)
        assert value >= gate["min"], (
            f"MRR {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_ndcg_gate(self):
        """NDCG@6 should meet the CI gate threshold."""
        gate = RAG_GATES["ndcg_at_6"]
        value = self._metrics.get("ndcg_at_6", 0)
        assert value >= gate["min"], (
            f"NDCG@6 {value:.4f} < threshold {gate['min']} ({gate['description']})"
        )

    def test_latency_reasonable(self):
        """P95 latency should be under 2 seconds (mock or real)."""
        value = self._metrics.get("latency_p95_ms", 0)
        assert value < 2000, f"P95 latency {value:.0f}ms is too high"

    def test_all_rag_gates(self):
        """Run all RAG gates and report violations."""
        violations = check_gates(self._metrics, RAG_GATES, "rag")
        if violations:
            msg_parts = ["RAG CI gate violations:"]
            for v in violations:
                msg_parts.append(
                    f"  - {v['metric']}: {v['value']:.4f} "
                    f"({'<' if v['direction'] == 'min' else '>'} {v['threshold']}) "
                    f"[{v['description']}]"
                )
            pytest.fail("\n".join(msg_parts))


# -----------------------------------------------------------------------
# Report generation test
# -----------------------------------------------------------------------

class TestReportGeneration:
    """Test that reports are generated correctly."""

    def test_report_structure(self):
        """Generated report should have proper structure."""
        results = {"subsystem": "test", "total_cases": 1, "aggregated_metrics": {"accuracy": 0.9}}
        report = generate_report(results)
        assert "timestamp" in report
        assert "results" in report

    def test_baseline_comparison(self):
        """Report with baseline should include comparison."""
        results = {"subsystem": "test", "aggregated_metrics": {"accuracy": 0.9, "f1": 0.8}}
        baseline = {"aggregated_metrics": {"accuracy": 0.85, "f1": 0.9}}
        report = generate_report(results, baseline)
        assert "baseline_comparison" in report
        comp = report["baseline_comparison"]
        assert "accuracy" in comp
        assert comp["accuracy"]["delta"] == pytest.approx(0.05, abs=0.001)
