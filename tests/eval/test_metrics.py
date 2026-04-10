"""
Unit tests for offline evaluation metrics.

Validates the correctness of all metric computation functions
independently from the benchmark runner.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.offline.metrics.react_metrics import (
    ReActCaseResult,
    ReActMetrics,
    aggregate_react_metrics,
    compute_iteration_efficiency,
    compute_react_metrics,
    compute_rouge_l,
    compute_tool_error_rate,
    compute_tool_f1,
    compute_tool_precision,
    compute_tool_recall,
)
from eval.offline.metrics.intent_metrics import (
    IntentCaseResult,
    compute_accuracy,
    compute_ambiguous_unhandled_rate,
    compute_confusion_matrix,
    compute_ece,
    compute_export_false_positive_rate,
    compute_export_recall,
    compute_intent_metrics,
    compute_per_class_metrics,
)
from eval.offline.metrics.rag_metrics import (
    RAGCaseResult,
    aggregate_rag_metrics,
    compute_hit_at_k,
    compute_latency_percentiles,
    compute_mrr,
    compute_ndcg_at_k,
)
from eval.offline.ci_gates import (
    INTENT_GATES,
    RAG_GATES,
    REACT_GATES,
    check_gates,
)


# =====================================================================
# ReAct Metrics
# =====================================================================

class TestToolPrecision:
    def test_perfect_precision(self):
        assert compute_tool_precision(["rag_search"], ["rag_search"]) == 1.0

    def test_no_predicted(self):
        assert compute_tool_precision([], ["rag_search"]) == 0.0

    def test_no_expected_no_predicted(self):
        assert compute_tool_precision([], []) == 1.0

    def test_extra_tools(self):
        assert compute_tool_precision(["rag_search", "read_file"], ["rag_search"]) == 0.5

    def test_partial_overlap(self):
        assert compute_tool_precision(["rag_search", "code_grep"], ["rag_search", "read_file"]) == 0.5


class TestToolRecall:
    def test_perfect_recall(self):
        assert compute_tool_recall(["rag_search"], ["rag_search"]) == 1.0

    def test_no_expected(self):
        assert compute_tool_recall(["rag_search"], []) == 1.0

    def test_missing_tool(self):
        assert compute_tool_recall(["rag_search"], ["rag_search", "read_file"]) == 0.5

    def test_no_predicted_with_expected(self):
        assert compute_tool_recall([], ["rag_search"]) == 0.0


class TestToolF1:
    def test_perfect_f1(self):
        assert compute_tool_f1(1.0, 1.0) == 1.0

    def test_zero_f1(self):
        assert compute_tool_f1(0.0, 0.0) == 0.0

    def test_balanced(self):
        f1 = compute_tool_f1(0.5, 0.5)
        assert f1 == pytest.approx(0.5)


class TestIterationEfficiency:
    def test_first_iteration(self):
        assert compute_iteration_efficiency(1, 3) == 1.0

    def test_max_iterations(self):
        assert compute_iteration_efficiency(3, 3) == 0.0

    def test_middle(self):
        assert compute_iteration_efficiency(2, 3) == pytest.approx(0.5)

    def test_single_max(self):
        assert compute_iteration_efficiency(1, 1) == 1.0


class TestToolErrorRate:
    def test_no_errors(self):
        steps = [{"observation": "Found result"}, {"observation": "Another result"}]
        assert compute_tool_error_rate(steps) == 0.0

    def test_all_errors(self):
        steps = [{"observation": "Tool error: API failed"}, {"observation": "Tool error: timeout"}]
        assert compute_tool_error_rate(steps) == 1.0

    def test_partial_errors(self):
        steps = [{"observation": "Found result"}, {"observation": "Tool error: API failed"}]
        assert compute_tool_error_rate(steps) == 0.5

    def test_empty_steps(self):
        assert compute_tool_error_rate([]) == 0.0


class TestRougeL:
    def test_identical(self):
        text = "The retriever combines FAISS and BM25 using RRF"
        assert compute_rouge_l(text, text) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_rouge_l("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self):
        pred = "The retriever uses FAISS and BM25"
        ref = "FAISS and BM25 are combined in the retriever"
        score = compute_rouge_l(pred, ref)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert compute_rouge_l("", "some text") == 0.0
        assert compute_rouge_l("some text", "") == 0.0


class TestComputeReActMetrics:
    def test_basic_metrics(self):
        result = ReActCaseResult(
            case_id="test_001",
            category="single_tool",
            query="test query",
            final_answer="Test answer",
            tools_called=["rag_search"],
            iterations=2,
            max_iterations=3,
        )
        metrics = compute_react_metrics(result, ["rag_search"], "Test answer")
        assert metrics.tool_precision == 1.0
        assert metrics.tool_recall == 1.0
        assert metrics.tool_f1 == 1.0
        assert metrics.iteration_efficiency == pytest.approx(0.5)
        assert metrics.rouge_l == pytest.approx(1.0)


class TestAggregateReActMetrics:
    def test_aggregation(self):
        metrics = [
            ReActMetrics(case_id="1", category="direct_answer", tool_f1=1.0, iteration_efficiency=1.0),
            ReActMetrics(case_id="2", category="single_tool", tool_f1=0.5, iteration_efficiency=0.5),
        ]
        agg = aggregate_react_metrics(metrics)
        assert agg["total_cases"] == 2
        assert agg["avg_tool_f1"] == pytest.approx(0.75)
        assert agg["avg_iteration_efficiency"] == pytest.approx(0.75)

    def test_empty(self):
        assert aggregate_react_metrics([]) == {}


# =====================================================================
# Intent Metrics
# =====================================================================

class TestIntentAccuracy:
    def test_perfect(self):
        results = [
            IntentCaseResult("1", "q1", "EXPORT_PDF", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q2", "GENERAL_CHAT", predicted_intent="GENERAL_CHAT"),
        ]
        assert compute_accuracy(results) == 1.0

    def test_half(self):
        results = [
            IntentCaseResult("1", "q1", "EXPORT_PDF", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q2", "GENERAL_CHAT", predicted_intent="EXPORT_PDF"),
        ]
        assert compute_accuracy(results) == 0.5

    def test_empty(self):
        assert compute_accuracy([]) == 0.0


class TestPerClassMetrics:
    def test_binary(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q", "EXPORT_PDF", predicted_intent="GENERAL_CHAT"),
            IntentCaseResult("3", "q", "GENERAL_CHAT", predicted_intent="GENERAL_CHAT"),
        ]
        per_class = compute_per_class_metrics(results)
        assert per_class["EXPORT_PDF"]["recall"] == 0.5
        assert per_class["GENERAL_CHAT"]["precision"] == 0.5


class TestConfusionMatrix:
    def test_basic(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q", "GENERAL_CHAT", predicted_intent="EXPORT_PDF"),
        ]
        cm = compute_confusion_matrix(results)
        assert cm["EXPORT_PDF"]["EXPORT_PDF"] == 1
        assert cm["GENERAL_CHAT"]["EXPORT_PDF"] == 1


class TestECE:
    def test_perfect_calibration(self):
        results = [
            IntentCaseResult("1", "q", "A", predicted_intent="A", predicted_confidence=1.0),
            IntentCaseResult("2", "q", "B", predicted_intent="B", predicted_confidence=1.0),
        ]
        assert compute_ece(results) == pytest.approx(0.0)

    def test_empty(self):
        assert compute_ece([]) == 0.0


class TestExportRecall:
    def test_perfect(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q", "EXPORT_PPT", predicted_intent="EXPORT_PPT"),
        ]
        assert compute_export_recall(results) == 1.0

    def test_missed(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", predicted_intent="GENERAL_CHAT"),
            IntentCaseResult("2", "q", "EXPORT_PPT", predicted_intent="EXPORT_PPT"),
        ]
        assert compute_export_recall(results) == 0.5


class TestExportFalsePositive:
    def test_no_false_positives(self):
        results = [
            IntentCaseResult("1", "q", "GENERAL_CHAT", predicted_intent="GENERAL_CHAT"),
        ]
        assert compute_export_false_positive_rate(results) == 0.0

    def test_false_positive(self):
        results = [
            IntentCaseResult("1", "q", "GENERAL_CHAT", predicted_intent="EXPORT_PDF"),
            IntentCaseResult("2", "q", "GENERAL_CHAT", predicted_intent="GENERAL_CHAT"),
        ]
        assert compute_export_false_positive_rate(results) == 0.5


class TestAmbiguousUnhandled:
    def test_all_handled(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", is_ambiguous=True, handled=True),
        ]
        assert compute_ambiguous_unhandled_rate(results) == 0.0

    def test_some_unhandled(self):
        results = [
            IntentCaseResult("1", "q", "EXPORT_PDF", is_ambiguous=True, handled=True),
            IntentCaseResult("2", "q", "EXPORT_PPT", is_ambiguous=True, handled=False),
        ]
        assert compute_ambiguous_unhandled_rate(results) == 0.5


# =====================================================================
# RAG Metrics
# =====================================================================

class TestHitAtK:
    def test_hit_at_1(self):
        assert compute_hit_at_k(["a.py", "b.py", "c.py"], ["a.py"], 1) == 1.0

    def test_miss_at_1(self):
        assert compute_hit_at_k(["a.py", "b.py", "c.py"], ["d.py"], 1) == 0.0

    def test_hit_at_3(self):
        assert compute_hit_at_k(["a.py", "b.py", "c.py"], ["c.py"], 3) == 1.0

    def test_no_golden(self):
        assert compute_hit_at_k(["a.py"], [], 3) == 1.0


class TestMRR:
    def test_first_position(self):
        assert compute_mrr(["a.py", "b.py"], ["a.py"]) == 1.0

    def test_second_position(self):
        assert compute_mrr(["a.py", "b.py"], ["b.py"]) == 0.5

    def test_not_found(self):
        assert compute_mrr(["a.py", "b.py"], ["c.py"]) == 0.0

    def test_no_golden(self):
        assert compute_mrr(["a.py"], []) == 1.0


class TestNDCG:
    def test_perfect_ranking(self):
        ndcg = compute_ndcg_at_k(["a.py", "b.py", "c.py"], ["a.py"], 6)
        assert ndcg == pytest.approx(1.0)

    def test_no_relevant(self):
        assert compute_ndcg_at_k(["a.py", "b.py"], ["c.py"], 6) == 0.0


class TestLatencyPercentiles:
    def test_basic(self):
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        pcts = compute_latency_percentiles(latencies)
        assert pcts["p50"] == pytest.approx(55.0, abs=1)
        assert pcts["p95"] > 90

    def test_empty(self):
        pcts = compute_latency_percentiles([])
        assert pcts["p50"] == 0.0
        assert pcts["p95"] == 0.0


class TestAggregateRAG:
    def test_basic(self):
        results = [
            RAGCaseResult("1", "q1", ["a.py"], ["a.py", "b.py"], 50.0),
            RAGCaseResult("2", "q2", ["c.py"], ["d.py", "c.py"], 100.0),
        ]
        metrics = aggregate_rag_metrics(results)
        assert metrics.total_cases == 2
        assert metrics.hit_at_1 == 0.5  # a.py hit at 1, c.py miss at 1
        assert metrics.hit_at_3 == 1.0  # both found within 3

    def test_empty(self):
        metrics = aggregate_rag_metrics([])
        assert metrics.total_cases == 0


# =====================================================================
# CI Gates
# =====================================================================

class TestCIGates:
    def test_passing_gates(self):
        metrics = {"avg_tool_f1": 0.9, "forced_termination_rate": 0.01, "avg_tool_error_rate": 0.0}
        violations = check_gates(metrics, REACT_GATES, "react")
        # Only check non-LLM gates
        non_llm = [v for v in violations if v["metric"] not in ("avg_answer_relevance", "avg_faithfulness")]
        assert len(non_llm) == 0

    def test_failing_gate(self):
        metrics = {"avg_tool_f1": 0.3, "forced_termination_rate": 0.5, "avg_tool_error_rate": 0.1}
        violations = check_gates(metrics, REACT_GATES, "react")
        assert len(violations) > 0

    def test_missing_metric_skipped(self):
        metrics = {}
        violations = check_gates(metrics, REACT_GATES, "react")
        assert len(violations) == 0

    def test_intent_gates(self):
        metrics = {"accuracy": 0.95, "export_recall": 0.95, "export_false_positive_rate": 0.01, "ece": 0.05}
        violations = check_gates(metrics, INTENT_GATES, "intent")
        assert len(violations) == 0

    def test_rag_gates(self):
        metrics = {"hit_at_3": 0.9, "hit_at_6": 0.95, "mrr": 0.8, "ndcg_at_6": 0.85}
        violations = check_gates(metrics, RAG_GATES, "rag")
        assert len(violations) == 0
