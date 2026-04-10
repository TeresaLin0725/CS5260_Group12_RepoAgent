"""
CI gate thresholds for offline evaluation.

When metrics fall below these thresholds, the CI pipeline should fail.
Thresholds are loaded by the pytest integration tests and the benchmark
runner's gate-check logic.
"""

# -----------------------------------------------------------------------
# ReAct thresholds
# -----------------------------------------------------------------------

REACT_GATES = {
    "avg_answer_relevance": {"min": 0.75, "description": "答案相关性 ≥ 0.75"},
    "avg_faithfulness": {"min": 0.80, "description": "答案忠实度 ≥ 0.80"},
    "avg_tool_f1": {"min": 0.70, "description": "工具调用 F1 ≥ 0.70"},
    "forced_termination_rate": {"max": 0.08, "description": "强制终止率 ≤ 8%"},
    "avg_tool_error_rate": {"max": 0.02, "description": "工具错误率 ≤ 2%"},
}

# -----------------------------------------------------------------------
# Intent classification thresholds
# -----------------------------------------------------------------------

INTENT_GATES = {
    "accuracy": {"min": 0.85, "description": "分类准确率 ≥ 0.85"},
    "export_recall": {"min": 0.90, "description": "EXPORT 类 Recall ≥ 0.90"},
    "export_false_positive_rate": {"max": 0.10, "description": "EXPORT 假阳率 ≤ 10%"},
    "ece": {"max": 0.15, "description": "期望校准误差 ≤ 0.15"},
}

# -----------------------------------------------------------------------
# RAG retrieval thresholds
# -----------------------------------------------------------------------

RAG_GATES = {
    "hit_at_3": {"min": 0.75, "description": "Hit@3 ≥ 0.75"},
    "hit_at_6": {"min": 0.85, "description": "Hit@6 ≥ 0.85"},
    "mrr": {"min": 0.60, "description": "MRR ≥ 0.60"},
    "ndcg_at_6": {"min": 0.65, "description": "NDCG@6 ≥ 0.65"},
}


def check_gates(
    metrics: dict,
    gates: dict,
    subsystem: str = "unknown",
) -> list:
    """Check metrics against gate thresholds.

    Returns a list of gate violation dicts. An empty list means all gates pass.
    """
    violations = []
    for metric_name, threshold in gates.items():
        value = metrics.get(metric_name)
        if value is None:
            continue  # Skip metrics that weren't computed (e.g., LLM-judge scores)

        if "min" in threshold and value < threshold["min"]:
            violations.append({
                "subsystem": subsystem,
                "metric": metric_name,
                "value": value,
                "threshold": threshold["min"],
                "direction": "min",
                "description": threshold.get("description", ""),
            })
        if "max" in threshold and value > threshold["max"]:
            violations.append({
                "subsystem": subsystem,
                "metric": metric_name,
                "value": value,
                "threshold": threshold["max"],
                "direction": "max",
                "description": threshold.get("description", ""),
            })

    return violations
