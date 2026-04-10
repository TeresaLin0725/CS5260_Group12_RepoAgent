"""
ReAct evaluation metrics.

Computes answer quality, tool usage efficiency, and iteration convergence
metrics for ReAct loop benchmark cases.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ReActCaseResult:
    """Raw execution result for a single ReAct benchmark case."""

    case_id: str
    category: str
    query: str
    final_answer: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 3
    forced_termination: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReActMetrics:
    """Computed metrics for a single ReAct benchmark case."""

    case_id: str
    category: str

    # LLM-as-Judge scores (0.0 - 1.0)
    answer_relevance: Optional[float] = None
    faithfulness: Optional[float] = None

    # Tool metrics
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0

    # Iteration metrics
    iteration_efficiency: float = 0.0
    forced_termination: bool = False

    # Error metrics
    tool_error_rate: float = 0.0

    # Text overlap (optional)
    rouge_l: Optional[float] = None


def compute_tool_precision(predicted: List[str], expected: List[str]) -> float:
    """Tool Precision = |predicted ∩ expected| / |predicted|."""
    if not predicted:
        return 1.0 if not expected else 0.0
    predicted_set = set(predicted)
    expected_set = set(expected)
    return len(predicted_set & expected_set) / len(predicted_set)


def compute_tool_recall(predicted: List[str], expected: List[str]) -> float:
    """Tool Recall = |predicted ∩ expected| / |expected|."""
    if not expected:
        return 1.0
    predicted_set = set(predicted)
    expected_set = set(expected)
    return len(predicted_set & expected_set) / len(expected_set)


def compute_tool_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_iteration_efficiency(actual_iterations: int, max_iterations: int) -> float:
    """Iteration Efficiency = 1 - (actual - 1) / (max - 1).

    Score of 1.0 means first-iteration convergence; 0.0 means all iterations used.
    """
    if max_iterations <= 1:
        return 1.0
    efficiency = 1.0 - (actual_iterations - 1) / (max_iterations - 1)
    return max(0.0, min(1.0, efficiency))


def compute_tool_error_rate(steps: List[Dict[str, Any]]) -> float:
    """Fraction of iterations where observation contains 'Tool error:'."""
    if not steps:
        return 0.0
    error_count = sum(
        1
        for step in steps
        if step.get("observation") and "Tool error:" in str(step["observation"])
    )
    return error_count / len(steps)


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest common subsequence length (DP)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    if not prediction or not reference:
        return 0.0
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -----------------------------------------------------------------------
# LLM-as-Judge prompts
# -----------------------------------------------------------------------

RELEVANCE_PROMPT = (
    "You are an evaluation judge. Score the following answer on how well it "
    "directly addresses the query.\n\n"
    "Query: {query}\n"
    "Answer: {answer}\n\n"
    "Output ONLY a JSON object: {{\"score\": <float 0.0-1.0>, \"reason\": \"<brief reason>\"}}"
)

FAITHFULNESS_PROMPT = (
    "You are an evaluation judge. Given the following observations collected "
    "during reasoning and the final answer, score how faithful the answer is "
    "to the evidence in the observations. Every claim in the answer should be "
    "supported by the observations.\n\n"
    "Observations:\n{observations}\n\n"
    "Final Answer:\n{answer}\n\n"
    "Output ONLY a JSON object: {{\"score\": <float 0.0-1.0>, \"reason\": \"<brief reason>\"}}"
)


def _parse_judge_score(response: str) -> Optional[float]:
    """Extract score from LLM judge response."""
    try:
        # Try direct JSON parse
        data = json.loads(response)
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    # Fallback regex
    match = re.search(r'"score"\s*:\s*([\d.]+)', response)
    if match:
        return float(match.group(1))
    return None


async def judge_answer_relevance(
    query: str,
    answer: str,
    llm_fn,
) -> Optional[float]:
    """Use LLM-as-Judge to score answer relevance (0.0 - 1.0)."""
    if not answer:
        return 0.0
    prompt = RELEVANCE_PROMPT.format(query=query, answer=answer)
    try:
        response = await llm_fn(prompt)
        return _parse_judge_score(response)
    except Exception:
        logger.warning("LLM judge failed for answer relevance.", exc_info=True)
        return None


async def judge_faithfulness(
    observations: List[str],
    answer: str,
    llm_fn,
) -> Optional[float]:
    """Use LLM-as-Judge to score answer faithfulness (0.0 - 1.0)."""
    if not answer:
        return 0.0
    if not observations:
        return None
    obs_text = "\n---\n".join(observations)
    prompt = FAITHFULNESS_PROMPT.format(observations=obs_text, answer=answer)
    try:
        response = await llm_fn(prompt)
        return _parse_judge_score(response)
    except Exception:
        logger.warning("LLM judge failed for faithfulness.", exc_info=True)
        return None


def compute_react_metrics(
    result: ReActCaseResult,
    expected_tools: List[str],
    golden_answer: Optional[str] = None,
) -> ReActMetrics:
    """Compute all deterministic ReAct metrics for a single case."""
    precision = compute_tool_precision(result.tools_called, expected_tools)
    recall = compute_tool_recall(result.tools_called, expected_tools)
    f1 = compute_tool_f1(precision, recall)
    efficiency = compute_iteration_efficiency(result.iterations, result.max_iterations)
    error_rate = compute_tool_error_rate(result.steps)

    rouge = None
    if golden_answer and result.final_answer:
        rouge = compute_rouge_l(result.final_answer, golden_answer)

    return ReActMetrics(
        case_id=result.case_id,
        category=result.category,
        tool_precision=round(precision, 4),
        tool_recall=round(recall, 4),
        tool_f1=round(f1, 4),
        iteration_efficiency=round(efficiency, 4),
        forced_termination=result.forced_termination,
        tool_error_rate=round(error_rate, 4),
        rouge_l=round(rouge, 4) if rouge is not None else None,
    )


def aggregate_react_metrics(metrics_list: List[ReActMetrics]) -> Dict[str, Any]:
    """Aggregate metrics across all ReAct benchmark cases."""
    if not metrics_list:
        return {}

    n = len(metrics_list)

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    relevance_scores = [m.answer_relevance for m in metrics_list if m.answer_relevance is not None]
    faithfulness_scores = [m.faithfulness for m in metrics_list if m.faithfulness is not None]
    rouge_scores = [m.rouge_l for m in metrics_list if m.rouge_l is not None]

    forced_count = sum(1 for m in metrics_list if m.forced_termination)

    # Per-category breakdown
    categories: Dict[str, List[ReActMetrics]] = {}
    for m in metrics_list:
        categories.setdefault(m.category, []).append(m)

    category_summary = {}
    for cat, cat_metrics in categories.items():
        category_summary[cat] = {
            "count": len(cat_metrics),
            "avg_tool_f1": _avg([m.tool_f1 for m in cat_metrics]),
            "avg_iteration_efficiency": _avg([m.iteration_efficiency for m in cat_metrics]),
            "forced_termination_rate": round(
                sum(1 for m in cat_metrics if m.forced_termination) / len(cat_metrics), 4
            ),
            "avg_tool_error_rate": _avg([m.tool_error_rate for m in cat_metrics]),
        }

    return {
        "total_cases": n,
        "avg_answer_relevance": _avg(relevance_scores) if relevance_scores else None,
        "avg_faithfulness": _avg(faithfulness_scores) if faithfulness_scores else None,
        "avg_tool_precision": _avg([m.tool_precision for m in metrics_list]),
        "avg_tool_recall": _avg([m.tool_recall for m in metrics_list]),
        "avg_tool_f1": _avg([m.tool_f1 for m in metrics_list]),
        "avg_iteration_efficiency": _avg([m.iteration_efficiency for m in metrics_list]),
        "forced_termination_rate": round(forced_count / n, 4),
        "avg_tool_error_rate": _avg([m.tool_error_rate for m in metrics_list]),
        "avg_rouge_l": _avg(rouge_scores) if rouge_scores else None,
        "by_category": category_summary,
    }
