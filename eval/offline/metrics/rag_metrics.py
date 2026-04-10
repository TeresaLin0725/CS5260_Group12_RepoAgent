"""
RAG retrieval evaluation metrics.

Computes Hit@K, MRR, NDCG@K, and retrieval latency statistics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGCaseResult:
    """Raw execution result for a single RAG retrieval case."""

    case_id: str
    query: str
    golden_files: List[str]
    retrieved_files: List[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class RAGMetrics:
    """Aggregated RAG retrieval metrics."""

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_6: float = 0.0
    mrr: float = 0.0
    ndcg_at_6: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_cases: int = 0


def compute_hit_at_k(retrieved: List[str], golden: List[str], k: int) -> float:
    """Hit@K: 1.0 if any golden file appears in top-K retrieved results, else 0.0."""
    if not golden:
        return 1.0
    top_k = set(retrieved[:k])
    golden_set = set(golden)
    return 1.0 if top_k & golden_set else 0.0


def compute_mrr(retrieved: List[str], golden: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first golden file found."""
    if not golden:
        return 1.0
    golden_set = set(golden)
    for i, doc in enumerate(retrieved):
        if doc in golden_set:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def compute_ndcg_at_k(retrieved: List[str], golden: List[str], k: int = 6) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: 1 if the file is in golden_files, 0 otherwise.
    """
    if not golden:
        return 1.0

    golden_set = set(golden)

    # Actual relevances in retrieved order
    relevances = [1.0 if doc in golden_set else 0.0 for doc in retrieved[:k]]

    # Ideal relevances (all golden files first)
    ideal_relevances = sorted(relevances, reverse=True)
    # Extend ideal with remaining golden files not retrieved
    remaining_golden = len(golden_set) - sum(1 for r in relevances if r > 0)
    ideal_relevances = [1.0] * min(k, len(golden_set)) + [0.0] * max(0, k - len(golden_set))

    dcg_val = _dcg(relevances, k)
    idcg_val = _dcg(ideal_relevances, k)

    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def compute_latency_percentiles(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """Compute P50 and P95 latency from a list of latency values in ms."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _percentile(vals: List[float], p: float) -> float:
        idx = (p / 100.0) * (len(vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(vals) - 1)
        frac = idx - lower
        return vals[lower] * (1 - frac) + vals[upper] * frac

    return {
        "p50": round(_percentile(sorted_lat, 50), 2),
        "p95": round(_percentile(sorted_lat, 95), 2),
    }


def compute_rag_case_metrics(result: RAGCaseResult) -> Dict[str, Any]:
    """Compute metrics for a single RAG case."""
    return {
        "case_id": result.case_id,
        "hit_at_1": compute_hit_at_k(result.retrieved_files, result.golden_files, 1),
        "hit_at_3": compute_hit_at_k(result.retrieved_files, result.golden_files, 3),
        "hit_at_6": compute_hit_at_k(result.retrieved_files, result.golden_files, 6),
        "mrr": compute_mrr(result.retrieved_files, result.golden_files),
        "ndcg_at_6": compute_ndcg_at_k(result.retrieved_files, result.golden_files, 6),
        "latency_ms": result.retrieval_latency_ms,
    }


def aggregate_rag_metrics(results: List[RAGCaseResult]) -> RAGMetrics:
    """Aggregate RAG metrics across all cases."""
    if not results:
        return RAGMetrics()

    n = len(results)

    def _avg(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    per_case = [compute_rag_case_metrics(r) for r in results]
    latencies = [r.retrieval_latency_ms for r in results]
    lat_pcts = compute_latency_percentiles(latencies)

    return RAGMetrics(
        hit_at_1=_avg([c["hit_at_1"] for c in per_case]),
        hit_at_3=_avg([c["hit_at_3"] for c in per_case]),
        hit_at_6=_avg([c["hit_at_6"] for c in per_case]),
        mrr=_avg([c["mrr"] for c in per_case]),
        ndcg_at_6=_avg([c["ndcg_at_6"] for c in per_case]),
        latency_p50_ms=lat_pcts["p50"],
        latency_p95_ms=lat_pcts["p95"],
        total_cases=n,
    )
