"""
Intent classification evaluation metrics.

Computes accuracy, per-class precision/recall/F1, confusion matrix,
calibration error (ECE), and ambiguity handling metrics.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IntentCaseResult:
    """Raw execution result for a single intent classification case."""

    case_id: str
    query: str
    expected_intent: str
    predicted_intent: Optional[str] = None
    predicted_confidence: float = 0.0
    expected_confidence_min: float = 0.0
    is_ambiguous: bool = False
    handled: bool = True
    source: str = "unknown"  # "rule" | "embedding" | "llm"
    category: str = ""


@dataclass
class IntentMetrics:
    """Aggregated intent classification metrics."""

    accuracy: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    ece: float = 0.0  # Expected Calibration Error
    export_recall: float = 0.0  # Recall for all EXPORT_* classes
    export_false_positive_rate: float = 0.0
    ambiguous_unhandled_rate: float = 0.0
    total_cases: int = 0


def compute_accuracy(results: List[IntentCaseResult]) -> float:
    """Overall classification accuracy."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.predicted_intent == r.expected_intent)
    return round(correct / len(results), 4)


def compute_per_class_metrics(
    results: List[IntentCaseResult],
) -> Dict[str, Dict[str, float]]:
    """Per-class Precision, Recall, F1."""
    # Collect all unique labels
    labels: Set[str] = set()
    for r in results:
        labels.add(r.expected_intent)
        if r.predicted_intent:
            labels.add(r.predicted_intent)

    per_class: Dict[str, Dict[str, float]] = {}
    for label in sorted(labels):
        tp = sum(
            1
            for r in results
            if r.expected_intent == label and r.predicted_intent == label
        )
        fp = sum(
            1
            for r in results
            if r.expected_intent != label and r.predicted_intent == label
        )
        fn = sum(
            1
            for r in results
            if r.expected_intent == label and r.predicted_intent != label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return per_class


def compute_confusion_matrix(
    results: List[IntentCaseResult],
) -> Dict[str, Dict[str, int]]:
    """Confusion matrix: matrix[actual][predicted] = count."""
    labels: Set[str] = set()
    for r in results:
        labels.add(r.expected_intent)
        if r.predicted_intent:
            labels.add(r.predicted_intent)

    matrix: Dict[str, Dict[str, int]] = {
        label: {l: 0 for l in sorted(labels)} for label in sorted(labels)
    }

    for r in results:
        predicted = r.predicted_intent or "NONE"
        if predicted not in matrix.get(r.expected_intent, {}):
            # Add dynamic column if needed
            for row in matrix.values():
                row.setdefault(predicted, 0)
            matrix.setdefault(predicted, {l: 0 for l in matrix})
        matrix[r.expected_intent][predicted] += 1

    return matrix


def compute_ece(results: List[IntentCaseResult], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |avg_confidence - accuracy|
    per bin, weighted by bin size.
    """
    if not results:
        return 0.0

    bins: List[List[IntentCaseResult]] = [[] for _ in range(n_bins)]
    for r in results:
        conf = max(0.0, min(1.0, r.predicted_confidence))
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    n = len(results)
    ece = 0.0
    for bin_cases in bins:
        if not bin_cases:
            continue
        bin_size = len(bin_cases)
        avg_confidence = sum(r.predicted_confidence for r in bin_cases) / bin_size
        accuracy = sum(
            1 for r in bin_cases if r.predicted_intent == r.expected_intent
        ) / bin_size
        ece += (bin_size / n) * abs(avg_confidence - accuracy)

    return round(ece, 4)


def compute_export_recall(results: List[IntentCaseResult]) -> float:
    """Recall specifically for EXPORT_* intents (missing an export is costly)."""
    export_cases = [r for r in results if r.expected_intent.startswith("EXPORT_")]
    if not export_cases:
        return 1.0
    correct = sum(
        1
        for r in export_cases
        if r.predicted_intent and r.predicted_intent.startswith("EXPORT_")
        and r.predicted_intent == r.expected_intent
    )
    return round(correct / len(export_cases), 4)


def compute_export_false_positive_rate(results: List[IntentCaseResult]) -> float:
    """Rate of GENERAL_CHAT being misclassified as EXPORT_*."""
    chat_cases = [r for r in results if r.expected_intent == "GENERAL_CHAT"]
    if not chat_cases:
        return 0.0
    false_exports = sum(
        1
        for r in chat_cases
        if r.predicted_intent and r.predicted_intent.startswith("EXPORT_")
    )
    return round(false_exports / len(chat_cases), 4)


def compute_ambiguous_unhandled_rate(results: List[IntentCaseResult]) -> float:
    """Rate of ambiguous cases (has_generation_intent=True) that ended as unhandled."""
    ambiguous = [r for r in results if r.is_ambiguous]
    if not ambiguous:
        return 0.0
    unhandled = sum(1 for r in ambiguous if not r.handled)
    return round(unhandled / len(ambiguous), 4)


def compute_intent_metrics(results: List[IntentCaseResult]) -> IntentMetrics:
    """Compute all intent classification metrics."""
    return IntentMetrics(
        accuracy=compute_accuracy(results),
        per_class=compute_per_class_metrics(results),
        confusion_matrix=compute_confusion_matrix(results),
        ece=compute_ece(results),
        export_recall=compute_export_recall(results),
        export_false_positive_rate=compute_export_false_positive_rate(results),
        ambiguous_unhandled_rate=compute_ambiguous_unhandled_rate(results),
        total_cases=len(results),
    )
