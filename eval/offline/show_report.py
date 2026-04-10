"""Pretty-print the latest evaluation report."""
import json
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def find_latest_report():
    candidates = sorted(REPORTS_DIR.glob("eval_report_*.json"), reverse=True)
    if not candidates:
        print("No reports found in", REPORTS_DIR)
        sys.exit(1)
    return candidates[0]


def print_report(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"  OFFLINE EVALUATION REPORT")
    print(f"  File: {path.name}")
    print(f"{'=' * 70}")

    from eval.offline.ci_gates import REACT_GATES, INTENT_GATES, RAG_GATES, check_gates

    gate_map = {"react": REACT_GATES, "intent": INTENT_GATES, "rag": RAG_GATES}

    for sub in ["react", "intent", "rag"]:
        if sub not in data:
            continue
        r = data[sub]["results"]
        agg = r.get("aggregated_metrics", {})
        total = r.get("total_cases", "?")

        print(f"\n{'─' * 70}")
        print(f"  [{sub.upper()}]  {total} test cases")
        print(f"{'─' * 70}")

        # Print non-dict metrics
        for k, v in agg.items():
            if k in ("by_category", "per_class_metrics", "confusion_matrix",
                      "latency_p50_ms", "latency_p95_ms"):
                continue
            if isinstance(v, float):
                print(f"  {k:40s} {v:.4f}")
            elif isinstance(v, int):
                print(f"  {k:40s} {v}")
            elif v is None:
                print(f"  {k:40s} N/A (需要 LLM-as-Judge)")

        # Latency (RAG)
        if "latency_p50_ms" in agg:
            print(f"  {'latency_p50_ms':40s} {agg['latency_p50_ms']:.2f} ms")
            print(f"  {'latency_p95_ms':40s} {agg['latency_p95_ms']:.2f} ms")

        # Category breakdown
        if "by_category" in agg:
            print(f"\n  Per-Category Breakdown:")
            for cat, cv in agg["by_category"].items():
                count = cv.get("count", "?")
                f1 = cv.get("avg_tool_f1", cv.get("avg_hit_at_3", "?"))
                label = f"    {cat} ({count} cases)"
                if isinstance(f1, float):
                    print(f"{label:45s} F1={f1:.4f}")
                else:
                    print(f"{label:45s}")

        # Per-class metrics (Intent)
        if "per_class_metrics" in agg:
            print(f"\n  Per-Class Metrics:")
            for cls_name, cls_v in agg["per_class_metrics"].items():
                p = cls_v.get("precision", 0)
                r_val = cls_v.get("recall", 0)
                f1 = cls_v.get("f1", 0)
                print(f"    {cls_name:24s} P={p:.3f}  R={r_val:.3f}  F1={f1:.3f}")

        # Confusion matrix (Intent)
        if "confusion_matrix" in agg:
            print(f"\n  Confusion Matrix:")
            cm = agg["confusion_matrix"]
            # cm is a nested dict: cm[actual][predicted] = count
            all_labels = set(cm.keys())
            for row_data in cm.values():
                if isinstance(row_data, dict):
                    all_labels.update(row_data.keys())
            labels = sorted(all_labels)
            header = "    " + " " * 24 + "  ".join(f"{l:>12s}" for l in labels)
            print(header)
            for actual in labels:
                row = f"    {actual:24s}"
                for predicted in labels:
                    count = cm.get(actual, {}).get(predicted, 0) if isinstance(cm.get(actual), dict) else 0
                    row += f"  {count:12d}"
                print(row)

        # CI Gate Check
        gates = gate_map.get(sub, {})
        violations = check_gates(agg, gates, sub)
        if violations:
            print(f"\n  ⚠ CI GATE VIOLATIONS ({len(violations)}):")
            for v in violations:
                direction = "≥" if v["direction"] == "min" else "≤"
                print(f"    FAIL: {v['metric']} = {v['value']:.4f}  (要求 {direction} {v['threshold']})")
        else:
            n_checked = sum(1 for m in gates if agg.get(m) is not None)
            print(f"\n  ✓ All CI gates passed ({n_checked}/{len(gates)} checked)")

        # Baseline comparison
        bc = data[sub].get("baseline_comparison", {})
        if bc:
            print(f"\n  Baseline Comparison:")
            for k, v in bc.items():
                delta_str = f"+{v['delta']:.4f}" if v['delta'] >= 0 else f"{v['delta']:.4f}"
                flag = " ⚠ REGRESSION" if v["regression"] else ""
                print(f"    {k:35s} {v['current']:.4f} vs {v['baseline']:.4f}  ({delta_str}){flag}")

    print(f"\n{'=' * 70}")
    print(f"  Report: {path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_report()
    if isinstance(report_path, str):
        report_path = Path(report_path)
    print_report(report_path)
