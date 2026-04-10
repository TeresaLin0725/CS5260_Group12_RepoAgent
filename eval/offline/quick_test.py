"""
Quick sanity test: runs a small subset of each benchmark to verify the
real Ollama backend works end-to-end.

Usage:
    python -m eval.offline.quick_test --provider ollama --model qwen3:8b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.offline.run_benchmark import (
    load_fixture,
    run_react_benchmark,
    run_intent_benchmark,
    run_rag_benchmark,
)

logger = logging.getLogger(__name__)


async def main_async(args):
    from eval.offline.real_backends import create_real_llm, create_real_intent_classifier

    llm_fn = create_real_llm(provider=args.provider, model=args.model)

    n = args.n  # cases per subsystem

    # ── Intent (fastest, no repo needed) ──
    print(f"\n{'='*60}")
    print(f"  INTENT BENCHMARK  (first {n} cases, real LLM)")
    print(f"{'='*60}")
    intent_cases = load_fixture("intent_testcases.json")[:n]
    classifier_fn = create_real_intent_classifier(
        mode="llm", provider=args.provider, model=args.model,
    )
    t0 = time.monotonic()
    intent_result = await run_intent_benchmark(
        cases=intent_cases,
        classifier_fn=classifier_fn,
    )
    t_intent = time.monotonic() - t0
    agg = intent_result["aggregated_metrics"]
    print(f"\n  Accuracy:             {agg['accuracy']:.4f}")
    print(f"  ECE:                  {agg['ece']:.4f}")
    print(f"  Export Recall:        {agg['export_recall']:.4f}")
    print(f"  Time:                 {t_intent:.1f}s  ({t_intent/n:.1f}s/case)")

    # Show per-case detail
    for r in intent_result.get("per_case_results", []):
        match = "✓" if r["predicted_intent"] == r["expected_intent"] else "✗"
        print(f"    {match}  {r['case_id']}  expected={r['expected_intent']}  "
              f"predicted={r['predicted_intent']}  conf={r['predicted_confidence']:.2f}")

    # ── ReAct (mock tools, real LLM) ──
    print(f"\n{'='*60}")
    print(f"  REACT BENCHMARK  (first {n} cases, real LLM + mock tools)")
    print(f"{'='*60}")
    react_cases = load_fixture("react_testcases.json")[:n]
    t0 = time.monotonic()
    react_result = await run_react_benchmark(
        cases=react_cases,
        llm_fn=llm_fn,
        judge_llm_fn=llm_fn,
    )
    t_react = time.monotonic() - t0
    agg = react_result["aggregated_metrics"]
    print(f"\n  Tool F1:              {agg.get('avg_tool_f1', 'N/A')}")
    print(f"  Iteration Efficiency: {agg.get('avg_iteration_efficiency', 'N/A')}")
    print(f"  Forced Termination:   {agg.get('forced_termination_rate', 'N/A')}")
    print(f"  ROUGE-L:              {agg.get('avg_rouge_l', 'N/A')}")
    print(f"  Answer Relevance:     {agg.get('avg_answer_relevance', 'N/A')}")
    print(f"  Faithfulness:         {agg.get('avg_faithfulness', 'N/A')}")
    print(f"  Time:                 {t_react:.1f}s  ({t_react/n:.1f}s/case)")

    # Per-case summary
    for m in react_result.get("per_case_metrics", []):
        print(f"    {m['case_id']}  F1={m.get('tool_f1',0):.2f}  "
              f"rel={m.get('answer_relevance','?')}  "
              f"faith={m.get('faithfulness','?')}  "
              f"rouge={m.get('rouge_l','?')}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Intent accuracy:  {intent_result['aggregated_metrics']['accuracy']:.4f}")
    print(f"  ReAct tool F1:    {react_result['aggregated_metrics'].get('avg_tool_f1', 'N/A')}")
    print(f"  Total time:       {t_intent + t_react:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Quick agent evaluation sanity test")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("-n", type=int, default=10, help="Cases per subsystem")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy HTTP logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
