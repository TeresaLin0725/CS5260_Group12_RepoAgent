"""
DeepWiki Agent Evaluation Framework.

Three-dimensional evaluation system:

  1. Offline Automation  — benchmark datasets, deterministic metrics, regression tests
  2. Online Observation  — Langfuse tracing, latency, tool-call telemetry
  3. Manual Review       — structured human annotation with aggregation & reporting

Quick start
-----------
Offline:
    from eval.offline.evaluator import OfflineEvaluator
    report = await OfflineEvaluator().run_benchmark("eval/offline/fixtures/react_testcases.json")

Online (wrap your agent):
    from eval.online.langfuse_tracer import AgentTracer
    tracer = AgentTracer()
    async with tracer.trace_react_run(query, session_id) as ctx:
        ...

Review API:
    # mount in main.py:
    from eval.review.review_api import router as review_router
    app.include_router(review_router, prefix="/eval")
"""
