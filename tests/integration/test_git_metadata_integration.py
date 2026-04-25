#!/usr/bin/env python3
"""
Integration tests for commit-history extraction.

These tests hit real resources:
  - Local .git directories (filesystem I/O)
  - git subprocess calls (ensure_full_history)
  - GitHub REST API (contributors, releases)

They are slower (~10s) and require network access. Run with:
    pytest tests/integration/test_git_metadata_integration.py -s

The `-s` flag disables pytest's stdout capture so you can actually see
the extracted timeline data printed by the demo tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Pick the first locally-cloned repo from these candidates, else fall back
# to the current project's .git. Add more as needed for iteration.
_DEMO_REPO_CANDIDATES = [
    ("~/.adalflow/repos/pallets_click", "https://github.com/pallets/click"),
    ("~/.adalflow/repos/tiangolo_typer", "https://github.com/tiangolo/typer"),
    ("~/.adalflow/repos/TeresaLin0725_CS5260_Group12_RepoAgent",
     "https://github.com/TeresaLin0725/CS5260_Group12_RepoAgent"),
]


def _pick_demo_repo():
    for path, url in _DEMO_REPO_CANDIDATES:
        expanded = os.path.expanduser(path)
        if os.path.isdir(os.path.join(expanded, ".git")):
            return expanded, url
    if (project_root / ".git").exists():
        return str(project_root), ""
    return None, None


# ---------------------------------------------------------------------------
# Real-data demo tests — print what the production pipeline sees.
# Run with `pytest -s` to view the output.
# ---------------------------------------------------------------------------

def test_demo_extract_real_timeline():
    """Print a human-readable dump of a real commit timeline."""
    from api.data_pipeline import ensure_full_history
    from api.git_metadata import extract_commit_timeline, format_timeline_for_prompt

    local_path, repo_url = _pick_demo_repo()
    if not local_path:
        pytest.skip("No local git repo available for demo")

    # Match production: upgrade shallow clones to full history first.
    ensure_full_history(local_path)

    timeline = extract_commit_timeline(local_path=local_path, repo_url=repo_url)
    assert not timeline.is_empty(), f"Expected non-empty timeline for {local_path}"

    print("\n" + "=" * 70)
    print(f"DEMO: CommitTimeline for {local_path}")
    print(f"  repo_url: {repo_url or '(local only — GitHub API skipped)'}")
    print("=" * 70)
    print(f"total_commits_scanned: {timeline.total_commits_scanned}")
    print(f"first_commit_date:     {timeline.first_commit_date[:10] if timeline.first_commit_date else '(none)'}")
    print(f"latest_commit_date:    {timeline.latest_commit_date[:10] if timeline.latest_commit_date else '(none)'}")
    print(f"contributors:          {len(timeline.contributors)}")
    print(f"releases:              {len(timeline.releases)}")
    print()

    print("-- Recent commits (first 10) --")
    for c in timeline.commits[:10]:
        date = c.date[:10] if c.date else "?"
        msg = c.message[:80] if c.message else ""
        print(f"  {date} [{c.sha}] {c.author}: {msg}")
    print()

    if timeline.contributors:
        print("-- Contributors --")
        for c in timeline.contributors[:8]:
            print(f"  {c.login} — {c.commit_count} commits")
        print()

    if timeline.releases:
        print("-- Releases --")
        for r in timeline.releases[:5]:
            date = r.date[:10] if r.date else "?"
            print(f"  {r.tag} ({date}): {r.name}")
        print()

    print("-- LLM prompt block preview --")
    preview = format_timeline_for_prompt(timeline)
    max_chars = 1200
    print(preview[:max_chars] + ("..." if len(preview) > max_chars else ""))
    print("=" * 70 + "\n")


def test_demo_pdf_evolution_section():
    """Print the exact PDF text that production will append to summary."""
    from api.data_pipeline import ensure_full_history
    from api.git_metadata import extract_commit_timeline
    from api.pdf_export import _format_evolution_section
    from types import SimpleNamespace

    local_path, repo_url = _pick_demo_repo()
    if not local_path:
        pytest.skip("No local git repo available for demo")

    ensure_full_history(local_path)
    timeline = extract_commit_timeline(local_path=local_path, repo_url=repo_url)
    analyzed = SimpleNamespace(
        commit_timeline=timeline,
        evolution_narrative="",  # simulate qwen3 returning empty narrative
    )
    section = _format_evolution_section(analyzed)

    print("\n" + "=" * 70)
    print("DEMO: PDF evolution section (what gets appended to summary_text)")
    print("=" * 70)
    print(section)
    print("=" * 70 + "\n")

    assert "Commit History" in section
    # Compact PDF format: SHA was dropped to fit the single-page budget.
    # Author + commit message must still appear.
    assert timeline.commits[0].author.split()[0] in section
    assert timeline.commits[0].message[:20] in section


# ---------------------------------------------------------------------------
# Production integration path test — verify analyze_repo_content() really
# does attach commit_timeline to AnalyzedContent. Mocks the heavy parts
# (RAG embedding + LLM) but exercises the real integration code.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_repo_content_attaches_commit_timeline(monkeypatch):
    """Verify the production pipeline attaches commit_timeline to AnalyzedContent.

    Mocks RAG + LLM so we don't touch embedding/OpenAI — but uses the real
    git_metadata extraction against a real local .git directory. This is the
    test that catches regressions in the WIRING between content_analyzer.py
    and git_metadata.py.
    """
    from api import content_analyzer
    from api.content_analyzer import RepoAnalysisRequest

    local_path, repo_url = _pick_demo_repo()
    if not local_path or not repo_url:
        pytest.skip("No local repo with GitHub URL available")

    # Mock RAG.prepare_retriever to expose the local_path (no embedding).
    class _FakeDbManager:
        repo_paths = {"save_repo_dir": local_path}

    class _FakeRag:
        db_manager = _FakeDbManager()
        def __init__(self, *a, **kw): pass
        def prepare_retriever(self, *a, **kw): pass

    monkeypatch.setattr(content_analyzer, "RAG", _FakeRag, raising=False)
    # Also patch the from-imports
    import api.rag
    monkeypatch.setattr(api.rag, "RAG", _FakeRag, raising=False)

    # Return non-empty context so the LLM-path isn't skipped.
    monkeypatch.setattr(
        content_analyzer, "_extract_repo_context",
        lambda rag, repo_name: "fake repo context content",
    )

    # Short-circuit the LLM: return a minimal valid JSON response.
    async def _fake_llm(**kwargs):
        return '{"project_overview":"fake","evolution_narrative":"Mocked narrative"}'
    monkeypatch.setattr(content_analyzer, "_run_llm_structured_analysis", _fake_llm)

    # Run through the production function.
    req = RepoAnalysisRequest(
        repo_url=repo_url,
        repo_name=Path(local_path).name,
        provider="openai",
        model="gpt-4o",
        language="en",
        repo_type="github",
    )
    result = await content_analyzer.analyze_repo_content(req)

    # The integration contract:
    assert result.commit_timeline is not None, "commit_timeline must be attached"
    assert not result.commit_timeline.is_empty(), "timeline must have data"
    assert len(result.commit_timeline.commits) > 0, "timeline must have commits"
    assert result.evolution_narrative == "Mocked narrative", \
        "evolution_narrative must be read from LLM JSON"

    print(f"\n[OK] analyze_repo_content attached {len(result.commit_timeline.commits)} "
          f"commits to AnalyzedContent for {req.repo_name}")
