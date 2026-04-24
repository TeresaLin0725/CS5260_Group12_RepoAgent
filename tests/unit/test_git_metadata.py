#!/usr/bin/env python3
"""
Unit tests for api/git_metadata.py.

These tests are pure — no network, no filesystem git calls, no subprocess.
Run with: pytest tests/unit/test_git_metadata.py

For tests against a real repo (I/O, GitHub API), see:
  tests/integration/test_git_metadata_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_parse_github_owner_repo_standard_url():
    from api.git_metadata import _parse_github_owner_repo
    assert _parse_github_owner_repo("https://github.com/pallets/click") == ("pallets", "click")


def test_parse_github_owner_repo_with_dot_git():
    from api.git_metadata import _parse_github_owner_repo
    assert _parse_github_owner_repo("https://github.com/pallets/click.git") == ("pallets", "click")


def test_parse_github_owner_repo_non_github_returns_none():
    from api.git_metadata import _parse_github_owner_repo
    assert _parse_github_owner_repo("https://gitlab.com/owner/repo") is None
    assert _parse_github_owner_repo("") is None
    assert _parse_github_owner_repo(None) is None  # type: ignore[arg-type]


def test_clean_author_strips_email():
    from api.git_metadata import _clean_author
    assert _clean_author("Seakon Liu <seakon@example.com>") == "Seakon Liu"
    assert _clean_author("Alice") == "Alice"
    assert _clean_author("") == ""


def test_clean_message_keeps_first_line():
    from api.git_metadata import _clean_message
    assert _clean_message("feat: add X\n\nLong body here") == "feat: add X"
    assert _clean_message("") == ""


def test_clean_message_caps_length():
    from api.git_metadata import _clean_message
    long_msg = "x" * 500
    result = _clean_message(long_msg)
    assert len(result) == 200
    assert result.endswith("...")


def test_commit_timeline_is_empty():
    from api.git_metadata import CommitTimeline, CommitTimelineEntry
    empty = CommitTimeline()
    assert empty.is_empty()

    with_commit = CommitTimeline(commits=[CommitTimelineEntry(sha="abc", message="x", author="a", date="d")])
    assert not with_commit.is_empty()


def test_format_timeline_for_prompt_empty():
    from api.git_metadata import CommitTimeline, format_timeline_for_prompt
    assert format_timeline_for_prompt(CommitTimeline()) == ""


def test_format_timeline_for_prompt_has_sections():
    from api.git_metadata import (
        CommitTimeline, CommitTimelineEntry, ContributorInfo, ReleaseInfo,
        format_timeline_for_prompt,
    )
    timeline = CommitTimeline(
        commits=[CommitTimelineEntry(sha="abc12345", message="init", author="Alice", date="2025-01-01T00:00:00")],
        contributors=[ContributorInfo(login="alice", commit_count=5)],
        releases=[ReleaseInfo(tag="v1.0", date="2025-02-01T00:00:00", name="First release")],
    )
    block = format_timeline_for_prompt(timeline)
    assert "RECENT COMMIT HISTORY" in block
    assert "Alice: init" in block
    assert "TOP CONTRIBUTORS" in block
    assert "alice (5 commits)" in block
    assert "RELEASES" in block
    assert "v1.0" in block


def test_extract_commit_timeline_missing_path_returns_empty():
    from api.git_metadata import extract_commit_timeline
    timeline = extract_commit_timeline(local_path="/nonexistent/path/xyz", repo_url="")
    assert timeline.is_empty()


def test_extract_commit_timeline_no_git_dir(tmp_path):
    """Directory exists but has no .git — should return empty, not raise."""
    from api.git_metadata import extract_commit_timeline
    timeline = extract_commit_timeline(local_path=str(tmp_path), repo_url="")
    assert timeline.is_empty()


def test_format_evolution_section_empty_timeline():
    """PDF section formatter returns empty string when no timeline data."""
    from api.pdf_export import _format_evolution_section
    from types import SimpleNamespace
    analyzed = SimpleNamespace(commit_timeline=None, evolution_narrative="")
    assert _format_evolution_section(analyzed) == ""


def test_format_evolution_section_with_data():
    """PDF section formatter renders commits + contributors + releases."""
    from api.git_metadata import (
        CommitTimeline, CommitTimelineEntry, ContributorInfo, ReleaseInfo,
    )
    from api.pdf_export import _format_evolution_section
    from types import SimpleNamespace

    timeline = CommitTimeline(
        commits=[CommitTimelineEntry(
            sha="abc12345", message="feat: initial commit",
            author="Alice", date="2025-01-01T00:00:00",
        )],
        contributors=[ContributorInfo(login="alice", commit_count=10)],
        releases=[ReleaseInfo(tag="v1.0", date="2025-02-01T00:00:00", name="First")],
        total_commits_scanned=1,
        first_commit_date="2025-01-01T00:00:00",
        latest_commit_date="2025-01-01T00:00:00",
    )
    analyzed = SimpleNamespace(
        commit_timeline=timeline,
        evolution_narrative="The project started as a CLI tool.",
    )
    section = _format_evolution_section(analyzed)
    assert "Commit History" in section
    assert "The project started as a CLI tool." in section
    assert "abc12345" in section
    assert "Alice" in section
    assert "alice (10)" in section
    assert "v1.0" in section
