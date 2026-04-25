#!/usr/bin/env python3
"""
Unit tests for api/video/onboard_5act subpackage.

Pure unit tests — no network, no filesystem video rendering.
Run with: pytest tests/unit/test_onboard_5act.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# icons.py
# ---------------------------------------------------------------------------

def test_input_icon_for_github_url():
    from api.video.onboard_5act import icons
    assert icons.guess_input_icon("GitHub URL") == "🔗"
    assert icons.guess_input_icon("a repository link") == "🔗"


def test_input_icon_falls_back_to_default():
    from api.video.onboard_5act import icons
    assert icons.guess_input_icon("xyzzy") == "📥"
    assert icons.guess_input_icon("") == "📥"


def test_output_icon_for_pdf_and_video():
    from api.video.onboard_5act import icons
    assert icons.guess_output_icon("PDF report") == "📄"
    assert icons.guess_output_icon("explainer video") == "🎬"


def test_process_icon_for_ai():
    from api.video.onboard_5act import icons
    assert icons.guess_process_icon("AI summarizes") == "🧠"
    assert icons.guess_process_icon("LLM analysis") == "🧠"


def test_setup_step_icons():
    from api.video.onboard_5act import icons
    assert icons.guess_setup_step_icon("git clone repo") == "📥"
    assert icons.guess_setup_step_icon("pip install -r requirements.txt") == "📦"
    assert icons.guess_setup_step_icon("python -m api.main") == "▶️"
    assert icons.guess_setup_step_icon("Open http://localhost:3000") == "🌐"


def test_rank_medals():
    from api.video.onboard_5act import icons
    assert icons.rank_medal(1) == "🥇"
    assert icons.rank_medal(2) == "🥈"
    assert icons.rank_medal(3) == "🥉"
    assert icons.rank_medal(4) == "▫️"
    assert icons.rank_medal(0) == "▫️"


def test_tech_icon_known_and_unknown():
    from api.video.onboard_5act import icons
    assert icons.tech_icon("Python") == "🐍"
    assert icons.tech_icon("Docker") == "🐳"
    assert icons.tech_icon("UnknownLang") == ""


# ---------------------------------------------------------------------------
# acts.py — always returns 5 acts in fixed order
# ---------------------------------------------------------------------------

def test_build_acts_returns_5_acts_in_fixed_order():
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name="test/repo")
    acts = build_acts(analyzed)
    assert len(acts) == 5
    sections = [a["section"] for a in acts]
    assert sections == ["intro", "metaphor", "io", "usecase", "setup"]
    assert [a["act_number"] for a in acts] == [1, 2, 3, 4, 5]


def test_build_acts_minimal_input_has_all_act_titles():
    """Even with empty AnalyzedContent, every act gets a title and narration."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name="x")
    for act in build_acts(analyzed):
        assert act["title"], f"act {act['act_number']} has empty title"
        assert act["narration"], f"act {act['act_number']} has empty narration"
        assert act["duration_seconds"] > 0


def test_act3_io_uses_onboard_boxes_when_present():
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent, OnboardSnapshot

    analyzed = AnalyzedContent(repo_name="x")
    analyzed.onboard = OnboardSnapshot(
        mental_model_3_boxes=["GitHub URL", "AI reads code", "Friendly PDF"],
    )
    acts = build_acts(analyzed)
    io = acts[2]
    assert io["section"] == "io"
    box_labels = [b["label"] for b in io["card"]["boxes"]]
    assert box_labels == ["GitHub URL", "AI reads code", "Friendly PDF"]
    box_icons = [b["icon"] for b in io["card"]["boxes"]]
    assert box_icons[0] == "🔗"     # URL → link icon
    assert box_icons[1] == "🧠"     # AI → brain icon
    assert box_icons[2] == "📄"     # PDF → document icon


def test_act3_io_falls_back_to_placeholders():
    """Missing onboard.mental_model_3_boxes → default Input/Process/Output."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name="x")
    acts = build_acts(analyzed)
    io = acts[2]
    box_labels = [b["label"] for b in io["card"]["boxes"]]
    assert len(box_labels) == 3
    assert all(box_labels)


def test_act5_setup_parses_numbered_steps():
    """`first_5_minutes` like '1. clone 2. install 3. run' should split into 3 steps."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent, OnboardSnapshot

    analyzed = AnalyzedContent(repo_name="x")
    analyzed.onboard = OnboardSnapshot(
        first_5_minutes="1. git clone repo  2. pip install  3. python -m api.main",
        prerequisites=["Python 3.10+"],
    )
    acts = build_acts(analyzed)
    setup = acts[4]
    assert setup["section"] == "setup"
    steps = setup["card"]["steps"]
    assert len(steps) >= 3
    step_texts = [s["text"] for s in steps]
    assert any("clone" in t for t in step_texts)
    assert any("install" in t for t in step_texts)
    assert any("python" in t for t in step_texts)


def test_act5_setup_provides_default_steps_when_missing():
    """No onboard.first_5_minutes → default 4 steps so video still renders."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name="x")
    setup = build_acts(analyzed)[4]
    assert len(setup["card"]["steps"]) >= 3


def test_act1_includes_stats_and_contributors_when_available():
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent
    from api.git_metadata import (
        CommitTimeline, CommitTimelineEntry, ContributorInfo, RepoStats,
    )

    analyzed = AnalyzedContent(repo_name="foo/bar")
    analyzed.commit_timeline = CommitTimeline(
        commits=[CommitTimelineEntry(sha="abc", author="alice", date="2025-01-01T00:00:00", message="init")],
        contributors=[
            ContributorInfo(login="alice", commit_count=50, followers=1000, public_repos=20),
            ContributorInfo(login="bob", commit_count=10, followers=10),
        ],
        stats=RepoStats(stars=42, forks=5, license="MIT", pushed_at="2026-01-01T00:00:00Z"),
        first_commit_date="2025-01-01T00:00:00",
        latest_commit_date="2026-01-01T00:00:00",
    )

    intro = build_acts(analyzed)[0]
    assert intro["section"] == "intro"
    assert intro["card"]["stats"]["stars"] == 42
    headliners = intro["card"]["headline_contributors"]
    assert len(headliners) == 2
    # Alice has higher reputation → should rank first
    assert headliners[0]["login"] == "alice"
    assert headliners[0]["medal"] == "🥇"


def test_act2_uses_metaphor_segments_when_present():
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent, MetaphorSegment

    analyzed = AnalyzedContent(repo_name="x")
    analyzed.metaphor_story = [
        MetaphorSegment(detail="A diner sits down.", brief="Diner: I'm hungry!"),
        MetaphorSegment(detail="Chef cooks.", brief="Chef: On it!"),
        MetaphorSegment(detail="Plate served.", brief="Diner: Yum!"),
    ]
    metaphor = build_acts(analyzed)[1]
    assert len(metaphor["card"]["segments"]) == 3
    assert "hungry" in metaphor["narration"].lower()


def test_act2_falls_back_when_no_metaphor():
    """No metaphor_story → segments empty, narration is generic but non-empty."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name="x")
    metaphor = build_acts(analyzed)[1]
    assert metaphor["card"]["segments"] == []
    assert metaphor["narration"]


def test_acts_never_raise_on_empty_input():
    """Total empty AnalyzedContent should still produce 5 valid acts."""
    from api.video.onboard_5act.acts import build_acts
    from api.content_analyzer import AnalyzedContent

    acts = build_acts(AnalyzedContent())
    assert len(acts) == 5
    for act in acts:
        assert isinstance(act["card"], dict)
