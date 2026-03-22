#!/usr/bin/env python3
"""Unit tests for baseline video export helpers."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_normalize_scenes_clamps_invalid_values():
    from api.video_export import _normalize_scenes

    scenes = _normalize_scenes(
        [
            {"title": "", "narration": "One. Two. Three.", "duration_seconds": "100"},
            {"title": "Skip me", "narration": "   ", "duration_seconds": 8},
        ],
        "RepoX",
    )

    assert len(scenes) == 1
    assert scenes[0]["title"] == "Scene 1"
    assert scenes[0]["duration_seconds"] == 24


def test_split_narration_to_bullets_prefers_sentences():
    from api.video_export import _split_narration_to_bullets

    bullets = _split_narration_to_bullets(
        "This project indexes repositories. It then retrieves relevant code context. Finally it produces a guided walkthrough video."
    )

    assert 1 <= len(bullets) <= 4
    assert bullets[0].startswith("This project indexes repositories")


def test_scene_to_card_content_includes_footer_and_bullets():
    from api.video_export import _scene_to_card_content

    card = _scene_to_card_content(
        {
            "title": "Architecture",
            "narration": "The backend processes repository content. The frontend presents the result.",
            "duration_seconds": 12,
        },
        "RepoHelper",
        2,
        5,
    )

    assert card["title"] == "Architecture"
    assert card["subtitle"] == "RepoHelper"
    assert card["footer"].endswith("Scene 2/5")
    assert len(card["bullets"]) >= 1
