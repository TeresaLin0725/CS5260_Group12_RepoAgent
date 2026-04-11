#!/usr/bin/env python3
"""
Unit tests for Gamma PPT export payload shaping.
"""

import asyncio
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.asyncio
async def test_build_generation_payload_uses_visual_theme_defaults(monkeypatch):
    import api.gamma_ppt_export as gamma

    monkeypatch.setattr(gamma, "GAMMA_CARD_SPLIT", "inputTextBreaks")
    monkeypatch.setattr(gamma, "GAMMA_CARD_DIMENSIONS", "16x9")
    monkeypatch.setattr(gamma, "GAMMA_THEME_ID", "theme-123")
    monkeypatch.setattr(gamma, "GAMMA_IMAGE_SOURCE", "aiGenerated")
    monkeypatch.setattr(
        gamma,
        "GAMMA_IMAGE_STYLE",
        "abstract gradient backgrounds for enterprise technical presentations",
    )
    monkeypatch.setattr(gamma, "GAMMA_ADDITIONAL_INSTRUCTIONS", "")

    payload = await gamma._build_generation_payload("Slide 1\n---\nSlide 2", "en", num_cards=12)

    assert payload["cardSplit"] == "inputTextBreaks"
    assert "numCards" not in payload
    assert payload["cardOptions"]["dimensions"] == "16x9"
    assert payload["themeId"] == "theme-123"
    assert payload["imageOptions"]["source"] == "aiGenerated"
    assert "gradient" in payload["imageOptions"]["style"]
    assert "plain white" in payload["additionalInstructions"].lower()


@pytest.mark.asyncio
async def test_build_generation_payload_keeps_num_cards_in_auto_mode(monkeypatch):
    import api.gamma_ppt_export as gamma

    monkeypatch.setattr(gamma, "GAMMA_CARD_SPLIT", "auto")
    monkeypatch.setattr(gamma, "GAMMA_THEME_ID", "")
    monkeypatch.setattr(gamma, "GAMMA_IMAGE_SOURCE", "webFreeToUseCommercially")
    monkeypatch.setattr(gamma, "GAMMA_ADDITIONAL_INSTRUCTIONS", "Keep layouts bold.")
    # Prevent real API call for auto theme
    monkeypatch.setattr(gamma, "_resolved_theme_id", "")

    payload = await gamma._build_generation_payload("Outline", "zh-CN", num_cards=7)

    assert payload["textOptions"]["language"] == "zh"
    assert payload["numCards"] == 7
    assert payload["imageOptions"] == {"source": "webFreeToUseCommercially"}
    assert "纯白" in payload["additionalInstructions"]
    assert "Keep layouts bold." in payload["additionalInstructions"]
