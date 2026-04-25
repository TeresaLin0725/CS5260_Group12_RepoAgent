"""
Onboard 5-Act video orchestrator.

Pipeline:
    AnalyzedContent
        ↓ acts.build_acts()
    [5 act dicts]
        ↓ TTS per act (api.tts_service)         ← imported (utility)
        ↓ HTML build per act (templates.py)      ← local
        ↓ Playwright render per act              ← local + legacy executor
        ↓ MoviePy clip + audio attach
        ↓ concatenate + write MP4
    bytes (MP4 video)

Importable as `render_onboard_5act_video(analyzed, ...) -> bytes`.

Selected via the new env switch ``VIDEO_PIPELINE=onboard_5act`` in
``api/video/orchestrator.py``. The legacy baseline pipeline at that
file is intentionally untouched.
"""

from __future__ import annotations

import asyncio as _asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from api.video.compose import _compose_final_video, _read_file_bytes
from api.video.constants import (
    AUDIO_PADDING_SECONDS,
    SCENE_DURATION_MAX,
    SCENE_DURATION_MIN,
    TRANSITION_SECONDS,
)
from api.video.onboard_5act.acts import build_acts
from api.video.onboard_5act.templates import render_act_to_png
from api.video.progress import update_progress

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)

COMPOSE_TIMEOUT_SECONDS = max(60, int(os.getenv("VIDEO_COMPOSE_TIMEOUT_SECONDS", "480")))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def render_onboard_5act_video(
    analyzed: "AnalyzedContent",
    *,
    job_id: Optional[str] = None,
) -> bytes:
    """Generate the 5-act onboard video and return raw MP4 bytes.

    All five acts always render — gaps in the AnalyzedContent are filled
    with safe defaults by acts.build_acts() so the resulting video has
    consistent length and pacing.
    """
    overall_start = time.perf_counter()
    logger.info("onboard_5act video requested for %s", analyzed.repo_name)
    update_progress(job_id, 1)

    # ── Step 1: build the 5 act specs ──────────────────────────────────
    acts = build_acts(analyzed)
    logger.info("onboard_5act: built %d acts", len(acts))

    with tempfile.TemporaryDirectory(prefix="repohelper_5act_") as tmpdir:
        tmp_path = Path(tmpdir)

        # ── Step 2: TTS audio per act (parallelized) ───────────────────
        update_progress(job_id, 2, "Generating narration audio...")
        tts_start = time.perf_counter()
        audio_paths = await _generate_act_audio(acts, tmp_path, analyzed.language)
        has_audio = any(p is not None for p in audio_paths)
        logger.info(
            "onboard_5act: TTS done in %.2fs (%d/%d acts with audio)",
            time.perf_counter() - tts_start,
            sum(1 for p in audio_paths if p),
            len(acts),
        )

        # Stretch each act's visual duration to match its audio when known.
        for act, audio_path in zip(acts, audio_paths):
            audio_duration = _audio_duration(audio_path)
            if audio_duration > 0:
                act["duration_seconds"] = max(
                    SCENE_DURATION_MIN,
                    min(audio_duration + AUDIO_PADDING_SECONDS, SCENE_DURATION_MAX),
                )

        # ── Step 3: render PNG per act (parallel) ──────────────────────
        update_progress(job_id, 3, "Rendering act visuals...")
        render_start = time.perf_counter()
        png_paths: list[str] = []
        render_coros = []
        for i, act in enumerate(acts, start=1):
            png = str(tmp_path / f"act{i:02d}_{act['section']}.png")
            png_paths.append(png)
            render_coros.append(render_act_to_png(act, png))
        await _asyncio.gather(*render_coros)
        logger.info(
            "onboard_5act: %d PNGs rendered in %.2fs",
            len(png_paths), time.perf_counter() - render_start,
        )

        # ── Step 4: assemble clips with audio + transitions ────────────
        update_progress(job_id, 4, "Assembling video clips...")
        clips = _build_clips(acts, png_paths, audio_paths)

        # Close the shared Playwright browser between renders and ffmpeg
        # so resources are released before the (heavy) MP4 encoding step.
        try:
            from api.scene_renderer import close_browser
            await close_browser()
        except Exception:
            pass

        # ── Step 5: MP4 composition ───────────────────────────────────
        update_progress(job_id, 5, "Composing final MP4...")
        compose_start = time.perf_counter()
        output_path = tmp_path / "onboard_5act.mp4"
        loop = _asyncio.get_running_loop()
        try:
            await _asyncio.wait_for(
                loop.run_in_executor(
                    None, _compose_final_video, clips, str(output_path), has_audio,
                ),
                timeout=COMPOSE_TIMEOUT_SECONDS,
            )
        except _asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"onboard_5act MP4 composition exceeded {COMPOSE_TIMEOUT_SECONDS}s"
            ) from exc
        logger.info(
            "onboard_5act: MP4 composed in %.2fs",
            time.perf_counter() - compose_start,
        )

        payload = _read_file_bytes(str(output_path))
        logger.info(
            "onboard_5act video ready for %s (%d bytes, audio=%s, total %.2fs)",
            analyzed.repo_name, len(payload), has_audio,
            time.perf_counter() - overall_start,
        )
        return payload


# ---------------------------------------------------------------------------
# TTS helpers — adapted to act dict shape
# ---------------------------------------------------------------------------

async def _generate_act_audio(
    acts: list[dict], tmp_path: Path, language: str,
) -> list[Optional[str]]:
    """Generate a TTS audio file per act. Reuses tts_service which expects
    scene-shaped dicts with a "narration" key — and our act dicts already
    carry one. Returns a list of file paths (or None when TTS failed for
    that act).
    """
    try:
        from api.tts_service import generate_all_scene_audio
    except Exception as e:
        logger.warning("TTS service import failed: %s", e)
        return [None] * len(acts)

    # tts_service.generate_all_scene_audio mutates each scene with
    # "audio_path" / "audio_duration" — we treat acts the same way.
    try:
        return await generate_all_scene_audio(acts, str(tmp_path), language=language)
    except Exception as e:
        logger.warning("TTS generation failed for onboard_5act, going silent: %s", e)
        return [None] * len(acts)


def _audio_duration(audio_path: Optional[str]) -> float:
    """Return the duration in seconds of a TTS file, or 0 if unavailable."""
    if not audio_path or not os.path.exists(audio_path):
        return 0.0
    # tts_service writes durations into the scene dict, but we may also
    # need to read directly from disk. Use moviepy as the source of truth.
    try:
        from moviepy import AudioFileClip
    except ImportError:
        try:
            from moviepy.editor import AudioFileClip
        except ImportError:
            return 0.0
    try:
        clip = AudioFileClip(audio_path)
        d = float(getattr(clip, "duration", 0) or 0)
        try:
            clip.close()
        except Exception:
            pass
        return d
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Clip assembly
# ---------------------------------------------------------------------------

def _build_clips(
    acts: list[dict], png_paths: list[str], audio_paths: list[Optional[str]],
) -> list:
    """Build a list of ImageClips (one per act) with audio + fade transitions."""
    try:
        from moviepy import ImageClip, AudioFileClip
    except ImportError:
        from moviepy.editor import ImageClip, AudioFileClip  # type: ignore

    clips = []
    for act, png, audio_path in zip(acts, png_paths, audio_paths):
        duration = float(act.get("duration_seconds", 6.0))
        clip = ImageClip(png)
        if hasattr(clip, "with_duration"):
            clip = clip.with_duration(duration)
        else:
            clip = clip.set_duration(duration)

        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path)
                # Safety belt: if TTS audio is somehow longer than the
                # visual (would happen if SCENE_DURATION_MAX clamped the
                # visual but TTS exceeded the cap), trim the audio so it
                # doesn't bleed into the next act and create overlapping
                # voices. Leave a 100ms gap so the cut isn't audible.
                if getattr(audio_clip, "duration", 0) and audio_clip.duration > duration:
                    trim_to = max(0.5, duration - 0.1)
                    if hasattr(audio_clip, "subclipped"):
                        audio_clip = audio_clip.subclipped(0, trim_to)
                    elif hasattr(audio_clip, "subclip"):
                        audio_clip = audio_clip.subclip(0, trim_to)
                if hasattr(clip, "with_audio"):
                    clip = clip.with_audio(audio_clip)
                else:
                    clip = clip.set_audio(audio_clip)
            except Exception as e:
                logger.warning("Failed to attach audio for act %d: %s",
                               act.get("act_number", -1), e)

        if hasattr(clip, "fadein"):
            clip = clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
        clips.append(clip)

    return clips
