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

# Per-subtitle-line timing (Step 2 / Step 4):
SUBLINE_MIN_SECONDS = 1.4         # never flash a subtitle by faster than this
SUBLINE_AUDIO_PAD = 0.15          # tiny tail of silence after each TTS line


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

        # ── Step 2: TTS audio per subtitle line (parallelized) ─────────
        # One TTS call per ``narration_lines`` entry so each subtitle
        # frame ships with its own audio. This keeps voice and subtitle
        # perfectly in sync — no proportional-duration arithmetic to get
        # wrong, and an act's total duration becomes simply the sum of
        # its line durations.
        update_progress(job_id, 2, "Generating narration audio...")
        tts_start = time.perf_counter()
        per_line_audio_paths = await _generate_subtitle_audio(
            acts, tmp_path, analyzed.language,
        )
        has_audio = any(p for line_list in per_line_audio_paths for p in line_list)
        line_count = sum(len(line_list) for line_list in per_line_audio_paths)
        ok_count = sum(1 for line_list in per_line_audio_paths for p in line_list if p)
        logger.info(
            "onboard_5act: TTS done in %.2fs (%d/%d subtitle lines with audio across %d acts)",
            time.perf_counter() - tts_start, ok_count, line_count, len(acts),
        )

        # Each act's total duration is now the sum of its sub-line
        # audio durations. Visual frames are sized to match each line's
        # audio (with a small floor for very short lines).
        for act, line_audios in zip(acts, per_line_audio_paths):
            durations = [
                max(SUBLINE_MIN_SECONDS, _audio_duration(p) + SUBLINE_AUDIO_PAD)
                for p in line_audios
            ]
            act["sub_durations"] = durations
            act["duration_seconds"] = max(SCENE_DURATION_MIN, sum(durations))

        # ── Step 3: render N PNGs per act (one per subtitle line) ───────
        # Each act renders one frame per ``narration_lines`` entry so the
        # subtitle bar at the bottom advances sentence-by-sentence in
        # time with the TTS. Single-line narrations still produce a
        # single PNG (back-compat).
        update_progress(job_id, 3, "Rendering act visuals...")
        render_start = time.perf_counter()
        per_act_pngs: list[list[str]] = []
        render_coros = []
        for i, act in enumerate(acts, start=1):
            lines = act.get("narration_lines") or [act.get("narration", "")]
            if not lines or (len(lines) == 1 and not lines[0].strip()):
                lines = [act.get("narration", "")]
            act_pngs: list[str] = []
            for sub_idx, line in enumerate(lines):
                # Render one frame per subtitle line: clone the act dict
                # and override its narration so _wrap_page draws this
                # specific line in the bottom subtitle bar.
                act_frame = dict(act)
                act_frame["narration"] = line
                png = str(
                    tmp_path
                    / f"act{i:02d}_{act['section']}_{sub_idx:02d}.png"
                )
                act_pngs.append(png)
                render_coros.append(render_act_to_png(act_frame, png))
            per_act_pngs.append(act_pngs)
        await _asyncio.gather(*render_coros)
        total_pngs = sum(len(p) for p in per_act_pngs)
        logger.info(
            "onboard_5act: %d PNGs rendered across %d acts in %.2fs",
            total_pngs, len(acts), time.perf_counter() - render_start,
        )

        # ── Step 4: assemble per-act composite clips ──────────────────
        # Each act becomes ONE clip whose visual is a chain of subtitle
        # frames, and each frame carries its OWN per-line TTS audio.
        # Sub-clips chain via concatenate_videoclips, so audios sequence
        # naturally with zero drift between voice and subtitle.
        update_progress(job_id, 4, "Assembling video clips...")
        clips = _build_clips(acts, per_act_pngs, per_line_audio_paths)

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

async def _generate_subtitle_audio(
    acts: list[dict], tmp_path: Path, language: str,
) -> list[list[Optional[str]]]:
    """Generate one TTS audio file per subtitle line.

    Returns a list of lists matching the structure of ``narration_lines``:
    ``result[act_index][line_index]`` is the path (or None on failure).

    Internally we flatten every line into pseudo-scenes for one batched
    ``generate_all_scene_audio`` call — same parallelism as before, just
    finer granularity — then re-group by act.
    """
    try:
        from api.tts_service import generate_all_scene_audio
    except Exception as e:
        logger.warning("TTS service import failed: %s", e)
        return [[None] * len(act.get("narration_lines") or [act.get("narration", "")])
                for act in acts]

    # Flatten subtitle lines into pseudo-scenes. We tag each pseudo-scene
    # with a sub-directory of tmp_path so individual MP3 files don't
    # clobber each other (tts_service writes scene_{idx:02d}.mp3).
    flat_dir = tmp_path / "sublines"
    flat_dir.mkdir(parents=True, exist_ok=True)

    flat_scenes: list[dict] = []
    mapping: list[tuple[int, int]] = []  # (act_idx, line_idx)
    for ai, act in enumerate(acts):
        lines = act.get("narration_lines") or [act.get("narration", "")]
        for li, line in enumerate(lines):
            flat_scenes.append({
                "narration": (line or "").strip(),
                "section": f"{act.get('section', 'act')}_{ai:02d}_{li:02d}",
            })
            mapping.append((ai, li))

    try:
        flat_paths = await generate_all_scene_audio(
            flat_scenes, str(flat_dir), language=language,
        )
    except Exception as e:
        logger.warning("TTS generation failed for onboard_5act, going silent: %s", e)
        flat_paths = [None] * len(flat_scenes)

    # Re-group into [[...act 1 lines...], [...act 2 lines...], ...]
    grouped: list[list[Optional[str]]] = [[] for _ in acts]
    for (ai, _li), path in zip(mapping, flat_paths):
        grouped[ai].append(path)
    return grouped


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
    acts: list[dict],
    per_act_pngs: list[list[str]],
    per_act_audio: list[list[Optional[str]]],
) -> list:
    """Build one composite clip per act.

    Each subtitle frame in the act:
      * carries its OWN per-line TTS audio (sized exactly to the audio)
      * runs back-to-back with the next frame via concatenate(method="chain")

    Result: voice and subtitle never drift because each pair is one
    self-contained sub-clip. Cross-fade transitions are only applied
    between acts (outer concatenate).
    """
    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
    except ImportError:  # moviepy 1.x compatibility
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips  # type: ignore

    clips = []
    for act, png_list, audio_list in zip(acts, per_act_pngs, per_act_audio):
        composite = _build_act_composite(
            act, png_list, audio_list, ImageClip, AudioFileClip, concatenate_videoclips,
        )
        if composite is None:
            continue
        if hasattr(composite, "fadein"):
            composite = composite.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
        clips.append(composite)

    return clips


def _build_act_composite(
    act: dict,
    png_list: list[str],
    audio_list: list[Optional[str]],
    ImageClip, AudioFileClip, concatenate_videoclips,
):
    """Build the per-act composite: chain of (image + matching audio)
    sub-clips, one per subtitle line.

    Visual duration of each sub-clip = its own audio duration + small
    padding (or SUBLINE_MIN_SECONDS when no audio). No proportional
    arithmetic — sync is exact by construction.
    """
    if not png_list:
        return None

    sub_clips = []
    for sub_idx, png in enumerate(png_list):
        audio_path = audio_list[sub_idx] if sub_idx < len(audio_list) else None
        audio_clip = None
        sub_duration = SUBLINE_MIN_SECONDS

        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path)
                a_dur = float(getattr(audio_clip, "duration", 0) or 0)
                sub_duration = max(SUBLINE_MIN_SECONDS, a_dur + SUBLINE_AUDIO_PAD)
            except Exception as e:
                logger.warning("Failed to load audio for act %d line %d: %s",
                               act.get("act_number", -1), sub_idx, e)
                audio_clip = None

        clip = ImageClip(png)
        if hasattr(clip, "with_duration"):
            clip = clip.with_duration(sub_duration)
        else:
            clip = clip.set_duration(sub_duration)

        if audio_clip is not None:
            if hasattr(clip, "with_audio"):
                clip = clip.with_audio(audio_clip)
            else:
                clip = clip.set_audio(audio_clip)

        sub_clips.append(clip)

    if len(sub_clips) == 1:
        return sub_clips[0]
    # method="chain" → hard cuts between subtitle frames (visual is
    # identical except for the subtitle bar; cross-fading would jitter).
    return concatenate_videoclips(sub_clips, method="chain")
