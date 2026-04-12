"""Main video pipeline orchestrator: AnalyzedContent → MP4 bytes."""

import asyncio as _asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from api.video.card_builder import _scene_to_card_content
from api.video.compose import _build_scene_clip, _compose_final_video, _read_file_bytes
from api.video.constants import (
    AUDIO_PADDING_SECONDS,
    SCENE_DURATION_MAX,
    SCENE_DURATION_MIN,
    TRANSITION_SECONDS,
)
from api.video.narration import generate_narration_script
from api.video.pillow_renderer import _render_scene_card_image
from api.video.progress import update_progress
from api.video.storyline import _build_storyline_scenes, _normalize_scenes

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)
COMPOSE_TIMEOUT_SECONDS = max(60, int(os.getenv("VIDEO_COMPOSE_TIMEOUT_SECONDS", "480")))


async def render_video_from_analyzed(  # noqa: C901
    analyzed: "AnalyzedContent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    job_id: Optional[str] = None,
) -> bytes:
    """
    Phase 2b-video plus Phase 3: AnalyzedContent to MP4 bytes.

    Strategy: if FAL_API_KEY is configured, try AI video API first.
    On failure (credits exhausted, network error, etc.), fall back to baseline.
    """
    overall_start = time.perf_counter()

    # Try API path if configured
    fal_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if fal_key:
        logger.info("Video export for %s: trying API path (fal.ai)", analyzed.repo_name)
        try:
            return await _render_via_api_path(analyzed, provider, model, job_id)
        except Exception as api_err:
            logger.warning("API video generation failed, falling back to baseline: %s", api_err)

    logger.info("Video export for %s: using baseline pipeline", analyzed.repo_name)

    # --- Baseline pipeline below ---
    update_progress(job_id, 1)
    update_progress(job_id, 2)  # Step 2: narration script
    narration_start = time.perf_counter()
    raw_scenes = await generate_narration_script(analyzed, provider=provider, model=model)
    normalized_raw = _normalize_scenes(raw_scenes, analyzed.repo_name)
    scenes = _build_storyline_scenes(analyzed, normalized_raw)
    scenes = _normalize_scenes(scenes, analyzed.repo_name)
    logger.info("Timing - video narration + storyline planning completed in %.2fs", time.perf_counter() - narration_start)
    total = len(scenes)

    if total == 0:
        raise ValueError("No valid narration scenes were generated for video export.")

    with tempfile.TemporaryDirectory(prefix="repohelper_video_") as tmpdir:
        tmp_path = Path(tmpdir)

        # --- TTS audio generation ---
        update_progress(job_id, 3)  # Step 3: TTS audio
        tts_start = time.perf_counter()
        has_audio = False
        try:
            from api.tts_service import generate_all_scene_audio
            audio_paths = await generate_all_scene_audio(
                scenes, str(tmp_path), language=analyzed.language,
            )
            has_audio = any(p is not None for p in audio_paths)
            logger.info(
                "Timing - TTS generation completed in %.2fs (%d/%d scenes with audio)",
                time.perf_counter() - tts_start,
                sum(1 for p in audio_paths if p),
                total,
            )
        except Exception as e:
            logger.warning("TTS generation failed, falling back to silent video: %s", e)
            audio_paths = [None] * total

        # --- Update scene durations based on audio length ---
        for scene in scenes:
            audio_duration = scene.get("audio_duration")
            if audio_duration and audio_duration > 0:
                scene["duration_seconds"] = max(
                    SCENE_DURATION_MIN,
                    min(audio_duration + AUDIO_PADDING_SECONDS, SCENE_DURATION_MAX),
                )

        # --- Render scene images + build clips (multi-frame with subtitles) ---
        use_playwright = True
        try:
            from api.scene_renderer import render_scene_to_png, close_browser
        except ImportError:
            use_playwright = False
            logger.warning("Playwright renderer not available, falling back to Pillow")

        try:
            from moviepy import ImageClip, AudioFileClip, concatenate_videoclips as _concat
        except ImportError:
            from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips as _concat

        # Pre-compute expansion indices for each scene
        expansion_counter = 0
        scene_expansion_indices = []
        for scene in scenes:
            if scene.get("section") == "expansion":
                expansion_counter += 1
            scene_expansion_indices.append(expansion_counter or 1)

        # --- Parallel PNG rendering phase ---
        update_progress(job_id, 4)  # Step 4: scene rendering
        render_start = time.perf_counter()

        scene_cards = []
        render_tasks = []
        for index, scene in enumerate(scenes, start=1):
            card = _scene_to_card_content(scene, analyzed, index, total)
            card["narration"] = scene.get("narration", "")
            scene_cards.append(card)
            segments = card.get("narration_segments") or [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]
            exp_idx = scene_expansion_indices[index - 1]

            if use_playwright and len(segments) > 1:
                for seg_idx, seg in enumerate(segments):
                    frame_path = str(tmp_path / f"scene_{index:02d}_f{seg_idx:02d}.png")
                    render_tasks.append((index, seg_idx, frame_path, card, exp_idx,
                                         seg["text"], seg["highlight_labels"]))
            else:
                image_path = str(tmp_path / f"scene_{index:02d}.png")
                subtitle_text = segments[0]["text"] if segments else ""
                highlight_labels = segments[0]["highlight_labels"] if segments else []
                render_tasks.append((index, -1, image_path, card, exp_idx,
                                     subtitle_text, highlight_labels))

        async def _render_one(task_info):
            index, seg_idx, path, card, exp_idx, sub_text, hl_labels = task_info
            if use_playwright:
                try:
                    await render_scene_to_png(
                        card, path,
                        expansion_index=exp_idx,
                        subtitle_text=sub_text,
                        highlight_labels=hl_labels,
                    )
                    return
                except Exception as render_err:
                    seg_label = f" frame {seg_idx}" if seg_idx >= 0 else ""
                    logger.warning("Playwright render failed for scene %d%s: %s", index, seg_label, render_err)
            _render_scene_card_image(card, path)

        await _asyncio.gather(*[_render_one(t) for t in render_tasks])
        logger.info(
            "Scene rendering finished using %s (%d frame tasks)",
            "Playwright" if use_playwright else "Pillow",
            len(render_tasks),
        )
        logger.info("Timing - all %d PNG frames rendered in %.2fs", len(render_tasks), time.perf_counter() - render_start)

        # --- Assemble clips from rendered PNGs ---
        clips = []
        for index, scene in enumerate(scenes, start=1):
            card = scene_cards[index - 1]
            scene_duration = scene["duration_seconds"]
            audio_path = audio_paths[index - 1] if index - 1 < len(audio_paths) else None
            segments = card.get("narration_segments") or [{"text": "", "highlight_labels": [], "duration_fraction": 1.0}]

            if use_playwright and len(segments) > 1:
                sub_clips = []
                for seg_idx, seg in enumerate(segments):
                    frame_path = str(tmp_path / f"scene_{index:02d}_f{seg_idx:02d}.png")
                    seg_duration = max(0.5, scene_duration * seg["duration_fraction"])
                    sub_clip = ImageClip(frame_path)
                    if hasattr(sub_clip, "with_duration"):
                        sub_clip = sub_clip.with_duration(seg_duration)
                    else:
                        sub_clip = sub_clip.set_duration(seg_duration)
                    sub_clips.append(sub_clip)

                scene_clip = _concat(sub_clips, method="compose")

                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_clip = AudioFileClip(audio_path)
                        if hasattr(scene_clip, "with_audio"):
                            scene_clip = scene_clip.with_audio(audio_clip)
                        else:
                            scene_clip = scene_clip.set_audio(audio_clip)
                    except Exception as e:
                        logger.warning("Failed to attach audio to scene %d: %s", index, e)

                if hasattr(scene_clip, "fadein"):
                    scene_clip = scene_clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
                clips.append(scene_clip)
            else:
                image_path = str(tmp_path / f"scene_{index:02d}.png")
                clips.append(_build_scene_clip(image_path, scene_duration, audio_path=audio_path))

        # Clean up Playwright browser
        if use_playwright:
            try:
                await close_browser()
            except Exception:
                pass

        output_path = tmp_path / "repo_overview.mp4"
        update_progress(job_id, 5)  # Step 5: MP4 composition
        compose_start = time.perf_counter()
        logger.info(
            "Composing final MP4 for %s with %d scenes (audio=%s, timeout=%ss)",
            analyzed.repo_name,
            total,
            has_audio,
            COMPOSE_TIMEOUT_SECONDS,
        )
        # Run blocking ffmpeg/moviepy in thread to avoid freezing the async event loop
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
                f"Video composition exceeded {COMPOSE_TIMEOUT_SECONDS} seconds and was aborted."
            ) from exc
        logger.info("Timing - final MP4 composition completed in %.2fs", time.perf_counter() - compose_start)

        read_start = time.perf_counter()
        payload = _read_file_bytes(str(output_path))
        logger.info(
            "Final MP4 composed for %s (%d bytes, audio=%s, readback %.2fs, total %.2fs)",
            analyzed.repo_name,
            len(payload),
            has_audio,
            time.perf_counter() - read_start,
            time.perf_counter() - overall_start,
        )
        return payload


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------

async def _render_via_api_path(
    analyzed: "AnalyzedContent",
    provider: Optional[str],
    model: Optional[str],
    job_id: Optional[str],
) -> bytes:
    """Generate video via external API, then overlay TTS narration audio."""
    from api.video.api_renderer import render_via_api

    update_progress(job_id, 1, "Analyzing repository...")
    update_progress(job_id, 2, "Generating narration scenes...")

    raw_scenes = await generate_narration_script(analyzed, provider=provider, model=model)
    normalized = _normalize_scenes(raw_scenes, analyzed.repo_name)
    scenes = _build_storyline_scenes(analyzed, normalized)
    scenes = _normalize_scenes(scenes, analyzed.repo_name)
    logger.info("API path: %d narration scenes prepared", len(scenes))

    update_progress(job_id, 3, "Generating AI video...")

    video_bytes = await render_via_api(analyzed, scenes)

    # --- Add background music ---
    update_progress(job_id, 4, "Adding background music...")
    video_bytes = await _add_bgm(video_bytes)

    update_progress(job_id, 5, "Video ready!")
    return video_bytes


async def _add_bgm(video_bytes: bytes) -> bytes:
    """Add background music to the API-generated video.

    Looks for a BGM file at api/video/assets/bgm.mp3.
    If not found, returns the silent video as-is.
    """
    import tempfile

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    bgm_path = None
    for name in ("bgm_long.mp3", "bgm.mp3", "bgm.wav", "bgm.m4a"):
        candidate = os.path.join(assets_dir, name)
        if os.path.exists(candidate):
            bgm_path = candidate
            break
    if not bgm_path:
        logger.info("No BGM file in %s, returning silent video", assets_dir)
        return video_bytes

    with tempfile.TemporaryDirectory(prefix="repohelper_bgm_") as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        try:
            from moviepy import VideoFileClip, AudioFileClip

            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(bgm_path)

            # Trim BGM to video duration, fade out at end
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclipped(0, video_clip.duration)
            if hasattr(audio_clip, "audio_fadeout"):
                audio_clip = audio_clip.audio_fadeout(1.0)

            # Lower volume to not overwhelm visuals
            audio_clip = audio_clip.with_volume_scaled(0.4)

            video_with_audio = video_clip.with_audio(audio_clip)
            video_with_audio.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast",
                logger=None,
            )

            with open(output_path, "rb") as f:
                result_bytes = f.read()

            video_clip.close()
            audio_clip.close()
            video_with_audio.close()

            logger.info("BGM overlay complete: %d bytes", len(result_bytes))
            return result_bytes

        except Exception as e:
            logger.warning("BGM merge failed: %s, returning silent video", e)
            return video_bytes


def render_video(summary_text: str, repo_name: str) -> bytes:
    """Backward-compatible wrapper for sync callers."""
    import asyncio
    from api.content_analyzer import AnalyzedContent

    analyzed = AnalyzedContent(repo_name=repo_name, project_overview=summary_text)
    return asyncio.run(render_video_from_analyzed(analyzed))
