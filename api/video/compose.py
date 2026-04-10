"""MP4 video composition: scene clips → final video file."""

import logging
import os
from typing import Any, List, Optional

from api.video.constants import TRANSITION_SECONDS, VIDEO_FPS

logger = logging.getLogger(__name__)


def _close_clip_tree(clip: Any, seen: Optional[set[int]] = None) -> None:
    """Recursively close a moviepy clip and any nested child/audio clips."""
    if clip is None:
        return

    if seen is None:
        seen = set()

    clip_id = id(clip)
    if clip_id in seen:
        return
    seen.add(clip_id)

    for attr_name in ("audio", "mask"):
        nested_clip = getattr(clip, attr_name, None)
        if nested_clip is not None:
            _close_clip_tree(nested_clip, seen)

    child_clips = getattr(clip, "clips", None)
    if child_clips:
        for child_clip in child_clips:
            _close_clip_tree(child_clip, seen)

    try:
        clip.close()
    except Exception:
        pass


def _build_scene_clip(image_path: str, duration: float, audio_path: Optional[str] = None):
    """Create a moviepy clip from a rendered image card, optionally with TTS audio."""
    try:
        from moviepy import ImageClip, AudioFileClip
    except ImportError:
        try:
            from moviepy.editor import ImageClip, AudioFileClip
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    clip = ImageClip(image_path)
    if hasattr(clip, "with_duration"):
        clip = clip.with_duration(duration)
    else:
        clip = clip.set_duration(duration)

    if audio_path and os.path.exists(audio_path):
        try:
            audio_clip = AudioFileClip(audio_path)
            if hasattr(clip, "with_audio"):
                clip = clip.with_audio(audio_clip)
            else:
                clip = clip.set_audio(audio_clip)
            logger.info("Audio attached to scene clip: %s (%.2fs)", audio_path, audio_clip.duration)
        except Exception as e:
            logger.warning("Failed to attach audio to scene clip: %s", e)

    if hasattr(clip, "fadein"):
        clip = clip.fadein(TRANSITION_SECONDS).fadeout(TRANSITION_SECONDS)
    return clip


def _compose_final_video(clips: List[Any], output_path: str, has_audio: bool = False) -> None:
    """Concatenate scene clips and write the final MP4 to disk."""
    try:
        from moviepy import concatenate_videoclips
    except ImportError:
        try:
            from moviepy.editor import concatenate_videoclips
        except ImportError as exc:
            raise ImportError("moviepy is required for video export. Install moviepy.") from exc

    try:
        final_clip = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_SECONDS)
    except TypeError:
        final_clip = concatenate_videoclips(clips, method="compose")
    try:
        write_kwargs = {
            "fps": VIDEO_FPS,
            "codec": "libx264",
            "preset": "ultrafast",
            "bitrate": "700k",
            "logger": None,
            "threads": max(1, min(4, os.cpu_count() or 1)),
        }
        if has_audio:
            temp_audiofile = os.path.join(
                os.path.dirname(output_path),
                f"{os.path.splitext(os.path.basename(output_path))[0]}_temp_audio.m4a",
            )
            write_kwargs["audio"] = True
            write_kwargs["audio_codec"] = "aac"
            write_kwargs["audio_bitrate"] = "128k"
            write_kwargs["temp_audiofile"] = temp_audiofile
            write_kwargs["remove_temp"] = True
        else:
            write_kwargs["audio"] = False
        logger.info(
            "Writing MP4 to %s with %d clips (audio=%s)",
            output_path,
            len(clips),
            has_audio,
        )
        final_clip.write_videofile(output_path, **write_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to render MP4 video. Ensure ffmpeg is installed and available to moviepy."
        ) from exc
    finally:
        seen: set[int] = set()
        _close_clip_tree(final_clip, seen)
        for clip in clips:
            _close_clip_tree(clip, seen)


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()
