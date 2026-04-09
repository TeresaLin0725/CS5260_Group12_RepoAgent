"""MP4 video composition: scene clips → final video file."""

import logging
import os
from typing import Any, List, Optional

from api.video.constants import TRANSITION_SECONDS, VIDEO_FPS

logger = logging.getLogger(__name__)


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
        }
        if has_audio:
            write_kwargs["audio"] = True
            write_kwargs["audio_codec"] = "aac"
            write_kwargs["audio_bitrate"] = "128k"
        else:
            write_kwargs["audio"] = False
        final_clip.write_videofile(output_path, **write_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to render MP4 video. Ensure ffmpeg is installed and available to moviepy."
        ) from exc
    finally:
        try:
            final_clip.close()
        except Exception:
            pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()
