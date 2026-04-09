"""
TTS Service — Text-to-Speech via edge-tts.

Converts narration text into MP3 audio files for video scenes.
Uses Microsoft Edge TTS (free, async, high quality, multi-language).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Language code → edge-tts voice mapping
_VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
}

_DEFAULT_VOICE = "en-US-AriaNeural"


def _get_voice(language: str) -> str:
    """Map language code to edge-tts voice name."""
    return _VOICE_MAP.get(language, _DEFAULT_VOICE)


async def generate_scene_audio(
    text: str,
    output_path: str,
    language: str = "en",
    rate: str = "+0%",
) -> Optional[float]:
    """
    Generate TTS audio for a single scene's narration.

    Args:
        text: Narration text to speak.
        output_path: Path to write the MP3 file.
        language: Language code (en, zh, ja, ko).
        rate: Speech rate adjustment (e.g., "+10%", "-5%").

    Returns:
        Duration in seconds of the generated audio, or None if TTS failed.
    """
    if not text or not text.strip():
        logger.warning("Empty narration text, skipping TTS")
        return None

    try:
        import edge_tts
    except ImportError:
        logger.warning("edge-tts not installed, skipping TTS audio generation")
        return None

    voice = _get_voice(language)
    logger.info("Generating TTS audio: voice=%s, text_len=%d, rate=%s", voice, len(text), rate)

    try:
        communicate = edge_tts.Communicate(text=text.strip(), voice=voice, rate=rate)
        await communicate.save(output_path)
    except Exception as e:
        logger.error("TTS generation failed: %s", e)
        return None

    duration = _get_audio_duration(output_path)
    if duration and duration > 0:
        logger.info("TTS audio generated: %.2fs, path=%s", duration, output_path)
        return duration

    logger.warning("TTS audio file generated but duration could not be determined: %s", output_path)
    return None


def _get_audio_duration(path: str) -> Optional[float]:
    """Get duration of an audio file in seconds."""
    # Try moviepy first (already a dependency)
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        pass

    try:
        from moviepy.editor import AudioFileClip
        clip = AudioFileClip(path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        pass

    # Fallback: estimate from file size (MP3 ~16kB/s for edge-tts quality)
    try:
        size = Path(path).stat().st_size
        if size > 0:
            estimated = size / 16000.0
            logger.info("Audio duration estimated from file size: %.2fs", estimated)
            return estimated
    except Exception:
        pass

    return None


async def generate_all_scene_audio(
    scenes: list[dict],
    output_dir: str,
    language: str = "en",
) -> list[Optional[str]]:
    """
    Generate TTS audio for all scenes in parallel.

    Args:
        scenes: List of scene dicts, each with a "narration" key.
        output_dir: Directory to write audio files.
        language: Language code.

    Returns:
        List of audio file paths (or None for scenes where TTS failed).
        Also updates each scene dict with "audio_path" and "audio_duration" keys.
    """
    import asyncio

    out = Path(output_dir)
    audio_paths: list[Optional[str]] = [None] * len(scenes)

    # Build tasks for scenes that have narration
    tasks: list[tuple[int, str, asyncio.Task]] = []
    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "")
        if not narration.strip():
            continue
        audio_path = str(out / f"scene_{i + 1:02d}.mp3")
        tasks.append((i, audio_path, generate_scene_audio(
            text=narration,
            output_path=audio_path,
            language=language,
        )))

    if tasks:
        results = await asyncio.gather(
            *(t[2] for t in tasks), return_exceptions=True,
        )
        for (i, audio_path, _), duration in zip(tasks, results):
            if isinstance(duration, Exception):
                logger.warning("TTS failed for scene %d: %s", i + 1, duration)
                continue
            if duration and duration > 0:
                audio_paths[i] = audio_path
                scenes[i]["audio_path"] = audio_path
                scenes[i]["audio_duration"] = duration

    return audio_paths