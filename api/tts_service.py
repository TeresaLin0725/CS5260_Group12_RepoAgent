"""Text-to-Speech service for video narration.

Uses OpenAI TTS API (tts-1 / tts-1-hd) as the primary engine.
Each scene's narration text is synthesised to an MP3 file, and its
``audio_duration`` field is set so the video pipeline can size clips.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TTS_MODEL = os.getenv("DEEPWIKI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("DEEPWIKI_TTS_VOICE", "nova")
TTS_SPEED = float(os.getenv("DEEPWIKI_TTS_SPEED", "1.0"))

# Language → voice mapping (override defaults for specific languages)
_LANG_VOICE: dict[str, str] = {
    "zh": "nova",
    "en": "nova",
    "ja": "nova",
}


def _voice_for_language(language: str) -> str:
    return _LANG_VOICE.get(language, TTS_VOICE)


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------

def _get_audio_duration(file_path: str) -> float:
    """Return the duration in seconds of an audio file using mutagen."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(file_path)
        return audio.info.length
    except Exception:
        pass

    # Fallback: estimate from file size (MP3 ≈ 16 KB/s at 128 kbps)
    try:
        size = os.path.getsize(file_path)
        return size / 16_000
    except Exception:
        return 5.0


# ---------------------------------------------------------------------------
# Single-scene TTS
# ---------------------------------------------------------------------------

async def _synthesise_scene(
    text: str,
    output_path: str,
    language: str,
) -> Optional[str]:
    """Generate TTS audio for a single piece of narration text.

    Returns the output file path on success, ``None`` on failure.
    """
    if not text or not text.strip():
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for TTS generation")

    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    voice = _voice_for_language(language)

    try:
        import openai

        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=text.strip(),
            speed=TTS_SPEED,
            response_format="mp3",
        )
        response.stream_to_file(output_path)
        logger.debug("TTS generated: %s (%d chars)", output_path, len(text))
        return output_path

    except Exception as exc:
        logger.warning("TTS synthesis failed for %s: %s", output_path, exc)
        return None


# ---------------------------------------------------------------------------
# Batch generation (called by orchestrator)
# ---------------------------------------------------------------------------

async def generate_all_scene_audio(
    scenes: List[dict],
    output_dir: str,
    *,
    language: str = "en",
) -> List[Optional[str]]:
    """Generate TTS audio for every scene that has narration text.

    For each scene ``scenes[i]``:
    * Reads ``scene["narration"]``
    * Writes ``{output_dir}/scene_{i:02d}.mp3``
    * Sets ``scene["audio_duration"]`` (float seconds)

    Returns a list of file paths (or ``None`` for scenes that failed /
    had no narration).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, scene in enumerate(scenes):
        narration = scene.get("narration", "").strip()
        if not narration:
            tasks.append(None)
            continue
        file_path = str(out / f"scene_{idx:02d}.mp3")
        tasks.append(_synthesise_scene(narration, file_path, language))

    # Run all TTS calls concurrently
    results: list[Optional[str]] = []
    pending = [(i, t) for i, t in enumerate(tasks) if t is not None]
    resolved = await asyncio.gather(
        *[t for _, t in pending],
        return_exceptions=True,
    )

    result_map: dict[int, Optional[str]] = {}
    for (i, _), res in zip(pending, resolved):
        if isinstance(res, Exception):
            logger.warning("TTS failed for scene %d: %s", i, res)
            result_map[i] = None
        else:
            result_map[i] = res

    audio_paths: List[Optional[str]] = []
    for idx in range(len(scenes)):
        path = result_map.get(idx)
        audio_paths.append(path)
        if path and os.path.exists(path):
            scenes[idx]["audio_duration"] = _get_audio_duration(path)
        else:
            scenes[idx]["audio_duration"] = 0

    return audio_paths
