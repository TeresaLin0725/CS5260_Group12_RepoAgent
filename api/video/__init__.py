"""Video export subpackage — structured pipeline for repo → MP4 video generation.

Modules:
    constants       - Shared constants (dimensions, durations, limits)
    progress        - Job progress tracking (in-memory)
    text_utils      - Pure text/keyword utilities
    narration       - Phase 2b: LLM narration script generation
    storyline       - Scene construction, normalization, fallbacks
    card_builder    - Scene → visual card content transformation
    pillow_renderer - Legacy Pillow PNG fallback (deprecated in favor of scene_renderer.py)
    compose         - MoviePy MP4 composition
    orchestrator    - Main pipeline entry point
    api_renderer    - Stub for future external video API integration
"""

# Re-export public API for backward compatibility
from api.video.progress import (  # noqa: F401
    PROGRESS_STEPS,
    clear_progress,
    get_progress,
    update_progress,
)
from api.video.constants import *  # noqa: F401,F403
from api.video.text_utils import (  # noqa: F401
    _bubble_caption,
    _chunk_list,
    _clean_entity_label,
    _clean_keyword,
    _keyword_phrases,
    _module_lookup,
    _segment_narration,
    _segment_narration_sequential,
    _short_desc,
    _split_narration_to_bullets,
    _truncate_narration,
)
from api.video.narration import (  # noqa: F401
    _analysis_to_prompt_json,
    _call_llm_raw,
    _parse_scene_array,
    generate_narration_script,
)
from api.video.storyline import (  # noqa: F401
    _adopt_llm_scene,
    _build_storyline_scenes,
    _fallback_narration_script,
    _normalize_scenes,
)
from api.video.card_builder import _scene_to_card_content  # noqa: F401
from api.video.pillow_renderer import _render_scene_card_image  # noqa: F401
from api.video.compose import (  # noqa: F401
    _build_scene_clip,
    _compose_final_video,
    _read_file_bytes,
)
from api.video.orchestrator import (  # noqa: F401
    render_video,
    render_video_from_analyzed,
)
