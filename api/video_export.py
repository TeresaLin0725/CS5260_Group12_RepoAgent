"""Backward-compatibility shim. Real code lives in api/video/."""

# Re-export everything from api.video subpackage.
# The star import covers public names; underscored names are re-exported
# explicitly below for test compatibility.
from api.video import *  # noqa: F401,F403
from api.video import (  # noqa: F401
    _analysis_to_prompt_json,
    _adopt_llm_scene,
    _bubble_caption,
    _build_scene_clip,
    _build_storyline_scenes,
    _call_llm_raw,
    _chunk_list,
    _clean_entity_label,
    _clean_keyword,
    _compose_final_video,
    _fallback_narration_script,
    _keyword_phrases,
    _module_lookup,
    _normalize_scenes,
    _parse_scene_array,
    _read_file_bytes,
    _render_scene_card_image,
    _scene_to_card_content,
    _segment_narration,
    _segment_narration_sequential,
    _short_desc,
    _split_narration_to_bullets,
    _truncate_narration,
)
