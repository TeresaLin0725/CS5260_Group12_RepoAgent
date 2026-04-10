"""Scene storyline construction: LLM adoption + fallback generation + normalization."""

import logging
from typing import TYPE_CHECKING, List, Optional

from api.video.constants import (
    MAX_EXPANSION_SCENES,
    MAX_SCENES,
    SCENE_DURATION_DEFAULT,
    SCENE_DURATION_MAX,
    SCENE_DURATION_MIN,
)
from api.video.text_utils import (
    _clean_entity_label,
    _clean_keyword,
    _keyword_phrases,
    _truncate_narration,
)

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent

logger = logging.getLogger(__name__)


def _build_storyline_scenes(analyzed: "AnalyzedContent", raw_scenes: List[dict]) -> List[dict]:
    """Build walkthrough scenes: trust LLM scene decisions, fill gaps with fallbacks.

    When raw_scenes from the LLM are available, respect the LLM's choices for
    scene count, entity count, and expansion grouping. Only generate fallback
    scenes for sections the LLM missed entirely (overview/core/summary).
    """

    def _pick_scenes(section: str) -> List[dict]:
        return [s for s in raw_scenes if str(s.get("section") or "").strip().lower() == section]

    llm_overview = _pick_scenes("overview")
    llm_core = _pick_scenes("core")
    llm_expansion = _pick_scenes("expansion")
    llm_summary = _pick_scenes("summary")

    core_modules = [m for m in analyzed.module_progression if getattr(m, "stage", "") == "core"]
    expansion_modules = [m for m in analyzed.module_progression if getattr(m, "stage", "") == "expansion"]

    scenes: list[dict] = []

    # --- Overview ---
    if llm_overview:
        for s in llm_overview:
            scenes.append(_adopt_llm_scene(s, "overview", "overview_map", analyzed))
    else:
        tech_anchor: list[str] = []
        tech_anchor.extend(analyzed.tech_stack.languages[:2])
        tech_anchor.extend(analyzed.tech_stack.frameworks[:2])
        overview_fallback = (
            f"So what is {analyzed.repo_name or 'this project'}? "
            f"{analyzed.project_overview.strip()} "
            f"{'It is built with ' + ', '.join(tech_anchor[:4]) + '.' if tech_anchor else ''}"
        ).strip()
        scenes.append({
            "title": analyzed.repo_name or "Repository Overview",
            "section": "overview", "visual_type": "overview_map", "visual_motif": "diagram",
            "entities": [
                {"label": _clean_entity_label(analyzed.repo_name or "Repository"), "kind": "file"},
                {"label": "Core path", "kind": "concept"},
                {"label": "Users", "kind": "user"},
                {"label": "Outputs", "kind": "concept"},
            ],
            "relations": [
                {"from": _clean_entity_label(analyzed.repo_name or "Repository"), "to": "Core path", "type": "feeds"},
                {"from": "Core path", "to": "Outputs", "type": "extends"},
                {"from": "Outputs", "to": "Users", "type": "helps"},
            ],
            "narration": _truncate_narration(overview_fallback),
            "duration_seconds": 6, "focus_modules": [],
        })

    # --- Core ---
    if llm_core:
        for s in llm_core:
            focus = [m.name for m in core_modules] if core_modules else []
            scenes.append(_adopt_llm_scene(s, "core", "core_diagram", analyzed, focus_modules=focus))
    else:
        core_focus = core_modules[:4]
        if core_focus:
            core_names = " and ".join(m.name for m in core_focus)
            core_roles = " ".join(f"{m.name} {m.role.rstrip('.')}." for m in core_focus[:2])
            core_default = (
                f"The minimum viable system is built on {core_names}. "
                f"{core_roles} "
                f"With just these pieces, users can already get the core value out of {analyzed.repo_name or 'the project'}."
            )
        else:
            core_default = (
                f"Let's look at the foundation. The smallest useful version of {analyzed.repo_name or 'this project'} "
                f"needs just a few key modules working together to deliver its core value."
            )
        scenes.append({
            "title": "The Core Backbone",
            "section": "core", "visual_type": "core_diagram", "visual_motif": "relay",
            "entities": [{"label": _clean_entity_label(m.name), "kind": "file"} for m in core_focus],
            "relations": [
                {"from": _clean_entity_label(core_focus[i].name), "to": _clean_entity_label(core_focus[i + 1].name), "type": "calls"}
                for i in range(max(0, len(core_focus) - 1))
            ],
            "narration": _truncate_narration(core_default),
            "duration_seconds": 7, "focus_modules": [m.name for m in core_focus],
        })

    # --- Expansion ---
    if llm_expansion:
        for s in llm_expansion[:MAX_EXPANSION_SCENES]:
            scenes.append(_adopt_llm_scene(s, "expansion", "expansion_ladder", analyzed))
    else:
        for index, module in enumerate(expansion_modules[:MAX_EXPANSION_SCENES], start=1):
            solves_text = module.solves.rstrip('.') if module.solves else "a gap in the system"
            role_text = module.role.rstrip('.') if module.role else "extends the core"
            default_narration = (
                f"At this point the core works, but there is a problem: {solves_text}. "
                f"{module.name} addresses this. It {role_text}. "
                f"With this in place, the system becomes more capable."
            )
            scenes.append({
                "title": f"Expansion: {_clean_entity_label(module.name)}",
                "section": "expansion", "visual_type": "expansion_ladder",
                "visual_motif": ["dialogue", "analogy", "relay", "diagram"][index % 4],
                "entities": [
                    {"label": _clean_entity_label(module.name), "kind": "file"},
                    {"label": "Core path", "kind": "concept"},
                    {"label": _clean_entity_label(_keyword_phrases(module.solves, 1)[0] if _keyword_phrases(module.solves, 1) else 'Capability'), "kind": "concept"},
                ],
                "relations": [
                    {"from": "Core path", "to": _clean_entity_label(module.name), "type": "extends"},
                    {"from": _clean_entity_label(module.name), "to": _clean_entity_label(_keyword_phrases(module.solves, 1)[0] if _keyword_phrases(module.solves, 1) else 'Capability'), "type": "helps"},
                ],
                "narration": _truncate_narration(default_narration),
                "duration_seconds": 6, "focus_modules": [module.name],
            })

    # --- Summary ---
    if llm_summary:
        for s in llm_summary:
            scenes.append(_adopt_llm_scene(s, "summary", "summary_usecases", analyzed))
    else:
        user_story = analyzed.target_users[:320] if analyzed.target_users else ""
        if user_story:
            summary_default = (
                f"Now let's put it all together. {user_story} "
                f"That is what {analyzed.repo_name or 'this project'} enables end to end."
            )
        else:
            summary_default = (
                f"With all these pieces in place, {analyzed.repo_name or 'this project'} "
                f"goes from a collection of files to a working system that users can rely on for their day to day workflow."
            )
        scenes.append({
            "title": "Complete System and Use Cases",
            "section": "summary", "visual_type": "summary_usecases", "visual_motif": "usecases",
            "entities": [
                {"label": "Users", "kind": "user"},
                {"label": "Workflow", "kind": "concept"},
                {"label": "Outcome", "kind": "concept"},
            ],
            "relations": [
                {"from": "Users", "to": "Workflow", "type": "calls"},
                {"from": "Workflow", "to": "Outcome", "type": "helps"},
            ],
            "narration": _truncate_narration(summary_default),
            "duration_seconds": 6, "focus_modules": [],
        })

    return scenes[:MAX_SCENES]


def _adopt_llm_scene(
    raw: dict,
    expected_section: str,
    default_visual_type: str,
    analyzed: "AnalyzedContent",
    focus_modules: Optional[List[str]] = None,
) -> dict:
    """Adopt an LLM-generated scene, enforcing rendering constraints."""
    valid_visual_types = {"overview_map", "core_diagram", "expansion_ladder", "summary_usecases"}
    vtype = str(raw.get("visual_type") or "").strip().lower()
    if vtype not in valid_visual_types:
        vtype = default_visual_type

    valid_motifs = {"diagram", "relay", "dialogue", "analogy", "usecases"}
    motif = str(raw.get("visual_motif") or "").strip().lower()
    if motif not in valid_motifs:
        motif = "diagram"

    entities = [
        {"label": _clean_entity_label(e.get("label", "")), "kind": e.get("kind", "concept")}
        for e in (raw.get("entities") or [])
        if isinstance(e, dict) and e.get("label")
    ]
    relations = [r for r in (raw.get("relations") or []) if isinstance(r, dict)]

    return {
        "title": _clean_keyword(str(raw.get("title") or f"{expected_section.title()} Scene").strip(), 50),
        "section": expected_section,
        "visual_type": vtype,
        "visual_motif": motif,
        "entities": entities,
        "relations": relations,
        "narration": _truncate_narration(str(raw.get("narration") or "").strip()),
        "duration_seconds": raw.get("duration_seconds", 6),
        "focus_modules": focus_modules or [],
    }


def _fallback_narration_script(analyzed: "AnalyzedContent") -> List[dict]:
    """Build a deterministic storyline-first script directly from structured data."""
    return _build_storyline_scenes(analyzed, raw_scenes=[])


def _normalize_scenes(raw_scenes: List[dict], repo_name: str) -> List[dict]:
    """Normalize raw scene data into a renderer-friendly structure."""
    normalized: list[dict] = []

    for index, raw_scene in enumerate(raw_scenes[:MAX_SCENES], start=1):
        if not isinstance(raw_scene, dict):
            continue

        title = str(raw_scene.get("title") or f"Scene {index}").strip() or f"Scene {index}"
        narration = str(raw_scene.get("narration") or "").strip()
        if not narration:
            continue

        duration = raw_scene.get("duration_seconds", SCENE_DURATION_DEFAULT)
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            duration = SCENE_DURATION_DEFAULT
        duration = max(SCENE_DURATION_MIN, min(duration, SCENE_DURATION_MAX))

        normalized.append({
            "title": title[:80],
            "section": str(raw_scene.get("section") or "").strip().lower(),
            "visual_type": str(raw_scene.get("visual_type") or "").strip().lower(),
            "visual_motif": str(raw_scene.get("visual_motif") or "").strip().lower(),
            "focus_modules": [str(item).strip() for item in raw_scene.get("focus_modules", []) if str(item).strip()],
            "entities": [item for item in raw_scene.get("entities", []) if isinstance(item, dict)],
            "relations": [item for item in raw_scene.get("relations", []) if isinstance(item, dict)],
            "narration": _truncate_narration(narration),
            "duration_seconds": duration,
        })

    if normalized:
        return normalized

    fallback_title = repo_name or "Repository Overview"
    return [{
        "title": fallback_title,
        "section": "overview", "visual_type": "overview_map", "visual_motif": "diagram",
        "focus_modules": [], "entities": [], "relations": [],
        "narration": f"This video provides a quick overview of {fallback_title}.",
        "duration_seconds": SCENE_DURATION_DEFAULT,
    }]
