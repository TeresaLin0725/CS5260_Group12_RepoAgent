"""Scene-to-card content transformation for the visual renderer."""

from typing import TYPE_CHECKING, List

from api.video.constants import MAX_KEYWORDS
from api.video.text_utils import (
    _clean_entity_label,
    _clean_keyword,
    _keyword_phrases,
    _module_lookup,
    _segment_narration,
    _segment_narration_sequential,
    _short_desc,
)

if TYPE_CHECKING:
    from api.content_analyzer import AnalyzedContent


def _scene_to_card_content(scene: dict, analyzed: "AnalyzedContent", index: int, total: int) -> dict:
    """Create a visual scene plan with keyword-first metadata for the renderer."""
    section = scene.get("section") or "overview"
    visual_type = scene.get("visual_type") or {
        "overview": "overview_map",
        "core": "core_diagram",
        "expansion": "expansion_ladder",
        "summary": "summary_usecases",
    }.get(section, "overview_map")

    focus_modules = scene.get("focus_modules") or []
    modules = _module_lookup(analyzed)
    scene_entities = [item for item in scene.get("entities", []) if isinstance(item, dict)]
    scene_relations = [item for item in scene.get("relations", []) if isinstance(item, dict)]
    tech_chips = [_clean_keyword(item, 24) for item in (analyzed.tech_stack.languages + analyzed.tech_stack.frameworks)[:4]]

    screen_keywords: list[str] = []
    for module_name in focus_modules:
        module = modules.get(module_name)
        if module:
            screen_keywords.append(_clean_keyword(module.name, 26))
            screen_keywords.extend(_keyword_phrases(module.solves, limit=2))
            screen_keywords.extend(_keyword_phrases(module.role, limit=1))
        else:
            screen_keywords.append(_clean_keyword(module_name, 26))

    if scene_entities:
        screen_keywords = [_clean_keyword(item.get("label", ""), 28) for item in scene_entities[:MAX_KEYWORDS]]
    elif section == "overview":
        screen_keywords = _keyword_phrases(analyzed.project_overview, limit=3) + tech_chips[:2]
    elif section == "summary":
        screen_keywords = _keyword_phrases(analyzed.target_users or scene["narration"], limit=4)

    deduped_keywords: list[str] = []
    for item in screen_keywords:
        cleaned = _clean_keyword(item, 32)
        if cleaned and cleaned.lower() not in {x.lower() for x in deduped_keywords}:
            deduped_keywords.append(cleaned)

    use_cases = [_clean_keyword(item, 34) for item in _keyword_phrases(analyzed.target_users or scene["narration"], limit=3)]
    microcopy = [_clean_keyword(item, 42) for item in _keyword_phrases(scene["narration"], limit=3)]

    module_details: list[dict] = []
    for module_name in focus_modules:
        module = modules.get(module_name)
        if module:
            module_details.append({
                "name": module.name,
                "role": _clean_keyword(module.role or "", 40),
                "solves": _clean_keyword(module.solves or "", 40),
                "stage": getattr(module, "stage", ""),
                "position": _clean_keyword(getattr(module, "position", ""), 40),
            })

    overview_descriptions: list[str] = []
    if section == "overview":
        overview_descriptions = [
            _short_desc(analyzed.project_overview) if analyzed.project_overview else "",
            ", ".join(analyzed.tech_stack.frameworks[:2] + analyzed.tech_stack.languages[:1]),
            _short_desc(analyzed.target_users) if analyzed.target_users else "",
            ", ".join(_short_desc(f.responsibility) for f in analyzed.key_modules[:2]) if analyzed.key_modules else "",
        ]

    core_descriptions: list[str] = []
    if section == "core":
        for m_name in focus_modules:
            m = modules.get(m_name)
            if m:
                core_descriptions.append(_short_desc(m.role, 40))

    # Build personas for comic-style overview and summary scenes
    _svg_cycle = ["person_thinking", "person_at_desk", "person_happy", "process_gear", "person_at_desk"]
    personas: list[dict] = []
    if section == "overview":
        if scene_entities:
            for i, ent in enumerate(scene_entities[:5]):
                label = _clean_entity_label(ent.get("label", ""))
                personas.append({"svg": _svg_cycle[i % len(_svg_cycle)], "label": label, "caption": label})
        else:
            user_role = "Developer"
            if analyzed.target_users:
                first_word = analyzed.target_users.split()[0:2]
                user_role = " ".join(first_word).rstrip("s,.")
                if len(user_role) > 12:
                    user_role = "Developer"
            tech_names = analyzed.tech_stack.frameworks[:2]
            tech_caption = " + ".join(t.split()[0] for t in tech_names) if tech_names else "Code"
            output_types = []
            for kw in ["doc", "diagram", "video", "chat", "pdf", "ppt", "export"]:
                if kw in (analyzed.project_overview or "").lower():
                    output_types.append(kw.capitalize())
            output_caption = " & ".join(output_types[:2]) if output_types else "Documentation"
            personas = [
                {"svg": "person_thinking", "label": user_role, "caption": "What is this repo?"},
                {"svg": "person_at_desk", "label": "Analyze", "caption": tech_caption},
                {"svg": "person_happy", "label": "Understand", "caption": output_caption},
            ]
    elif section == "summary":
        if scene_entities:
            for i, ent in enumerate(scene_entities[:5]):
                label = _clean_entity_label(ent.get("label", ""))
                personas.append({"svg": _svg_cycle[i % len(_svg_cycle)], "label": label, "caption": label})
        else:
            personas = [
                {"svg": "person_at_desk", "label": "User", "caption": "Submit repo URL"},
                {"svg": "process_gear", "label": "Process", "caption": "AI analysis"},
                {"svg": "person_happy", "label": "Result", "caption": "Get walkthrough"},
            ]

    built_entities = [{"label": _clean_entity_label(item.get("label", "")), "kind": item.get("kind", "concept")} for item in scene_entities[:6]]
    narration_text = scene.get("narration", "")

    if section in ("overview", "summary") and personas:
        narration_segments = _segment_narration_sequential(
            narration_text, [p["label"] for p in personas]
        )
    else:
        narration_segments = _segment_narration(narration_text, built_entities)

    return {
        "title": _clean_keyword(scene["title"], 50),
        "subtitle": _clean_keyword(analyzed.repo_name or "Repository Walkthrough", 36),
        "section": section,
        "visual_type": visual_type,
        "visual_motif": str(scene.get("visual_motif") or "").strip().lower(),
        "focus_modules": [_clean_keyword(item, 26) for item in focus_modules[:6]],
        "entities": built_entities,
        "relations": [
            {
                "from": _clean_entity_label(item.get("from", "")),
                "to": _clean_entity_label(item.get("to", "")),
                "type": _clean_keyword(item.get("type", ""), 18),
            }
            for item in scene_relations[:8]
        ],
        "tech_chips": tech_chips,
        "keywords": deduped_keywords[:MAX_KEYWORDS],
        "microcopy": microcopy[:3],
        "use_cases": use_cases[:3],
        "module_details": module_details,
        "overview_descriptions": overview_descriptions,
        "core_descriptions": core_descriptions,
        "personas": personas,
        "narration_segments": narration_segments,
        "footer": f"{analyzed.repo_name or 'Repo'} | {index}/{total}",
    }
