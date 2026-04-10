"""
Template: Poster export tool customization.

INSTRUCTIONS:
  Use this template when modifying or extending the poster export feature.
  Copy the relevant section and adapt to your use case.

This file is a reference — do NOT import it directly.
"""

# ============================================================================
# SECTION A: Export Tool Registration
# Location: api/agent/tools/export_tools.py → build_export_tool_registry()
# ============================================================================

"""
registry.register(
    AgentTool(
        name="GENERATE_POSTER",
        action_tag="[ACTION:GENERATE_POSTER]",
        description="Generate a repository illustrated poster via NanoBanana.",
        keywords=(
            # English
            "poster",
            "illustrated poster",
            "pictorial",
            "infographic",
            # Chinese
            "画报",
            "海报",
            "图文海报",
            "画报制作",
        ),
    )
)
"""


# ============================================================================
# SECTION B: Scheduler Preamble Messages
# Location: api/agent/scheduler.py → _tool_preamble() both mappings
# ============================================================================

"""
# In the Chinese mapping dict:
"GENERATE_POSTER": "我将通过 NanoBanana 为该仓库生成一份图文画报。",

# In the English mapping dict:
"GENERATE_POSTER": "I will generate an illustrated poster for this repository via NanoBanana.",
"""


# ============================================================================
# SECTION C: Stage 2 Inference Signals
# Location: api/agent/scheduler.py → _infer_from_recommendation()
# ============================================================================

"""
poster_signals = ("poster", "画报", "海报", "infographic", "图文海报", "pictorial")
# ... add to scores dict:
scores["[ACTION:GENERATE_POSTER]"] = poster_score
"""


# ============================================================================
# SECTION D: NanoBanana Payload Customization
# Location: api/poster_export.py → render_poster_from_analyzed()
# ============================================================================

"""
# To add a new style or custom field to the NanoBanana payload:
nb_payload = {
    "repo_name": analyzed.repo_name,
    "language": analyzed.language,
    "sections": sections,
    "style": "infographic",       # Change to: "timeline", "comparison", etc.
    # "custom_field": "value",    # Add custom fields as needed
}
"""


# ============================================================================
# SECTION E: Poster Layout Prompt
# Location: api/prompts.py → POSTER_LAYOUT_PROMPT
# ============================================================================

"""
POSTER_LAYOUT_PROMPT = '''You are a creative technical illustrator. ...

Respond ONLY with a valid JSON array (no markdown fences):
[
  {{
    "title": "<section title>",
    "content": "<concise summary text for this section>",
    "visual_hint": "<description of suggested visual element>"
  }}
]

Target 5-8 sections covering: ...

Project analysis:
{analysis_json}
'''
"""
