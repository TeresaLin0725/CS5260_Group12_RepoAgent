---
name: poster-export
description: 'Generate illustrated posters (画报/海报/infographic) from repository code via NanoBanana. Use when: creating visual posters from repos, configuring NanoBanana integration, customizing poster layout prompts, debugging poster export pipeline, extending poster rendering options. Covers the full workflow from user query to rendered poster image.'
---

# Poster Export (NanoBanana Integration)

## When to Use

- User asks to generate a **poster**, **infographic**, **画报**, or **海报** from a repository
- Debugging or extending the NanoBanana rendering pipeline
- Customizing poster layout prompts or visual styles
- Configuring NanoBanana connection (URL, API key, timeout)
- Adding new poster templates or section types

## Architecture Overview

Load [architecture reference](./references/architecture.md) for the full system map.

```
User Query ("生成画报" / "create a poster")
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Agent Scheduler (Stage 1)                           │
│  ├─ Planner keyword match: poster/画报/海报/infographic │
│  └─ Returns [ACTION:GENERATE_POSTER]                 │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Export Service                                      │
│  ├─ Phase 1: Content analysis (shared pipeline)      │
│  ├─ Phase 2a: Structured analysis (AnalyzedContent)  │
│  ├─ Phase 2b: LLM → poster layout spec (JSON)       │
│  └─ Phase 3: NanoBanana render → PNG bytes           │
└──────────────────────────────────────────────────────┘
```

## Key Files

| File | Role |
|------|------|
| `api/agent/tools/export_tools.py` | `GENERATE_POSTER` tool registration + keywords |
| `api/agent/scheduler.py` | Preamble messages + Stage 2 poster signals |
| `api/poster_export.py` | Phase 2b layout generation + Phase 3 NanoBanana call |
| `api/export_service.py` | `ExportFormat.POSTER` enum + `_render_poster()` |
| `api/prompts.py` | `POSTER_LAYOUT_PROMPT` template |
| `tests/unit/test_agent_tools.py` | Poster-specific test classes |

## Configuration

NanoBanana is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOBANANA_BASE_URL` | `http://localhost:8787` | NanoBanana service endpoint |
| `NANOBANANA_API_KEY` | (empty) | Bearer token for authentication |
| `NANOBANANA_TIMEOUT` | `120` | Request timeout in seconds |

## Pipeline Details

### Phase 2b: Layout Generation

The LLM converts `AnalyzedContent` into a poster layout spec using `POSTER_LAYOUT_PROMPT`.

**Input**: Structured analysis JSON (repo name, architecture, tech stack, modules, data flow)

**Output**: JSON array of sections:
```json
[
  {
    "title": "Project Identity",
    "content": "Concise summary text",
    "visual_hint": "Description of suggested icon/diagram"
  }
]
```

Target: 5–8 sections covering project identity, architecture, tech stack, key components, data flow, and audience.

### Phase 3: NanoBanana Render

The layout spec is sent to NanoBanana's `/api/v1/render` endpoint:

```json
POST /api/v1/render
{
  "repo_name": "owner/repo",
  "language": "en",
  "sections": [...],
  "style": "infographic"
}
```

Response: PNG image bytes (HTTP 200) or error detail (non-200).

## Keyword Triggers

The following keywords activate the poster tool:

| Language | Keywords |
|----------|----------|
| English | `poster`, `illustrated poster`, `pictorial`, `infographic` |
| Chinese | `画报`, `海报`, `图文海报`, `画报制作` |

These are registered in `export_tools.py` and must NOT overlap with PDF/PPT/Video keywords.

## Customizing the Layout Prompt

Edit `POSTER_LAYOUT_PROMPT` in `api/prompts.py` to:
- Change the number of target sections (default: 5–8)
- Add specific visual styles or branding
- Adjust section types (e.g., add "security overview", "performance metrics")
- Change the output schema (must stay JSON-parseable)

## Extending Poster Styles

To add a new poster style (e.g., "timeline", "comparison"):

1. Add a `style` parameter to the NanoBanana payload in `poster_export.py`
2. Optionally create a new prompt variant in `prompts.py`
3. Add keywords to distinguish styles in `export_tools.py` if needed

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "NanoBanana render failed (HTTP 5xx)" | NanoBanana service down | Check `NANOBANANA_BASE_URL`, ensure service is running |
| "NanoBanana render failed (HTTP 401)" | Missing/wrong API key | Set `NANOBANANA_API_KEY` environment variable |
| Timeout errors | Large repo / slow NanoBanana | Increase `NANOBANANA_TIMEOUT` |
| Layout has too few sections | LLM output parsing issue | Check `_extract_json_from_llm` in `content_analyzer.py` |
| "poster" triggers PDF instead | Keyword overlap | Verify keywords in `export_tools.py` have no overlap |

## Checklist

Before deploying poster export:

- [ ] `NANOBANANA_BASE_URL` is set and reachable
- [ ] `NANOBANANA_API_KEY` is set (if service requires auth)
- [ ] `POSTER_LAYOUT_PROMPT` produces valid JSON sections
- [ ] No keyword overlap with existing tools: `pytest tests/unit/test_agent_tools.py::TestPosterToolRegistration::test_poster_no_keyword_overlap -v`
- [ ] All poster tests pass: `pytest tests/unit/test_agent_tools.py -k Poster -v`
- [ ] Frontend handles `[ACTION:GENERATE_POSTER]` tag and `image/png` response

## References

- [Architecture deep-dive](./references/architecture.md)
- [NanoBanana API reference](./references/nanobanana-api.md)
