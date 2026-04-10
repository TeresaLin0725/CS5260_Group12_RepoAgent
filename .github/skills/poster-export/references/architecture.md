# Poster Export Architecture

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Poster Export Pipeline                       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ Phase 1+2a   │   │ Phase 2b     │   │ Phase 3            │  │
│  │ Content      │──▶│ LLM Layout   │──▶│ NanoBanana Render  │  │
│  │ Analysis     │   │ Generation   │   │ (External Service) │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│        │                   │                      │             │
│  AnalyzedContent    JSON sections[]         PNG image bytes     │
└─────────────────────────────────────────────────────────────────┘
```

## File Map

```
api/
├── poster_export.py          # Phase 2b + Phase 3 (this module)
│   ├── _analysis_to_poster_payload()  # AnalyzedContent → dict
│   ├── generate_poster_layout()       # Phase 2b: LLM → sections JSON
│   ├── _call_nanobanana()             # Phase 3: HTTP POST → PNG bytes
│   └── render_poster_from_analyzed()  # Sync entry point
│
├── export_service.py         # Orchestrator
│   ├── ExportFormat.POSTER   # Enum value
│   └── _render_poster()      # Routes to poster_export
│
├── content_analyzer.py       # Phase 1+2a (shared with PDF/PPT/Video)
│   └── analyze_repo_content()
│
├── prompts.py
│   └── POSTER_LAYOUT_PROMPT  # LLM prompt template
│
└── agent/
    ├── tools/export_tools.py # GENERATE_POSTER registration
    ├── scheduler.py          # Preamble + Stage 2 signals
    └── planner.py            # Keyword matching (shared)
```

## Data Flow

### 1. User Query → Action Tag

```
"生成画报" ──▶ RuleBasedPlanner
                 ├─ keyword match: "画报" ∈ GENERATE_POSTER.keywords
                 ├─ generation marker: "生成" ∈ generation_markers
                 └─ ToolPlan(should_invoke=True, tool=GENERATE_POSTER)
              ──▶ AgentScheduler
                 └─ "我将通过 NanoBanana 为该仓库生成一份图文画报。\n[ACTION:GENERATE_POSTER]"
```

### 2. Action Tag → Rendered Poster

```
[ACTION:GENERATE_POSTER]
    │
    ▼
export_service.export_repo(request, ExportFormat.POSTER)
    │
    ├─ analyze_repo_content(request)  → AnalyzedContent
    │
    └─ _render_poster(analyzed)
        │
        ├─ generate_poster_layout(analyzed)
        │     └─ LLM call with POSTER_LAYOUT_PROMPT
        │     └─ Returns: [{"title":..., "content":..., "visual_hint":...}, ...]
        │
        └─ _call_nanobanana(payload)
              └─ POST /api/v1/render → PNG bytes
```

### 3. NanoBanana Request/Response

```
Request:
  POST {NANOBANANA_BASE_URL}/api/v1/render
  Headers:
    Content-Type: application/json
    Authorization: Bearer {NANOBANANA_API_KEY}  (if set)
  Body:
    {
      "repo_name": "owner/repo",
      "language": "en",
      "sections": [...],
      "style": "infographic"
    }

Response:
  200 OK → binary PNG image
  4xx/5xx → error text (first 500 chars logged)
```

## Integration with Existing Agent System

The poster tool follows the exact same pattern as PDF/PPT/Video:

1. **Registration**: `AgentTool` in `export_tools.py` with unique keywords
2. **Planning**: `RuleBasedPlanner` keyword match → `ToolPlan`
3. **Scheduling**: `AgentScheduler` Stage 1 action tag or Stage 2 inference
4. **Rendering**: `export_service.py` routes to format-specific renderer
5. **Output**: Binary content + filename + MIME type via `ExportResult`
