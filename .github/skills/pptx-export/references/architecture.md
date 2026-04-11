# Gamma PPTX Export Architecture

## Pipeline Flow

```
+------------------------------------------------------------------+
|                   Gamma PPTX Export Pipeline                      |
|                                                                   |
|  +--------------+   +--------------+   +----------------------+   |
|  | Phase 1+2a   |   | Phase 2b     |   | Phase 3              |   |
|  | Content      |-->| LLM Outline  |-->| Gamma API            |   |
|  | Analysis     |   | Generation   |   | (Create+Poll+Download|   |
|  +--------------+   +--------------+   +----------------------+   |
|        |                   |                      |               |
|  AnalyzedContent    Text outline           PPTX file bytes       |
+------------------------------------------------------------------+
```

## File Map

```
api/
+-- gamma_ppt_export.py       # Phase 2b + Phase 3 (this module)
|   +-- _analysis_to_gamma_payload()     # AnalyzedContent -> dict
|   +-- generate_gamma_outline()         # Phase 2b: LLM -> text outline
|   +-- _create_generation()             # Phase 3a: POST /generations
|   +-- _poll_generation()               # Phase 3b: GET /generations/{id}
|   +-- _download_pptx()                 # Phase 3c: fetch exportUrl
|   +-- render_gamma_ppt_from_analyzed() # Async entry point
|
+-- export_service.py         # Orchestrator
|   +-- ExportFormat.GAMMA_PPT  # Enum value
|   +-- _render_gamma_ppt()     # Routes to gamma_ppt_export
|
+-- content_analyzer.py       # Phase 1+2a (shared with PDF/PPT/Video/Poster)
|   +-- analyze_repo_content()
|
+-- prompts.py
|   +-- GAMMA_PPT_OUTLINE_PROMPT  # LLM prompt template
|
+-- agent/
    +-- tools/export_tools.py # GENERATE_GAMMA_PPT registration
    +-- scheduler.py          # Preamble + Stage 2 signals
    +-- planner.py            # Keyword matching (shared)
```

## Data Flow

### 1. User Query -> Action Tag

```
"生成gamma ppt" --> RuleBasedPlanner
                 +- keyword match: "gamma" in GENERATE_GAMMA_PPT.keywords
                 +- generation marker: "生成" in generation_markers
                 +- ToolPlan(should_invoke=True, tool=GENERATE_GAMMA_PPT)
              --> AgentScheduler
                 +- "我将通过 Gamma 为该仓库生成一份精美 PPT 演示文稿。\n[ACTION:GENERATE_GAMMA_PPT]"
```

### 2. Action Tag -> PPTX File

```
[ACTION:GENERATE_GAMMA_PPT]
    |
    v
export_service.export_repo(request, ExportFormat.GAMMA_PPT)
    |
    +- analyze_repo_content(request)  -> AnalyzedContent
    |
    +- _render_gamma_ppt(analyzed)
        |
        +- generate_gamma_outline(analyzed)
        |     +- LLM call with GAMMA_PPT_OUTLINE_PROMPT
        |     +- Returns: text outline with --- slide separators
        |
        +- _create_generation(outline)
        |     +- POST /generations -> generationId
        |
        +- _poll_generation(generationId)
        |     +- GET /generations/{id} every 5s until completed
        |     +- Returns: {status: "completed", exportUrl: "https://..."}
        |
        +- _download_pptx(exportUrl)
              +- GET signed URL -> PPTX bytes
```

### 3. Gamma API Request Flow

```
Request 1 — Create:
  POST https://public-api.gamma.app/v1.0/generations
  Headers:
    X-API-KEY: {GAMMA_API_KEY}
    Content-Type: application/json
  Body:
    {
      "inputText": "<outline text>",
      "textMode": "condense",
      "format": "presentation",
      "numCards": 10,
      "exportAs": "pptx",
      ...
    }
  Response: {"generationId": "gen_abc123", ...}

Request 2 — Poll (repeated):
  GET https://public-api.gamma.app/v1.0/generations/gen_abc123
  Headers:
    X-API-KEY: {GAMMA_API_KEY}
  Response: {"status": "completed", "exportUrl": "https://signed-url...", "gammaUrl": "https://gamma.app/..."}

Request 3 — Download:
  GET https://signed-url...
  Response: binary PPTX file
```

## Integration with Existing Agent System

The Gamma PPT tool follows the exact same pattern as PDF/PPT/Video/Poster:

1. **Registration**: `AgentTool` in `export_tools.py` with unique keywords
2. **Planning**: `RuleBasedPlanner` keyword match -> `ToolPlan`
3. **Scheduling**: `AgentScheduler` Stage 1 action tag or Stage 2 inference
4. **Rendering**: `export_service.py` routes to format-specific renderer
5. **Output**: Binary content + filename + MIME type via `ExportResult`
