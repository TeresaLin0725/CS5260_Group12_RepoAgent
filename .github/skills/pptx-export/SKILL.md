---
name: pptx-export
description: 'Generate professionally designed PPTX presentations from repository code via Gamma.app API. Use when: creating polished slide decks from repos, configuring Gamma integration, customizing outline prompts, debugging Gamma export pipeline, extending presentation options. Covers the full workflow from user query to downloaded PPTX file.'
---

# PPTX Export (Gamma.app Integration)

## When to Use

- User asks to generate a **Gamma PPT**, **gamma slides**, **精美ppt**, **设计感ppt**, or **AI ppt** from a repository
- Debugging or extending the Gamma API integration
- Customizing the outline prompt that feeds Gamma
- Configuring Gamma connection (API key, timeout, card count)
- Understanding the difference between basic PPT (python-pptx) and Gamma PPT

## Architecture Overview

Load [architecture reference](./references/architecture.md) for the full system map.

```
User Query ("生成gamma ppt" / "create gamma slides")
    |
    v
+------------------------------------------------------+
|  Agent Scheduler (Stage 1)                           |
|  +- Planner keyword match: gamma/精美ppt/ai ppt     |
|  +- Returns [ACTION:GENERATE_GAMMA_PPT]              |
+------------------------------------------------------+
    |
    v
+------------------------------------------------------+
|  Export Service                                       |
|  +- Phase 1: Content analysis (shared pipeline)      |
|  +- Phase 2a: Structured analysis (AnalyzedContent)  |
|  +- Phase 2b: LLM -> outline text for Gamma          |
|  +- Phase 3: Gamma API create + poll + download PPTX |
+------------------------------------------------------+
```

## Key Files

| File | Role |
|------|------|
| `api/agent/tools/export_tools.py` | `GENERATE_GAMMA_PPT` tool registration + keywords |
| `api/agent/scheduler.py` | Preamble messages + Stage 2 gamma PPT signals |
| `api/gamma_ppt_export.py` | Phase 2b outline generation + Phase 3 Gamma API calls |
| `api/export_service.py` | `ExportFormat.GAMMA_PPT` enum + `_render_gamma_ppt()` |
| `api/prompts.py` | `GAMMA_PPT_OUTLINE_PROMPT` template |
| `src/app/api/export/repo-gamma-ppt/route.ts` | Next.js proxy route |
| `src/app/chat/page.tsx` | Frontend action tag handling for `GENERATE_GAMMA_PPT` |

## Configuration

Gamma is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GAMMA_API_KEY` | (empty, **required**) | API key from Gamma Account Settings > API Keys |
| `GAMMA_POLL_INTERVAL` | `5` | Seconds between status polls |
| `GAMMA_TIMEOUT` | `300` | Maximum seconds to wait for generation |
| `GAMMA_NUM_CARDS` | `10` | Number of slides/cards to generate |

**Note**: Gamma API requires a paid plan (Pro, Ultra, Teams, or Business).

## Pipeline Details

### Phase 2b: Outline Generation

The LLM converts `AnalyzedContent` into a rich text outline using `GAMMA_PPT_OUTLINE_PROMPT`.

**Input**: Structured analysis JSON (repo name, architecture, tech stack, modules, data flow)

**Output**: Plain text outline with `---` slide separators:
```
Project Name: MyRepo

---

Project Overview
- MyRepo is a microservice framework that...
- It provides...

---

Architecture & Design
- Uses event-driven architecture with...
- Key design decisions include...
```

### Phase 3: Gamma API (Async Generation)

The pipeline uses Gamma's asynchronous generation API:

1. **POST `/generations`** — start generation job
   ```json
   {
     "inputText": "<outline from Phase 2b>",
     "textMode": "condense",
     "format": "presentation",
     "numCards": 10,
     "exportAs": "pptx",
     "textOptions": {
       "tone": "professional",
       "audience": "developers and technical stakeholders",
       "amount": "detailed",
       "language": "en"
     },
     "imageOptions": {
       "source": "webFreeToUseCommercially"
     }
   }
   ```

2. **GET `/generations/{id}`** — poll until `status == "completed"`
   - Polls every `GAMMA_POLL_INTERVAL` seconds
   - Times out after `GAMMA_TIMEOUT` seconds

3. **Download** — fetch PPTX from the signed `exportUrl` in the response

## Keyword Triggers

The following keywords activate the Gamma PPT tool:

| Language | Keywords |
|----------|----------|
| English | `gamma ppt`, `gamma slides`, `gamma presentation`, `gamma deck`, `gamma`, `ai ppt` |
| Chinese | `精美ppt`, `gamma演示`, `设计感ppt`, `gamma幻灯片` |

These are registered in `export_tools.py` and must NOT overlap with PDF/PPT/Video/Poster keywords.

## Gamma PPT vs Basic PPT

| Feature | Basic PPT (`GENERATE_PPT`) | Gamma PPT (`GENERATE_GAMMA_PPT`) |
|---------|---------------------------|----------------------------------|
| Rendering | Local python-pptx | Gamma.app cloud API |
| Design quality | Basic (dark blue theme) | Professional (AI-designed) |
| Images | None | Auto-sourced from web |
| Speed | Fast (~10s) | Slower (~30-120s) |
| Requires API key | No | Yes (`GAMMA_API_KEY`) |
| Offline | Yes | No |

## Customizing the Outline Prompt

Edit `GAMMA_PPT_OUTLINE_PROMPT` in `api/prompts.py` to:
- Change the number of target slides
- Adjust the outline structure or topic order
- Add specific content areas (e.g., security, performance)
- Change the writing style or tone

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "GAMMA_API_KEY is not set" | Missing API key | Set `GAMMA_API_KEY` environment variable |
| "Gamma generation creation failed (HTTP 401)" | Invalid API key | Regenerate key in Gamma Account Settings |
| "Gamma generation creation failed (HTTP 429)" | Rate limit exceeded | Wait and retry; check daily limits |
| "Gamma generation timed out" | Gamma service slow | Increase `GAMMA_TIMEOUT` |
| "Gamma generation failed" | Content issue | Check outline quality; simplify input |
| "No exportUrl found" | API response format changed | Check Gamma API docs for updates |
| "gamma" triggers basic PPT instead | Keyword overlap | Verify keywords in `export_tools.py` have no overlap |

## Checklist

Before deploying Gamma PPT export:

- [ ] `GAMMA_API_KEY` is set and valid (paid Gamma plan required)
- [ ] `GAMMA_PPT_OUTLINE_PROMPT` produces clean text outlines
- [ ] No keyword overlap with existing tools: `pytest tests/unit/test_agent_tools.py -k "GammaPpt" -v`
- [ ] Frontend handles `[ACTION:GENERATE_GAMMA_PPT]` tag and PPTX response
- [ ] Gamma API is reachable from deployment environment

## References

- [Architecture deep-dive](./references/architecture.md)
- [Gamma API reference](./references/gamma-api.md)
