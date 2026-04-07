# RepoHelper — Claude Code Project Guide

## Project Overview

RepoHelper is an AI-powered repository documentation generator (NUS CS5260 Group 12). It analyzes GitHub/GitLab/Bitbucket repos and produces structured docs, Mermaid diagrams, interactive AI chat (RAG), and multi-format exports (PDF, PPT, Video).

**Tech stack:** Next.js 15 + React 19 frontend (`src/`) | FastAPI Python backend (`api/`) | Multi-LLM provider support

## Active Development

- **Branch:** `SK_video` — video export feature
- **Dev plan:** `.local/docs/video-pipeline-dev-plan.md`
- **Current sprint:** P0 (TTS audio) + P1 (narrative prompt optimization)

## Key Architecture

### Video Pipeline (main focus)
```
content_analyzer.py  → Phase 1 (FAISS retrieval) + Phase 2a (structured LLM analysis → AnalyzedContent)
video_export.py      → Phase 2b (narration script via LLM) + Phase 3 (Pillow PNG → MoviePy → MP4)
export_service.py    → Orchestration layer routing to PDF/PPT/Video renderers
prompts.py           → All prompt templates (STRUCTURED_ANALYSIS_PROMPT, VIDEO_NARRATION_PROMPT, etc.)
```

### Frontend
```
src/app/chat/page.tsx    → Chat UI with export action tracking
src/app/page.tsx         → Home page with repo URL input
```

## Conventions

- Respond in Chinese unless asked otherwise
- Prefer concise, direct communication
- When facing build-vs-buy decisions, give a clear recommendation with trade-off analysis
- Video design principle: comprehension over aesthetics — every element must aid understanding
