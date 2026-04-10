# NanoBanana API Reference

## Overview

NanoBanana is an external poster/infographic rendering service. It accepts structured layout data and returns a rendered image.

## Base Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOBANANA_BASE_URL` | `http://localhost:8787` | Service endpoint |
| `NANOBANANA_API_KEY` | (empty) | Bearer token for `Authorization` header |
| `NANOBANANA_TIMEOUT` | `120` | Request timeout (seconds) |

## Endpoints

### POST `/api/v1/render`

Render an infographic poster from structured sections.

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <NANOBANANA_API_KEY>  (optional, only if key is set)
```

**Request Body:**
```json
{
  "repo_name": "owner/repo",
  "language": "en",
  "sections": [
    {
      "title": "Project Overview",
      "content": "A brief, visual-friendly summary of the project.",
      "visual_hint": "Globe icon with connected nodes representing microservices"
    },
    {
      "title": "Tech Stack",
      "content": "Python, FastAPI, React, PostgreSQL",
      "visual_hint": "Horizontal bar chart of language distribution"
    }
  ],
  "style": "infographic"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_name` | string | yes | Repository name (used for title/header) |
| `language` | string | yes | Language code (`en`, `zh`, etc.) |
| `sections` | array | yes | Layout sections (see below) |
| `style` | string | no | Rendering style. Default: `"infographic"` |

**Section Object:**

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Section header text |
| `content` | string | Body text (1–3 sentences, concise) |
| `visual_hint` | string | Description of suggested visual element |

**Response (Success):**
```
HTTP 200
Content-Type: image/png
Body: <binary PNG data>
```

**Response (Error):**
```
HTTP 4xx/5xx
Content-Type: text/plain or application/json
Body: Error description
```

## Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Use response body as PNG image |
| 400 | Invalid request | Check sections format |
| 401 | Unauthorized | Set/fix `NANOBANANA_API_KEY` |
| 422 | Validation error | Check required fields |
| 500 | Server error | Check NanoBanana service logs |
| 503 | Service unavailable | NanoBanana may be starting up; retry |

## Available Styles

| Style | Description |
|-------|-------------|
| `infographic` | Default. Clean, icon-heavy layout with section cards |

Additional styles may be added by the NanoBanana service. Consult NanoBanana documentation for the latest list.

## Rate Limits

NanoBanana may enforce rate limiting. If you receive HTTP 429, implement exponential backoff. The current integration does not auto-retry — it surfaces the error to the user.

## Local Development

To run NanoBanana locally for development:

```bash
# Start NanoBanana (see NanoBanana's own docs for full setup)
docker run -p 8787:8787 nanobanana/server:latest

# Set environment variable
export NANOBANANA_BASE_URL=http://localhost:8787
```
