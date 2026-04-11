# Gamma API Reference

## Overview

Gamma (gamma.app) is an AI-powered presentation generation service. It accepts text input and produces professionally designed presentations that can be exported as PPTX, PDF, or PNG.

## Base Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GAMMA_API_KEY` | (empty, **required**) | API key from Account Settings > API Keys |
| `GAMMA_POLL_INTERVAL` | `5` | Seconds between status polls |
| `GAMMA_TIMEOUT` | `300` | Maximum wait time in seconds |
| `GAMMA_NUM_CARDS` | `10` | Target number of slides |

## Authentication

- Header: `X-API-KEY: <your-key>` (NOT `Authorization: Bearer`)
- Generate keys: Account Settings > API Keys tab
- Requires paid plan: Pro, Ultra, Teams, or Business

## Endpoints

### POST `/generations`

Create a new async generation job.

**Base URL**: `https://public-api.gamma.app/v1.0`

**Headers:**
```
X-API-KEY: <GAMMA_API_KEY>
Content-Type: application/json
```

**Request Body:**
```json
{
  "inputText": "Topic: Project Overview\n- Key point 1\n- Key point 2\n---\nNext slide topic...",
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

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputText` | string | yes | Content for the presentation (up to 400k chars) |
| `textMode` | string | yes | `generate` (expand brief), `condense` (summarize), `preserve` (keep as-is) |
| `format` | string | yes | `presentation`, `document`, `social`, `webpage` |
| `numCards` | integer | no | Target number of slides/cards |
| `exportAs` | string | no | `pptx`, `pdf`, `png` |
| `themeId` | string | no | Theme ID from GET /themes |
| `textOptions.tone` | string | no | e.g., "professional", "casual" |
| `textOptions.audience` | string | no | e.g., "executives", "students" |
| `textOptions.amount` | string | no | `brief`, `medium`, `detailed`, `extensive` |
| `textOptions.language` | string | no | ISO language code (50+ supported) |
| `imageOptions.source` | string | no | `aiGenerated`, `webFreeToUseCommercially`, `noImages`, `pexels`, `giphy`, `pictographic` |

**Response:**
```json
{
  "generationId": "gen_abc123"
}
```

### GET `/generations/{id}`

Poll generation status.

**Response (pending):**
```json
{
  "status": "pending"
}
```

**Response (completed):**
```json
{
  "status": "completed",
  "exportUrl": "https://signed-download-url...",
  "gammaUrl": "https://gamma.app/docs/..."
}
```

**Response (failed):**
```json
{
  "status": "failed",
  "error": "Error description"
}
```

### GET `/themes`

List available workspace themes (standard + custom).

### GET `/folders`

List workspace folders for organizing output.

## Slide Breaks

Use `\n---\n` in `inputText` to force slide/card breaks. This is how our outline prompt structures the content.

## Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200/201 | Success | Parse response body |
| 400 | Invalid request | Check payload format |
| 401 | Unauthorized | Check `GAMMA_API_KEY` |
| 429 | Rate limited | Wait; check `x-ratelimit-remaining-*` headers |
| 500 | Server error | Check Gamma status page |

## Rate Limits

Response headers include:
- `x-ratelimit-remaining-burst` — short-term limit
- `x-ratelimit-remaining` — hourly limit
- `x-ratelimit-remaining-daily` — daily limit

## Official Documentation

- Developer docs: https://developers.gamma.app
- Getting started: https://developers.gamma.app/docs/getting-started
- API access/pricing: https://developers.gamma.app/docs/get-access
- Generate API parameters: https://developers.gamma.app/guides/generate-api-parameters-explained
- POST /generations reference: https://developers.gamma.app/generations/create-generation
