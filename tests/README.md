# RepoHelper Tests

This directory contains all tests for the RepoHelper project.

## Directory Structure

```
test/                         # Root-level quick tests (also used by pytest.ini)
  └── test_extract_repo_name.py   # URL / path → repo name extraction
tests/
  ├── unit/                   # Fast, isolated unit tests
  │   ├── test_config.py          # Configuration loading & defaults
  │   ├── test_data_pipeline.py   # Token counting, pipeline preparation
  │   └── test_embedders.py       # Embedder config, factory, env-var selection
  ├── integration/            # Tests that touch multiple components
  │   └── test_embedder_pipeline.py  # Config → factory → live embedder calls
  ├── api/                    # HTTP endpoint tests (require running server)
  │   └── test_api.py             # Health, auth, models, chat stream
  └── run_tests.py            # Convenience test runner
```

## Running Tests

```bash
# All tests
pytest -v

# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# API tests (start backend first: python -m api.main)
pytest tests/api -v

# Via the runner script
python tests/run_tests.py
python tests/run_tests.py --unit
python tests/run_tests.py --check-env
```

## Environment Variables

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` | Google embedder live tests |
| `OPENAI_API_KEY` | OpenAI embedder live tests |
| `DEEPWIKI_EMBEDDER_TYPE` | Override default embedder type (`openai` / `google` / `ollama` / `bedrock`) |

Tests that require API keys are automatically skipped when the keys are not set.

## Adding New Tests

1. Pick the right category: `unit/`, `integration/`, or `api/`.
2. Name the file `test_<component>.py`.
3. Add the project root path setup at the top:
   ```python
   from pathlib import Path
   import sys
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   ```
4. Use `pytest.mark.skipif` for tests that need external services.
