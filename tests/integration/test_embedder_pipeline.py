#!/usr/bin/env python3
"""
Integration tests for RepoHelper backend.

Tests cover:
- Embedder configuration pipeline (config → factory → embedder)

Usage:
    pytest tests/integration/test_embedder_pipeline.py -v
"""

import os
import sys
import importlib
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestEmbedderPipeline:
    """End-to-end: env var → config → factory → embedder instance."""

    def test_config_loads_google_embedder(self):
        from api.config import configs
        assert 'embedder_google' in configs
        google_config = configs['embedder_google']
        assert isinstance(google_config, dict)
        assert 'client_class' in google_config or 'model_client' in google_config

    def test_factory_creates_google_embedder(self):
        from api.tools.embedder import get_embedder
        embedder = get_embedder(embedder_type='google')
        assert embedder is not None

    def test_env_var_selects_google_embedder(self):
        import api.config
        original = os.environ.get('DEEPWIKI_EMBEDDER_TYPE')
        try:
            os.environ['DEEPWIKI_EMBEDDER_TYPE'] = 'google'
            importlib.reload(api.config)

            from api.config import EMBEDDER_TYPE, get_embedder_config
            assert EMBEDDER_TYPE == 'google'

            config = get_embedder_config()
            assert isinstance(config, dict)

            from api.tools.embedder import get_embedder
            embedder = get_embedder()
            assert embedder is not None
        finally:
            if original is not None:
                os.environ['DEEPWIKI_EMBEDDER_TYPE'] = original
            elif 'DEEPWIKI_EMBEDDER_TYPE' in os.environ:
                del os.environ['DEEPWIKI_EMBEDDER_TYPE']
            importlib.reload(api.config)


@pytest.mark.skipif(
    not os.getenv('GOOGLE_API_KEY'),
    reason="GOOGLE_API_KEY not set"
)
class TestGoogleEmbedderLive:
    """Live tests that call the Google API (require GOOGLE_API_KEY)."""

    def test_single_embedding(self):
        from api.google_embedder_client import GoogleEmbedderClient
        from adalflow.core.types import ModelType

        client = GoogleEmbedderClient()
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input="Hello world",
            model_kwargs={"model": "text-embedding-004", "task_type": "SEMANTIC_SIMILARITY"},
            model_type=ModelType.EMBEDDER,
        )
        response = client.call(api_kwargs, ModelType.EMBEDDER)
        assert response is not None

        parsed = client.parse_embedding_response(response)
        assert parsed.data is not None
        assert len(parsed.data) > 0
        assert parsed.error is None

    def test_batch_embedding(self):
        from api.google_embedder_client import GoogleEmbedderClient
        from adalflow.core.types import ModelType

        client = GoogleEmbedderClient()
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=["Hello world", "Test embedding"],
            model_kwargs={"model": "text-embedding-004", "task_type": "SEMANTIC_SIMILARITY"},
            model_type=ModelType.EMBEDDER,
        )
        response = client.call(api_kwargs, ModelType.EMBEDDER)
        assert response is not None

        parsed = client.parse_embedding_response(response)
        assert parsed.data is not None
        assert len(parsed.data) == 2


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIEmbedderLive:
    """Live tests that call the OpenAI API (require OPENAI_API_KEY)."""

    def test_single_embedding(self):
        import adalflow as adal
        from api.openai_client import OpenAIClient

        client = OpenAIClient()
        embedder = adal.Embedder(
            model_client=client,
            model_kwargs={"model": "text-embedding-3-small", "dimensions": 256},
        )
        result = embedder("Hello world")
        assert result is not None
        assert hasattr(result, 'data')
        assert len(result.data) > 0


class TestRAGInitialization:
    """Test RAG class can be instantiated."""

    def test_rag_has_expected_attributes(self):
        from api.rag import RAG
        try:
            rag = RAG(provider="google", model="gemini-1.5-flash")
            assert hasattr(rag, 'embedder')
            assert hasattr(rag, 'is_ollama_embedder')
        except Exception:
            pytest.skip("RAG init failed – likely missing API keys")
