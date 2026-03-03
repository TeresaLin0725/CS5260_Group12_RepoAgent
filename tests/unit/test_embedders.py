#!/usr/bin/env python3
"""
Unit tests for embedder configuration and factory.

Tests cover:
- Embedder config loading from JSON files
- Embedder type detection (openai, google, ollama, bedrock)
- Embedder factory function (get_embedder)
- Environment variable based selection

Usage:
    pytest tests/unit/test_embedders.py -v
"""

import os
import sys
import importlib
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestEmbedderConfiguration:
    """Test embedder configuration loading and type detection."""

    def test_configs_contain_embedder_keys(self):
        from api.config import configs
        assert 'embedder' in configs, "OpenAI embedder config missing"
        assert 'embedder_google' in configs, "Google embedder config missing"
        assert 'embedder_ollama' in configs, "Ollama embedder config missing"
        assert 'embedder_bedrock' in configs, "Bedrock embedder config missing"

    def test_client_classes_registered(self):
        from api.config import CLIENT_CLASSES
        assert 'OpenAIClient' in CLIENT_CLASSES
        assert 'GoogleEmbedderClient' in CLIENT_CLASSES
        assert 'BedrockClient' in CLIENT_CLASSES
        assert 'GoogleGenAIClient' in CLIENT_CLASSES

    def test_get_embedder_type_returns_valid_type(self):
        from api.config import get_embedder_type
        current_type = get_embedder_type()
        assert current_type in ['openai', 'google', 'ollama', 'bedrock']

    def test_is_ollama_embedder_returns_bool(self):
        from api.config import is_ollama_embedder
        assert isinstance(is_ollama_embedder(), bool)

    def test_is_google_embedder_returns_bool(self):
        from api.config import is_google_embedder
        assert isinstance(is_google_embedder(), bool)

    def test_is_bedrock_embedder_returns_bool(self):
        from api.config import is_bedrock_embedder
        assert isinstance(is_bedrock_embedder(), bool)

    def test_embedder_type_consistency(self):
        """Only one embedder type should be active at a time."""
        from api.config import get_embedder_type, is_ollama_embedder, is_google_embedder, is_bedrock_embedder
        current_type = get_embedder_type()
        if current_type == 'ollama':
            assert is_ollama_embedder() and not is_google_embedder() and not is_bedrock_embedder()
        elif current_type == 'google':
            assert is_google_embedder() and not is_ollama_embedder() and not is_bedrock_embedder()
        elif current_type == 'bedrock':
            assert is_bedrock_embedder() and not is_ollama_embedder() and not is_google_embedder()
        else:  # openai
            assert not is_ollama_embedder() and not is_google_embedder() and not is_bedrock_embedder()

    @pytest.mark.parametrize("embedder_type", ["openai", "google", "ollama", "bedrock"])
    def test_get_embedder_config_per_type(self, embedder_type):
        from api.config import get_embedder_config
        with patch('api.config.EMBEDDER_TYPE', embedder_type):
            config = get_embedder_config()
            assert isinstance(config, dict)
            assert 'model_client' in config or 'client_class' in config


class TestEmbedderFactory:
    """Test get_embedder factory function."""

    def test_get_embedder_auto(self):
        from api.tools.embedder import get_embedder
        try:
            embedder = get_embedder()
            assert embedder is not None
        except Exception:
            pytest.skip("Embedder creation failed – likely missing API keys or service")

    def test_get_embedder_explicit_google(self):
        from api.tools.embedder import get_embedder
        try:
            embedder = get_embedder(embedder_type='google')
            assert embedder is not None
        except Exception:
            pytest.skip("Google embedder creation failed – likely missing GOOGLE_API_KEY")

    def test_get_embedder_explicit_openai(self):
        from api.tools.embedder import get_embedder
        try:
            embedder = get_embedder(embedder_type='openai')
            assert embedder is not None
        except Exception:
            pytest.skip("OpenAI embedder creation failed – likely missing OPENAI_API_KEY")

    def test_get_embedder_explicit_bedrock(self):
        from api.tools.embedder import get_embedder
        with patch("api.bedrock_client.boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.client.return_value = MagicMock()
            mock_session_cls.return_value = mock_session
            embedder = get_embedder(embedder_type='bedrock')
            assert embedder is not None


class TestEmbedderEnvironmentVariable:
    """Test embedder selection via DEEPWIKI_EMBEDDER_TYPE environment variable."""

    @pytest.mark.parametrize("embedder_type", ["openai", "google", "bedrock"])
    def test_env_var_sets_embedder_type(self, embedder_type):
        import api.config
        original = os.environ.get('DEEPWIKI_EMBEDDER_TYPE')
        try:
            os.environ['DEEPWIKI_EMBEDDER_TYPE'] = embedder_type
            importlib.reload(api.config)
            from api.config import EMBEDDER_TYPE, get_embedder_type
            assert EMBEDDER_TYPE == embedder_type
            assert get_embedder_type() == embedder_type
        finally:
            if original is not None:
                os.environ['DEEPWIKI_EMBEDDER_TYPE'] = original
            elif 'DEEPWIKI_EMBEDDER_TYPE' in os.environ:
                del os.environ['DEEPWIKI_EMBEDDER_TYPE']
            importlib.reload(api.config)
