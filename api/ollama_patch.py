from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import requests
import os
import time
import random

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class OllamaModelNotFoundError(Exception):
    """Custom exception for when Ollama model is not found"""
    pass

def check_ollama_model_exists(model_name: str, ollama_host: str = None) -> bool:
    """
    Check if an Ollama model exists before attempting to use it.
    
    Args:
        model_name: Name of the model to check
        ollama_host: Ollama host URL, defaults to localhost:11434
        
    Returns:
        bool: True if model exists, False otherwise
    """
    if ollama_host is None:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        # Remove /api prefix if present and add it back
        if ollama_host.endswith('/api'):
            ollama_host = ollama_host[:-4]
        
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model.get('name', '').split(':')[0] for model in models_data.get('models', [])]
            model_base_name = model_name.split(':')[0]  # Remove tag if present
            
            is_available = model_base_name in available_models
            if is_available:
                logger.info(f"Ollama model '{model_name}' is available")
            else:
                logger.warning(f"Ollama model '{model_name}' is not available. Available models: {available_models}")
            return is_available
        else:
            logger.warning(f"Could not check Ollama models, status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to Ollama to check models: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking Ollama model availability: {e}")
        return False

class OllamaDocumentProcessor(DataComponent):
    """
    Process documents for Ollama embeddings by processing one document at a time.
    Adalflow Ollama Client does not support batch embedding, so we need to process each document individually.
    """
    def __init__(self, embedder: adal.Embedder) -> None:
        super().__init__()
        self.embedder = embedder
    
    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents individually for Ollama embeddings")

        successful_docs: List[Document] = []
        expected_embedding_size = None

        # ---- Ollama 直连配置 ----
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        # ---- knobs：按机器调整 ----
        base_sleep = float(os.getenv("OLLAMA_EMBED_SLEEP_SEC", "0.4"))        # 每次请求前节流（建议 >=0.4）
        max_retries = int(os.getenv("OLLAMA_EMBED_MAX_RETRIES", "10"))        # 503/429 重试次数
        max_backoff = float(os.getenv("OLLAMA_EMBED_MAX_BACKOFF", "10.0"))    # 最大退避秒数
        timeout_sec = float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "120"))     # 单次请求超时

        def _sleep_with_jitter(seconds: float) -> None:
            time.sleep(max(0.0, seconds + random.uniform(0, 0.2)))

        def _fetch_embedding(text: str) -> List[float]:
            """
            直接调用 Ollama embeddings API，绕开 adalflow/httpx
            """
            resp = requests.post(
                f"{ollama_host}/api/embeddings",
                json={"model": model, "prompt": text},   # 你已验证 prompt 方式可用
                timeout=timeout_sec,
                headers={"Connection": "close"},         # 关键：禁用 keep-alive，避免连接池异常
            )
            # 429/503 让上层重试；其他错误直接抛
            if resp.status_code in (429, 503):
                raise RuntimeError(f"transient_http_{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()

            # Ollama embeddings 常见返回：{"embedding":[...]}
            if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]

            # 兼容极少数返回形态
            if isinstance(data, dict) and "data" in data and data["data"]:
                emb = data["data"][0].get("embedding")
                if isinstance(emb, list):
                    return emb

            raise RuntimeError(f"Unexpected embeddings response: keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
            file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')

            text = doc.text
            if text is None:
                logger.warning(f"Document '{file_path}' has empty text, skipping")
                continue
            text = str(text)

            if len(text.strip()) < 10:
                logger.info(f"Document '{file_path}' too short (<10 chars), skipping")
                continue

            # 每个文档请求前节流一下
            _sleep_with_jitter(base_sleep)

            embedding = None
            for attempt in range(max_retries):
                try:
                    embedding = _fetch_embedding(text)
                    break
                except Exception as e:
                    msg = str(e).lower()

                    # 只对临时错误重试（429/503/timeout/连接问题）
                    is_transient = (
                        "transient_http_503" in msg
                        or "transient_http_429" in msg
                        or "timeout" in msg
                        or "timed out" in msg
                        or "connection" in msg
                        or "reset" in msg
                        or "refused" in msg
                    )
                    if not is_transient or attempt == max_retries - 1:
                        logger.error(f"Error processing document '{file_path}': {e}, skipping")
                        embedding = None
                        break

                    backoff = min(max_backoff, 0.5 * (2 ** attempt))
                    logger.warning(
                        f"Transient Ollama embedding error for '{file_path}' "
                        f"(attempt {attempt+1}/{max_retries}): {e}. Retrying in {backoff:.2f}s"
                    )
                    _sleep_with_jitter(backoff)

            if not embedding:
                logger.warning(f"Failed to get embedding for document '{file_path}', skipping")
                continue

            # 维度一致性校验
            if expected_embedding_size is None:
                expected_embedding_size = len(embedding)
                logger.info(f"Expected embedding size set to: {expected_embedding_size}")
            elif len(embedding) != expected_embedding_size:
                logger.warning(
                    f"Document '{file_path}' has inconsistent embedding size "
                    f"{len(embedding)} != {expected_embedding_size}, skipping"
                )
                continue

            output[i].vector = embedding
            successful_docs.append(output[i])

        logger.info(f"Successfully processed {len(successful_docs)}/{len(output)} documents with consistent embeddings")
        return successful_docs



    # def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
    #     output = deepcopy(documents)
    #     logger.info(f"Processing {len(output)} documents individually for Ollama embeddings")

    #     successful_docs = []
    #     expected_embedding_size = None

    #     for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
    #         try:
    #             # Get embedding for a single document
    #             result = self.embedder(input=doc.text)
    #             if result.data and len(result.data) > 0:
    #                 embedding = result.data[0].embedding

    #                 # Validate embedding size consistency
    #                 if expected_embedding_size is None:
    #                     expected_embedding_size = len(embedding)
    #                     logger.info(f"Expected embedding size set to: {expected_embedding_size}")
    #                 elif len(embedding) != expected_embedding_size:
    #                     file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
    #                     logger.warning(f"Document '{file_path}' has inconsistent embedding size {len(embedding)} != {expected_embedding_size}, skipping")
    #                     continue

    #                 # Assign the embedding to the document
    #                 output[i].vector = embedding
    #                 successful_docs.append(output[i])
    #             else:
    #                 file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
    #                 logger.warning(f"Failed to get embedding for document '{file_path}', skipping")
    #         except Exception as e:
    #             file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
    #             logger.error(f"Error processing document '{file_path}': {e}, skipping")

    #     logger.info(f"Successfully processed {len(successful_docs)}/{len(output)} documents with consistent embeddings")
    #     return successful_docs