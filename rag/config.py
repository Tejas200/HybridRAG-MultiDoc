from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class RagConfig:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "chunks")
    qdrant_timeout_seconds: int = _env_int("QDRANT_TIMEOUT_SECONDS", 180)
    qdrant_grpc_port: int = _env_int("QDRANT_GRPC_PORT", 6334)

    dense_model: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    dense_vector: str = os.getenv("QDRANT_DENSE_VECTOR", "dense")
    dense_dim: int = _env_int("EMBEDDING_DIMENSIONS", 384)
    bge_query_prefix: str = os.getenv(
        "BGE_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )

    sparse_model: str = os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")
    sparse_vector: str = os.getenv("QDRANT_SPARSE_VECTOR", "bm25")

    top_k_candidates: int = _env_int("TOP_K_CANDIDATES", 50)
    context_k: int = _env_int("CONTEXT_K", 8)

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_models_url: str = f"{openrouter_base_url}/models"
    openrouter_chat_url: str = f"{openrouter_base_url}/chat/completions"
    openrouter_primary_model: str = os.getenv("PRIMARY_LLM_MODEL", "openrouter/auto")
    openrouter_free_model_refresh_seconds: int = _env_int("FREE_MODEL_REFRESH_INTERVAL", 3600)
    openrouter_timeout_seconds: int = _env_int("LLM_TIMEOUT_SECONDS", 60)
    openrouter_max_retries: int = _env_int("LLM_MAX_RETRIES", 3)
    openrouter_backoff_base_seconds: float = _env_float("LLM_BACKOFF_BASE", 2.0)


CONFIG = RagConfig()
