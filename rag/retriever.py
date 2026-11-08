from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from .config import RagConfig, CONFIG

LOGGER = logging.getLogger(__name__)

_DENSE_MODEL: TextEmbedding | None = None
_SPARSE_MODEL: SparseTextEmbedding | None = None


def get_models() -> tuple[TextEmbedding, SparseTextEmbedding]:
    """Load query embedding models lazily and cache them process-wide."""
    global _DENSE_MODEL, _SPARSE_MODEL
    if _DENSE_MODEL is None:
        _DENSE_MODEL = TextEmbedding(model_name=CONFIG.dense_model)
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name=CONFIG.sparse_model)
    return _DENSE_MODEL, _SPARSE_MODEL


def get_client(config: RagConfig = CONFIG) -> QdrantClient:
    return QdrantClient(
        url=config.qdrant_url,
        timeout=config.qdrant_timeout_seconds,
        grpc_port=config.qdrant_grpc_port,
        prefer_grpc=False,
    )


@dataclass(frozen=True)
class Hit:
    rank: int
    doc_id: str
    page: int
    chunk_type: str
    heading: str
    chunk: str
    payload_html: str | None = None

    @property
    def source(self) -> str:
        return f"{self.doc_id}, p.{self.page}"


def doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        ]
    )


def _as_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def validate_collection(
    client: QdrantClient | None = None,
    collection: str | None = None,
    config: RagConfig = CONFIG,
) -> None:
    """Fail early if the Qdrant collection does not match this retriever."""
    client = client or get_client(config)
    name = collection or config.qdrant_collection

    if not client.collection_exists(name):
        raise RuntimeError(f"Collection {name!r} does not exist")

    info = client.get_collection(name)
    vectors = info.config.params.vectors or {}
    if config.dense_vector not in vectors:
        raise RuntimeError(f"Collection {name!r} has no named {config.dense_vector!r} vector")

    dense_params = vectors[config.dense_vector]
    if dense_params.size != config.dense_dim:
        raise RuntimeError(
            f"Dense size mismatch: stored={dense_params.size}, "
            f"expected={config.dense_dim}"
        )
    if _as_value(dense_params.distance) != "cosine":
        raise RuntimeError(
            f"Dense distance mismatch: stored={dense_params.distance!r}, expected cosine"
        )

    sparse_vectors = info.config.params.sparse_vectors or {}
    if config.sparse_vector not in sparse_vectors:
        raise RuntimeError(
            f"Collection {name!r} has no named {config.sparse_vector!r} sparse vector"
        )
    modifier = sparse_vectors[config.sparse_vector].modifier
    if _as_value(modifier) != "idf":
        raise RuntimeError(
            f"Sparse modifier mismatch: stored={modifier!r}, expected idf"
        )

    LOGGER.debug("Validated Qdrant collection %s", name)


class Retriever:
    def __init__(
        self,
        config: RagConfig = CONFIG,
        client: QdrantClient | None = None,
        *,
        validate: bool = True,
    ) -> None:
        self.config = config
        self.client = client or get_client(config)
        if validate:
            validate_collection(self.client, self.config.qdrant_collection, self.config)

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        doc_id: str | None = None,
    ) -> list[Hit]:
        """Run one hybrid RRF search against the configured chunks collection."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        limit = int(limit or self.config.top_k_candidates)
        candidate_limit = max(self.config.top_k_candidates, limit)
        dense_model, sparse_model = get_models()

        dense_vector = list(dense_model.query_embed(self.config.bge_query_prefix + query))[0]
        sparse_vector = list(sparse_model.query_embed(query))[0]
        dense_list = dense_vector.tolist() if hasattr(dense_vector, "tolist") else list(dense_vector)

        qdrant_sparse = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        flt = doc_filter(doc_id) if doc_id else None
        prefetch = [
            models.Prefetch(
                query=dense_list,
                using=self.config.dense_vector,
                limit=candidate_limit,
                filter=flt,
            ),
            models.Prefetch(
                query=qdrant_sparse,
                using=self.config.sparse_vector,
                limit=candidate_limit,
                filter=flt,
            ),
        ]

        response = self.client.query_points(
            collection_name=self.config.qdrant_collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            with_payload=True,
            limit=limit,
        )

        return [_hit_from_point(index, point) for index, point in enumerate(response.points, 1)]


def _hit_from_point(rank: int, point: Any) -> Hit:
    payload = dict(point.payload or {})
    payload_html = payload.get("payload_html")
    return Hit(
        rank=rank,
        doc_id=str(payload.get("doc_id", "")),
        page=int(payload.get("page", 0)),
        chunk_type=str(payload.get("chunk_type", "")),
        heading=str(payload.get("heading", "")),
        chunk=str(payload.get("chunk", "")),
        payload_html=str(payload_html) if payload_html is not None else None,
    )
