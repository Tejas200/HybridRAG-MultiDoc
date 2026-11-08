from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import tiktoken
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    qdrant_url: str = "http://localhost:6333"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    collection: str = "chunks"
    batch_size: int = 32
    max_dense_tokens: int = 512


CONFIG = Config()

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
EMBED_TEXT_FORMAT = "heading + newline + chunk"
POINT_NAMESPACE = uuid.UUID("6f1c2b7e-2f1a-5c4d-9e8b-3a7d6c5b4a21")
DEFAULT_CHUNKS = Path(__file__).resolve().parent / "chunks.jsonl"
TOKENIZER = tiktoken.get_encoding("cl100k_base")

_DENSE_MODEL: TextEmbedding | None = None
_SPARSE_MODEL: SparseTextEmbedding | None = None


def get_models() -> tuple[TextEmbedding, SparseTextEmbedding]:
    """Initialize the embedding models once."""
    global _DENSE_MODEL, _SPARSE_MODEL
    if _DENSE_MODEL is None:
        LOGGER.info("Loading dense model %s", CONFIG.dense_model)
        _DENSE_MODEL = TextEmbedding(model_name=CONFIG.dense_model)
    if _SPARSE_MODEL is None:
        LOGGER.info("Loading sparse model %s", CONFIG.sparse_model)
        _SPARSE_MODEL = SparseTextEmbedding(model_name=CONFIG.sparse_model)
    return _DENSE_MODEL, _SPARSE_MODEL


def get_client() -> QdrantClient:
    return QdrantClient(url=CONFIG.qdrant_url, timeout=180)


def build_embed_text(record: dict[str, Any]) -> str:
    """Single searchable representation for both dense and BM25."""
    return f'{record["heading"]}\n{record["chunk"]}'


def truncate_for_embedding(text: str) -> tuple[str, int]:
    """Cap both dense and sparse input; keep the full text in the payload."""
    token_ids = TOKENIZER.encode(text)
    if len(token_ids) <= CONFIG.max_dense_tokens:
        return text, len(token_ids)
    return TOKENIZER.decode(token_ids[: CONFIG.max_dense_tokens]), len(token_ids)


def embed_batch(texts: Sequence[str]) -> tuple[list[Any], list[Any]]:
    """Embed passages only; never use the query methods while indexing."""
    dense_model, sparse_model = get_models()
    dense_vectors = list(dense_model.passage_embed(texts, batch_size=len(texts)))
    sparse_vectors = list(sparse_model.passage_embed(texts, batch_size=len(texts)))
    if len(dense_vectors) != len(texts) or len(sparse_vectors) != len(texts):
        raise RuntimeError(
            f"embedding count mismatch: {len(texts)} texts, "
            f"{len(dense_vectors)} dense, {len(sparse_vectors)} sparse"
        )
    return dense_vectors, sparse_vectors


def _collection_metadata(dense_dim: int) -> dict[str, Any]:
    return {
        "dense_model": CONFIG.dense_model,
        "sparse_model": CONFIG.sparse_model,
        "embed_text_format": EMBED_TEXT_FORMAT,
        "dense_dim": dense_dim,
    }


def _as_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _validate_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    info = client.get_collection(name)
    metadata = info.config.metadata or {}
    expected = _collection_metadata(dense_dim)

    mismatches = [
        f"{key}: stored={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Collection {name!r} configuration does not match index.py "
            f"({'; '.join(mismatches)}). Re-run with --recreate to rebuild it."
        )

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR not in vectors:
        raise RuntimeError(f"Collection {name!r} has no named {DENSE_VECTOR!r} vector")
    dense_params = vectors[DENSE_VECTOR]
    if dense_params.size != dense_dim:
        raise RuntimeError(
            f"Dense size mismatch: stored={dense_params.size}, expected={dense_dim}"
        )
    if _as_value(dense_params.distance) != "cosine":
        raise RuntimeError(
            f"Dense distance mismatch: stored={dense_params.distance!r}, expected cosine"
        )

    sparse_vectors = info.config.params.sparse_vectors or {}
    if SPARSE_VECTOR not in sparse_vectors:
        raise RuntimeError(f"Collection {name!r} has no named {SPARSE_VECTOR!r} sparse vector")
    modifier = sparse_vectors[SPARSE_VECTOR].modifier
    if _as_value(modifier) != "idf":
        raise RuntimeError(
            f"Sparse modifier mismatch: stored={modifier!r}, expected idf"
        )


def create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create the dual-vector collection, or validate and reuse it."""
    if client.collection_exists(name):
        _validate_collection(client, name, dense_dim)
        LOGGER.info("Reusing collection %s", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata=_collection_metadata(dense_dim),
    )
    LOGGER.info("Created collection %s (dense_dim=%d)", name, dense_dim)


def delete_document(client: QdrantClient, doc_id: str, collection: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
        wait=True,
    )


def _load_records(
    chunks_path: Path,
) -> tuple[OrderedDict[str, list[dict[str, Any]]], Counter, int]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    skipped: Counter = Counter()
    total_lines = 0
    required = ("doc_id", "page", "chunk_type", "heading", "chunk")

    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines += 1
            if not line.strip():
                skipped["blank_line"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped["invalid_json"] += 1
                LOGGER.warning("%s:%d invalid JSON: %s", chunks_path, line_number, exc)
                continue

            if not isinstance(record, dict):
                skipped["not_an_object"] += 1
                LOGGER.warning("%s:%d record is not a JSON object", chunks_path, line_number)
                continue

            missing = [field for field in required if field not in record]
            if missing:
                reason = f"missing:{','.join(missing)}"
                skipped[reason] += 1
                LOGGER.warning("%s:%d skipped (%s)", chunks_path, line_number, reason)
                continue

            chunk_type = record["chunk_type"]
            if chunk_type not in ("text", "table"):
                skipped["invalid_chunk_type"] += 1
                LOGGER.warning(
                    "%s:%d skipped (invalid_chunk_type=%r)",
                    chunks_path,
                    line_number,
                    chunk_type,
                )
                continue

            doc_id = record["doc_id"]
            if not isinstance(doc_id, str) or not doc_id.strip():
                skipped["invalid_doc_id"] += 1
                LOGGER.warning("%s:%d skipped (invalid doc_id)", chunks_path, line_number)
                continue

            page = record["page"]
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                skipped["invalid_page"] += 1
                LOGGER.warning("%s:%d skipped (invalid page=%r)", chunks_path, line_number, page)
                continue

            if not isinstance(record["heading"], str):
                skipped["invalid_heading"] += 1
                LOGGER.warning("%s:%d skipped (heading is not a string)", chunks_path, line_number)
                continue

            chunk = record["chunk"]
            if not isinstance(chunk, str) or not chunk.strip():
                skipped["empty_chunk"] += 1
                LOGGER.warning("%s:%d skipped (empty chunk)", chunks_path, line_number)
                continue

            if chunk_type == "table":
                payload_html = record.get("payload")
                if not isinstance(payload_html, str) or not payload_html.strip():
                    skipped["missing_table_payload"] += 1
                    LOGGER.warning(
                        "%s:%d skipped (table without payload HTML)",
                        chunks_path,
                        line_number,
                    )
                    continue

            clean = dict(record)
            clean["chunk_index"] = len(groups.get(doc_id, []))
            groups.setdefault(doc_id, []).append(clean)

    return groups, skipped, total_lines


def _batches(records: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _point(record: dict[str, Any], dense: Any, sparse: Any, token_count: int) -> models.PointStruct:
    payload = {
        "doc_id": record["doc_id"],
        "page": record["page"],
        "chunk_type": record["chunk_type"],
        "heading": record["heading"],
        "chunk": record["chunk"],
        "chunk_index": record["chunk_index"],
        "token_count": token_count,
    }
    if record["chunk_type"] == "table":
        payload["payload_html"] = record["payload"]

    point_id = str(uuid.uuid5(POINT_NAMESPACE, f'{record["doc_id"]}:{record["chunk_index"]}'))
    return models.PointStruct(
        id=point_id,
        vector={
            DENSE_VECTOR: dense.tolist(),
            SPARSE_VECTOR: models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ),
        },
        payload=payload,
    )


def index_document(
    chunks_path: str | Path,
    collection: str = CONFIG.collection,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Index one JSONL file. Documents are replaced, not appended."""
    path = Path(chunks_path)
    client = client or get_client()
    dense_model, _ = get_models()
    create_collection(client, collection, dense_model.embedding_size)

    groups, skipped, total_lines = _load_records(path)
    valid_records = sum(len(records) for records in groups.values())

    if skipped:
        LOGGER.warning(
            "Skipped %d of %d lines: %s",
            sum(skipped.values()),
            total_lines,
            ", ".join(f"{reason}={count}" for reason, count in sorted(skipped.items())),
        )
    else:
        LOGGER.info("Loaded %d valid records from %d lines", valid_records, total_lines)

    over_length: Counter = Counter()
    indexed = 0

    for doc_id, records in groups.items():
        delete_document(client, doc_id, collection)
        embed_seconds = 0.0
        upsert_seconds = 0.0

        for batch in _batches(records, CONFIG.batch_size):
            texts: list[str] = []
            token_counts: list[int] = []
            for record in batch:
                text, token_count = truncate_for_embedding(build_embed_text(record))
                texts.append(text)
                token_counts.append(token_count)

            started = time.perf_counter()
            dense_vectors, sparse_vectors = embed_batch(texts)
            embed_seconds += time.perf_counter() - started

            points = [
                _point(record, dense, sparse, token_count)
                for record, dense, sparse, token_count in zip(
                    batch, dense_vectors, sparse_vectors, token_counts
                )
            ]

            started = time.perf_counter()
            client.upsert(collection_name=collection, points=points, wait=True)
            upsert_seconds += time.perf_counter() - started

            for record, token_count in zip(batch, token_counts):
                if token_count > CONFIG.max_dense_tokens:
                    over_length[record["chunk_type"]] += 1

        indexed += len(records)
        LOGGER.info(
            "%s: %d chunks | embed %.2fs | upsert %.2fs",
            doc_id,
            len(records),
            embed_seconds,
            upsert_seconds,
        )

    total_over = sum(over_length.values())
    LOGGER.info(
        "Over-length source texts (>%d cl100k tokens, truncated for embedding): "
        "table=%d, text=%d, total=%d",
        CONFIG.max_dense_tokens,
        over_length["table"],
        over_length["text"],
        total_over,
    )

    return {
        "documents": len(groups),
        "indexed": indexed,
        "skipped": sum(skipped.values()),
        "skipped_reasons": dict(skipped),
        "over_length": dict(over_length),
    }


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Index chunk JSONL into Qdrant.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and recreate the collection before indexing",
    )
    args = parser.parse_args(argv)

    client = get_client()
    if args.recreate and client.collection_exists(CONFIG.collection):
        client.delete_collection(CONFIG.collection)
        LOGGER.info("Deleted collection %s", CONFIG.collection)

    result = index_document(args.chunks, CONFIG.collection, client=client)
    LOGGER.info(
        "Done: %d documents, %d chunks indexed, %d skipped",
        result["documents"],
        result["indexed"],
        result["skipped"],
    )


if __name__ == "__main__":
    main()
