from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import RagConfig, CONFIG
from .generator import GenerationResult, Generator
from .retriever import Hit, Retriever

LOGGER = logging.getLogger(__name__)

ZERO_HIT_ANSWER = (
    "I could not find any relevant information in the ingested documents "
    "to answer your question."
)


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[Hit]
    generation_model: str | None = None
    error: str | None = None
    zero_hit: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_seconds(self) -> float:
        return self.timings.get("total", 0.0)


class Pipeline:
    def __init__(
        self,
        config: RagConfig = CONFIG,
        *,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        validate: bool = True,
    ) -> None:
        self.config = config
        self.retriever = retriever or Retriever(config, validate=validate)
        self.generator = generator or Generator(config)

    def ask(
        self,
        question: str,
        *,
        k: int | None = None,
        doc_id: str | None = None,
    ) -> AnswerResult:
        """Run retrieval + synthesis for one user question."""
        total_start = time.perf_counter()
        retrieval_limit = int(k or self.config.context_k)

        started = time.perf_counter()
        hits = self.retriever.search(
            question,
            limit=retrieval_limit,
            doc_id=doc_id,
        )
        retrieval_seconds = time.perf_counter() - started

        if not hits:
            total_seconds = time.perf_counter() - total_start
            return AnswerResult(
                question=question,
                answer=ZERO_HIT_ANSWER,
                sources=[],
                generation_model="zero-hit-refusal",
                zero_hit=True,
                timings={
                    "retrieval": retrieval_seconds,
                    "generation": 0.0,
                    "total": total_seconds,
                },
            )

        started = time.perf_counter()
        generation = self.generator.generate(question, hits)
        generation_seconds = time.perf_counter() - started
        total_seconds = time.perf_counter() - total_start

        return _result_from_generation(
            question=question,
            answer=generation.answer,
            sources=hits,
            generation_model=generation.model,
            error=generation.error,
            timings={
                "retrieval": retrieval_seconds,
                "generation": generation_seconds,
                "total": total_seconds,
            },
        )


def _result_from_generation(
    *,
    question: str,
    answer: str,
    sources: list[Hit],
    generation_model: str | None,
    error: str | None,
    timings: dict[str, float],
) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        generation_model=generation_model,
        error=error,
        zero_hit=not sources,
        timings=timings,
    )
