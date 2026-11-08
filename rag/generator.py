from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import requests

from .config import RagConfig, CONFIG
from .retriever import Hit

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise, grounded research assistant.
Answer the user's question using ONLY the provided context chunks.
For every claim, cite the source document and page number using the format [source: <doc_id>, p.<page>].
If the context does not contain enough information to answer the question, say so clearly.
Do NOT hallucinate or use outside knowledge."""

USER_PROMPT_TEMPLATE = """Context chunks (ordered by relevance):
{context}

---
Question: {question}

Answer (with citations):"""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def format_context(hits: Sequence[Hit]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, 1):
        heading = f" — {hit.heading}" if hit.heading else ""
        parts.append(
            f"[{i}] doc: {hit.doc_id}, p.{hit.page}{heading}\n{hit.chunk}"
        )
    return "\n\n".join(parts)


class Generator:
    def __init__(
        self,
        config: RagConfig = CONFIG,
        *,
        max_free_fallbacks: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.config = config
        self.max_free_fallbacks = max_free_fallbacks
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._free_models: list[str] = []
        self._models_fetched_at = 0.0
        self._lock = threading.Lock()

    def generate(self, question: str, context: Sequence[Hit]) -> GenerationResult:
        """Generate a grounded answer from retrieved hits."""
        if not context:
            return GenerationResult(
                answer=(
                    "I could not find any relevant information in the ingested "
                    "documents to answer your question."
                ),
                model="zero-hit-refusal",
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=format_context(context),
            question=question,
        )
        return self._call_with_fallback(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def call_llm(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a helpful assistant.",
    ) -> GenerationResult:
        return self._call_with_fallback(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

    def model_chain(self) -> list[str]:
        now = time.time()
        with self._lock:
            if now - self._models_fetched_at > self.config.openrouter_free_model_refresh_seconds:
                self._free_models = self._fetch_free_models()
                self._models_fetched_at = now

        chain = [self.config.openrouter_primary_model]
        for model_id in self._free_models:
            if model_id not in chain:
                chain.append(model_id)
        return chain

    def _call_with_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationResult:
        if not self.config.openrouter_api_key:
            return GenerationResult(
                answer="Error: OPENROUTER_API_KEY is not set. Please configure it in your .env file.",
                error="OPENROUTER_API_KEY is not set.",
            )

        last_error = "Unknown error"
        for model_id in self.model_chain():
            result = self._call_model_with_retry(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            if result.ok:
                return result
            last_error = result.error or last_error
            LOGGER.warning("Model %s failed: %s — trying next.", model_id, last_error)

        message = (
            f"Error: All available free models failed. Last error: {last_error}. "
            "Please try again shortly."
        )
        return GenerationResult(answer=message, error=message)

    def _call_model_with_retry(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationResult:
        headers = self._headers()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = "Unknown error"
        for attempt in range(1, self.config.openrouter_max_retries + 1):
            try:
                response = requests.post(
                    self.config.openrouter_chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.openrouter_timeout_seconds,
                )

                if response.status_code == 429:
                    wait = self.config.openrouter_backoff_base_seconds ** attempt
                    LOGGER.warning(
                        "Rate-limited by %s (attempt %d/%d). Waiting %.1fs.",
                        model_id,
                        attempt,
                        self.config.openrouter_max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = "HTTP 429 rate limit"
                    continue

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    LOGGER.info("Model %s responded successfully.", model_id)
                    return GenerationResult(answer=text, model=model_id)

                return GenerationResult(
                    answer=f"Error: HTTP {response.status_code}: {response.text[:200]}",
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.config.openrouter_timeout_seconds}s"
                LOGGER.warning("Timeout calling %s (attempt %d/%d).", model_id, attempt, self.config.openrouter_max_retries)
                if attempt < self.config.openrouter_max_retries:
                    time.sleep(self.config.openrouter_backoff_base_seconds ** attempt)
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
                break

        return GenerationResult(
            answer=f"Error: {last_error}",
            error=last_error,
        )

    def _fetch_free_models(self) -> list[str]:
        try:
            response = requests.get(
                self.config.openrouter_models_url,
                headers=self._headers(),
                timeout=15,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            free_ids: list[str] = []
            for model in data:
                pricing = model.get("pricing", {})
                try:
                    prompt_cost = float(pricing.get("prompt", 1))
                    completion_cost = float(pricing.get("completion", 1))
                except (TypeError, ValueError):
                    continue
                model_id = str(model.get("id", ""))
                if prompt_cost == 0.0 and completion_cost == 0.0 and model_id:
                    free_ids.append(model_id)
            LOGGER.info("Fetched %d free models from OpenRouter.", len(free_ids))
            return free_ids[: self.max_free_fallbacks]
        except Exception as exc:
            LOGGER.warning("Failed to fetch free model list: %s", exc)
            return []

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agentic-rag",
            "X-Title": "Agentic RAG",
        }
