from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

from rag.pipeline import AnswerResult, Pipeline


def _configure_windows_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _source_rows(result: AnswerResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in result.sources:
        row = {
            "rank": hit.rank,
            "doc_id": hit.doc_id,
            "page": hit.page,
            "chunk_type": hit.chunk_type,
            "heading": hit.heading,
            "chunk": hit.chunk,
        }
        if hit.payload_html is not None:
            row["payload_html"] = hit.payload_html
        rows.append(row)
    return rows


def _to_json(result: AnswerResult, show_context: bool) -> str:
    payload = {
        "question": result.question,
        "answer": result.answer,
        "sources": _source_rows(result),
        "show_context": show_context,
        "generation_model": result.generation_model,
        "zero_hit": result.zero_hit,
        "error": result.error,
        "timings": {
            key: round(value, 3) for key, value in result.timings.items()
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _print_text(result: AnswerResult, show_context: bool) -> None:
    print()
    print("Answer:")
    print((result.answer or "I could not find any relevant information in the ingested documents to answer your question.").strip())
    print()
    print("Sources:")
    print("  Rank | Page | Type  | Heading")
    print("  -----+------+-------+----------------------------------------")
    for hit in result.sources:
        heading = hit.heading.replace("\n", " ")
        if len(heading) > 55:
            heading = heading[:52] + "..."
        print(
            f"  {hit.rank:>4} | {hit.page:>4} | {hit.chunk_type:<5} | {heading}"
        )
        print(f"        doc: {hit.doc_id}")

    if show_context:
        print()
        print("Retrieved context:")
        for hit in result.sources:
            print(f"  [{hit.rank}] {hit.doc_id}, p.{hit.page} — {hit.heading}")
            print("      " + hit.chunk.replace("\n", "\n      "))
            if hit.payload_html:
                print(f"      payload_html ({len(hit.payload_html)} chars):")
                if len(hit.payload_html) <= 4000:
                    print("        " + hit.payload_html)
                else:
                    print("        " + hit.payload_html[:400].replace("\n", " ") + "...")
            print()

    retrieval = result.timings.get("retrieval", 0.0)
    generation = result.timings.get("generation", 0.0)
    total = result.timings.get("total", retrieval + generation)
    print()
    print(
        f"Timings: retrieval {retrieval:.2f}s | generation {generation:.2f}s | "
        f"total {total:.2f}s"
    )
    print(f"Model: {result.generation_model or 'none'}")
    print("Note: free-tier generation latency can vary widely.")


def run_one(
    pipeline: Pipeline,
    question: str,
    *,
    k: int,
    json_output: bool,
    show_context: bool,
) -> int:
    result = pipeline.ask(question, k=k)
    if json_output:
        print(_to_json(result, show_context))
    else:
        _print_text(result, show_context)
    return 0 if result.ok else 1


def interactive(pipeline: Pipeline, *, k: int, json_output: bool, show_context: bool) -> int:
    print("Interactive RAG query. Type blank line, Ctrl+C, or Ctrl+Z to exit.")
    if not json_output:
        print(f"Context size: top {k} chunks.")
    last_status = 0
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return last_status

        if not question:
            print("Exiting.")
            return last_status

        last_status = run_one(
            pipeline,
            question,
            k=k,
            json_output=json_output,
            show_context=show_context,
        )
        if json_output:
            print()


def main(argv: list[str] | None = None) -> int:
    _configure_windows_console()
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Query the Qdrant-backed RAG index.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--k", type=int, default=None, help="Number of context chunks to use.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks/payload info.")
    args = parser.parse_args(argv)

    pipeline = Pipeline()
    k = args.k if args.k is not None else pipeline.config.context_k
    if k < 1:
        parser.error("--k must be at least 1")

    question = " ".join(args.question).strip()
    if question:
        return run_one(
            pipeline,
            question,
            k=k,
            json_output=args.json,
            show_context=args.show_context,
        )

    return interactive(
        pipeline,
        k=k,
        json_output=args.json,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    raise SystemExit(main())
