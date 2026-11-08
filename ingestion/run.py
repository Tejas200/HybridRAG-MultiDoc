"""
run.py - production ingestion orchestrator: extract -> chunk -> index.

Chains the three ingestion stages for the configured corpus directories and
rebuilds the Qdrant collection from scratch, so points from removed documents
never survive in the production index.

Stale artifacts from previous runs (``*.md``, per-doc ``*.jsonl``,
``review_report.csv``) are archived to ``ingestion/output/_archive/<ts>/``
before a fresh extraction, keeping ``chunk.py`` from re-chunking old corpora.

Usage:
    python ingestion/run.py
    python ingestion/run.py --dirs academic contract
    python ingestion/run.py --sample 2 --seed 0        # quick smoke run
    python ingestion/run.py --skip-extract             # re-chunk/re-index only
    python ingestion/run.py --no-recreate              # reuse the collection
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import chunk as chunk_mod  # noqa: E402
import extract as extract_mod  # noqa: E402
import index as index_mod  # noqa: E402

DEFAULT_DIRS = ("academic", "contract")
OUTPUT_DIR = extract_mod.DEFAULT_OUTPUT
CHUNKS_PATH = chunk_mod.DEFAULT_OUTPUT


def resolve_dir(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / "corpus" / "raw" / name


def archive_stale(output_dir: Path) -> None:
    """Move previous-run artifacts out so chunk.py sees only fresh markdown."""
    stale = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix in {".md", ".jsonl", ".csv"}
    ]
    if not stale:
        return
    archive = output_dir / "_archive" / time.strftime("%Y%m%d-%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for path in stale:
        shutil.move(str(path), archive / path.name)
    print(f"archived {len(stale)} stale artifact(s) -> {archive}")


def run_stage(label: str, fn, argv: list[str]) -> None:
    started = time.perf_counter()
    print(f"\n=== {label} ===")
    rc = fn(argv)
    elapsed = time.perf_counter() - started
    if rc:
        raise SystemExit(f"{label} failed with exit code {rc} after {elapsed:.1f}s")
    print(f"=== {label} done in {elapsed:.1f}s ===")


def main(argv: list[str] | None = None) -> int:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Full ingestion: extract -> chunk -> index.")
    parser.add_argument("--dirs", nargs="*", default=list(DEFAULT_DIRS),
                        help="corpus subfolder names under corpus/raw (or full paths)")
    parser.add_argument("--sample", type=int, default=None,
                        help="pass --sample N to extract.py (smoke runs only)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample")
    parser.add_argument("--no-fresh", action="store_true",
                        help="keep existing output/*.md instead of archiving them")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--no-recreate", action="store_true",
                        help="reuse the existing collection (per-doc replace only)")
    args = parser.parse_args(argv)

    if not args.skip_extract and not args.no_fresh:
        archive_stale(OUTPUT_DIR)

    for name in args.dirs:
        input_dir = resolve_dir(name)
        if not input_dir.is_dir():
            raise SystemExit(f"corpus directory not found: {input_dir}")
        if args.skip_extract:
            break
        extract_argv = ["--input", str(input_dir), "--output", str(OUTPUT_DIR),
                        "--seed", str(args.seed)]
        if args.sample is not None:
            extract_argv += ["--sample", str(args.sample)]
        run_stage(f"extract [{input_dir.name}]", extract_mod.main, extract_argv)
        report = OUTPUT_DIR / "review_report.csv"
        if report.is_file():
            shutil.move(str(report), OUTPUT_DIR / f"review_report_{input_dir.name}.csv")

    if not args.skip_chunk:
        run_stage("chunk", chunk_mod.main,
                  ["--input", str(OUTPUT_DIR), "--output", str(CHUNKS_PATH)])

    if not args.skip_index:
        index_argv = ["--chunks", str(CHUNKS_PATH)]
        if not args.no_recreate:
            index_argv.append("--recreate")
        run_stage("index", index_mod.main, index_argv)

    print(f"\nfull ingestion finished in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
