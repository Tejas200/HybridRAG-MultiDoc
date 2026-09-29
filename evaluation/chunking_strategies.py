"""
chunking_strategies.py — Candidate chunking strategies built on LangChain
(evaluation/plan.md Section 5 candidate list).

Strategies (sliding-window variant intentionally NOT implemented, per user
instruction 2026-09-28):

  fixed-size    LangChain RecursiveCharacterTextSplitter, token-based via
                tiktoken (cl100k_base), 450 tokens / 80 overlap — matches the
                production app's MAX_CHUNK_TOKENS / CHUNK_OVERLAP_TOKENS.
  semantic      Embedding-similarity breakpoint chunker (the standard
                semantic-chunking algorithm: consecutive-sentence cosine
                distance, breakpoint at the p95 percentile).  LangChain
                ships no semantic splitter in current releases, so this is
                built on LangChain's Embeddings interface (a
                BAAI/bge-small-en-v1.5 wrapper) + LangChain's
                RecursiveCharacterTextSplitter for sub-splitting oversized
                segments.
  layout-aware  PDF font-size / bold heading detection (PyMuPDF span data,
                table regions excluded) -> Markdown -> LangChain
                MarkdownHeaderTextSplitter.  Oversized sections are
                sub-split with RecursiveCharacterTextSplitter.
  table-aware   PyMuPDF find_tables(): every detected table is kept INTACT as
                a single chunk (index tables separately); remaining prose is
                split with RecursiveCharacterTextSplitter.

All strategies take `pages` (the same [{"page_number", "text", "pdf_path"}]
shape the app's doc_agent._extract_pages produces) and return standard chunk
dicts: text, chunk_id, source_file, doc_id, page_number, strategy_used,
section_title (same schema as src/chunkers/base.make_chunk).

Usage:
  python evaluation/chunking_strategies.py <pdf_id> [--strategy all|fixed-size|semantic|layout-aware|table-aware]
  python evaluation/chunking_strategies.py --test     # one PDF per doc_type bucket
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF  # noqa: E402
import tiktoken  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.agents.doc_agent import _extract_pages  # noqa: E402
from src.chunkers.base import make_chunk  # noqa: E402
from src.config import (  # noqa: E402
    BGE_QUERY_PREFIX,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_TOKENS,
    SEMANTIC_BREAKPOINT_PERCENTILE,
)

CORPUS_RAW = ROOT / "corpus" / "raw"
TEST_OUT = ROOT / "evaluation" / "chunking_test"

CHUNK_TOKENS = MAX_CHUNK_TOKENS          # 450 (production token budget)
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS     # 80
ENC = tiktoken.get_encoding("cl100k_base")
STRATEGIES = ["fixed-size", "semantic", "layout-aware", "table-aware"]

MIN_CHUNK_WORDS = 5  # same short-chunk filter as src/chunkers/base.BaseChunker


def ntok(text: str) -> int:
    return len(ENC.encode(text))


# ── corpus helpers ─────────────────────────────────────────────────────────────

def find_pdf(pdf_id: str) -> Path:
    for bucket in sorted(CORPUS_RAW.iterdir()):
        if not bucket.is_dir() or bucket.name.startswith("_"):
            continue
        p = bucket / f"{pdf_id}.pdf"
        if p.exists():
            return p
    raise FileNotFoundError(f"no corpus PDF for pdf_id {pdf_id!r}")


# ── LangChain Embeddings wrapper over the production BGE model ─────────────────

class BGEEmbeddings(Embeddings):
    """LangChain Embeddings interface over BAAI/bge-small-en-v1.5 (L2-normalised)."""

    def __init__(self, model) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            list(texts), normalize_embeddings=True, batch_size=64
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            BGE_QUERY_PREFIX + text, normalize_embeddings=True
        ).tolist()


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
    )


def _keep(text: str) -> bool:
    return len(text.split()) >= MIN_CHUNK_WORDS


# ── strategy 1: fixed-size (RecursiveCharacterTextSplitter) ────────────────────

def chunk_fixed_size(pages: list[dict], source_file: str, doc_id: str) -> list[dict]:
    splitter = _recursive_splitter()
    out: list[dict] = []
    for page in pages:
        for t in splitter.split_text(page["text"]):
            t = t.strip()
            if _keep(t):
                out.append(make_chunk(t, source_file, doc_id,
                                      page["page_number"], "fixed-size"))
    return out


# ── strategy 2: semantic (embedding-breakpoint) ────────────────────────────────

def chunk_semantic(pages: list[dict], source_file: str, doc_id: str,
                   model) -> list[dict]:
    """
    Standard semantic-chunking algorithm:
      1. split each page into paragraphs (blank-line separated)
      2. embed all paragraphs (L2-normalised BGE)
      3. consecutive cosine distances; breakpoint where distance >= p95
      4. paragraphs between breakpoints form a chunk
      5. oversized chunks (>450 tokens) sub-split with RecursiveCharacterTextSplitter
    """
    import numpy as np

    paras: list[str] = []
    para_pages: list[int] = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        for p in re.split(r"\n\s*\n", text):
            p = " ".join(p.split())
            if len(p.split()) >= 8:
                paras.append(p)
                para_pages.append(page["page_number"])

    if len(paras) < 2:
        # not enough structure for breakpoints — degrade to fixed-size
        return chunk_fixed_size(pages, source_file, doc_id)

    emb = np.asarray(
        BGEEmbeddings(model).embed_documents(paras), dtype="float32"
    )
    dists = [1.0 - float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    thr = float(np.percentile(dists, SEMANTIC_BREAKPOINT_PERCENTILE))
    breakpoints = {i + 1 for i, d in enumerate(dists) if d >= thr}

    splitter = _recursive_splitter()
    out: list[dict] = []

    def flush(group: list[str], start_page: int) -> None:
        text = " ".join(group).strip()
        if not _keep(text):
            return
        if ntok(text) > CHUNK_TOKENS:
            for t in splitter.split_text(text):
                t = t.strip()
                if _keep(t):
                    out.append(make_chunk(t, source_file, doc_id,
                                          start_page, "semantic"))
        else:
            out.append(make_chunk(text, source_file, doc_id,
                                  start_page, "semantic"))

    group: list[str] = []
    start_page = para_pages[0]
    for i, p in enumerate(paras):
        if i in breakpoints and group:
            flush(group, start_page)
            group = []
        if not group:
            start_page = para_pages[i]
        group.append(p)
    flush(group, start_page)
    return out


# ── strategy 3: layout-aware (Markdown + MarkdownHeaderTextSplitter) ───────────

HEADING_SIZE_DELTA = 1.5   # pts above median font size => heading candidate
MIN_HEADING_WORDS = 1
MAX_HEADING_WORDS = 20
PAGE_MARKER = "[PAGE="     # marker embedded so sections keep page attribution


def _line_bbox_in_tables(line_bbox, table_rects) -> bool:
    for r in table_rects:
        if r.intersects(line_bbox):
            return True
    return False


def _build_markdown(pdf_path: str) -> list[str]:
    """Convert PDF to markdown lines: font-size/bold headings -> '# ', with
    [PAGE=N] markers at each page boundary.  Lines inside table regions are
    never treated as headings (bold table labels would create bogus sections)."""
    doc = fitz.open(pdf_path)
    md: list[str] = []
    sizes: list[float] = []
    structured: list[list[dict]] = []
    table_rects_by_page: list[list[fitz.Rect]] = []

    for page in doc:
        try:
            rects = [fitz.Rect(t.bbox) for t in page.find_tables().tables]
        except Exception:  # noqa: BLE001
            rects = []
        table_rects_by_page.append(rects)
        page_lines: list[dict] = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                txt = "".join(s.get("text", "") for s in spans).strip()
                if not txt:
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold_chars = sum(len(s.get("text", "")) for s in spans
                                 if s.get("flags", 0) & 2**4)
                bold_frac = bold_chars / len(txt) if txt else 0.0
                sizes.append(size)
                page_lines.append({
                    "text": txt, "size": size, "bold_frac": bold_frac,
                    "bbox": fitz.Rect(line["bbox"]),
                })
        structured.append(page_lines)
    doc.close()

    median = statistics.median(sizes) if sizes else 10.0
    size_thr = median + HEADING_SIZE_DELTA
    bold_size_thr = median - 0.5
    BOLD_FRAC_MIN = 0.8
    # checkbox / glyph lines on cover pages ("☒ ANNUAL REPORT ...") are noise
    GLYPH_RE = re.compile(r"[\u2610\u2611\u2612\u2713\u2714\u2717\u2718]")

    # Running headers/footers: identical lines on many pages are page
    # furniture, not section headings (they would reset the section per page).
    line_pages: dict[str, set[int]] = {}
    for page_idx, page_lines in enumerate(structured):
        for ln in page_lines:
            key = re.sub(r"\s+", " ", ln["text"]).strip().casefold()
            if key:
                line_pages.setdefault(key, set()).add(page_idx)
    n_pages = max(len(structured), 1)
    running_lines = {
        key for key, pgs in line_pages.items()
        if len(pgs) >= max(3, int(0.10 * n_pages))
    }

    for page_idx, page_lines in enumerate(structured):
        md.append(f"{PAGE_MARKER}{page_idx + 1}]")
        rects = table_rects_by_page[page_idx]
        for ln in page_lines:
            in_table = _line_bbox_in_tables(ln["bbox"], rects)
            words = len(ln["text"].split())
            # page 1 (cover) never yields headings; its text stays as body
            on_cover = page_idx == 0
            is_running = re.sub(r"\s+", " ", ln["text"]).strip().casefold() in running_lines
            is_heading = (
                not in_table
                and not on_cover
                and not is_running
                and not GLYPH_RE.search(ln["text"])
                and MIN_HEADING_WORDS <= words <= MAX_HEADING_WORDS
                and (
                    ln["size"] >= size_thr
                    or (ln["bold_frac"] >= BOLD_FRAC_MIN and ln["size"] >= bold_size_thr)
                )
            )
            if is_heading:
                title = re.sub(r"\s+", " ", ln["text"]).strip().lstrip("#").strip()
                md.append("# " + title)
            else:
                md.append(ln["text"])
    return md


def chunk_layout(pages: list[dict], source_file: str, doc_id: str) -> list[dict]:
    pdf_path = pages[0]["pdf_path"]
    md = "\n".join(_build_markdown(pdf_path))
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    sub = _recursive_splitter()
    out: list[dict] = []
    for doc in splitter.split_text(md):
        content = doc.page_content
        title = (
            doc.metadata.get("h1")
            or doc.metadata.get("h2")
            or doc.metadata.get("h3")
            or ""
        )
        page_marks = [int(m) for m in re.findall(r"\[PAGE=(\d+)\]", content)]
        body = "\n".join(
            l for l in content.split("\n") if not l.startswith(PAGE_MARKER)
        )
        body = re.sub(r"\n{2,}", "\n", body).strip()
        if not _keep(body):
            continue
        page_no = page_marks[0] if page_marks else 1
        if ntok(body) > CHUNK_TOKENS:
            for t in sub.split_text(body):
                t = t.strip()
                if _keep(t):
                    out.append(make_chunk(t, source_file, doc_id, page_no,
                                          "layout-aware", section_title=title))
        else:
            out.append(make_chunk(body, source_file, doc_id, page_no,
                                  "layout-aware", section_title=title))
    return out


# ── strategy 4: table-aware (tables intact + recursive prose) ──────────────────

def chunk_table(pages: list[dict], source_file: str, doc_id: str) -> list[dict]:
    pdf_path = pages[0]["pdf_path"]
    doc = fitz.open(pdf_path)
    sub = _recursive_splitter()
    out: list[dict] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1
        try:
            tables = page.find_tables().tables
        except Exception:  # noqa: BLE001
            tables = []
        rects = [fitz.Rect(t.bbox) for t in tables]

        # 1) each table stays intact as its own chunk
        for t in tables:
            rows = t.extract() or []
            lines = []
            for row in rows:
                cells = [(c or "").replace("\n", " ").strip() for c in row]
                if any(cells):
                    lines.append(" | ".join(cells))
            text = "\n".join(lines).strip()
            if _keep(text):
                out.append(make_chunk(text, source_file, doc_id, page_no,
                                      "table-aware", section_title="table"))

        # 2) prose = all lines outside table regions, recursively split
        prose_lines: list[str] = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox = fitz.Rect(line["bbox"])
                if any(r.intersects(bbox) for r in rects):
                    continue
                txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if txt:
                    prose_lines.append(txt)
        prose = "\n".join(prose_lines)
        for t in sub.split_text(prose):
            t = t.strip()
            if _keep(t):
                out.append(make_chunk(t, source_file, doc_id, page_no, "table-aware"))
    doc.close()
    return out


# ── dispatch + reporting ───────────────────────────────────────────────────────

def run_strategy(pdf_id: str, strategy: str, model=None) -> tuple[Path, list[dict]]:
    pdf_path = find_pdf(pdf_id)
    pages = _extract_pages(str(pdf_path))
    source_file = pdf_path.name
    if strategy == "fixed-size":
        chunks = chunk_fixed_size(pages, source_file, pdf_id)
    elif strategy == "semantic":
        if model is None:
            raise ValueError("semantic strategy requires a loaded SentenceTransformer")
        chunks = chunk_semantic(pages, source_file, pdf_id, model)
    elif strategy == "layout-aware":
        chunks = chunk_layout(pages, source_file, pdf_id)
    elif strategy == "table-aware":
        chunks = chunk_table(pages, source_file, pdf_id)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return pdf_path, chunks


def _stats(chunks: list[dict]) -> dict:
    toks = [ntok(c["text"]) for c in chunks]
    pages_hit = sorted({c["page_number"] for c in chunks})
    n_tables = sum(1 for c in chunks if c.get("section_title") == "table")
    return {
        "n_chunks": len(chunks),
        "tokens": {
            "min": min(toks) if toks else 0,
            "median": int(statistics.median(toks)) if toks else 0,
            "mean": round(sum(toks) / len(toks), 1) if toks else 0,
            "max": max(toks) if toks else 0,
        },
        "chunks_over_450_tokens": sum(1 for t in toks if t > CHUNK_TOKENS),
        "pages_covered": len(pages_hit),
        "table_chunks": n_tables,
    }


def report_for(pdf_id: str, strategy: str, chunks: list[dict], pdf_path: Path,
               elapsed: float) -> dict:
    stats = _stats(chunks)
    stats["pdf_pages"] = fitz.open(str(pdf_path)).page_count
    return {
        "pdf_id": pdf_id,
        "pdf_file": pdf_path.name,
        "strategy": strategy,
        "elapsed_s": round(elapsed, 1),
        **stats,
        "sample_chunks": [
            {"page_number": c["page_number"],
             "section_title": c.get("section_title", ""),
             "text_head": c["text"][:200]}
            for c in chunks[:3]
        ],
    }


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'pdf_id':<42} {'strategy':<13} {'chunks':>7} {'med_tok':>8} {'max_tok':>8} {'>450':>5} {'pages':>6} {'tbl':>4} {'s':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = r["tokens"]
        print(f"{r['pdf_id'][:42]:<42} {r['strategy']:<13} {r['n_chunks']:>7} "
              f"{t['median']:>8} {t['max']:>8} {r['chunks_over_450_tokens']:>5} "
              f"{r['pages_covered']:>6}/{r['pdf_pages']:<3} {r['table_chunks']:>4} "
              f"{r['elapsed_s']:>6}")


def cmd_test(_args) -> int:
    # one PDF per in-scope doc_type bucket
    test_pdfs = [
        ("financial_pdf", "AAPL_10-K_0000320193-25-000079"),
        ("contract", "AzulSa_20170303_F-1A_EX-10.3_9943903_EX-10.3_Maintenance Agreement1"),
        ("academic", "cs_CL_2609.26693v1"),
        ("manual", "advanced-excel-formulas-functions-glossary"),
        ("slide_deck", "5724245-Health-Powerpoint-Final"),
    ]
    from sentence_transformers import SentenceTransformer

    print(f"loading {EMBEDDING_MODEL_NAME} …", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    TEST_OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for bucket, pdf_id in test_pdfs:
        for strategy in STRATEGIES:
            t0 = time.time()
            pdf_path, chunks = run_strategy(pdf_id, strategy, model)
            row = report_for(pdf_id, strategy, chunks, pdf_path, time.time() - t0)
            row["bucket"] = bucket
            all_rows.append(row)
            out = TEST_OUT / f"{pdf_id}__{strategy}.json"
            out.write_text(json.dumps(row, indent=2, ensure_ascii=True),
                           encoding="utf-8")
    _print_table(all_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_budget_tokens": CHUNK_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "semantic_breakpoint_percentile": SEMANTIC_BREAKPOINT_PERCENTILE,
        "strategies": STRATEGIES,
        "results": all_rows,
    }
    (TEST_OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nreports written to {TEST_OUT.relative_to(ROOT)}/")
    return 0


def cmd_single(args) -> int:
    model = None
    if "semantic" in (args.strategy or STRATEGIES):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    rows = []
    for strategy in (args.strategy or STRATEGIES):
        t0 = time.time()
        pdf_path, chunks = run_strategy(args.pdf, strategy, model)
        row = report_for(args.pdf, strategy, chunks, pdf_path, time.time() - t0)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=True))
    _print_table(rows)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", nargs="?", help="pdf_id (stem of the corpus PDF)")
    p.add_argument("--strategy", nargs="*", choices=STRATEGIES, default=None)
    p.add_argument("--test", action="store_true",
                   help="run all strategies on one PDF per doc_type bucket")
    args = p.parse_args()
    if args.test:
        return cmd_test(args)
    if not args.pdf:
        p.error("pdf_id required (or use --test)")
    return cmd_single(args)


if __name__ == "__main__":
    sys.exit(main())
