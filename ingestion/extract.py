"""
extract.py - Pymupdf4llm per-page extraction for the RAG ingestion step.

Converts every PDF in an input folder into per-page structured records. Tables
are emitted as HTML (``table_output="html"``) so merged cells survive; the
Markdown produced downstream is chunked and embedded for retrieval.

Installed-version contract (verified against the environment, not assumed):
    pymupdf4llm.to_markdown(
        doc, *, pages=None, page_chunks=False, table_output="markdown",
        use_ocr=True, force_ocr=False, ocr_language="eng", ...
    )
The Layout module (``pymupdf.layout``) is active by default, so ``to_markdown``
routes through ``_layout_to_markdown`` and OCR is available offline. With the
Layout path, ``page_chunks=True`` returns one dict per page with keys:
    {"metadata", "toc_items", "page_boxes", "text"}
There is no ``tables`` key on this path, so table counts are derived from the
emitted ``<table>`` markup and cross-checked with ``pymupdf`` ``find_tables()``.

Usage:
    python ingestion/extract.py --input corpus/raw
    python ingestion/extract.py --pdf path/to/one.pdf
    python ingestion/extract.py --input pdfs --sample 3 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import html
import statistics
from html.parser import HTMLParser
from pathlib import Path
from typing import TypedDict

import pymupdf
import pymupdf4llm
import time

# ── paths ─────────────────────────────────────────────────────────────────────
# extract.py lives at <root>/ingestion/extract.py
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "corpus" / "raw"
DEFAULT_OUTPUT = ROOT / "ingestion" / "output"

# ── tuning thresholds ─────────────────────────────────────────────────────────
SCAN_TEXT_THRESHOLD = 50        # chars of raw page text below which a page looks scanned
SCAN_IMAGE_COVERAGE = 0.5       # fraction of page area covered by images
SHORT_LINE_MAX_WORDS = 6        # a "short" line has at most this many tokens
NUMERIC_RATIO_MIN = 0.6         # fraction of tokens that must be numeric
NUMERIC_RUN_MIN = 4             # consecutive numeric short lines => suspicious table
BORDERLESS_MIN_TOKENS_PER_LINE = 2.0  # real table rows hold multiple numbers
BORDERLESS_MAX_GAP_FACTOR = 3.0       # rows are vertically adjacent (x line height)
TABLE_CONTENT_PRESERVED_MIN = 0.8     # detected-table cells already in the markdown

_NUM_TOKEN_RE = re.compile(r"[\(\$\-\+]?\d[\d,\.%\)]*")
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_NOISE_RE = re.compile(r"<[^>]+>|[*_`#>|\\]")
_WHITESPACE_RE = re.compile(r"\s+")

# The Layout module wraps figure/chart text in these markers. Numbers found
# there are axis labels, not a borderless table.
_PICTURE_START = "<!-- Start of picture text -->"
_PICTURE_END = "<!-- End of picture text -->"
_PICTURE_BLOCK_RE = re.compile(
    re.escape(_PICTURE_START) + r"(.*?)" + re.escape(_PICTURE_END),
    re.DOTALL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("extract")


class PageRecord(TypedDict):
    """One extracted PDF page."""

    doc_id: str
    page_number: int
    markdown_content: str
    table_count: int
    is_scanned: bool
    needs_review: bool
    review_reasons: list[str]


# ── HTML table parsing (column-count consistency) ─────────────────────────────

class _TableRowsParser(HTMLParser):
    """Collect, per table, rows of ``(colspan, rowspan)`` cell spans."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[tuple[int, int]]]] = []
        self._rows: list[list[tuple[int, int]]] | None = None
        self._row: list[tuple[int, int]] | None = None

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        for key, value in attrs:
            if key == name and value:
                try:
                    return max(int(value), 1)
                except ValueError:
                    return 1
        return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._row.append(
                (self._span(attrs, "colspan"), self._span(attrs, "rowspan"))
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._row is not None and self._rows is not None:
            self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            self.tables.append(self._rows)
            self._rows = None


def _occupied_widths(rows: list[list[tuple[int, int]]]) -> list[int]:
    """Columns occupied by each row, honoring both colspan and rowspan.

    A ``rowspan`` cell occupies its columns across every row it spans, even
    though those later rows contain no ``<td>`` for it. Placement must be
    positional (skip columns already filled by an earlier rowspan), so a grid
    occupancy map is used rather than a per-row column sum.
    """
    occupied: set[tuple[int, int]] = set()
    for r, cells in enumerate(rows):
        col = 0
        for colspan, rowspan in cells:
            while (r, col) in occupied:
                col += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied.add((r + dr, col + dc))
            col += colspan

    max_col = max((c for _, c in occupied), default=-1) + 1
    return [
        sum(1 for c in range(max_col) if (r, c) in occupied)
        for r in range(len(rows))
    ]


def _inconsistent_column_tables(markdown: str) -> int:
    """Return how many HTML tables have rows with differing occupied widths.

    Rows are compared after resolving merged cells, so a legitimate rowspan /
    colspan table is not flagged, while a genuinely ragged table still is.
    """
    parser = _TableRowsParser()
    try:
        parser.feed(markdown)
    except Exception:  # malformed markup should not kill the page
        return 0
    bad = 0
    for rows in parser.tables:
        widths = {width for width in _occupied_widths(rows) if width > 0}
        if len(widths) > 1:
            bad += 1
    return bad


# ── page-level heuristics ─────────────────────────────────────────────────────

def _image_coverage(page: pymupdf.Page) -> float:
    """Fraction of the page area covered by raster images (0.0-1.0)."""
    page_area = abs(page.rect)
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        covered += abs(pymupdf.Rect(info["bbox"]) & page.rect)
    return min(covered / page_area, 1.0)


def _is_scanned(page: pymupdf.Page) -> bool:
    """Heuristic: almost no extractable text but a mostly-image page."""
    raw_text = page.get_text("text").strip()
    if len(raw_text) < SCAN_TEXT_THRESHOLD and _image_coverage(page) >= SCAN_IMAGE_COVERAGE:
        return True
    return len(raw_text) == 0 and bool(page.get_images())


def _is_short_numeric_line(line: str) -> bool:
    tokens = line.split()
    if not tokens or len(tokens) > SHORT_LINE_MAX_WORDS:
        return False
    numeric = sum(1 for tok in tokens if _NUM_TOKEN_RE.fullmatch(tok))
    return numeric / len(tokens) >= NUMERIC_RATIO_MIN


def _text_lines_with_rects(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """All text lines with bounding boxes, in reading order."""
    lines: list[tuple[str, pymupdf.Rect]] = []
    blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text:
                lines.append((text, pymupdf.Rect(line["bbox"])))
    return lines


def _longest_short_numeric_run(
    page: pymupdf.Page,
) -> list[tuple[str, pymupdf.Rect]]:
    """Longest run of consecutive short, number-heavy lines (with geometry)."""
    best: list[tuple[str, pymupdf.Rect]] = []
    current: list[tuple[str, pymupdf.Rect]] = []
    for text, rect in _text_lines_with_rects(page):
        if _is_short_numeric_line(text):
            current.append((text, rect))
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def _looks_like_borderless_table(run: list[tuple[str, pymupdf.Rect]]) -> bool:
    """Geometry guard for borderless-table candidates.

    Real table rows hold several numbers per line and sit vertically adjacent.
    Section numbers in a margin are single-number lines spread down the page,
    so requiring both a multi-number row and tight row spacing rejects them.
    """
    if len(run) < NUMERIC_RUN_MIN:
        return False

    tokens_per_line = [len(_numeric_tokens(text)) for text, _ in run]
    if sum(tokens_per_line) / len(tokens_per_line) < BORDERLESS_MIN_TOKENS_PER_LINE:
        return False

    rects = [rect for _, rect in run]
    heights = [rect.height for rect in rects if rect.height > 0]
    if not heights:
        return False
    median_height = statistics.median(heights)
    centers = [(rect.y0 + rect.y1) / 2 for rect in rects]
    gaps = [abs(centers[i + 1] - centers[i]) for i in range(len(centers) - 1)]
    median_gap = statistics.median(gaps) if gaps else 0.0
    return median_gap <= BORDERLESS_MAX_GAP_FACTOR * median_height


def _numeric_tokens(text: str) -> list[str]:
    """All number-like tokens in ``text`` (e.g. ``0.42``, ``1,234``, ``(5)``)."""
    return _NUM_TOKEN_RE.findall(text)


def _normalize_for_compare(text: str) -> str:
    """Lowercase, strip markup and whitespace for loose text containment."""
    text = html.unescape(text)
    text = _MARKDOWN_NOISE_RE.sub("", text)
    return _WHITESPACE_RE.sub("", text).lower()


def _meaningful_table_cells(table: object) -> list[str] | None:
    """Normalized table cells that carry alphanumeric content.

    ``None`` means the cells could not be read (treated as not preserved).
    Symbol-only cells (checkbox glyphs such as ``☐``/``☒``) are ignored.
    """
    try:
        rows = table.extract()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - unknown/odd table object
        return None
    cells: list[str] = []
    for row in rows or []:
        for cell in row:
            if not cell:
                continue
            normalized = _normalize_for_compare(cell)
            if any(char.isalnum() for char in normalized):
                cells.append(normalized)
    return cells


def _table_preserved_in_markdown(table: object, markdown: str) -> bool:
    """True when a detected table's text already appears in the markdown.

    The renderer may deliberately emit a detected grid as plain text (for
    example the SEC cover-page filer checkboxes). When its cells are already
    present, no content was lost and the page should not be flagged.
    """
    cells = _meaningful_table_cells(table)
    if cells is None:
        return False  # unreadable cells: stay conservative and flag
    if not cells:
        return True  # nothing meaningful that could be lost
    flat_markdown = _normalize_for_compare(markdown)
    present = sum(1 for cell in cells if cell in flat_markdown)
    return present / len(cells) >= TABLE_CONTENT_PRESERVED_MIN


def _picture_text(markdown: str) -> str:
    """Text the Layout module wrapped in picture-text markers, or ``""``."""
    return "\n".join(_PICTURE_BLOCK_RE.findall(markdown))


def _run_is_picture_text(run_lines: list[str], markdown: str) -> bool:
    """True when every number in the run also appears inside picture text."""
    picture = _picture_text(markdown)
    if not picture.strip():
        return False
    run_tokens = _numeric_tokens(" ".join(run_lines))
    if not run_tokens:
        return False
    picture_tokens = set(_numeric_tokens(picture))
    return all(token in picture_tokens for token in run_tokens)


# ── extraction ────────────────────────────────────────────────────────────────

def _extract_single_page(pdf_path: Path, page_index: int, force_ocr: bool) -> str:
    """Extract one 0-based page via pymupdf4llm, returning its Markdown."""
    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=[page_index],
        page_chunks=True,
        table_output="html",
        use_ocr=True,
        force_ocr=force_ocr,
    )
    if not chunks:
        return ""
    return chunks[0]["text"]


def _build_record(
    doc_id: str,
    pdf_path: Path,
    page: pymupdf.Page,
    page_number: int,
    markdown: str,
    prev_markdown: str,
) -> PageRecord:
    """Assemble one PageRecord and run the table/scanned quality checks."""
    reasons: list[str] = []

    is_scanned = _is_scanned(page)
    if is_scanned:
        reasons.append("scanned_or_low_text")
        if len(markdown.strip()) < SCAN_TEXT_THRESHOLD:
            try:
                ocr_md = _extract_single_page(pdf_path, page_number - 1, force_ocr=True)
                if len(ocr_md.strip()) > len(markdown.strip()):
                    markdown = ocr_md
                    reasons.append("ocr_applied")
            except Exception as exc:  # noqa: BLE001 - OCR failure must not drop the page
                reasons.append(f"ocr_failed: {exc}")

    table_count = markdown.lower().count("<table")

    # check 1: find_tables() sees a table but nothing was emitted as markup.
    # Only flag when a detected table's text is actually missing from the
    # markdown; grids the renderer emitted as plain text lost no content.
    try:
        detected_tables = page.find_tables().tables
    except Exception:  # noqa: BLE001 - table detection can fail on odd pages
        detected_tables = []
    if detected_tables and table_count == 0:
        lost = any(
            not _table_preserved_in_markdown(table, markdown)
            for table in detected_tables
        )
        if lost:
            reasons.append("table_detected_but_no_markup")

    # check 2: many short number-heavy lines in a row (missed borderless table).
    # Ignore runs inside picture-text blocks (chart axes) and single-number
    # margins that are not vertically adjacent (e.g. contract section numbers).
    numeric_run = _longest_short_numeric_run(page)
    run_texts = [text for text, _ in numeric_run]
    if (
        _looks_like_borderless_table(numeric_run)
        and table_count == 0
        and not _run_is_picture_text(run_texts, markdown)
    ):
        reasons.append(f"possible_borderless_table ({len(numeric_run)} numeric lines)")

    # check 3: emitted HTML tables with inconsistent row column counts
    bad_tables = _inconsistent_column_tables(markdown)
    if bad_tables:
        reasons.append(f"inconsistent_table_columns ({bad_tables} table(s))")

    # check 4: this page opens with a table and the previous page closed one
    if markdown.lstrip().lower().startswith("<table") and prev_markdown.rstrip().lower().endswith("</table>"):
        reasons.append("possible_split_table_from_previous_page")

    return {
        "doc_id": doc_id,
        "page_number": page_number,
        "markdown_content": markdown,
        "table_count": table_count,
        "is_scanned": is_scanned,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }


def extract_document(
    pdf_path: Path,
    sample: int | None,
    rng: random.Random,
) -> list[PageRecord]:
    """Extract all (or a random sample of) pages from one PDF."""
    doc_id = pdf_path.stem
    doc = pymupdf.open(pdf_path)
    try:
        indices = list(range(doc.page_count))
        if sample is not None:
            indices = sorted(rng.sample(indices, min(sample, len(indices))))
        if not indices:
            return []

        # First pass: parse the whole (selected) document in one go.
        md_by_page: dict[int, str] = {}
        try:
            chunks = pymupdf4llm.to_markdown(
                str(pdf_path),
                pages=indices,
                page_chunks=True,
                table_output="html",
                use_ocr=True,
            )
            for chunk in chunks:
                md_by_page[chunk["metadata"]["page_number"]] = chunk["text"]
        except Exception as exc:  # noqa: BLE001 - fall back to per-page isolation
            log.warning("%s: batch parse failed (%s); falling back to per-page", doc_id, exc)

        records: list[PageRecord] = []
        prev_markdown = ""
        for index in indices:
            page_number = index + 1
            try:
                markdown = md_by_page.get(page_number)
                if markdown is None:
                    markdown = _extract_single_page(pdf_path, index, force_ocr=False)
                record = _build_record(
                    doc_id, pdf_path, doc[index], page_number, markdown, prev_markdown
                )
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the doc
                record = {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "markdown_content": "",
                    "table_count": 0,
                    "is_scanned": False,
                    "needs_review": True,
                    "review_reasons": [f"extraction_error: {exc}"],
                }
            records.append(record)
            prev_markdown = record["markdown_content"]
        return records
    finally:
        doc.close()


# ── output ────────────────────────────────────────────────────────────────────

def write_document_outputs(doc_id: str, records: list[PageRecord], output_dir: Path) -> None:
    """Write ``{doc_id}.jsonl`` and ``{doc_id}.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{doc_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    md_path = output_dir / f"{doc_id}.md"
    sections = [
        f"<!-- page {record['page_number']} -->\n\n{record['markdown_content']}"
        for record in records
    ]
    md_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def write_review_report(rows: list[tuple[str, int, list[str]]], output_dir: Path) -> None:
    """Write ``review_report.csv`` (doc_id, page_number, reasons)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "review_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_id", "page_number", "reasons"])
        for doc_id, page_number, reasons in rows:
            writer.writerow([doc_id, page_number, "; ".join(reasons)])


def find_pdfs(input_dir: Path) -> list[Path]:
    """All PDFs under ``input_dir`` (recursive), sorted, hidden files skipped."""
    return sorted(
        path
        for path in input_dir.rglob("*.pdf")
        if not path.name.startswith(".")
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Per-page PDF -> Markdown extraction.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=Path, default=None,
                        help="single PDF file to process (overrides --input)")
    source.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="folder to scan for PDFs (recursive)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="folder for jsonl/md/csv outputs")
    parser.add_argument("--sample", type=int, default=None,
                        help="process only N random pages per PDF (quick testing)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample")
    args = parser.parse_args(argv)

    print(f"pymupdf      : {pymupdf.__version__}")
    print(f"pymupdf4llm  : {pymupdf4llm.__version__}")
    print(f"layout module: {getattr(pymupdf4llm, '_use_layout', None)}")

    if args.pdf is not None:
        if not args.pdf.is_file():
            log.error("pdf not found: %s", args.pdf)
            return 1
        pdfs = [args.pdf]
    else:
        if not args.input.exists():
            log.error("input folder not found: %s", args.input)
            return 1
        pdfs = find_pdfs(args.input)
        if not pdfs:
            log.error("no PDFs found under %s", args.input)
            return 1

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    review_rows: list[tuple[str, int, list[str]]] = []

    for pdf_path in pdfs:
        try:
            records = extract_document(pdf_path, args.sample, rng)
        except Exception as exc:  # noqa: BLE001 - skip a broken PDF, keep going
            log.error("%s: failed entirely: %s", pdf_path.name, exc)
            continue

        write_document_outputs(pdf_path.stem, records, args.output)
        flagged = 0
        for record in records:
            if record["needs_review"]:
                flagged += 1
                review_rows.append(
                    (record["doc_id"], record["page_number"], record["review_reasons"])
                )
                log.warning(
                    "%s p.%d -> %s",
                    record["doc_id"], record["page_number"],
                    "; ".join(record["review_reasons"]),
                )
        log.info("%s: %d pages, %d flagged", pdf_path.name, len(records), flagged)

    write_review_report(review_rows, args.output)
    log.info("done: %d PDF(s); outputs in %s", len(pdfs), args.output)
    end = time.perf_counter()
    print(f"Execution time: {end - start} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
