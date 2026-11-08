"""
chunk.py - turn pymupdf4llm markdown into table/text chunks for RAG.

Input
-----
Markdown files produced by ``extract.py`` (``ingestion/output/*.md``),
each page prefixed by a ``<!-- page N -->`` marker.

Output
------
One combined JSONL file of chunk records:

    table: {"doc_id", "page", "chunk_type", "heading", "chunk", "payload"}
    text : {"doc_id", "page", "chunk_type", "heading", "chunk"}

``chunk`` never contains its own heading; ``heading`` is stored separately.
This module only chunks - it does not embed or index anything.

Usage
-----
    python ingestion/chunk.py
    python ingestion/chunk.py --input DIR --output FILE [--doc STEM]
    python ingestion/chunk.py --test [--test-file PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── paths ─────────────────────────────────────────────────────────────────────
# chunk.py lives at <root>/ingestion/chunk.py
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ingestion" / "output"
DEFAULT_OUTPUT = ROOT / "ingestion" / "chunks.jsonl"

# ── tuning ────────────────────────────────────────────────────────────────────
TARGET_TOKENS = 450
MAX_HEADING_CHARS = 100

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MAX_TOKENS = 512
TABLE_MAX_ROWS = 20
TABLE_TARGET_TOKENS = 400

PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
# "Table 3: ...", "Exhibit 10.1 ...", "Note 5 ...", "Figure 2 - ..."
CAPTION_RE = re.compile(
    r"^(?:table|exhibit|figure|schedule|note)\b\s*(?:[\dIVXivx]|[:.\-\u2014])",
    re.IGNORECASE,
)
TABLE_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SPLITTER: RecursiveCharacterTextSplitter | None = None
_BGE_TOKENIZER = None


def bge_token_count(text: str) -> int:
    """Count tokens with the same tokenizer used by the embedding model."""
    global _BGE_TOKENIZER
    if _BGE_TOKENIZER is None:
        from transformers import AutoTokenizer

        _BGE_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return len(_BGE_TOKENIZER.encode(text, add_special_tokens=True, verbose=False))


@dataclass
class Block:
    """A document-ordered block: heading, paragraph, or table."""

    kind: str  # "heading" | "paragraph" | "table"
    page: int
    text: str = ""
    level: int = 0
    html: str = ""


@dataclass
class TablePart:
    """One row-grouped part of a Markdown table."""

    start: int
    end: int
    rows: list[str]
    text: str
    token_count: int
    over_limit: bool


# ── parsing ───────────────────────────────────────────────────────────────────

def split_pages(markdown: str) -> list[tuple[int, str]]:
    """Split markdown on ``<!-- page N -->`` markers into (page, text) pairs."""
    matches = list(PAGE_RE.finditer(markdown))
    if not matches:
        return [(1, markdown)]

    segments: list[tuple[int, str]] = []
    if markdown[: matches[0].start()].strip():
        segments.append((1, markdown[: matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        segments.append((int(match.group(1)), markdown[match.end():end]))
    return segments


def top_level_table_spans(text: str) -> list[tuple[int, int]]:
    """Offsets of top-level ``<table>...</table>`` blocks (nested ones included).

    Only locates boundaries so the untouched HTML can be stored in ``payload``
    and removed from the text stream; cell parsing is done by BeautifulSoup.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for match in TABLE_TAG_RE.finditer(text):
        if match.group(0).lower().startswith("</"):
            if depth:
                depth -= 1
                if depth == 0:
                    spans.append((start, match.end()))
        else:
            if depth == 0:
                start = match.start()
            depth += 1
    return spans


def clean_heading(text: str) -> str:
    """Strip markdown/HTML noise and collapse whitespace in a heading."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" #\t")
    return text


def parse_blocks(markdown: str) -> list[Block]:
    """Document-ordered blocks with page attribution.

    Paragraphs may span page boundaries; the page of a paragraph is the page
    where its first line starts. Tables always end a paragraph and are kept
    as one block each.
    """
    blocks: list[Block] = []
    paragraph_lines: list[str] = []
    paragraph_page: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_page
        text = "\n".join(paragraph_lines).strip()
        if text:
            blocks.append(Block("paragraph", paragraph_page or 1, text=text))
        paragraph_lines = []
        paragraph_page = None

    def consume_text(text: str, page: int) -> None:
        nonlocal paragraph_lines, paragraph_page
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue

            heading = HEADING_RE.match(stripped)
            if heading:
                flush_paragraph()
                heading_text = clean_heading(heading.group(2))
                if heading_text:
                    blocks.append(
                        Block("heading", page, text=heading_text,
                              level=len(heading.group(1)))
                    )
                continue

            if paragraph_page is None:
                paragraph_page = page
            paragraph_lines.append(stripped)

    for page, segment in split_pages(markdown):
        spans = top_level_table_spans(segment)
        cursor = 0
        for start, end in spans:
            consume_text(segment[cursor:start], page)
            flush_paragraph()
            blocks.append(Block("table", page, html=segment[start:end]))
            cursor = end
        consume_text(segment[cursor:], page)
        # Do not flush here: a paragraph may continue onto the next page.

    flush_paragraph()
    return blocks


# ── tables ────────────────────────────────────────────────────────────────────

def _span(cell, name: str, limit: int = 50) -> int:
    try:
        value = int(cell.get(name, 1))
    except (TypeError, ValueError):
        value = 1
    return max(1, min(value, limit))


def _cell_text(cell) -> str:
    for br in cell.find_all("br"):
        br.replace_with(" ")
    text = cell.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def table_grid(table) -> list[list[str]]:
    """Expand rowspan/colspan into a rectangular grid (values repeated)."""
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    occupied: dict[tuple[int, int], str] = {}
    max_col = 0

    for row_index, row in enumerate(rows):
        col = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (row_index, col) in occupied:
                col += 1
            text = _cell_text(cell)
            colspan = _span(cell, "colspan")
            rowspan = _span(cell, "rowspan")
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(row_index + dr, col + dc)] = text
            col += colspan
        max_col = max(max_col, col)

    if not occupied:
        return []

    max_row = max(row for row, _ in occupied) + 1
    max_col = max(max_col, max(col for _, col in occupied) + 1)
    return [
        [occupied.get((row, col), "") for col in range(max_col)]
        for row in range(max_row)
    ]


def pipe_table(grid: list[list[str]]) -> str:
    """Render a grid as a pipe-style markdown table (header = first row)."""
    if not grid:
        return ""

    def row_markdown(cells: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    lines = [row_markdown(grid[0]), row_markdown(["---"] * len(grid[0]))]
    lines.extend(row_markdown(row) for row in grid[1:])
    return "\n".join(lines)


def grid_to_html(grid: list[list[str]]) -> str:
    """Render a rectangular grid as a simple HTML table."""
    if not grid:
        return ""

    def row_html(cells: list[str], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in cells) + "</tr>"

    lines = ["<table>", "<thead>", row_html(grid[0], "th"), "</thead>", "<tbody>"]
    lines.extend(row_html(row, "td") for row in grid[1:])
    lines.extend(["</tbody>", "</table>"])
    return "".join(lines)


def split_pipe_table(pipe_md: str) -> tuple[str, str, list[str]]:
    """Split a pipe table into header, separator, and data rows."""
    lines = pipe_md.splitlines()
    if len(lines) < 2:
        return "", "", []
    return lines[0], lines[1], lines[2:]


def format_table_label(title: str, part: int, total: int, start: int, end: int) -> str:
    return f"Table: {title or 'Table'} | Part {part} of {total} | Rows {start}-{end}"


def render_table_part(
    title: str,
    part: int,
    total: int,
    start: int,
    end: int,
    header: str,
    separator: str,
    rows: list[str],
) -> str:
    label = format_table_label(title, part, total, start, end)
    return "\n".join([label, header, separator, *rows])


def group_table_rows(
    title: str,
    header: str,
    separator: str,
    rows: list[str],
    max_rows: int = TABLE_MAX_ROWS,
    target_tokens: int = TABLE_TARGET_TOKENS,
) -> list[TablePart]:
    """Pack rows into ordered parts without splitting any row."""
    if not rows:
        text = render_table_part(title, 1, 1, 0, 0, header, separator, [])
        tokens = bge_token_count(text)
        return [TablePart(0, 0, [], text, tokens, tokens > target_tokens)]

    grouped = [rows[i:i + max_rows] for i in range(0, len(rows), max_rows)]
    while True:
        total = len(grouped)
        parts: list[TablePart] = []
        start = 1
        for part_number, part_rows in enumerate(grouped, 1):
            end = start + len(part_rows) - 1
            text = render_table_part(
                title, part_number, total, start, end, header, separator, part_rows
            )
            tokens = bge_token_count(text)
            parts.append(
                TablePart(start, end, part_rows, text, tokens, tokens > target_tokens)
            )
            start = end + 1

        oversized = next(
            (
                index
                for index, part in enumerate(parts)
                if part.token_count > target_tokens and len(part.rows) > 1
            ),
            None,
        )
        if oversized is None:
            return parts

        part_rows = grouped[oversized]
        middle = len(part_rows) // 2
        grouped[oversized:oversized + 1] = [
            part_rows[:middle],
            part_rows[middle:],
        ]


def _unique(cells: list[str]) -> list[str]:
    """De-duplicate while preserving order (colspan repeats header text)."""
    return list(dict.fromkeys(cell for cell in cells if cell))


def _inference_header(grid: list[list[str]]) -> list[str]:
    """Best row for heading inference, skipping a spanning title row."""
    for row in grid[:3]:
        cells = [clean_heading(cell) for cell in row]
        if len(_unique(cells)) >= 2:
            return cells
    return [clean_heading(cell) for cell in grid[0]]


def parse_table(
    html: str,
) -> tuple[str, str, list[str], list[str], list[list[str]]]:
    """Return pipe markdown, caption, header cells, first column, and grid."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return "", "", [], [], []

    caption_tag = table.find("caption")
    caption = ""
    if caption_tag is not None:
        caption = clean_heading(caption_tag.get_text(" ", strip=True))

    grid = table_grid(table)
    if not grid or not any(cell.strip() for row in grid for cell in row):
        return "", caption, [], [], []

    header = _inference_header(grid)
    first_column = [clean_heading(row[0]) for row in grid[1:] if row]
    return pipe_table(grid), caption, header, first_column, grid


def infer_table_heading(header: list[str], first_column: list[str]) -> str:
    """Short deterministic heading from the header row / first column."""
    parts = _unique(header)[:4]
    if not parts:
        parts = _unique(first_column)[:3]
    if not parts:
        return "Table"
    heading = re.sub(r"\s+", " ", ", ".join(parts)).strip()
    if len(heading) > MAX_HEADING_CHARS:
        heading = heading[:MAX_HEADING_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return heading or "Table"


def take_caption_line(text: str) -> tuple[str, str]:
    """Split an immediately-above caption line off a paragraph, if present."""
    lines = text.splitlines()
    index = len(lines) - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return "", text

    candidate = clean_heading(lines[index].strip())
    if not CAPTION_RE.match(candidate):
        return "", text

    remaining = "\n".join(lines[:index] + lines[index + 1:]).strip()
    return candidate, remaining


# ── text ──────────────────────────────────────────────────────────────────────

def text_splitter() -> RecursiveCharacterTextSplitter:
    """Token-counted splitter that only breaks between paragraphs."""
    global _SPLITTER
    if _SPLITTER is None:
        _SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=TARGET_TOKENS,
            chunk_overlap=0,
            separators=["\n\n"],
        )
    return _SPLITTER


def _page_at_offset(offsets: list[tuple[int, int]], index: int) -> int:
    page = offsets[0][1] if offsets else 1
    for start, paragraph_page in offsets:
        if start <= index:
            page = paragraph_page
        else:
            break
    return page


def flush_text(
    doc_id: str,
    paragraphs: list[Block],
    heading: str,
    chunks: list[dict],
) -> None:
    """Split a run of paragraphs into ~TARGET_TOKENS text chunks."""
    paragraphs = [block for block in paragraphs if block.text.strip()]
    if not paragraphs:
        return

    full = "\n\n".join(block.text for block in paragraphs)
    offsets: list[tuple[int, int]] = []
    position = 0
    for block in paragraphs:
        offsets.append((position, block.page))
        position += len(block.text) + 2

    cursor = 0
    for raw in text_splitter().split_text(full):
        text = raw.strip()
        if not text:
            continue
        index = full.find(text, cursor)
        if index < 0:
            index = full.find(text)
        if index < 0:
            index = cursor
        chunks.append(
            {
                "doc_id": doc_id,
                "page": _page_at_offset(offsets, index),
                "chunk_type": "text",
                "heading": heading,
                "chunk": text,
            }
        )
        cursor = max(cursor, index + len(text))


# ── orchestration ─────────────────────────────────────────────────────────────

def document_title(blocks: list[Block]) -> str:
    """First level-1 heading, if any."""
    for block in blocks:
        if block.kind == "heading" and block.level == 1 and block.text:
            return block.text
    return ""


def build_chunks(doc_id: str, markdown: str) -> list[dict]:
    """Build document-ordered table and text chunks for one markdown file."""
    blocks = parse_blocks(markdown)
    fallback_heading = document_title(blocks) or doc_id

    chunks: list[dict] = []
    buffer: list[Block] = []
    current_heading = ""
    last_kind: str | None = None

    for block in blocks:
        if block.kind == "heading":
            flush_text(doc_id, buffer, current_heading or fallback_heading, chunks)
            buffer = []
            current_heading = block.text
            last_kind = "heading"

        elif block.kind == "table":
            caption_line = ""
            if buffer:
                caption_line, remaining = take_caption_line(buffer[-1].text)
                if caption_line:
                    if remaining:
                        buffer[-1].text = remaining
                    else:
                        buffer.pop()

            flush_text(doc_id, buffer, current_heading or fallback_heading, chunks)
            buffer = []

            pipe_md, caption_tag, header, first_column, grid = parse_table(block.html)
            if pipe_md:
                if caption_tag:
                    heading = caption_tag
                elif caption_line:
                    heading = caption_line
                elif last_kind == "heading":
                    heading = current_heading
                else:
                    heading = infer_table_heading(header, first_column)

                header_line, separator, data_rows = split_pipe_table(pipe_md)
                parts = group_table_rows(heading, header_line, separator, data_rows)
                data_grid = grid[1:]
                for part_number, part in enumerate(parts, 1):
                    part_grid = [grid[0]]
                    if part.rows:
                        part_grid.extend(data_grid[part.start - 1:part.end])
                    chunks.append(
                        {
                            "doc_id": doc_id,
                            "page": block.page,
                            "chunk_type": "table",
                            "heading": heading,
                            "chunk": part.text,
                            "payload": grid_to_html(part_grid),
                            "table_part": part_number,
                            "table_parts": len(parts),
                            "row_start": part.start,
                            "row_end": part.end,
                            "token_count": part.token_count,
                            "over_token_limit": part.over_limit,
                        }
                    )
            last_kind = "table"

        else:  # paragraph
            buffer.append(block)
            last_kind = "paragraph"

    flush_text(doc_id, buffer, current_heading or fallback_heading, chunks)
    return chunks


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def report_over_limit(records: list[dict]) -> None:
    """Print table parts that could not be reduced below the token target."""
    over = [
        record
        for record in records
        if record.get("chunk_type") == "table" and record.get("over_token_limit")
    ]
    if not over:
        print(f"table parts over {TABLE_TARGET_TOKENS} BGE tokens: 0")
        return

    print(
        f"table parts over {TABLE_TARGET_TOKENS} BGE tokens: {len(over)}",
        file=sys.stderr,
    )
    for record in over:
        print(
            f"  doc={record['doc_id']} page={record['page']} "
            f"part={record['table_part']}/{record['table_parts']} "
            f"rows={record['row_start']}-{record['row_end']} "
            f"tokens={record['token_count']}",
            file=sys.stderr,
        )


def find_markdown_files(input_path: Path, doc: str | None = None) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".md":
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.md"))
    if doc:
        files = [path for path in files if path.stem == doc]
    return files


# ── self-test ─────────────────────────────────────────────────────────────────

def assert_clean(records: list[dict], source_markdown: str | None = None) -> None:
    """No chunk may start with its own heading line or with a '#' heading.

    When ``source_markdown`` is given, a first line equal to the heading is
    allowed only if that same text also occurs as a plain paragraph in the
    source (documents sometimes repeat a heading phrase as body text).
    """
    plain_lines: set[str] = set()
    if source_markdown is not None:
        for line in source_markdown.splitlines():
            stripped = line.strip()
            if stripped and not HEADING_RE.match(stripped):
                plain_lines.add(
                    re.sub(r"\s+", " ", clean_heading(stripped)).lower()
                )

    for record in records:
        chunk = record["chunk"].lstrip()
        assert chunk, f"empty chunk in {record['doc_id']} p.{record['page']}"
        assert not re.match(r"^#{1,6}\s", chunk), (
            f"chunk starts with a heading line: {chunk[:80]!r}"
        )
        heading = (record.get("heading") or "").strip()
        if not heading:
            continue

        flat_heading = re.sub(r"\s+", " ", heading).lower()
        first_line = chunk.splitlines()[0]
        flat_first = re.sub(r"\s+", " ", clean_heading(first_line)).lower()
        if flat_first != flat_heading:
            continue

        if source_markdown is not None:
            assert flat_first in plain_lines, (
                f"chunk starts with its heading: heading={heading!r} "
                f"chunk={chunk[:80]!r}"
            )
        else:
            raise AssertionError(
                f"chunk starts with its heading: heading={heading!r} "
                f"chunk={chunk[:80]!r}"
            )


def default_sample(input_path: Path) -> Path | None:
    files = find_markdown_files(input_path)
    if not files:
        return None
    for path in files:
        if path.name.startswith("AAPL_"):
            return path
    return files[0]


def run_self_test(markdown_path: Path) -> list[dict]:
    markdown = markdown_path.read_text(encoding="utf-8")
    records = build_chunks(markdown_path.stem, markdown)
    tables = [record for record in records if record["chunk_type"] == "table"]
    texts = [record for record in records if record["chunk_type"] == "text"]
    average = (
        sum(len(_ENCODING.encode(record["chunk"])) for record in texts) / len(texts)
        if texts else 0.0
    )

    print(f"file        : {markdown_path.name}")
    print(f"table chunks: {len(tables)}")
    print(f"text chunks : {len(texts)}")
    print(f"avg tokens / text chunk: {average:.1f}")

    print("\n3 sample table chunks:")
    for record in tables[:3]:
        print(json.dumps(record, ensure_ascii=False, indent=2))

    print("\n3 sample text chunks:")
    for record in texts[:3]:
        print(json.dumps(record, ensure_ascii=False, indent=2))

    report_over_limit(records)
    assert_clean(records, markdown)
    print("\nassertions passed: no chunk starts with its heading or a '#' line")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Chunk pymupdf4llm markdown into table/text RAG chunks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="markdown folder (or one .md file)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="combined chunks JSONL path")
    parser.add_argument("--doc", default=None,
                        help="only chunk this document stem")
    parser.add_argument("--test", action="store_true",
                        help="run the self-test on one sample markdown file")
    parser.add_argument("--test-file", type=Path, default=None,
                        help="markdown file for --test (default: AAPL, else first)")
    args = parser.parse_args(argv)

    if args.test:
        sample = args.test_file or default_sample(args.input)
        if sample is None or not sample.is_file():
            print(f"no markdown file found under {args.input}")
            return 1
        run_self_test(sample)
        print(f"\nExecution time: {time.perf_counter() - start:.2f} seconds")
        return 0

    files = find_markdown_files(args.input, args.doc)
    if not files:
        print(f"no markdown files found under {args.input}")
        return 1

    all_chunks: list[dict] = []
    for index, path in enumerate(files, 1):
        markdown = path.read_text(encoding="utf-8")
        records = build_chunks(path.stem, markdown)
        all_chunks.extend(records)
        tables = sum(1 for record in records if record["chunk_type"] == "table")
        print(f"[{index}/{len(files)}] {path.name}: "
              f"{len(records)} chunks ({tables} tables)")

    write_jsonl(all_chunks, args.output)
    report_over_limit(all_chunks)
    print(f"wrote {len(all_chunks)} chunks to {args.output}")
    print(f"Execution time: {time.perf_counter() - start:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
