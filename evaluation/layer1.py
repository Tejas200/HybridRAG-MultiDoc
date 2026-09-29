"""
layer1.py — Layer 1 per-PDF RAG eval harness (evaluation/plan.md Section 4).

Layer 1 produces strategy-agnostic Q&A pairs per PDF, each tagged with the
structural passage it was generated from (source_passage_id) — the retrieval
ground truth Layer 2 scores chunking strategies against.

Division of labor (2026-09-24 decision: the in-session assistant is the
generator AND the faithfulness judge, per user instruction):

  --extract      deterministic, no LLM: parses each PDF into structural
                 units per plan.md Section 4 step 1 (heading-anchored
                 paragraph runs, table units, per-page units for short
                 pages like slides) and writes staging files the assistant
                 reads and authors Q&As from:
                   evaluation/passages/<stem>.txt  (human/model-readable)
                   evaluation/passages/index.json  (passage_id -> exact text)
  --finalize-qa  JSONL of assistant-authored pairs
                 {pdf_id, question, answer, source_passage_id}; validates
                 the passage_id exists, attaches passage_text/page/section
                 and the Layer 0 doc_type, and appends to the single
                 evaluation/layer1_qa.jsonl (dedup on pdf_id+question).

Usage (from the repo root):
    python evaluation/layer1.py --extract [--input DIR]
    python evaluation/layer1.py --finalize-qa MY_PAIRS.jsonl
"""

import argparse
import glob
import json
import os
import statistics
import sys

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import Layer1QA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PASSAGES_DIR = os.path.join(HERE, "passages")
INDEX_FILE = os.path.join(PASSAGES_DIR, "index.json")
FEATURES_FILE = os.path.join(HERE, "layer0_features.jsonl")
QA_OUTPUT = os.path.join(HERE, "layer1_qa.jsonl")

MAX_PASSAGE_CHARS = 2000      # long paragraph runs / big tables -> continuations
MIN_PASSAGE_WORDS = 20        # smaller units can't ground a Q&A pair
HEADING_SIZE_RATIO = 1.12     # span size vs page body-size mode
HEADING_MAX_CHARS = 120


def _page_lines(page):
    """[(text, size, bold, bbox)] per line, in reading order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            size = max(s["size"] for s in line["spans"])
            bold = any(s["flags"] & 16 for s in line["spans"])  # bit 4 = bold
            lines.append((text, size, bold, pymupdf.Rect(line["bbox"])))
    return lines


def _is_heading(text, size, bold, body_size):
    if len(text) > HEADING_MAX_CHARS or len(text.split()) < 2:
        return False
    if size >= body_size * HEADING_SIZE_RATIO:
        return True
    return bool(bold) and size >= body_size + 0.3


def _table_text(table):
    rows = table.extract() or []
    lines = [" | ".join("" if c is None else c.replace("\n", " ") for c in row)
             for row in rows]
    return "\n".join(lines).strip()


def _inside_table(rect, table_rects, overlap=0.5):
    for t in table_rects:
        inter = rect & t
        if not inter.is_empty and inter.get_area() > overlap * rect.get_area():
            return True
    return False


def extract_passages(pdf_path):
    """Structural units per evaluation/plan.md Section 4, step 1.

    Units: heading-anchored paragraph runs (split into continuations past
    MAX_PASSAGE_CHARS) plus table units. Lines covered by a detected table
    are excluded from the paragraph stream so tables are anchored by their
    own unit. Short pages (slides, one-pagers) keep small units.

    Returns [ {passage_id, page, section, kind, text} ].
    """
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = pymupdf.open(pdf_path)
    passages = []
    for page_index, page in enumerate(doc):
        lines = _page_lines(page)
        if not lines:
            continue
        table_rects = []
        tables_here = []
        try:
            for t in page.find_tables().tables:
                bbox = t.bbox
                text = _table_text(t)
                if text:
                    table_rects.append(pymupdf.Rect(bbox))
                    tables_here.append(text)
        except Exception as e:  # find_tables can fail on odd layouts
            print(f"    [warn] find_tables failed on {stem} p{page_index + 1} ({e})", flush=True)

        body_size = statistics.mode([round(s, 1) for _, s, _, _ in lines])
        page_word_total = sum(len(t.split()) for t, _, _, _ in lines)
        min_words = 1 if page_word_total < 150 else MIN_PASSAGE_WORDS  # slides

        section = ""
        buf, buf_len = [], 0
        seq = 0

        def flush():
            nonlocal seq, buf, buf_len
            text = " ".join(buf).strip()
            if len(text.split()) >= min_words:
                seq += 1
                passages.append({
                    "passage_id": f"{stem}:{page_index + 1:04d}:{seq}",
                    "page": page_index + 1,
                    "section": section,
                    "kind": "section" if section else "paragraph",
                    "text": text[:MAX_PASSAGE_CHARS],
                })
            buf, buf_len = [], 0

        for text, size, bold, bbox in lines:
            if _inside_table(bbox, table_rects):
                continue  # owned by a table unit
            if _is_heading(text, size, bold, body_size):
                flush()
                section = text
                continue
            if buf and buf_len + len(text) > MAX_PASSAGE_CHARS:
                flush()  # continuation of the same section
            buf.append(text)
            buf_len += len(text)
        flush()

        for i, ttext in enumerate(tables_here):
            for j in range(0, max(1, len(ttext)), MAX_PASSAGE_CHARS):
                seq += 1
                chunk = ttext[j:j + MAX_PASSAGE_CHARS]
                if len(chunk.split()) < min_words:
                    continue
                passages.append({
                    "passage_id": f"{stem}:{page_index + 1:04d}:{seq}",
                    "page": page_index + 1,
                    "section": section,
                    "kind": "table",
                    "text": chunk,
                })
    doc.close()
    return passages


def cmd_extract(input_dir):
    os.makedirs(PASSAGES_DIR, exist_ok=True)
    # Direct subdirs of the input are the doc-type buckets. Skip
    # underscore-prefixed staging dirs (e.g. corpus/raw/_cuad_download).
    if os.path.basename(input_dir) == "raw" or any(
            os.path.isdir(os.path.join(input_dir, d)) and not d.startswith("_")
            for d in os.listdir(input_dir)):
        dirs = [d for d in sorted(os.listdir(input_dir))
                if not d.startswith("_") and os.path.isdir(os.path.join(input_dir, d))]
        pdfs = []
        for d in dirs:
            pdfs += glob.glob(os.path.join(input_dir, d, "*.pdf"))
    else:
        pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    print(f"Layer 1 extract: {len(pdfs)} PDFs from {input_dir}", flush=True)
    index = {}
    for path in pdfs:
        stem = os.path.splitext(os.path.basename(path))[0]
        passages = extract_passages(path)
        with open(os.path.join(PASSAGES_DIR, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"[pdf: {stem}] [passages: {len(passages)}]\n")
            for p in passages:
                f.write(f"\n=== {p['passage_id']} | page {p['page']} | {p['kind']} "
                        f"| section: {p['section']!r} | words {len(p['text'].split())} ===\n")
                f.write(p["text"] + "\n")
                index[p["passage_id"]] = {
                    "pdf_id": stem, "page": p["page"], "section": p["section"],
                    "kind": p["kind"], "text": p["text"],
                }
        print(f"  {stem:52s} passages={len(passages)}", flush=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=0)
    print(f"\n{len(index)} passages -> {PASSAGES_DIR} (index.json)")


def cmd_finalize_qa(pairs_file):
    with open(INDEX_FILE, encoding="utf-8") as f:
        index = json.load(f)

    doc_types = {}
    if os.path.exists(FEATURES_FILE):
        for l in open(FEATURES_FILE, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                doc_types[r["pdf_id"]] = r["doc_type"]

    existing = set()
    if os.path.exists(QA_OUTPUT):
        for l in open(QA_OUTPUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                existing.add((r["pdf_id"], r["question"]))

    added = skipped = dropped = 0
    with open(QA_OUTPUT, "a", encoding="utf-8") as out:
        for line in open(pairs_file, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["pdf_id"], r["question"])
            if key in existing:
                skipped += 1
                continue
            p = index.get(r["source_passage_id"])
            if p is None or p["pdf_id"] != r["pdf_id"]:
                print(f"  DROPPED (bad passage_id): {r['source_passage_id']}")
                dropped += 1
                continue
            row = Layer1QA(
                pdf_id=r["pdf_id"],
                question=r["question"],
                answer=r["answer"],
                source_passage_id=r["source_passage_id"],
                passage_page=p["page"],
                passage_section=p["section"],
                passage_text=p["text"],
                doc_type=doc_types.get(r["pdf_id"], "unknown"),
                generation_model="assistant-in-session (generator + faithfulness judge)",
            )
            out.write(row.model_dump_json() + "\n")
            existing.add(key)
            added += 1
    print(f"added {added}, duplicate {skipped}, dropped {dropped} -> {QA_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Layer 1 per-PDF RAG eval harness")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--input", default=os.path.join(ROOT, "corpus", "raw"))
    parser.add_argument("--finalize-qa", metavar="PAIRS_JSONL")
    args = parser.parse_args()
    if args.extract:
        cmd_extract(args.input)
    elif args.finalize_qa:
        cmd_finalize_qa(args.finalize_qa)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
