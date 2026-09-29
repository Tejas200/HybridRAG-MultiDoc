"""
layer0.py — Layer 0 feature extraction (plan.md section 3, layer0_plan.md).

Computes one PDFFeatures row per PDF:
  - page_count, avg_paragraph_length, column_layout : mechanical, from PyMuPDF blocks
  - table_density : page.find_tables() tables per page, bucketed with calibrated thresholds
  - ocr_quality   : "n/a" when a usable text layer exists; "poor" + review flag when absent
                    (actual OCR is out of scope — no PaddleOCR install needed for this batch)
  - doc_type      : rule-based heuristic cross-checked with one LLM zero-shot call
                    (OpenRouter, key from .env; primary openrouter/auto + free-model
                    fallback chain, mirrors src/agents/synthesis_agent.py). LLM is the
                    source of truth; disagreement with the heuristic -> flagged_for_review.

Per-PDF parse results (blocks, table counts) are cached under evaluation/cache/
keyed on (mtime, size), so re-running after a threshold change does not
re-parse the PDFs (layer0_plan.md Step 8.3 / Notes).

Usage (from the repo root):
    python evaluation/layer0.py [--input DIR] [--output FILE]
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

import pymupdf
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import PDFFeatures  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HERE, "cache")

# Recalibrated on the full 75-PDF mixed corpus (layer0_plan.md Step 10.2):
# tables-per-page median is 0.0 for contract/slide_deck, ~0.12 academic,
# ~0.25 manual, ~0.79 financial (2026-09-24 score distribution).
# high = table-centric (a table on roughly half of pages);
# low = tables essentially absent.
TABLE_DENSITY_HIGH = 0.5
TABLE_DENSITY_LOW = 0.1

DOC_TYPE_TAXONOMY = ("financial", "contract", "academic", "manual", "slide_deck", "scanned", "unknown")

# Calibrated against the full 75-PDF mixed corpus (2026-09-24, LLM labels as
# ground truth): 63/75 rule==LLM, the other 12 are abstentions (unknown).
# "FISCAL YEAR" was dropped — it is a defined term in many contracts and
# caused cross-type ties. SEC exhibit contracts all contain one boilerplate
# "SECURITIES AND EXCHANGE COMMISSION" mention; ties abstain instead of
# guessing. slide_deck deliberately has no keywords: deck 2-page text carries
# no reliable lexical signal.
DOC_TYPE_RULES = {
    "financial": ["SECURITIES AND EXCHANGE COMMISSION", "FORM 10-K", "FORM 10-Q",
                  "ANNUAL REPORT PURSUANT", "QUARTERLY REPORT PURSUANT"],
    "contract": ["THIS AGREEMENT", "THIS CONTRACT", "WITNESSETH", "IN WITNESS WHEREOF",
                 "PARTIES AGREE", "BY AND BETWEEN", "EFFECTIVE AS OF", "EFFECTIVE DATE",
                 "TERMINATION", "INDEMNIF", "SOLE DISCRETION", "CONFIDENTIAL INFORMATION"],
    "academic": ["ABSTRACT", "REFERENCES", "DOI.ORG/10.", "JOURNAL OF", "PROCEEDINGS OF",
                 "ARXIV", "PREPRINT", "METHODOLOGY", "FUTURE WORK", "ACKNOWLEDGEMENT",
                 "RELATED WORK", "EXPERIMENTAL", "JSTOR", "THESIS", "DISSERTATION"],
    "manual": ["USER MANUAL", "OPERATING INSTRUCTIONS", "MAINTENANCE MANUAL", "QUICK START",
               "USER GUIDE", "QUICK START GUIDE", "MANUAL", "GETTING STARTED",
               "TROUBLESHOOTING", "INSTALLATION", "SPECIFICATIONS", "SAFETY INSTRUCTIONS"],
    "scanned": ["DIGITAL COPY OF A BOOK", "SCANNED BY GOOGLE", "PUBLIC DOMAIN BOOK"],
}


# ── one pass over the PDF ────────────────────────────────────────────────────

def _parse_pdf(pdf_path):
    """Per page: text blocks (x0, text), table count, page width.

    This is the expensive part (find_tables on long 10-Ks takes minutes),
    so callers go through extract_page_data(), which caches the result.
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        blocks = [(b[0], b[4]) for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        pages.append({"blocks": blocks, "tables": len(page.find_tables().tables),
                      "width": page.rect.width})
    n = doc.page_count
    doc.close()
    return n, pages


def extract_page_data(pdf_path, use_cache=True):
    """Cached wrapper around _parse_pdf.

    Cache file per PDF under evaluation/cache/, keyed on (mtime, size);
    a changed source file is re-parsed automatically. Atomic write so a
    killed run cannot leave a corrupt cache entry.
    """
    cache_path = os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
    if use_cache:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            st = os.stat(pdf_path)
            if cached["mtime"] == st.st_mtime and cached["size"] == st.st_size:
                return cached["page_count"], cached["pages"]
        except (FileNotFoundError, KeyError, ValueError):
            pass  # miss, missing key, or corrupt/partial cache -> re-parse
    n, pages = _parse_pdf(pdf_path)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        st = os.stat(pdf_path)
        payload = {"pdf": os.path.basename(pdf_path), "mtime": st.st_mtime,
                   "size": st.st_size, "page_count": n, "pages": pages}
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, cache_path)
    return n, pages


# ── one feature per function (swap-able, per layer0_plan.md notes) ───────────

def compute_column_layout(pages):
    """'multi' when most pages show two distinct block-x bands (gutter gap).

    Both bands need substantial paragraph content: financial-statement
    column gaps, TOC pages, and offset-body render artifacts (e.g. GS
    filings printing body text at ~45% width with only a header left)
    otherwise register as fake multi-column layouts.
    """
    multi = judged = 0
    for p in pages:
        blocks = [b for b in p["blocks"] if len(b[1].split()) >= 3]
        xs = sorted(b[0] for b in blocks)
        if len(xs) < 8:
            continue
        judged += 1
        best_gap, best_at = 0.0, None
        for a, b in zip(xs, xs[1:]):
            if b - a > best_gap:
                best_gap, best_at = b - a, a
        if best_gap > 0.12 * p["width"]:
            left_para = sum(1 for b in blocks if b[0] <= best_at and len(b[1].split()) >= 10)
            right_para = sum(1 for b in blocks if b[0] > best_at and len(b[1].split()) >= 10)
            if min(left_para, right_para) >= 3:
                multi += 1
    return "multi" if judged and multi / judged >= 0.5 else "single"


def compute_avg_paragraph_length(pages):
    """Mean word count over paragraph-like blocks (>= 10 words skips headers)."""
    paragraphs = [len(b[1].split()) for p in pages for b in p["blocks"]
                  if len(b[1].split()) >= 10]
    return round(statistics.fmean(paragraphs), 1) if paragraphs else 0.0


def compute_table_density(pages):
    score = sum(p["tables"] for p in pages) / len(pages)
    bucket = "high" if score >= TABLE_DENSITY_HIGH else "medium" if score >= TABLE_DENSITY_LOW else "low"
    return bucket, round(score, 3)


def compute_ocr_quality(pages):
    """'n/a' with a usable text layer; 'poor' + flag when the doc is scanned
    or its text layer is corrupt (fonts without ToUnicode maps extract as
    control-byte garbage that a raw char count would read as usable text)."""
    texts = [b[1] for p in pages for b in p["blocks"]]
    if sum(len(t) for t in texts) / len(pages) < 20:
        return "poor", True
    all_text = "".join(texts)
    if sum(c.isprintable() or c.isspace() for c in all_text) / len(all_text) < 0.5:
        return "poor", True
    # A scan whose first page carries wrapper text (e.g. the Google Books
    # "digital copy of a book" notice) still averages enough chars per page
    # to pass the check above; catch it via page coverage instead.
    pages_with_text = sum(1 for p in pages if any(b[1].strip() for b in p["blocks"]))
    if pages_with_text / len(pages) < 0.5:
        return "poor", True
    return "n/a", False


def classify_doc_type_rules(sample_text):
    # Normalize whitespace first: joining blocks with "\n" splits phrases
    # across lines ("EFFECTIVE\nAS OF"), which exact-substring counting misses.
    text = " ".join(sample_text.upper().split())
    hits = {t: sum(text.count(k) for k in kws) for t, kws in DOC_TYPE_RULES.items()}
    ordered = sorted(hits.items(), key=lambda kv: -kv[1])
    if ordered[0][1] == 0 or ordered[1][1] == ordered[0][1]:
        return "unknown", 0.0  # no signal, or a tie between types -> abstain
    return ordered[0][0], round(min(0.99, 0.5 + 0.1 * ordered[0][1]), 2)


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
PRIMARY_LLM_MODEL = "openrouter/auto"


def fetch_free_models(api_key, limit=5):
    """Top-N zero-priced model IDs from GET /models (mirrors SynthesisAgent).

    Empty list on failure — the primary model is still tried.
    """
    try:
        r = requests.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        free = []
        for m in r.json().get("data", []):
            p = m.get("pricing", {})
            try:
                if float(p.get("prompt", 1)) == 0.0 and float(p.get("completion", 1)) == 0.0:
                    free.append(m["id"])
            except (TypeError, ValueError):
                continue
        return free[:limit]
    except Exception as e:
        print(f"    [warn] free-model fetch failed ({e})", flush=True)
        return []


def _llm_chat(api_key, model_id, prompt, max_retries=3, timeout=60):
    """One model, exponential backoff on 429 (mirrors SynthesisAgent).

    Returns (text, None) on success, (None, error) on final failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model_id,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.1, "max_tokens": 200},
                timeout=timeout,
            )
            if r.status_code == 429:
                time.sleep(2.0 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], None
            return None, f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return None, f"timeout after {timeout}s"
        except requests.exceptions.RequestException as e:
            return None, str(e)
    return None, f"exhausted {max_retries} retries"


def classify_doc_type_llm(sample_text, api_key):
    """Classify via PRIMARY_LLM_MODEL, then fall back to free models.

    Returns (label, confidence, model_id); raises if the whole chain fails.
    """
    prompt = (
        "Classify this document excerpt as exactly one of: financial, contract, "
        "academic, manual, slide_deck, scanned, unknown.\n\n"
        f"Excerpt:\n---\n{sample_text[:4000]}\n---\n\n"
        'Reply with JSON only: {"doc_type": "...", "confidence": 0.0-1.0}'
    )
    chain = [PRIMARY_LLM_MODEL]
    chain += [m for m in fetch_free_models(api_key) if m != PRIMARY_LLM_MODEL]
    last_err = "no models available"
    for model_id in chain:
        content, err = _llm_chat(api_key, model_id, prompt)
        if content is None:
            last_err = err
            continue
        m = re.search(r"\{.*\}", content, re.S)
        if m is None:
            last_err = f"{model_id}: no JSON in response"  # model answered in prose -> try next
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            last_err = f"{model_id}: bad JSON ({e})"
            continue
        label = str(data.get("doc_type", "")).strip().lower().replace(" ", "_")
        if label not in DOC_TYPE_TAXONOMY:
            label = "unknown"
        return label, round(float(data.get("confidence", 0.0)), 2), model_id
    raise RuntimeError(f"all models failed; last error: {last_err}")


# ── per-PDF pipeline ─────────────────────────────────────────────────────────

def analyze_pdf(pdf_path, api_key=None):
    page_count, pages = extract_page_data(pdf_path)
    sample = "\n".join(b[1] for p in pages[:2] for b in p["blocks"])

    rule_label, rule_conf = classify_doc_type_rules(sample)
    doc_type, doc_conf, flagged = rule_label, rule_conf, False
    llm_model = None

    if api_key:
        try:
            doc_type, doc_conf, llm_model = classify_doc_type_llm(sample, api_key)  # source of truth
            if rule_label != "unknown" and doc_type != rule_label:
                # heuristic abstention (unknown) is no opinion, not a
                # disagreement — only a real rule/LLM clash goes to review
                flagged = True
        except Exception as e:
            print(f"    [warn] LLM classification failed ({e}); using rule-based", flush=True)

    ocr_quality, ocr_flag = compute_ocr_quality(pages)
    if ocr_flag:  # no usable text layer: treat as scanned
        doc_type, doc_conf, flagged = "scanned", 0.9, True

    density, density_score = compute_table_density(pages)
    return PDFFeatures(
        pdf_id=os.path.splitext(os.path.basename(pdf_path))[0],
        page_count=page_count,
        doc_type=doc_type,
        doc_type_confidence=doc_conf,
        table_density=density,
        table_density_score=density_score,
        column_layout=compute_column_layout(pages),
        ocr_quality=ocr_quality,
        avg_paragraph_length=compute_avg_paragraph_length(pages),
        flagged_for_review=flagged,
    ), llm_model


def main():
    parser = argparse.ArgumentParser(description="Layer 0 PDF feature extraction")
    parser.add_argument("--input", default=os.path.join(ROOT, "corpus", "raw", "financial_pdf"))
    parser.add_argument("--output", default=os.path.join(HERE, "layer0_features.jsonl"))
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    api_key = os.environ.get("OPENROUTER_API_KEY")

    pdfs = sorted(glob.glob(os.path.join(args.input, "*.pdf")))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.input}")

    # Upsert: keep rows the input dir won't regenerate, so per-bucket runs
    # accumulate into the single layer0_features.jsonl (idempotent re-runs).
    refreshing = {os.path.splitext(os.path.basename(p))[0] for p in pdfs}
    kept = []
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and json.loads(line)["pdf_id"] not in refreshing:
                    kept.append(line)

    print(f"Layer 0: {len(pdfs)} PDFs from {args.input} "
          f"({'with LLM cross-check' if api_key else 'rule-based only, no API key'}; "
          f"{len(kept)} rows for other PDFs kept)", flush=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for line in kept:
            out.write(line + "\n")
        for path in pdfs:
            row, llm_model = analyze_pdf(path, api_key)
            out.write(row.model_dump_json() + "\n")
            out.flush()
            src = f"llm:{llm_model}" if llm_model else "rules"
            print(f"  {row.pdf_id:48s} type={row.doc_type}({row.doc_type_confidence:.2f}) [{src}] "
                  f"tables={row.table_density}({row.table_density_score}) "
                  f"cols={row.column_layout} ocr={row.ocr_quality} "
                  f"avg_para={row.avg_paragraph_length} flag={row.flagged_for_review}", flush=True)

    print(f"\nWrote {len(pdfs)} rows -> {args.output}")


if __name__ == "__main__":
    main()
