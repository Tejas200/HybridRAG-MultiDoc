# Layer 0 — Feature Extraction: Step-by-Step Implementation Plan

Goal: for every PDF in the corpus, produce a `pdf_features` JSON object containing
`doc_type`, `table_density`, `column_layout`, `ocr_quality`, `avg_paragraph_length`
(and any extra features you want to add later, e.g. `page_count`).

Test batch: the 15 financial filings (8×10-K, 7×10-Q) already downloaded and
converted to PDF. Validate the full pipeline on these before running on the
rest of the corpus.

---

## Step 1 — Set up environment
- Install core libraries:
  ```
  pip install pymupdf unstructured camelot-py[cv] paddleocr paddlepaddle pdf2image pandas pydantic
  ```
- No separate OCR binary install needed — PaddleOCR ships its own models (downloaded automatically on first run). Just `pip install`.
- Confirm your 15 test PDFs are all in one folder, e.g. `corpus/raw/financial_pdf/`.

---

## Step 2 — Define the output schema (Pydantic model)
Create a `PDFFeatures` model to enforce consistent output structure across every PDF:
```python
from pydantic import BaseModel
from typing import Literal

class PDFFeatures(BaseModel):
    pdf_id: str
    page_count: int
    doc_type: Literal["financial", "contract", "academic", "manual", "slide_deck", "scanned", "unknown"]
    doc_type_confidence: float
    table_density: Literal["high", "medium", "low"]
    table_density_score: float
    column_layout: Literal["single", "multi"]
    ocr_quality: Literal["n/a", "good", "poor"]
    avg_paragraph_length: float
    flagged_for_review: bool = False
```
This becomes one row per PDF in your final Layer 0 output table.

---

## Step 3 — Extract raw layout data (per PDF)
Using `PyMuPDF`:
1. Open the PDF, get `page_count`.
2. For each page, extract text blocks with bounding boxes (`page.get_text("blocks")` or `"dict"` for finer detail).
3. Check if a real text layer exists (i.e., blocks contain non-trivial extracted text). If not → flag as scanned, go to Step 6 (OCR) before continuing.
4. Store the raw block list per PDF for reuse in Steps 4 and 5 (avoid re-parsing the PDF for each feature).

---

## Step 4 — Compute `column_layout` and `avg_paragraph_length`
Purely mechanical — no thresholds to "decide," just compute:
- **`avg_paragraph_length`**: for each text block classified as a paragraph (filter out short/header-like blocks by length or font-size heuristics), count words/tokens, take the mean across the document.
- **`column_layout`**: cluster block x-coordinates (left edge of bounding box) per page. If blocks reliably fall into ≥2 distinct x-coordinate bands across most pages → `"multi"`, else `"single"`. A simple k-means (k=1 vs k=2) or a fixed x-gap threshold both work — start with the threshold approach for simplicity.

---

## Step 5 — Compute `table_density`
1. Run table detection with `camelot` (works well on financial filings with ruled/bordered tables) or `unstructured`'s table detection as a fallback for borderless tables.
2. For each PDF: `table_density_score = (number of detected tables OR table-covered area) / page_count`.
3. Calibrate bucket thresholds using your 15-PDF test batch as the first calibration sample (e.g., look at the score distribution, then set something like: `high` if score > 75th percentile, `low` if < 25th percentile, `medium` otherwise). Revisit thresholds once the full corpus is in, since 15 financial-only PDFs will skew high — you want thresholds that make sense across doc types, not just within financial.

---

## Step 6 — Compute `ocr_quality`
1. If Step 3 found a usable text layer → set `ocr_quality = "n/a"` (no OCR needed) — this will be the case for essentially all 15 EDGAR PDFs since they originate from HTML.
2. If no text layer (true scanned documents, relevant once you add the "scanned" bucket later): rasterize pages with `pdf2image`, run `PaddleOCR` (`PaddleOCR(lang="en")` then `.ocr(image)`), which returns per-line/word confidence scores, average confidence across the document, bucket into `"good"` (e.g. >80 avg confidence) or `"poor"` (below that).

---

## Step 7 — Classify `doc_type`
1. **Rule-based cross-check** (cheap, fast): keyword/structure heuristics — e.g., presence of "10-K", "10-Q", "SECURITIES AND EXCHANGE COMMISSION", fiscal-period headers → strong financial signal. Run this first since it's nearly free.
2. **LLM classification** (source of truth): sample first 1-2 pages of extracted text, prompt Qwen with a fixed taxonomy (`financial`, `contract`, `academic`, `manual`, `slide_deck`, `scanned`, `unknown`) and ask for a label + confidence score.
3. If rule-based and LLM disagree → set `flagged_for_review = true` for human review later; otherwise take the LLM label as final, store `doc_type_confidence`.
4. For your 15 test PDFs, this step should trivially return `"financial"` for all — useful as a correctness check on the classifier itself (if it doesn't, something's wrong in the prompt/taxonomy before you scale up).

---

## Step 8 — Assemble and store results
1. Combine Steps 3-7 into one `PDFFeatures` object per PDF.
2. Write all results to a single table: `layer0_features.jsonl` or `.parquet` (one row per PDF, one file for the whole corpus — easy to load with pandas later).
3. Keep raw intermediate outputs (block lists, table detections) in a separate cache folder in case you need to re-derive features after changing a threshold, without re-parsing PDFs from scratch.

---

## Step 9 — Validate on the 15-PDF test batch
1. Run the full Step 1-8 pipeline on the 15 financial PDFs only.
2. Manually spot-check 3-5 outputs against the actual PDFs:
   - Does `table_density` roughly match what you see visually (10-Ks with big financial statement sections should score high)?
   - Does `column_layout` correctly read as single-column (EDGAR filings are almost always single-column)?
   - Does `doc_type` return `"financial"` with high confidence for all 15?
3. Fix any obvious pipeline bugs (e.g., wkhtmltopdf table-clipping issues affecting table detection) before running on the rest of the corpus.

---

## Step 10 — Scale to full corpus
1. Once validated, point the pipeline at the full `corpus/raw/` directory (as other doc_type buckets get downloaded).
2. Re-calibrate `table_density` thresholds using the full, mixed-doc_type distribution — not just the financial-only sample from Step 5.
3. Review all `flagged_for_review = true` rows (doc_type disagreements) as a batch.
4. Freeze `layer0_features.jsonl` as the input for Layer 1/2.

---

## Notes
- Steps 3-7 should be implemented as independent functions that take parsed PDF data and return one feature each — makes it easy to swap tools later (e.g., replace `camelot` with `unstructured`'s table detector) without touching the rest of the pipeline.
- Cache intermediate parsing results — re-running OCR or table detection unnecessarily is the most expensive part of this layer.
