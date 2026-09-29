# Golden Dataset Generation & Evaluation — Plan
## Use case: Given a PDF, recommend the correct chunking strategy

## 1. Objective
Build a golden dataset that maps **PDF characteristics → best chunking strategy**, so that a target model can be evaluated on (or eventually trained/prompted to perform) chunking-strategy recommendation for arbitrary PDFs.

Because "correct chunking strategy" has no inherent ground truth, the golden label for each PDF is **empirically derived**: multiple candidate chunking strategies are run against the PDF, scored via a small retrieval/QA eval built specifically for that document, and the winning strategy becomes the golden answer.

This requires **three layers**:
- **Layer 0**: feature extraction — derive structural `pdf_features` for each PDF (doc type, table density, layout, OCR quality, etc.)
- **Layer 1**: a per-PDF, strategy-agnostic RAG eval harness (Q&A pairs used only to score retrieval quality)
- **Layer 2**: the actual golden dataset — PDF → winning chunking strategy, derived using Layer 1 as the scoring mechanism

---

## 2. Roles / Models
To avoid circular or biased evaluation, keep these roles separate:

| Role | Model | Notes |
|---|---|---|
| **Generator** (Layer 1 Q&A generation) | Qwen-32B-class model | Generates passage-grounded questions + answers per PDF |
| **Target model** (being evaluated) | Your model under test | Never used to generate golden data; only consumes questions and produces chunking recommendations |
| **Judge** (scoring retrieval/answers) | The in-session assistant itself (this Qwen Code agent, qwen3.8-27b-class, and its subagents) — the same "assistant-in-session" pattern used for Layer 1 generation + faithfulness judging | Bulk scoring is deterministic + local cross-encoder; the assistant judge acts only as a targeted tiebreaker on near-tie PDFs |

---

## 3. Layer 0 — Feature Extraction (runs before Layer 1/2)

**Purpose**: Produce the `pdf_features` block for each PDF. These features are what the target model will ultimately consume to predict a chunking strategy, so they must be derived consistently and cheaply for every PDF in the corpus.

Features fall into three tiers by how they're decided:

| Feature | Method | Tools |
|---|---|---|
| `column_layout`, `avg_paragraph_length` | Fully mechanical — computed directly from layout parser output (block bounding boxes, word/token counts) | `PyMuPDF (fitz)`, `unstructured` |
| `table_density`, `ocr_quality` | Mechanical + calibrated thresholds | Table detection: `unstructured`, `camelot`, `LayoutParser`/Table Transformer. OCR: check for extractable text layer first; if absent, run OCR (`Tesseract`/`PaddleOCR`) and use per-word confidence scores |
| `doc_type` (financial / contract / academic / manual / slide_deck / scanned*) | LLM zero-shot classification against a fixed taxonomy, using the first 1–2 pages (or a sampled set); cheap heuristics (keyword/structure cues) as a low-cost cross-check to flag disagreement | LLM call (Qwen or similar) + rule-based cross-check |

\*`scanned` is out of scope for v1 (see Section 5 note) — kept in the taxonomy for future extensibility, but no PDFs are currently sourced for this bucket.

**Pipeline (per PDF):**
1. Run layout-aware parsing → extract blocks, tables, columns, text layer presence.
2. Compute `column_layout` (cluster block x-coordinates into bands) and `avg_paragraph_length` (mean token count of paragraph blocks) directly.
3. Compute `table_density` = table-covered area (or table count) / total content area or page count; bucket into high/medium/low using thresholds calibrated on your corpus.
4. Determine `ocr_quality`: if no extractable text layer, mark as scanned and run OCR, capturing confidence scores; else mark "n/a" or "good" if text layer is clean. **v1 note**: since the `scanned` bucket is out of scope for v1 (Section 5), `ocr_quality` will be `"n/a"` for every PDF in the current corpus — this feature is a placeholder for a future version and currently contributes no signal to Layer 2.
5. Classify `doc_type` via one LLM call on sampled pages against a fixed taxonomy; run a cheap rule-based heuristic in parallel (e.g., legal boilerplate keywords → contract, currency/fiscal terms → financial) and flag PDFs where the two disagree for human review.
6. Output: `pdf_features` object per PDF, stored alongside the PDF ID — this feeds directly into the Layer 2 golden dataset schema (Section 4) and is also what the target model receives at eval time.

**Note**: Keep Layer 0 outputs versioned separately from Layer 2 labels — if thresholds or the doc_type taxonomy change later, you can recompute features without regenerating the (expensive) Layer 1/2 strategy evaluations.

---

## 4. Layer 1 — Per-PDF RAG Eval Harness (intermediate, disposable)

**Purpose**: Provide a strategy-agnostic way to measure retrieval quality, so Layer 2 can compare chunking strategies fairly on the same PDF.

**Why not generate from the whole document or from pre-chunked text:**
- Whole-document generation skews toward salient content (intros/conclusions) and provides no ground-truth passage to check retrieval against.
- Generating questions from chunks that already use one of the 5 candidate strategies would bias the eval toward that strategy (circular).
- You need a **strategy-agnostic anchor** — a natural structural unit (paragraph, section, table, slide) — independent of any candidate chunk_size/splitting method.

**Pipeline (per PDF):**
1. Parse PDF structure — headings, sections, paragraphs, tables, slides (use layout-aware PDF parsing, not raw text dump).
2. For each structural unit, prompt Qwen to generate 1–3 questions **answerable only from this passage**, explicitly instructing it not to require outside context or other sections of the document.
3. For each question, prompt Qwen to answer **using only the passage**, and instruct it to respond with a fixed marker (e.g., `INSUFFICIENT_CONTEXT`) instead of answering if the passage alone doesn't contain enough information — this stops the model from silently pulling in outside knowledge or content from elsewhere in the document. Discard any pair that returns this marker.
4. Tag every surviving Q&A pair with `source_passage_id` (the structural unit it came from) — this is the retrieval ground truth.
5. **Faithfulness check (second, independent LLM pass)** — this is the safety net against the single biggest risk in Layer 1: the answer silently depending on content outside the tagged passage, which would make `source_passage_id` wrong and corrupt Layer 2's retrieval scoring downstream. The check specifically verifies two things, not just general correctness:
   - **Groundedness**: is every claim in the answer directly supported by the passage text?
   - **Single-passage sufficiency**: could the answer have been produced *using only this passage*, with no outside information, no other section of the document, and no world knowledge required?
   
   Prompt pattern: *"Given ONLY this passage and this answer, could someone verify the answer is fully correct using nothing but the passage? Answer strictly yes/no, and if no, state what's missing."* Any pair that fails either check is discarded, not just flagged — a failed pair means `source_passage_id` can't be trusted as ground truth, so it has no place in the eval set at all.
6. **Explicitly exclude multi-hop questions by construction.** Because generation happens per-structural-unit (Step 2) rather than across the whole document, questions requiring two or more separate passages to answer (e.g., "how does the risk factor on page 12 relate to the mitigation plan on page 40?") shouldn't arise in the first place — the model never sees more than one passage at a time when generating a question. The faithfulness check (Step 5) is the backstop that catches the rare case where the model answers using outside knowledge anyway despite this constraint.
7. Output: **~20–30 surviving Q&A pairs per PDF** (raised from an initial 10–20 — see note below), each with `{question, answer, source_passage_id, passage_text}`. Note this count is *after* filtering — generate more than 20-30 up front (e.g., 30-40) to account for pairs discarded in Steps 3 and 5, so the target count is still met post-filtering.

**Why 20-30, not 10-20**: the winning-strategy decision in Layer 2 is only as reliable as the number of questions it's scored against *for that one document*. With only ~10 questions, a strategy "winning" by 1-2 more correct answers than a competitor is barely distinguishable from noise — that near-tie margin can flip with a different random sample of questions. Raising the per-PDF count to 20-30 gives each document's strategy comparison more statistical footing before it becomes a golden label. This is a cheap change (more LLM calls per PDF, not more PDFs), and it directly reduces the risk of training Layer 2's downstream classifier on noisy, coin-flip labels.

**What a mismatched-passage failure would look like if not caught**: the question is generated from passage 47, but the LLM's answer actually draws on information from passage 12 as well (e.g., it infers a connection not stated in passage 47 alone). If undetected, Layer 2 would penalize a chunking strategy for "failing" to retrieve passage 47 — when the real issue is that the ground-truth tag itself was wrong, not that the strategy performed poorly. Steps 3 and 5 exist specifically to prevent this from silently entering the dataset.

**What is *not* a problem, and should not be filtered out**: if a chunking strategy splits the correct passage across two chunks (e.g., a paragraph is cut in half by a naive fixed-size splitter) and that hurts its retrieval score, that's the eval working correctly, not a labeling error. A good strategy keeps source passages intact; a bad one fragments them — that fragmentation showing up as a recall drop is exactly the signal Layer 2 is meant to capture.

This set is **not the final golden dataset** — it exists only to score candidate strategies in Layer 2.

---

## 5. Layer 2 — PDF → Chunking Strategy Golden Set (final dataset)

**Purpose**: For each PDF, determine and record which chunking strategy performs best, using Layer 1 as the scoring harness.

**Status (2026-09-28)**: Layer 1 is complete — 14,430 Q&A rows across 70/71 PDFs in `evaluation/layer1_qa.jsonl` (JPM 10-K skipped by user instruction). The v1 Layer 2 design below was agreed 2026-09-28.

**Candidate chunking strategies to test** (adjust as needed):
- Fixed-size / recursive character splitting (various chunk_size + overlap combos)
- Semantic chunking (embedding-similarity based splits)
- Layout-aware chunking (respects headings/sections)
- Table-aware chunking (keeps tables intact / separately indexed)
- Sliding window with overlap variants

**v1 candidate set (finalized 2026-09-28 — user-confirmed: built with LangChain, sliding-window variant skipped)**: the 4 strategies are implemented in **`evaluation/chunking_strategies.py`** (standard chunk-dict output, same schema as `src/chunkers/base.make_chunk`; CLI: `python evaluation/chunking_strategies.py <pdf_id> [--strategy ...]` or `--test`):
- `fixed-size` — LangChain `RecursiveCharacterTextSplitter`, token-based (tiktoken), 450 tokens / 80 overlap (production budget).
- `semantic` — p95 cosine-distance breakpoint on paragraph embeddings (`BAAI/bge-small-en-v1.5`, via LangChain's `Embeddings` interface); oversized segments sub-split with `RecursiveCharacterTextSplitter`. (Current LangChain releases ship no semantic-breakpoint splitter class, so this is the standard semantic-chunking algorithm assembled from LangChain primitives.)
- `layout-aware` — PDF font-size/bold heading detection (PyMuPDF span data; excludes table regions, the cover page, repeated running headers/footers, and checkbox-glyph lines) → Markdown → LangChain `MarkdownHeaderTextSplitter`; oversized sections sub-split at the 450-token budget.
- `table-aware` — PyMuPDF `find_tables()`: every detected table kept **intact** as its own chunk (may exceed 450 tokens — the intended trade-off; BGE truncates at 512 at embed time); remaining prose split with `RecursiveCharacterTextSplitter`.

Tested 2026-09-28 on one PDF per in-scope bucket (AAPL 10-K financial, AzulSa contract, arXiv paper academic, Excel glossary manual, health slide deck) — all 20 runs passed; per-run reports in `evaluation/chunking_test/` (+ `summary.json`). The old `src/chunkers/` implementations are **not** the Layer 2 candidate set — the LangChain set above is. The plan's "start with 3–5" is satisfied (4).

**Pipeline (per PDF × strategy) — v1 design agreed 2026-09-28:**

1. **Chunk + index**: for each candidate strategy (via `evaluation/chunking_strategies.run_strategy`), chunk the PDF, embed every chunk with the production embedding model (`BAAI/bge-small-en-v1.5`, L2-normalized, cosine) into a dedicated **ephemeral Chroma collection per (PDF, strategy)** (cosine/HNSW). Disposable per run; questions never search across PDFs, so no cross-document contamination.
2. **Two-stage retrieval (mirrors the production app)**: per Layer 1 question — embed the question with the BGE query prefix → dense **top-30** candidates from the collection → cross-encoder (`BAAI/bge-reranker-base`, the app's production reranker) re-ranks the 30 (question, chunk) pairs → **top-5**. The wider 30-candidate pool (vs the app's 10) gives the reranker room to rescue a correct passage that a fragmenting strategy pushed down in raw vector rank. All strategies share this identical pipeline; chunking is the only variable.
3. **Score each question with three signals**:
   - `retrieval_recall` (deterministic — this section's original definition): 1 if any top-5 chunk contains ≥50% of the source passage's content tokens, else 0.
   - `numeric_coverage` (deterministic): fraction of the golden answer's number tokens (regex `\d+(?:,\d+)*(?:\.\d+)?%?`, normalized: strip commas/%) that appear literally in the top-5 text. Guards against the xent/LLM scoring a same-table/wrong-row chunk as "supporting" the answer.
   - `answer_support` (local cross-encoder): max over the top-5 of (golden_answer, chunk) relevance — the Ragas-faithfulness direction (is the golden answer grounded in the retrieved context?), with no answer generation. **Capped at 0.5 whenever `numeric_coverage` < 1.0** (missing key figures).

   **Score-row diagnostics (agreed 2026-09-28):** each `(pdf_id, strategy, question)` row in `evaluation/layer2_scores.jsonl` additionally stores `max_chunk_overlap` (raw per-chunk max fraction), `union_overlap` (raw fraction of passage tokens covered by the union of the top-5 text), `answer_support_raw` (uncapped), and `retrieved_pages`. Diagnostics only — the per-chunk max remains the labeling signal, so the fragmentation tax (union high, per-chunk low) is auditable per strategy at aggregate time without changing the metric.
4. **LLM judge as targeted tiebreaker (the in-session assistant)**: recompute `answer_support` with a standard faithfulness-style prompt (question + golden answer + top-5 context + precomputed numeric coverage as a hint → 0/0.5/1, same numeric cap) **only for the questions of PDFs whose provisional top-2 margin is < ~15%** — the only place a judge can flip a golden label. The judge is the in-session assistant (this Qwen Code agent, qwen3.8-27b-class, delegating to subagents for volume) — the same "assistant-in-session (generator + faithfulness judge)" pattern as Layer 1 — so there is **no external API budget, rate-limit, or credits constraint** on judging. It is still spent only on near-tie PDFs because a full ~58k-judgment sweep (14,430 questions × 4 strategies) buys nothing where the deterministic + cross-encoder signals already separate the strategies.
5. **Aggregate**: per-strategy score = mean over all of the PDF's Layer 1 questions (the corpus has far more than the 20–30 originally assumed — up to ~1,400 per PDF — no downsampling, strictly more statistical power). `combined = 0.5·retrieval_recall + 0.5·answer_support`.
6. **Check the margin between the top two strategies.** If the leading strategy beats the runner-up by less than a defined threshold (<10% on the combined score, i.e. (top1 − top2)/top1), label the PDF `"ambiguous"` instead of forcing a single winner — exclude it from classifier training later (Section 8), and track it separately (e.g., what fraction of the corpus ended up ambiguous). This is what the per-PDF question count is meant to support: fewer near-ties, and the ones that remain get flagged instead of silently treated as clean labels.
7. Otherwise, winning strategy (highest combined retrieval + answer score) = golden label.

**Ragas**: **not used as a library in v1** — its per-metric 1+N LLM calls per sample would multiply any API budget 3–5×, and two of the three signals are not Ragas metrics. `answer_support` simply follows the standard faithfulness definition (is the golden answer grounded in the retrieved context?), documented here rather than imported from the library.

**Reproducibility**: each golden record's metadata pins the chunking strategy (module `evaluation/chunking_strategies.py` + its params), embedding / reranker / judge model IDs, the judge prompt hash, k values (30/5), thresholds (passage-overlap 0.50, tiebreak 0.15, margin 0.10), and the numeric-token regex.

**Run path (decided 2026-09-28)**: the full sweep runs on a **free Kaggle GPU (T4/P100)** — the local machine is a 15 W CPU with no GPU and is ~13–16 days from done at measured throughput (below). The Kaggle run is the self-contained `evaluation/kaggle/layer2_kaggle.py` (bundle: `evaluation/kaggle/layer2_kaggle_bundle.zip`, runbook: `evaluation/kaggle/README_kaggle.md`; 71-PDF corpus + `layer1_qa.jsonl` only, no project code or secrets). One documented deviation from pipeline steps 1–2: the Kaggle run computes the dense top-30 as **exact cosine over the per-(PDF, strategy) chunk matrix via numpy** instead of a Chroma HNSW approximate-ANN index — per-document chunk counts are small enough that exact cosine is the faithful "dense top-30" and removes HNSW recall variance as a confound. Everything else (models, k=30/5, thresholds, row schema, signal formulas) is identical to `evaluation/layer2.py`, which remains the local path for per-PDF spot-checks and for the aggregate step. Kaggle output `layer2_scores_kaggle.jsonl` becomes `evaluation/layer2_scores.jsonl` (sole scores source) on return.

**Compute profile (corrected 2026-09-28, measured)**: each question costs cross-encoder encodings for the *entire* dense candidate pool, not just the final top-5 — 30 (question, chunk) rerank pairs + 5 (golden answer, chunk) support pairs. Total: 14,430 questions × 4 strategies × 35 pairs ≈ **2.0M cross-encoder pair encodings**. Measured locally (Intel i5-10210U, no GPU): ~1.5–2 pairs/sec → ~13–16 days local (the original "289k pairs / hours" estimate counted only the 5 support pairs). On a Kaggle T4/P100 the same sweep is expected to take ~1–3 h, resumable via the checkpointed scores file. The LLM tiebreak (near-tie PDFs only) is only the marginal cost.

**Final golden dataset schema (per PDF):**
```json
{
  "pdf_id": "uuid",
  "pdf_features": {
    "page_count": 42,
    "doc_type": "financial_report | contract | academic_paper | slide_deck | manual | scanned",
    "table_density": "high | medium | low",
    "column_layout": "single | multi",
    "ocr_quality": "n/a | good | poor",  // always "n/a" in v1 — scanned bucket out of scope, see note below
    "avg_paragraph_length": 120
  },
  "golden_chunking_strategy": {
    "method": "layout-aware",  // one of: fixed-size | semantic | layout-aware | table-aware
    "impl": "evaluation/chunking_strategies.py",
    "params": {"chunk_tokens": 450, "overlap_tokens": 80, "semantic_breakpoint_percentile": 95}
  },
  "label_status": "clear | ambiguous",  // "ambiguous" if top-2 strategies were within the margin threshold; excluded from classifier training
  "winning_margin": 0.12,  // (top1 - top2)/top1 on the combined score
  "strategy_scores": {
    "fixed-size": {"retrieval_recall": 0.62, "numeric_coverage": 0.61, "answer_support": 0.66, "combined": 0.64},
    "semantic": {"retrieval_recall": 0.78, "numeric_coverage": 0.74, "answer_support": 0.80, "combined": 0.79},
    "layout-aware": {"retrieval_recall": 0.91, "numeric_coverage": 0.92, "answer_support": 0.89, "combined": 0.90},
    "table-aware": {"retrieval_recall": 0.78, "numeric_coverage": 0.80, "answer_support": 0.74, "combined": 0.76}
  },
  "layer1_eval_set_ref": "path/id to the Layer 1 Q&A set used",
  "generation_model": "generator+judge=assistant-in-session (Qwen Code agent); embed=BAAI/bge-small-en-v1.5; reranker=BAAI/bge-reranker-base",
  "reviewed_by": "human | llm | none",
  "created_at": "timestamp"
}
```

**Corpus diversity requirement**: PDFs must span structurally distinct types. Without diversity, the golden set converges to "one strategy always wins," which defeats the purpose of a recommender.

**v1 scope (current)**: 5 doc_type buckets — financial, contract, academic, manual, slide_deck (~15 PDFs each, ~75 total). The `scanned` bucket is **explicitly out of scope for v1** — no scanned/OCR'd PDFs are being sourced. `ocr_quality` remains in the feature schema for forward-compatibility but will be `"n/a"` for every PDF in this version, so it contributes no signal to Layer 2 in v1. If scanned documents are added in a future version, this is the feature that will start varying and become useful.

---

## 6. Stack

### Core generation & orchestration
- **Qwen-32B** (self-hosted or via API) — Layer 1 question/answer generation.
- **Python 3.10+** — orchestration.
- **PDF parsing**: `PyMuPDF (fitz)`, `unstructured`, or `pdfplumber` — layout-aware extraction (headings, tables, columns), not raw text dump.
- **LangChain** (`langchain-text-splitters==0.3.9` + `langchain-core`) — candidate chunking strategies, implemented 2026-09-28 in `evaluation/chunking_strategies.py` (see Section 5 v1 candidate set).
- **Pydantic** — structured output schemas for Q&A generation and strategy scoring.

### Indexing / retrieval (Layer 2)
- **Vector DB**: Chroma or FAISS (local/self-hosted, good for per-PDF small-scale indexing) or Pinecone/Qdrant if scaling to many PDFs.
- **Embedding model**: match whatever your production retrieval stack uses (e.g., `bge-large`, `text-embedding-3-large`, or Qwen embeddings) so results transfer to your real system.

### Evaluation / scoring
- ~~**Ragas**~~ — removed from v1 (see Section 5 Ragas note); the faithfulness-style `answer_support` signal is implemented directly without the library.
- **DeepEval** — alternative/complementary, supports custom correctness metrics.
- **LLM-as-judge** — for answer correctness scoring where lexical overlap metrics are insufficient.

### Data storage & management
- **JSONL/Parquet** — Layer 1 sets (per PDF) and Layer 2 golden dataset.
- **Pandas** — aggregation of strategy scores, building the final golden table.
- **DVC / Git-LFS** — version the golden dataset as PDFs/strategies are added.

### Quality control
- Faithfulness check (Layer 1 answers vs passage) — LLM pass or sampled human review.
- Human review of a sample of final Layer 2 labels (spot-check that the "winning" strategy is sane given `pdf_features`).
- **Label Studio / Argilla** — review UI if human-in-the-loop QA is needed at scale.

### Experiment tracking
- **Weights & Biases / MLflow** — log strategy comparison runs per PDF, track score distributions across the corpus.

---

## 7. Step-by-Step Plan
1. Assemble a structurally diverse PDF corpus across the 5 in-scope buckets (financial, contract, academic, manual, slide_deck). Scanned documents are out of scope for v1.
2. Build layout-aware PDF parser → extract structural units per PDF.
3. **Layer 0**: extract `pdf_features` per PDF (doc_type via LLM classification, table_density, column_layout, ocr_quality, avg_paragraph_length).
4. Generate Layer 1 Q&A sets per PDF (Qwen-32B), ~20-30 per PDF, tagged with `source_passage_id`; faithfulness-filter.
5. Implement candidate chunking strategies (start with 3–5) in a common pipeline interface. **v1 (done 2026-09-28): the 4 LangChain-based strategies in `evaluation/chunking_strategies.py` (fixed-size, semantic, layout-aware, table-aware — sliding-window variant skipped per user instruction); tested on one PDF per bucket with reports in `evaluation/chunking_test/` — see the v1 candidate set note in Section 5.**
6. For each PDF × strategy: chunk → embed (production bge-small) into an ephemeral per-(PDF, strategy) Chroma collection → per question: dense top-30 → cross-encoder rerank (bge-reranker-base) → top-5 → score with the three Section-5 signals (deterministic retrieval recall, deterministic numeric coverage, cross-encoder answer support with numeric cap). No answer generation; the golden answer is scored against the retrieved context (Ragas-faithfulness direction).
7. Aggregate scores per strategy (mean over all of the PDF's Layer 1 questions), select the winning strategy per PDF, applying the margin threshold — clear wins become golden labels, near-ties are marked `"ambiguous"` and excluded from classifier training (Section 5, Section 8). The in-session assistant (this Qwen Code agent / its subagents — the same "assistant-in-session" judge as Layer 1) runs **only on near-tie PDFs** (provisional top-2 margin < ~15%) as a tiebreaker that can flip labels (Section 5 v1 design).
8. Human-review a sample of final labels for sanity (and a sample of Layer 0 doc_type classifications where heuristic/LLM disagree).
9. Freeze & version the Layer 2 golden dataset (e.g., `golden-chunking-v1.0`).
10. Evaluate your target model: feed it PDFs (or extracted `pdf_features`), have it predict a chunking strategy, compare against the golden label.
11. Report accuracy/agreement metrics; iterate on corpus diversity or strategy set as gaps appear.

---

## 8. Key Risks & Mitigations
- **Layer 1 questions biased toward salient content** → passage-based, structure-driven generation (not whole-doc) mitigates this.
- **Circularity** (using a candidate strategy's own chunks to generate eval questions) → always generate Layer 1 from strategy-agnostic structural units.
- **Non-diverse PDF corpus** → explicitly curate by document type; track type distribution.
- **LLM judge favoring certain phrasing/strategy** → the judge is only a near-tie tiebreaker; the two deterministic signals (retrieval recall, numeric coverage) remain the primary cross-check on any judge decision. The judge is the same in-session assistant as the Layer 1 generator — accepted for v1 because it only overturns a near-tie *after* the deterministic + cross-encoder sweep, and every judge override is recorded in the score rows for audit.
- **Golden label instability** (close scores between strategies) → store full `strategy_scores`, not just the winner, so near-ties are visible and can be handled (e.g., mark as "ambiguous" rather than forcing a single label).
