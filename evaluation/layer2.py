"""
layer2.py — Layer 2: PDF -> chunking-strategy golden set (evaluation/plan.md §5, §7 steps 5-7).

v1 pipeline (agreed 2026-09-28), per PDF x strategy:
  1. Chunk the PDF with the candidate strategy — the 4 LangChain-based strategies in
     evaluation/chunking_strategies.py (fixed-size | semantic | layout-aware | table-aware).
  2. Embed every chunk with the production embedder (BAAI/bge-small-en-v1.5, L2-normalized,
     cosine) into an EPHEMERAL Chroma collection per (pdf, strategy). Disposable per run;
     questions never search across PDFs, so no cross-document contamination.
  3. Per Layer 1 question: embed (BGE query prefix) -> dense top-30 -> cross-encoder rerank
     (BAAI/bge-reranker-base, the app's production reranker, max_length=512) -> top-5.
     The wider 30-candidate pool (vs the app's 10) gives the reranker room to rescue a
     correct passage that a fragmenting strategy pushed down in raw vector rank.
  4. Score each question deterministically — no LLM, no answer generation:
       retrieval_recall : 1 if any top-5 chunk holds >= 50% of the source passage's content
                          tokens, else 0. Strict per-chunk: a passage fragmented across
                          chunks scores 0 even if all fragments were retrieved (the intended
                          fragmentation penalty, plan §4).
       numeric_coverage : fraction of the golden answer's number tokens (regex
                          \\d+(?:,\\d+)*(?:\\.\\d+)?%?, commas/% stripped) present in the union
                          of the top-5 text. 1.0 when the answer has no numbers.
       answer_support   : sigmoid(max over top-5 of CrossEncoder(golden_answer, chunk)) —
                          the Ragas-faithfulness direction (golden answer grounded in the
                          retrieved context?). Capped at 0.5 when numeric_coverage < 1.0.
     Diagnostics (never used for labeling): max_chunk_overlap, union_overlap,
     answer_support_raw, retrieved_pages.
  5. Each (pdf, strategy, question) row appends to evaluation/layer2_scores.jsonl;
     completed triples are skipped on rerun (resumable).

aggregate (plan §7 step 7):
  - Per PDF: per-strategy mean of the three signals;
    combined = 0.5 * retrieval_recall + 0.5 * answer_support.
  - Winner = highest combined; (top1 - top2)/top1 < 0.10 -> label_status "ambiguous"
    (excluded from classifier training), else "clear".
  - PDFs with provisional margin < 0.15 are listed for the in-session assistant judge
    tiebreak (the Qwen Code agent runs that after the sweep — no external LLM in the
    bulk path).
  - Writes evaluation/layer2_golden.jsonl (plan §5 golden schema) with pdf_features
    merged in from evaluation/layer0_features.jsonl.

Usage:
  python evaluation/layer2.py score [--pdf STEM ...] [--strategy fixed-size semantic ...] [--limit N]
  python evaluation/layer2.py aggregate
  python evaluation/layer2.py status
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunking_strategies import STRATEGIES, run_strategy  # noqa: E402
from src.config import (  # noqa: E402
    BGE_QUERY_PREFIX,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_TOKENS,
    RERANKER_MODEL,
    SEMANTIC_BREAKPOINT_PERCENTILE,
)

LAYER1_QA = ROOT / "evaluation" / "layer1_qa.jsonl"
LAYER0_FEAT = ROOT / "evaluation" / "layer0_features.jsonl"
SCORES_FILE = ROOT / "evaluation" / "layer2_scores.jsonl"
GOLDEN_FILE = ROOT / "evaluation" / "layer2_golden.jsonl"

DENSE_TOP_K = 30                 # plan §5 step 2: candidate pool for the reranker
RERANK_TOP_K = 5                 # plan §5 step 2: final context chunks
OVERLAP_HIT_THRESHOLD = 0.50     # plan reproducibility line: passage-overlap threshold
MARGIN_THRESHOLD = 0.10          # plan §5 step 6: <10% top-2 gap => ambiguous
NEAR_TIE_TIEBREAK = 0.15         # plan §5 step 4: provisional margin below this => in-session judge
NUMERIC_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?%?")
TOKEN_RE = re.compile(r"[a-z0-9]+")
XENT_BATCH = 32

STRATEGY_PARAMS = {
    "fixed-size": {"chunk_tokens": MAX_CHUNK_TOKENS,
                   "overlap_tokens": CHUNK_OVERLAP_TOKENS},
    "semantic": {"breakpoint_percentile": SEMANTIC_BREAKPOINT_PERCENTILE,
                 "max_chunk_tokens": MAX_CHUNK_TOKENS, "overlap_tokens": 0},
    "layout-aware": {"heading_detection": "font-size/bold (PyMuPDF spans)",
                     "max_section_tokens": MAX_CHUNK_TOKENS},
    "table-aware": {"tables": "intact (PyMuPDF find_tables)",
                    "prose_chunk_tokens": MAX_CHUNK_TOKENS,
                    "prose_overlap_tokens": CHUNK_OVERLAP_TOKENS},
}


# ── data helpers ───────────────────────────────────────────────────────────────

def load_layer1() -> dict[str, list[dict]]:
    by_pdf: dict[str, list[dict]] = defaultdict(list)
    with open(LAYER1_QA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_pdf[r["pdf_id"]].append(r)
    return dict(by_pdf)


def load_layer0() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(LAYER0_FEAT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["pdf_id"]] = r
    return out


# ── text normalization + signals ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """lowercase -> de-hyphenate line-break hyphens -> collapse whitespace."""
    text = text.lower()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text)


def content_tokens(text: str) -> set[str]:
    """Distinct content tokens: [a-z0-9]+ runs, length >= 3 (numbers kept)."""
    return {t for t in TOKEN_RE.findall(normalize_text(text)) if len(t) >= 3}


def numeric_tokens(text: str) -> set[str]:
    """Number tokens normalized: commas and trailing % stripped ('4,132' -> '4132')."""
    return {t.rstrip("%").replace(",", "") for t in NUMERIC_RE.findall(text)}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def load_done() -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if SCORES_FILE.exists():
        with open(SCORES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("score_ok"):
                    done.add((r["pdf_id"], r["strategy"], r["question"]))
    return done


# ── per (pdf, strategy) scoring ───────────────────────────────────────────────

def score_pdf_strategy(
    pdf_id: str,
    strategy: str,
    questions: list[dict],
    model,
    reranker,
    done: set[tuple[str, str, str]],
    limit: int | None = None,
) -> int:
    import chromadb

    _pdf_path, chunks = run_strategy(pdf_id, strategy, model)
    if not chunks:
        raise RuntimeError(f"strategy {strategy} produced 0 chunks for {pdf_id}")
    texts = [c["text"] for c in chunks]
    pages = [int(c["page_number"]) for c in chunks]

    embs = model.encode(
        texts, normalize_embeddings=True, batch_size=64,
        show_progress_bar=len(texts) > 200,
    ).tolist()
    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(
        "l2_eval", metadata={"hnsw:space": "cosine"}
    )
    coll.upsert(
        ids=[f"c{i}" for i in range(len(chunks))],
        embeddings=embs,
        documents=texts,
        metadatas=[{"page_number": p, "seq": i} for i, p in enumerate(pages)],
    )

    q_items = questions if limit is None else questions[:limit]
    pending = [(qi, row) for qi, row in enumerate(q_items)
               if (pdf_id, strategy, row["question"]) not in done]
    if not pending:
        return 0

    q_embs = model.encode(
        [BGE_QUERY_PREFIX + row["question"] for _qi, row in pending],
        normalize_embeddings=True, batch_size=64,
    ).tolist()

    n_res = min(DENSE_TOP_K, len(chunks))
    raw = coll.query(
        query_embeddings=q_embs, n_results=n_res,
        include=["documents", "metadatas"],
    )
    qdata = [
        (row, list(zip(docs, metas)))
        for (_qi, row), docs, metas in zip(pending, raw["documents"], raw["metadatas"])
    ]

    # stage 1: batched cross-encoder rerank of every (question, candidate) pair
    rr_pairs: list[tuple[str, str]] = []
    for row, top30 in qdata:
        for c_text, _meta in top30:
            rr_pairs.append((row["question"], c_text))
    rr_scores = reranker.predict(
        rr_pairs, batch_size=XENT_BATCH, show_progress_bar=len(rr_pairs) > 400
    )

    top5list: list[tuple[dict, list[tuple[str, int]]]] = []
    cursor = 0
    for row, top30 in qdata:
        k = len(top30)
        ranked = sorted(zip(top30, rr_scores[cursor:cursor + k]),
                        key=lambda t: t[1], reverse=True)
        top5_pairs = [(c_text, meta)
                      for (c_text, meta), _score in ranked[:RERANK_TOP_K]]
        top5list.append((row, [(c_text, int(meta["page_number"]))
                               for c_text, meta in top5_pairs]))
        cursor += k

    # stage 2: batched cross-encoder answer support on (golden answer, top-5 chunk)
    as_pairs: list[tuple[str, str]] = []
    for row, t5 in top5list:
        for c_text, _p in t5:
            as_pairs.append((row["answer"], c_text))
    as_scores = reranker.predict(
        as_pairs, batch_size=XENT_BATCH, show_progress_bar=len(as_pairs) > 400
    )

    # stage 3: deterministic signals + checkpointed rows
    written = 0
    cursor = 0
    out_f = open(SCORES_FILE, "a", encoding="utf-8")
    try:
        for row, t5 in top5list:
            ctx_texts = [c_text for c_text, _p in t5]
            p_tokens = content_tokens(row.get("passage_text", ""))
            chunk_token_sets = [content_tokens(t) for t in ctx_texts]
            if p_tokens:
                max_ov = max(len(p_tokens & cs) / len(p_tokens)
                             for cs in chunk_token_sets)
                union = set().union(*chunk_token_sets)
                union_ov = len(p_tokens & union) / len(p_tokens)
            else:
                max_ov = union_ov = 0.0
            ans_nums = numeric_tokens(row["answer"])
            ctx_nums = numeric_tokens(" ".join(ctx_texts))
            ncov = 1.0 if not ans_nums else len(ans_nums & ctx_nums) / len(ans_nums)
            raw_sup = max(_sigmoid(s)
                          for s in as_scores[cursor:cursor + len(t5)])
            cursor += len(t5)
            support = min(raw_sup, 0.5) if ncov < 1.0 else raw_sup

            rec = {
                "pdf_id": pdf_id,
                "strategy": strategy,
                "question": row["question"],
                "source_passage_id": row["source_passage_id"],
                "retrieval_recall": int(max_ov >= OVERLAP_HIT_THRESHOLD),
                "max_chunk_overlap": round(max_ov, 4),
                "union_overlap": round(union_ov, 4),
                "numeric_coverage": round(ncov, 4),
                "answer_support": round(support, 4),
                "answer_support_raw": round(raw_sup, 4),
                "retrieved_pages": [p for _t, p in t5],
                "score_ok": True,
            }
            out_f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            out_f.flush()
            done.add((pdf_id, strategy, row["question"]))
            written += 1
    finally:
        out_f.close()
    return written


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_score(args) -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    qa = load_layer1()
    pdfs = args.pdf or sorted(qa.keys())
    strategies = args.strategy or STRATEGIES
    done = load_done()

    print(f"loading embedder {EMBEDDING_MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"loading reranker {RERANKER_MODEL} ...", flush=True)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    total_new = 0
    for pdf_id in pdfs:
        questions = qa.get(pdf_id)
        if not questions:
            print(f"[warn] no Layer 1 rows for {pdf_id!r} — skipped", flush=True)
            continue
        for strategy in strategies:
            t0 = time.time()
            n = score_pdf_strategy(
                pdf_id, strategy, questions, model, reranker, done, args.limit
            )
            total_new += n
            print(f"[done] {pdf_id} x {strategy}: {n} new rows "
                  f"({len(questions)} questions) in {time.time() - t0:.0f}s",
                  flush=True)
    print(f"[done] {total_new} new (pdf, strategy, question) rows "
          f"-> {SCORES_FILE.name}", flush=True)
    return 0


def cmd_aggregate(_args) -> int:
    scores: list[dict] = []
    with open(SCORES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("score_ok"):
                    scores.append(r)

    qa = load_layer1()
    feats = load_layer0()

    by_ps: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scores:
        by_ps[(r["pdf_id"], r["strategy"])].append(r)

    now = datetime.now(timezone.utc).isoformat()
    n_clear = n_amb = 0
    win_counts: dict[str, int] = defaultdict(int)
    near_ties: list[tuple[str, float, str]] = []
    frag_tax: dict[str, list[float]] = defaultdict(list)

    with open(GOLDEN_FILE, "w", encoding="utf-8") as out:
        for pdf_id in sorted(qa.keys()):
            per: dict[str, dict] = {}
            for strategy in STRATEGIES:
                rows = by_ps.get((pdf_id, strategy))
                if not rows:
                    continue
                rec = sum(r["retrieval_recall"] for r in rows) / len(rows)
                ncov = sum(r["numeric_coverage"] for r in rows) / len(rows)
                sup = sum(r["answer_support"] for r in rows) / len(rows)
                per[strategy] = {
                    "retrieval_recall": round(rec, 4),
                    "numeric_coverage": round(ncov, 4),
                    "answer_support": round(sup, 4),
                    "combined": round(0.5 * rec + 0.5 * sup, 4),
                    "_n": len(rows),
                }
                for r in rows:
                    frag_tax[strategy].append(
                        1.0 if (r["union_overlap"] >= 0.8
                                and r["retrieval_recall"] == 0) else 0.0)
            if not per:
                continue

            ranked = sorted(per.items(),
                            key=lambda kv: kv[1]["combined"], reverse=True)
            top_name, top = ranked[0]
            second = ranked[1][1]["combined"] if len(ranked) > 1 else 0.0
            margin = ((top["combined"] - second) / top["combined"]
                      if top["combined"] > 0 else 0.0)
            ambiguous = margin < MARGIN_THRESHOLD
            if ambiguous:
                n_amb += 1
            else:
                n_clear += 1
                win_counts[top_name] += 1
            if margin < NEAR_TIE_TIEBREAK:
                near_ties.append((pdf_id, round(margin, 4), top_name))

            golden = {
                "pdf_id": pdf_id,
                "pdf_features": {
                    k: feats.get(pdf_id, {}).get(k)
                    for k in ("page_count", "doc_type", "table_density",
                              "column_layout", "ocr_quality",
                              "avg_paragraph_length")
                },
                "golden_chunking_strategy": None if ambiguous else {
                    "method": top_name,
                    "impl": "evaluation/chunking_strategies.py",
                    "params": STRATEGY_PARAMS[top_name],
                },
                "label_status": "ambiguous" if ambiguous else "clear",
                "winning_margin": round(margin, 4),
                "strategy_scores": {
                    s: {"retrieval_recall": v["retrieval_recall"],
                        "numeric_coverage": v["numeric_coverage"],
                        "answer_support": v["answer_support"],
                        "combined": v["combined"],
                        "n_questions": v["_n"]}
                    for s, v in per.items()
                },
                "layer1_eval_set_ref": f"evaluation/layer1_qa.jsonl#{pdf_id}",
                "generation_model": ("generator+judge=assistant-in-session "
                                     "(Qwen Code agent); "
                                     f"embed={EMBEDDING_MODEL_NAME}; "
                                     f"reranker={RERANKER_MODEL}"),
                "reviewed_by": "none",
                "created_at": now,
            }
            out.write(json.dumps(golden, ensure_ascii=True) + "\n")

    print(f"golden labels: {n_clear} clear / {n_amb} ambiguous "
          f"-> {GOLDEN_FILE.name}")
    print("winners:", dict(win_counts))
    print(f"near-tie PDFs (margin < {NEAR_TIE_TIEBREAK} — candidates for the "
          f"in-session judge tiebreak):")
    for pdf_id, margin, top_name in near_ties:
        print(f"  {pdf_id}: margin={margin} provisional winner {top_name}")
    if any(frag_tax.values()):
        print("fragmentation tax (union_overlap>=0.8 but recall=0, share of "
              "scored questions):")
        for s in STRATEGIES:
            if frag_tax[s]:
                print(f"  {s}: {sum(frag_tax[s]) / len(frag_tax[s]):.3f}")
    return 0


def cmd_status(_args) -> int:
    qa = load_layer1()
    n_q = sum(len(v) for v in qa.values())
    expected = n_q * len(STRATEGIES)
    if not SCORES_FILE.exists():
        print(f"no scores yet (expected total: {expected} rows = "
              f"{n_q} questions x {len(STRATEGIES)} strategies)")
        return 0
    rows = [json.loads(l) for l in
            open(SCORES_FILE, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("score_ok")]
    done_pairs = {(r["pdf_id"], r["strategy"]) for r in ok}
    n_full_pdfs = sum(
        1 for pid in qa
        if sum(1 for s in STRATEGIES if (pid, s) in done_pairs)
        == len(STRATEGIES))
    print(f"score rows: {len(ok)} ok / {len(rows) - len(ok)} bad "
          f"of {expected} expected")
    print(f"completed (pdf, strategy) pairs: {len(done_pairs)} "
          f"of {len(qa) * len(STRATEGIES)}")
    print(f"PDFs fully scored (all {len(STRATEGIES)} strategies): "
          f"{n_full_pdfs} of {len(qa)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score",
                        help="score (pdf x strategy x question) triples")
    sp.add_argument("--pdf", nargs="*", default=None,
                    help="limit to these pdf_ids")
    sp.add_argument("--strategy", nargs="*", default=None,
                    choices=STRATEGIES)
    sp.add_argument("--limit", type=int, default=None,
                    help="only first N questions per PDF (smoke testing)")
    sub.add_parser("aggregate", help="build layer2_golden.jsonl from scores")
    sub.add_parser("status", help="show scoring progress")
    args = p.parse_args()
    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
