# Agentic RAG

Hybrid RAG over a mixed corpus of research papers and legal contracts.

## Overview

- Question answering over a corpus of 35 PDFs split between academic papers and legal agreements.
- Combines **BM25 sparse retrieval** and **dense vector retrieval**, fuses the two rankings with Reciprocal Rank Fusion (RRF), and sends the top matched chunks to an LLM via OpenRouter.
- Why hybrid: BM25 is stronger on exact terms (clause names, defined terms, citations); dense embeddings catch semantics (a sentence about "termination notice" matches a question about "how to end the agreement").

## Pipeline

1. **Ingestion** — `ingestion/extract.py` parses each PDF with `pymupdf4llm` and stores per-page Markdown plus HTML tables. `ingestion/chunk.py` then splits prose with `RecursiveCharacterTextSplitter` using a 450-token target and **no overlap**. HTML tables are parsed with BeautifulSoup, expanded into a rectangular grid, and split into row groups of at most 20 rows / 400 BGE tokens.
2. **Indexing** — `ingestion/index.py` embeds each chunk with `BAAI/bge-small-en-v1.5` (384-dim, cosine) and a `Qdrant/bm25` sparse model, then stores both vectors in a local Qdrant collection. Each point carries `doc_id`, `page`, `chunk_type`, heading, chunk text, `token_count`, and original table HTML.
3. **Retrieval** — `rag/retriever.py` runs dense + BM25 prefetches, each configured with `TOP_K_CANDIDATES=50`, then fuses results with RRF.
4. **Selection** — `rag/pipeline.py` takes the top `CONTEXT_K=8` fused hits and passes them to the generator. There is **no reranker** in the current code path.
5. **Generation** — `rag/generator.py` calls OpenRouter (`openrouter/auto` first, then fetched free models), builds a grounded prompt from retrieved `Hit` chunks, and instructs the model to cite `[source: doc_id, p.N]` for every claim.

## Tech Stack

| Purpose | Library |
|---|---|
| PDF parsing / chunking | `pymupdf==1.28.2` |
| Text splitting | `langchain-text-splitters==0.3.9` |
| Token counting | `tiktoken==0.14.0` |
| HTML table parsing | `beautifulsoup4==4.15.0` |
| Dense/sparse embedding | `fastembed==0.9.0` |
| Vector store client | `qdrant-client==1.19.1` |
| LLM requests | `requests==2.34.2` |
| Config loading | `python-dotenv==1.2.3` |
| Tokenizer backend | `transformers==5.16.1` |

## Project Structure

```
agentic_rag/
├── query.py                  # CLI entry point: ask one question
├── rag/
│   ├── config.py              # env-driven settings
│   ├── retriever.py           # dense + BM25 prefetch and RRF fusion
│   ├── pipeline.py            # retrieval → generation pipeline
│   └── generator.py           # OpenRouter synthesis and fallback chain
├── ingestion/
│   ├── extract.py             # PDF-to-Markdown parser
│   ├── chunk.py               # prose and table chunking
│   ├── index.py               # Qdrant upsert
│   └── run.py                 # ingestion orchestration
├── requirements.txt
└── storage/                   # Qdrant collection on disk
```

## Setup

```bash
git clone <repo_url>
cd agentic_rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows only**: `.venv\Scripts\activate`. On Linux/macOS use `source .venv/bin/activate`.

**Start Qdrant** on `localhost:6333`:

```bash
docker run -d --name qdrant -p 6333:6333 -v "$PWD/storage:/qdrant/storage" qdrant/qdrant
```

**Configuration**: `.env.example` is read directly by the query pipeline. Set `OPENROUTER_API_KEY` there.

## Usage

**Rebuild the index from the corpus**:

```bash
python ingestion/run.py --dirs academic contract
```

This runs extraction → chunking → indexing and rebuilds the Qdrant collection.

**Ask a question**:

```bash
python query.py "What is the maintenance term in the contract?"
```

Output shape (actual answer and timings depend on the model and corpus):

```
Answer:
  <grounded answer, with citations like [source: doc_id, p.N]>

Sources:
  Rank | Page | Type | Heading
  -----+------+--------+-------------------------------------
     1 | ...  | ...    | ...
        doc: ...

Timings: retrieval X.XXs | generation X.XXs | total X.XXs
Model: <model id>
```

## Design Decisions

- **Why 50 candidates → 8 context chunks**: RRF fusion across dense and BM25 can give high relevance to chunks that neither method ranks #1 alone. Taking 50 fused candidates from each branch gives enough headroom so the final top-8 avoids missing chunks that score high on one dimension but mid on the other.
- **Why hybrid retrieval**: exact legal terms (`WHEREAS`, `Section 4.2`, defined terms like `Company`) are reliably surfaced by BM25, while semantic similarity catches paraphrased or concept-level asks ("how does the contract define delivery"). RRF gives a cheap, effective compromise over both.

## Limitations

- The current corpus is small (35 PDFs), so retrieval behavior may not generalize to new document sets.
- No quantitative evaluation yet — there is no recall@k, hit-rate metric, or strategy comparison.
- No reranker is used to sharpen the final 8 hits.
- OpenRouter free models are rate-limited and may return 429s without notice.

## Future Work

- Add evaluation with a known-answer set: recall@k, hit rate, and hybrid vs BM25-only vs dense-only comparison.
- Add a reranker after RRF to sharpen final top-8 selection.
- Support a larger or externally supplied corpus instead of hardcoded PDFs.

## Notes

- **Corpus source**: PDFs live under `corpus/raw/` inside the repo. No licensing terms were found in the code; **TODO: share source licensing details**.
- **Never commit `.env`**. The API key stays local and out of version control.
- Current code does **not** include document routing or dynamic chunking; chunking is a fixed 450-token / 0-overlap strategy.
