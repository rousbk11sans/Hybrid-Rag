# Hybrid RAG Search Engine

Hybrid retrieval pipeline: **BM25 + dense (BGE) → Reciprocal Rank Fusion → cross-encoder rerank → LLM query rewriting**, evaluated by ablation on BEIR SciFact.

```
query ──► (rewrite) ──► BM25 ──┐
                               ├─► RRF ─► cross-encoder ─► top-k
                       Dense ──┘
```

## Results (BEIR SciFact, test split)

| method        | nDCG@10 | Recall@10 | MRR@10 |
|---------------|---------|-----------|--------|
| bm25          | TODO    | TODO      | TODO   |
| dense         | TODO    | TODO      | TODO   |
| hybrid (own RRF) | TODO | TODO      | TODO   |
| langchain_ensemble | TODO | TODO   | TODO   |
| hybrid_rerank | TODO    | TODO      | TODO   |
| full          | TODO    | TODO      | TODO   |

**Analysis:** TODO (which stage helped most, where BM25 beat dense, whether rewriting helped).

## LangChain comparison

`langchain_ensemble` runs LangChain's `EnsembleRetriever` (weighted RRF, c=60) over the same BM25 and dense
retrievers, wrapped as `BaseRetriever` classes (`src/lc_retrievers.py`). It serves as a baseline to check the
from-scratch RRF in `src/fusion.py`; with equal weights the rankings should match.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add ANTHROPIC_API_KEY (only needed for the "full" mode)
python run_eval.py                  # all methods
python run_eval.py bm25 dense hybrid hybrid_rerank   # skip rewriting (no API key)
streamlit run app.py
```

## Stack

rank_bm25, LangChain (comparison baseline only), sentence-transformers (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`), Claude Haiku 4.5 for rewriting, Streamlit.

## Layout

```
src/   data, bm25, dense, fusion, rerank, rewrite, lc_retrievers, metrics, pipeline
run_eval.py   ablation over all methods -> results/ablation.csv
app.py        side-by-side demo UI
```
