# Hybrid RAG Search Engine

Hybrid retrieval pipeline: **BM25 + dense (BGE) → Reciprocal Rank Fusion → cross-encoder rerank → LLM query rewriting**, evaluated by ablation on BEIR SciFact.

```
query ──► (rewrite) ──► BM25 ──┐
                               ├─► RRF ─► cross-encoder ─► top-k
                       Dense ──┘
```

## Results (BEIR SciFact, test split)

| method             | nDCG@10 | Recall@10 | MRR@10 |
|--------------------|---------|-----------|--------|
| bm25               | 0.6519  | 0.7740    | 0.6186 |
| dense (BGE-small)  | **0.7127** | **0.8362** | **0.6822** |
| hybrid (own RRF)   | 0.7085  | 0.8149    | 0.6804 |
| langchain_ensemble | 0.7085  | 0.8149    | 0.6804 |
| hybrid_rerank      | 0.6965  | 0.8289    | 0.6642 |

**Analysis:**
- Dense retrieval beats BM25 by about 6 nDCG points (0.652 → 0.713), as SciFact claims are paraphrased relative to the abstracts.
- Hybrid fusion did not beat dense alone (0.7085 vs 0.7127): BM25 is the weaker retriever here, so equal-weight RRF pulls the ranking slightly down.
- LangChain's `EnsembleRetriever` matches the from-scratch RRF exactly on every metric, which validates the implementation in `src/fusion.py`.
- The cross-encoder reranker lowered nDCG (0.7085 → 0.6965). Likely cause: `ms-marco-MiniLM` is trained on web-search question/passage pairs, not scientific claim verification. A domain-matched reranker would be the next thing to try.
- Query rewriting was not evaluated.

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
