from .bm25 import BM25
from .data import load
from .dense import Dense
from .fusion import rrf
from .rerank import Reranker
from .rewrite import rewrite

MODES = ["bm25", "dense", "hybrid", "langchain_ensemble", "hybrid_rerank", "full"]


class Pipeline:
    def __init__(self):
        self.doc_ids, self.doc_texts, self.queries, self.qrels = load()
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self.bm25 = BM25(self.doc_ids, self.doc_texts)
        self.dense = Dense(self.doc_ids, self.doc_texts)
        self.reranker = Reranker(self.doc_ids, self.doc_texts)
        self._lc_ensemble = None

    def run(self, query, mode, k=10):
        if mode == "bm25":
            return self.bm25.search(query, k)
        if mode == "dense":
            return self.dense.search(query, k)
        if mode == "hybrid":
            return rrf([self.bm25.search(query), self.dense.search(query)], top=k)
        if mode == "langchain_ensemble":  # LangChain EnsembleRetriever baseline
            if self._lc_ensemble is None:
                from .lc_retrievers import build_ensemble
                self._lc_ensemble = build_ensemble(
                    self.bm25, self.dense, self.doc_ids, self.doc_texts
                )
            docs = self._lc_ensemble.invoke(query)
            return [d.metadata["doc_id"] for d in docs][:k]
        if mode == "hybrid_rerank":
            cands = rrf([self.bm25.search(query), self.dense.search(query)], top=50)
            return self.reranker.rerank(query, cands, top=k)
        if mode == "full":  # rewrite + hybrid + rerank
            q2 = rewrite(query)
            lists = [
                self.bm25.search(query), self.dense.search(query),
                self.bm25.search(q2), self.dense.search(q2),
            ]
            cands = rrf(lists, top=50)
            return self.reranker.rerank(query, cands, top=k)
        raise ValueError(f"unknown mode: {mode}")
