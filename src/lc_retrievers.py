"""LangChain wrappers around the from-scratch retrievers, used only to benchmark
LangChain's EnsembleRetriever against our own RRF implementation."""
from typing import Any, Dict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

try:  # langchain >= 1.0
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # langchain < 1.0
    from langchain.retrievers import EnsembleRetriever


class EngineRetriever(BaseRetriever):
    """Wraps any object with .search(query, k) -> list[doc_id] as a LangChain retriever."""

    engine: Any
    texts: Dict[str, str]
    k: int = 100
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        ids = self.engine.search(query, self.k)
        return [Document(page_content=self.texts[i], metadata={"doc_id": i}) for i in ids]


def build_ensemble(bm25, dense, doc_ids, doc_texts, k=100):
    texts = dict(zip(doc_ids, doc_texts))
    return EnsembleRetriever(
        retrievers=[
            EngineRetriever(engine=bm25, texts=texts, k=k),
            EngineRetriever(engine=dense, texts=texts, k=k),
        ],
        weights=[0.5, 0.5],
        c=60,
        id_key="doc_id",  # dedupe by doc id, not by page content
    )
