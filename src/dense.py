import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Dense:
    def __init__(self, doc_ids, doc_texts, cache="cache/doc_emb.npy"):
        self.doc_ids = doc_ids
        self.model = SentenceTransformer(MODEL)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if os.path.exists(cache):
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(
                doc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
            )
            np.save(cache, self.emb)

    def search(self, query, k=100):
        q = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.emb @ q
        top = scores.argsort()[::-1][:k]
        return [self.doc_ids[i] for i in top]
