from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, doc_ids, doc_texts):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.text = dict(zip(doc_ids, doc_texts))

    def rerank(self, query, candidates, top=10):
        pairs = [(query, self.text[d]) for d in candidates]
        scores = self.model.predict(pairs, batch_size=32)
        order = scores.argsort()[::-1][:top]
        return [candidates[i] for i in order]
