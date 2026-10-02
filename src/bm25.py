import re
from rank_bm25 import BM25Okapi


def tokenize(t):
    return re.findall(r"\w+", t.lower())


class BM25:
    def __init__(self, doc_ids, doc_texts):
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi([tokenize(t) for t in doc_texts])

    def search(self, query, k=100):
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [self.doc_ids[i] for i in top]
