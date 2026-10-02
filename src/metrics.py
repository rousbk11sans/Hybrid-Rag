import math


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=10):
    rel_docs = {d for d, r in rel.items() if r > 0}
    return len(rel_docs & set(ranked[:k])) / len(rel_docs) if rel_docs else 0.0


def mrr_at_k(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
