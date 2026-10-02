from collections import defaultdict


def rrf(rank_lists, k=60, top=100):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores = defaultdict(float)
    for lst in rank_lists:
        for rank, doc in enumerate(lst):
            scores[doc] += 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])][:top]
