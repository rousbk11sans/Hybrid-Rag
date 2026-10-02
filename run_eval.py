import sys

import pandas as pd
from tqdm import tqdm

from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.pipeline import MODES, Pipeline

modes = sys.argv[1:] or MODES
p = Pipeline()
rows = []
for m in modes:
    nd, rc, mr = [], [], []
    for qid, q in tqdm(p.queries.items(), desc=m):
        if qid not in p.qrels:
            continue
        ranked = p.run(q, m, k=10)
        rel = p.qrels[qid]
        nd.append(ndcg_at_k(ranked, rel))
        rc.append(recall_at_k(ranked, rel))
        mr.append(mrr_at_k(ranked, rel))
    rows.append({
        "method": m,
        "nDCG@10": sum(nd) / len(nd),
        "Recall@10": sum(rc) / len(rc),
        "MRR@10": sum(mr) / len(mr),
    })

df = pd.DataFrame(rows).round(4)
df.to_csv("results/ablation.csv", index=False)
print(df.to_markdown(index=False))
