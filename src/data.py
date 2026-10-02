from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load(name="scifact"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    path = util.download_and_unzip(url, "data")
    corpus, queries, qrels = GenericDataLoader(path).load(split="test")
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]
    return doc_ids, doc_texts, queries, qrels
