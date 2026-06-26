"""Factory for creating retriever instances — caches loaded indexes."""

from services.retrieval.representations import Retriever, TFIDFRetriever, BM25Retriever, DenseRetriever

_cache: dict[str, Retriever] = {}


def get_retriever(dataset: str, repr_type: str) -> Retriever:
    """Get or create a retriever instance (cached per dataset+type)."""
    key = f"{dataset}_{repr_type}"
    if key not in _cache:
        if repr_type == "tfidf":
            _cache[key] = TFIDFRetriever(dataset)
        elif repr_type == "bm25":
            _cache[key] = BM25Retriever(dataset)
        elif repr_type == "dense":
            _cache[key] = DenseRetriever(dataset)
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}")
    return _cache[key]
