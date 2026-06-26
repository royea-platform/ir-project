"""Retriever interface and implementations — Strategy pattern.

Each retriever scores the cleaned/embedding index to get top-k doc_ids, then
hydrates the ORIGINAL document text from MongoDB (the raw doc store). Original
text — never the stemmed index text — is what gets returned to the user.
"""

from __future__ import annotations
import pickle
from typing import Protocol

import numpy as np
from scipy.sparse import load_npz

from common.config import settings
from common.schemas import ScoredDoc
from common.logging import get_logger
from common import doc_db

logger = get_logger("retrieval.representations")


def _hydrate(dataset: str, scored: list[tuple[str, float]], top_k: int) -> list[ScoredDoc]:
    """Map (doc_id, score) pairs to ScoredDoc with ORIGINAL text from MongoDB."""
    ids = [doc_id for doc_id, _ in scored]
    raw = doc_db.fetch_docs(dataset, ids)
    results = []
    for rank, (doc_id, score) in enumerate(scored[:top_k]):
        info = raw.get(doc_id, {})
        results.append(ScoredDoc(
            doc_id=doc_id,
            score=float(score),
            rank=rank + 1,
            title=info.get("title", ""),
            text=info.get("text", ""),
        ))
    return results


class Retriever(Protocol):
    """Common interface for all retrieval strategies."""

    def search(
        self,
        query_tokens: list[str],
        query_text: str,
        top_k: int = 10,
        **kwargs,
    ) -> list[ScoredDoc]: ...


class TFIDFRetriever:
    def __init__(self, dataset: str):
        self.dataset = dataset
        index_path = settings.index_dir / dataset
        logger.info(f"Loading TF-IDF index for {dataset}...")
        self.tfidf_matrix = load_npz(index_path / "tfidf_matrix.npz")
        with open(index_path / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(index_path / "doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)
        logger.info(f"TF-IDF index loaded for {dataset}")

    def search(self, query_tokens: list[str], query_text: str = "", top_k: int = 10, **kwargs) -> list[ScoredDoc]:
        query_str = " ".join(query_tokens)
        query_vec = self.vectorizer.transform([query_str])

        # Cosine similarity (TF-IDF vectors are L2-normalized by sklearn)
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]

        scored = [(self.doc_ids[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        return _hydrate(self.dataset, scored, top_k)


class BM25Retriever:
    def __init__(self, dataset: str):
        import bm25s
        self.dataset = dataset
        index_path = settings.index_dir / dataset
        logger.info(f"Loading BM25 index for {dataset}...")
        self.retriever = bm25s.BM25.load(str(index_path / "bm25_index"), load_corpus=False)
        with open(index_path / "doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)
        logger.info(f"BM25 index loaded for {dataset}")

    def search(self, query_tokens: list[str], query_text: str = "", top_k: int = 10, **kwargs) -> list[ScoredDoc]:
        import bm25s
        query_array = bm25s.tokenize([" ".join(query_tokens)], stemmer=None)
        doc_indices, scores = self.retriever.retrieve(query_array, k=min(top_k, len(self.doc_ids)))

        scored = [
            (self.doc_ids[idx], score)
            for idx, score in zip(doc_indices[0], scores[0]) if score > 0
        ]
        return _hydrate(self.dataset, scored, top_k)


class DenseRetriever:
    def __init__(self, dataset: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.dataset = dataset
        index_path = settings.index_dir / dataset
        logger.info(f"Loading FAISS index for {dataset}...")
        self.index = faiss.read_index(str(index_path / "faiss.index"))
        with open(index_path / "doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)
        self.model = SentenceTransformer(settings.embedding_model)
        logger.info(f"FAISS index loaded for {dataset}")

    def search(self, query_tokens: list[str], query_text: str = "", top_k: int = 10, **kwargs) -> list[ScoredDoc]:
        # Use original query text for embedding (not stemmed tokens)
        text = query_text if query_text else " ".join(query_tokens)
        query_emb = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)

        scored = [
            (self.doc_ids[idx], score)
            for idx, score in zip(indices[0], scores[0]) if idx >= 0
        ]
        return _hydrate(self.dataset, scored, top_k)

    def rerank(self, query_text: str, candidates: list[ScoredDoc], top_k: int = 10) -> list[ScoredDoc]:
        """Rerank a candidate set using dense embeddings (for serial hybrid)."""
        query_emb = self.model.encode([query_text], normalize_embeddings=True).astype("float32")

        texts = [c.text for c in candidates]
        doc_embs = self.model.encode(texts, normalize_embeddings=True).astype("float32")

        scores = (doc_embs @ query_emb.T).ravel()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            c = candidates[idx]
            results.append(ScoredDoc(
                doc_id=c.doc_id,
                score=float(scores[idx]),
                rank=rank + 1,
                title=c.title,
                text=c.text,
            ))
        return results
