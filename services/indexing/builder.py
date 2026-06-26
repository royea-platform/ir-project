"""Index builder — constructs TF-IDF, BM25, FAISS indexes and the raw doc store.

Offline phase only. Cleaned/stemmed text feeds the lexical models (TF-IDF, BM25);
the *original* text feeds the embedding model and is persisted to MongoDB for
retrieval-time hydration. Nothing here runs during online query serving.
"""

import pickle
from pathlib import Path

import numpy as np
import ir_datasets
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from common.config import settings, DATASET_REGISTRY
from common.logging import get_logger
from common import doc_db
from services.preprocessing.pipeline import preprocess_text

logger = get_logger("indexing.builder")


def _get_index_path(dataset: str) -> Path:
    path = settings.index_dir / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def _doc_raw_text(doc) -> tuple[str, str]:
    """Return (title, original_text) for a corpus doc."""
    title = doc.title if hasattr(doc, "title") and doc.title else ""
    return title, doc.text


def load_corpus(dataset: str) -> tuple[list[str], list[str], list[str]]:
    """Single pass over the corpus.

    Returns (doc_ids, cleaned_texts, embed_texts):
      - cleaned_texts : stemmed/normalized tokens joined — for TF-IDF & BM25
      - embed_texts   : original title+text — for embeddings (NOT stemmed)
    """
    ds_info = DATASET_REGISTRY[dataset]
    corpus = ir_datasets.load(ds_info["corpus_key"])

    doc_ids: list[str] = []
    cleaned_texts: list[str] = []
    embed_texts: list[str] = []

    logger.info(f"Loading + preprocessing corpus for {dataset} ({corpus.docs_count()} docs)...")
    for i, doc in enumerate(corpus.docs_iter()):
        title, text = _doc_raw_text(doc)
        raw = (title + " " + text).strip() if title else text
        tokens = preprocess_text(raw, language=ds_info["language"])

        doc_ids.append(doc.doc_id)
        cleaned_texts.append(" ".join(tokens))
        embed_texts.append(raw)

        if (i + 1) % 50000 == 0:
            logger.info(f"  Preprocessed {i + 1} docs...")

    logger.info(f"  Done: {len(doc_ids)} docs.")
    return doc_ids, cleaned_texts, embed_texts


def ingest_raw_docs(dataset: str) -> int:
    """Stream original documents into MongoDB (raw doc store). Standalone."""
    ds_info = DATASET_REGISTRY[dataset]
    corpus = ir_datasets.load(ds_info["corpus_key"])
    logger.info(f"Ingesting raw docs for {dataset} into MongoDB...")

    batch: list[dict] = []
    total = 0
    for doc in corpus.docs_iter():
        title, text = _doc_raw_text(doc)
        batch.append({"doc_id": doc.doc_id, "title": title, "text": text})
        if len(batch) >= 5000:
            total += doc_db.store_docs(dataset, batch)
            batch = []
            if total % 50000 == 0:
                logger.info(f"  [{dataset}] stored {total} raw docs...")
    if batch:
        total += doc_db.store_docs(dataset, batch)
    logger.info(f"  [{dataset}] raw doc store complete: {total} docs")
    return total


def build_tfidf_index(doc_ids: list[str], texts: list[str], index_path: Path):
    """Build and save TF-IDF vectorizer + document matrix (cleaned text)."""
    logger.info("  Building TF-IDF index...")
    vectorizer = TfidfVectorizer(max_features=100000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts)

    save_npz(index_path / "tfidf_matrix.npz", tfidf_matrix)
    with open(index_path / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(index_path / "doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    logger.info(f"  TF-IDF index saved: {tfidf_matrix.shape}")


def build_bm25_index(doc_ids: list[str], texts: list[str], index_path: Path):
    """Build and save BM25 index (cleaned text) using bm25s."""
    import bm25s
    logger.info("  Building BM25 index...")
    corpus_tokens = [text.split() for text in texts]

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(index_path / "bm25_index"))
    with open(index_path / "bm25_corpus_tokens.pkl", "wb") as f:
        pickle.dump(corpus_tokens, f)
    logger.info("  BM25 index saved")


def build_faiss_index(embed_texts: list[str], index_path: Path):
    """Build sentence embeddings + FAISS index from ORIGINAL (non-stemmed) text."""
    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info(f"  Building FAISS index with model {settings.embedding_model}...")
    model = SentenceTransformer(settings.embedding_model)

    logger.info("  Encoding original documents (this may take a while)...")
    batch_size = 512
    all_embeddings = []
    for i in range(0, len(embed_texts), batch_size):
        batch = embed_texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(emb)
        if (i + batch_size) % 10000 == 0:
            logger.info(f"    Encoded {min(i + batch_size, len(embed_texts))}/{len(embed_texts)} docs")

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    dim = embeddings_matrix.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    index.add(embeddings_matrix)

    faiss.write_index(index, str(index_path / "faiss.index"))
    np.save(index_path / "embeddings.npy", embeddings_matrix)
    logger.info(f"  FAISS index saved: {embeddings_matrix.shape}")


def rebuild_faiss_only(dataset: str):
    """Rebuild ONLY the FAISS index on original text (TF-IDF/BM25 untouched)."""
    index_path = _get_index_path(dataset)
    _, _, embed_texts = load_corpus(dataset)
    build_faiss_index(embed_texts, index_path)


def build_all_indexes(dataset: str):
    """Build all index types + raw doc store for a dataset (full offline build)."""
    index_path = _get_index_path(dataset)
    logger.info(f"Building all indexes for '{dataset}' at {index_path}")

    doc_ids, cleaned_texts, embed_texts = load_corpus(dataset)

    ingest_raw_docs(dataset)                               # raw docs → MongoDB
    build_tfidf_index(doc_ids, cleaned_texts, index_path)  # cleaned text
    build_bm25_index(doc_ids, cleaned_texts, index_path)   # cleaned text

    # Free the cleaned-text list before embedding — keeps peak memory down.
    del cleaned_texts
    import gc; gc.collect()

    build_faiss_index(embed_texts, index_path)             # original text

    logger.info(f"All indexes built for '{dataset}'")
