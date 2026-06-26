"""Offline build for document clustering (#15) + topic detection (#17).

Reuses artifacts already produced by the indexing offline phase:
  - embeddings.npy        → KMeans clustering (semantic groups)
  - tfidf_matrix.npz      → per-cluster top terms = the cluster's topic label
  - tfidf_vectorizer.pkl  → vocabulary for those terms
  - doc_ids.pkl           → align cluster labels back to corpus doc_ids

Nothing here runs at query time. Outputs:
  - clusters.npy          → int cluster label per doc (aligned to doc_ids order)
  - cluster_meta.json     → {n_clusters, topics:{cid:[terms]}, sizes:{cid:n}}
"""

import json
import pickle

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans

from common.config import settings
from common.logging import get_logger

logger = get_logger("features.clustering")


def _top_terms_per_cluster(tfidf_matrix, labels, vocab, n_clusters, topic_terms):
    """Mean TF-IDF weight per term within each cluster → highest-weighted terms."""
    topics: dict[str, list[str]] = {}
    for cid in range(n_clusters):
        rows = np.where(labels == cid)[0]
        if rows.size == 0:
            topics[str(cid)] = []
            continue
        # Mean TF-IDF vector over the cluster's docs (sparse → dense 1×V).
        centroid = np.asarray(tfidf_matrix[rows].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[::-1][:topic_terms]
        topics[str(cid)] = [vocab[i] for i in top_idx if centroid[i] > 0]
    return topics


def build_clusters(dataset: str) -> dict:
    """Cluster a dataset's documents and label each cluster with a topic."""
    index_path = settings.index_dir / dataset
    n_clusters = settings.n_clusters

    logger.info(f"[{dataset}] loading embeddings + tfidf for clustering...")
    embeddings = np.load(index_path / "embeddings.npy")
    tfidf_matrix = load_npz(index_path / "tfidf_matrix.npz")
    with open(index_path / "tfidf_vectorizer.pkl", "rb") as f:
        vocab = pickle.load(f).get_feature_names_out()
    with open(index_path / "doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    logger.info(f"[{dataset}] KMeans on {embeddings.shape[0]} docs → {n_clusters} clusters...")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=4096)
    labels = km.fit_predict(embeddings)

    logger.info(f"[{dataset}] extracting topic terms per cluster...")
    topics = _top_terms_per_cluster(tfidf_matrix, labels, vocab, n_clusters, settings.topic_terms)
    sizes = {str(cid): int((labels == cid).sum()) for cid in range(n_clusters)}

    np.save(index_path / "clusters.npy", labels.astype("int32"))
    meta = {"n_clusters": n_clusters, "topics": topics, "sizes": sizes, "n_docs": len(doc_ids)}
    with open(index_path / "cluster_meta.json", "w") as f:
        json.dump(meta, f)

    logger.info(f"[{dataset}] clustering done. Sizes: {sizes}")
    return meta
