"""Features service — bonus IR features.

Serves document clustering (#15) and topic detection (#17) from the offline
artifacts built by `services.features.clustering`. Loads cluster labels +
topic metadata per dataset (cached) and hydrates sample docs from the MongoDB
raw store, same as the retrieval service.
"""

import json
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException

from common.config import settings, DATASET_REGISTRY
from common.schemas import (
    ClusterInfo, ClustersResponse, ClusterDocsResponse, DocClusterResponse,
    ScoredDoc, HealthResponse,
)
from common.logging import get_logger
from common import doc_db

logger = get_logger("features")
app = FastAPI(title="Features Service", version="1.0.0")

# dataset -> {"labels": np.ndarray, "doc_ids": list[str], "meta": dict}
_cache: dict[str, dict] = {}


def _load(dataset: str) -> dict:
    if dataset not in DATASET_REGISTRY:
        raise HTTPException(404, f"Unknown dataset: {dataset}")
    if dataset not in _cache:
        index_path = settings.index_dir / dataset
        meta_path = index_path / "cluster_meta.json"
        labels_path = index_path / "clusters.npy"
        if not meta_path.exists() or not labels_path.exists():
            raise HTTPException(
                503,
                f"Clusters not built for '{dataset}'. Run `make build-clusters`.",
            )
        logger.info(f"Loading clusters for {dataset}...")
        labels = np.load(labels_path)
        with open(index_path / "doc_ids.pkl", "rb") as f:
            doc_ids = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        _cache[dataset] = {"labels": labels, "doc_ids": doc_ids, "meta": meta}
    return _cache[dataset]


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="features")


@app.get("/clusters/{dataset}", response_model=ClustersResponse)
async def list_clusters(dataset: str):
    """All clusters for a dataset with their detected topic + size (#15, #17)."""
    data = _load(dataset)
    meta = data["meta"]
    clusters = [
        ClusterInfo(
            cluster_id=cid,
            topic=meta["topics"].get(str(cid), []),
            size=meta["sizes"].get(str(cid), 0),
        )
        for cid in range(meta["n_clusters"])
    ]
    clusters.sort(key=lambda c: c.size, reverse=True)
    return ClustersResponse(dataset=dataset, n_clusters=meta["n_clusters"], clusters=clusters)


@app.get("/cluster/{dataset}/{cluster_id}", response_model=ClusterDocsResponse)
async def cluster_docs(dataset: str, cluster_id: int, limit: int = 10):
    """Sample documents from one cluster (titles hydrated from the raw store)."""
    data = _load(dataset)
    meta = data["meta"]
    if cluster_id < 0 or cluster_id >= meta["n_clusters"]:
        raise HTTPException(404, f"cluster_id out of range (0..{meta['n_clusters'] - 1})")

    member_idx = np.where(data["labels"] == cluster_id)[0][:limit]
    ids = [data["doc_ids"][i] for i in member_idx]
    raw = doc_db.fetch_docs(dataset, ids)
    docs = [
        ScoredDoc(
            doc_id=did,
            score=0.0,
            rank=rank + 1,
            title=raw.get(did, {}).get("title", ""),
            text=raw.get(did, {}).get("text", ""),
            dataset=dataset,
        )
        for rank, did in enumerate(ids)
    ]
    return ClusterDocsResponse(
        dataset=dataset,
        cluster_id=cluster_id,
        topic=meta["topics"].get(str(cluster_id), []),
        docs=docs,
    )


@app.get("/doc_cluster/{dataset}/{doc_id}", response_model=DocClusterResponse)
async def doc_cluster(dataset: str, doc_id: str):
    """Which cluster (and topic) a given document belongs to."""
    data = _load(dataset)
    try:
        idx = data["doc_ids"].index(doc_id)
    except ValueError:
        raise HTTPException(404, f"Document {doc_id} not found in {dataset}")
    cid = int(data["labels"][idx])
    return DocClusterResponse(
        dataset=dataset,
        doc_id=doc_id,
        cluster_id=cid,
        topic=data["meta"]["topics"].get(str(cid), []),
    )
