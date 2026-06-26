"""Application configuration loaded from environment / .env file."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service URLs
    gateway_url: str = "http://localhost:8000"
    preprocessing_url: str = "http://localhost:8001"
    indexing_url: str = "http://localhost:8002"
    retrieval_url: str = "http://localhost:8003"
    ranking_eval_url: str = "http://localhost:8004"
    query_refinement_url: str = "http://localhost:8005"
    features_url: str = "http://localhost:8006"

    # Paths
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/indexes")

    # MongoDB — raw document store (read by ID at query time)
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "irapp"

    # Datasets
    datasets: str = "beir/quora,beir/webis-touche2020"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # BM25 defaults
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Clustering / topic detection (offline)
    n_clusters: int = 20            # KMeans clusters per dataset
    topic_terms: int = 8            # top TF-IDF terms used as a cluster's topic label

    # Frontend
    frontend_url: str = "http://localhost:8501"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def dataset_list(self) -> list[str]:
        return [d.strip() for d in self.datasets.split(",")]


settings = Settings()

# Dataset registry: maps display name to ir_datasets keys
DATASET_REGISTRY = {
    "quora": {
        "corpus_key": "beir/quora",
        "test_key": "beir/quora/test",
        "language": "en",
        "description": "Duplicate question retrieval (~523K docs, 10K test queries, binary relevance)",
    },
    "touche": {
        "corpus_key": "beir/webis-touche2020",
        "test_key": "beir/webis-touche2020",
        "language": "en",
        "description": "Argument retrieval (~382K docs, 49 test queries, graded relevance 0/1/2)",
    },
}
