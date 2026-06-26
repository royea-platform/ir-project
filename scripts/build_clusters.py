"""Build document clusters + topic labels for all datasets (offline phase).

Requires the indexes (embeddings.npy + tfidf) already built. Run after
`make build-index`.
"""

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from services.features.clustering import build_clusters

logger = get_logger("build_clusters")


if __name__ == "__main__":
    for dataset in DATASET_REGISTRY:
        logger.info(f"Building clusters for: {dataset}")
        build_clusters(dataset)
    logger.info("All clusters built.")
