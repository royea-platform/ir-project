"""Rebuild ONLY the FAISS embedding index on ORIGINAL text (offline phase).

TF-IDF and BM25 indexes are left untouched. Use after fixing the embedding
input to original (non-stemmed) text.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from services.indexing.builder import rebuild_faiss_only

logger = get_logger("rebuild_embeddings")


def main():
    for dataset in DATASET_REGISTRY:
        logger.info(f"Rebuilding FAISS (original text) for: {dataset}")
        rebuild_faiss_only(dataset)
        logger.info(f"Done: {dataset}")
    logger.info("All embedding indexes rebuilt.")


if __name__ == "__main__":
    main()
