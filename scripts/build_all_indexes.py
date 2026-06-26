"""Build all indexes for all datasets."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from services.indexing.builder import build_all_indexes

logger = get_logger("build_all_indexes")


def main():
    for dataset_name in DATASET_REGISTRY:
        logger.info(f"Building indexes for: {dataset_name}")
        build_all_indexes(dataset_name)
        logger.info(f"Done: {dataset_name}")

    logger.info("All indexes built.")


if __name__ == "__main__":
    main()
