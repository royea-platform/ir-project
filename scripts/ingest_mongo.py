"""Ingest original documents into MongoDB for every dataset (offline phase)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from common import doc_db
from services.indexing.builder import ingest_raw_docs

logger = get_logger("ingest_mongo")


def main():
    for dataset in DATASET_REGISTRY:
        logger.info(f"Ingesting raw docs → MongoDB: {dataset}")
        total = ingest_raw_docs(dataset)
        logger.info(f"  {dataset}: {total} docs stored ({doc_db.count(dataset)} in collection)")
    logger.info("Mongo ingest complete.")


if __name__ == "__main__":
    main()
