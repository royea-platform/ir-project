"""Download and verify both datasets using ir_datasets."""

import sys
import ir_datasets
from common.config import DATASET_REGISTRY
from common.logging import get_logger

logger = get_logger("download_datasets")


def download_and_verify():
    for name, info in DATASET_REGISTRY.items():
        logger.info(f"Loading dataset: {name} ({info['corpus_key']})")

        # Load corpus
        corpus = ir_datasets.load(info["corpus_key"])
        doc_count = corpus.docs_count()
        logger.info(f"  Corpus docs: {doc_count}")

        if doc_count and doc_count < 200000:
            logger.warning(f"  WARNING: {name} has < 200K docs ({doc_count})")

        # Load test split (queries + qrels)
        test_ds = ir_datasets.load(info["test_key"])
        queries = list(test_ds.queries_iter())
        qrels = list(test_ds.qrels_iter())
        logger.info(f"  Test queries: {len(queries)}")
        logger.info(f"  Qrels: {len(qrels)}")

        # Show sample
        if queries:
            logger.info(f"  Sample query: {queries[0]}")
        if qrels:
            logger.info(f"  Sample qrel: {qrels[0]}")

    logger.info("All datasets verified successfully.")


if __name__ == "__main__":
    download_and_verify()
