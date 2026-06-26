"""Run full evaluation across all datasets and representations."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from services.ranking_evaluation.metrics import run_evaluation

logger = get_logger("run_evaluation")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    all_results = []

    for dataset in DATASET_REGISTRY:
        for repr_type in ["tfidf", "bm25", "dense", "hybrid"]:
            logger.info(f"Evaluating {dataset} / {repr_type}...")
            try:
                metrics = run_evaluation(
                    dataset=dataset,
                    repr_type=repr_type,
                    top_k=100,
                    num_queries=50,
                )
                row = {"dataset": dataset, "repr_type": repr_type, **metrics}
                all_results.append(row)
                logger.info(f"  {metrics}")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = REPORTS_DIR / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
