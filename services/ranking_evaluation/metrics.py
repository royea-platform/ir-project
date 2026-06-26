"""IR metrics evaluation using ir_measures."""

import ir_datasets
import ir_measures
from ir_measures import MAP, nDCG, P, Recall

from common.config import DATASET_REGISTRY
from common.logging import get_logger
from services.preprocessing.pipeline import preprocess_text
from services.retrieval.factory import get_retriever

logger = get_logger("ranking_evaluation.metrics")


def run_evaluation(
    dataset: str,
    repr_type: str = "tfidf",
    top_k: int = 100,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    hybrid_mode: str = "parallel",
    fusion_method: str = "rrf",
    fusion_weights: dict[str, float] | None = None,
    num_queries: int = 50,
) -> dict[str, float]:
    """Run evaluation: retrieve for test queries, compute metrics against qrels."""
    ds_info = DATASET_REGISTRY[dataset]
    test_ds = ir_datasets.load(ds_info["test_key"])

    # Get queries and qrels.
    # IMPORTANT: restrict qrels to the sampled queries — otherwise ir_measures
    # averages over every judged query (e.g. Quora's 10k), scoring the
    # un-run queries as 0 and collapsing the aggregate metrics.
    queries = list(test_ds.queries_iter())[:num_queries]
    sampled_qids = {q.query_id for q in queries}
    qrels = [qr for qr in test_ds.qrels_iter() if qr.query_id in sampled_qids]

    # Build run (retrieval results per query)
    run = []
    retriever = get_retriever(dataset, repr_type if repr_type != "hybrid" else "tfidf")

    for q in queries:
        tokens = preprocess_text(q.text, language=ds_info["language"])

        if repr_type == "hybrid":
            # Simple parallel hybrid for eval
            from services.retrieval.fusion import fuse_results
            bm25_ret = get_retriever(dataset, "bm25")
            dense_ret = get_retriever(dataset, "dense")

            bm25_results = bm25_ret.search(query_tokens=tokens, query_text=q.text, top_k=top_k)
            dense_results = dense_ret.search(query_tokens=tokens, query_text=q.text, top_k=top_k)

            results = fuse_results(
                results_dict={"bm25": bm25_results, "dense": dense_results},
                method=fusion_method,
                weights=fusion_weights or {"bm25": 0.5, "dense": 0.5},
                top_k=top_k,
            )
        else:
            results = retriever.search(
                query_tokens=tokens,
                query_text=q.text,
                top_k=top_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            )

        for doc in results:
            run.append(ir_measures.ScoredDoc(q.query_id, doc.doc_id, doc.score))

    # Compute metrics
    metrics_to_compute = [MAP, nDCG @ 10, P @ 10, Recall @ 100]
    results = ir_measures.calc_aggregate(metrics_to_compute, qrels, run)

    metrics_dict = {}
    for metric, value in results.items():
        metrics_dict[str(metric)] = round(value, 4)

    logger.info(f"Evaluation for {dataset}/{repr_type}: {metrics_dict}")
    return metrics_dict
