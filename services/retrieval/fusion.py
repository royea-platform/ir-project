"""Fusion methods for parallel hybrid retrieval."""

from common.schemas import ScoredDoc


def fuse_results(
    results_dict: dict[str, list[ScoredDoc]],
    method: str = "rrf",
    weights: dict[str, float] | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[ScoredDoc]:
    """Fuse multiple ranked lists into one."""
    if method == "rrf":
        return _rrf_fusion(results_dict, top_k, rrf_k)
    elif method == "weighted":
        return _weighted_fusion(results_dict, weights or {}, top_k)
    elif method == "combmnz":
        return _combmnz_fusion(results_dict, top_k)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def _rrf_fusion(results_dict: dict[str, list[ScoredDoc]], top_k: int, k: int = 60) -> list[ScoredDoc]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1/(k + rank(d))."""
    doc_scores: dict[str, float] = {}
    doc_info: dict[str, ScoredDoc] = {}

    for ranker_name, results in results_dict.items():
        for doc in results:
            doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0.0) + 1.0 / (k + doc.rank)
            if doc.doc_id not in doc_info:
                doc_info[doc.doc_id] = doc

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        ScoredDoc(
            doc_id=doc_id,
            score=score,
            rank=rank + 1,
            title=doc_info[doc_id].title,
            text=doc_info[doc_id].text,
        )
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _weighted_fusion(
    results_dict: dict[str, list[ScoredDoc]],
    weights: dict[str, float],
    top_k: int,
) -> list[ScoredDoc]:
    """CombSUM with min-max normalization and per-representation weights."""
    doc_scores: dict[str, float] = {}
    doc_info: dict[str, ScoredDoc] = {}

    for ranker_name, results in results_dict.items():
        w = weights.get(ranker_name, 1.0)
        raw_scores = [d.score for d in results]
        norm_scores = _min_max_normalize(raw_scores)

        for doc, ns in zip(results, norm_scores):
            doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0.0) + w * ns
            if doc.doc_id not in doc_info:
                doc_info[doc.doc_id] = doc

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        ScoredDoc(
            doc_id=doc_id,
            score=score,
            rank=rank + 1,
            title=doc_info[doc_id].title,
            text=doc_info[doc_id].text,
        )
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]


def _combmnz_fusion(results_dict: dict[str, list[ScoredDoc]], top_k: int) -> list[ScoredDoc]:
    """CombMNZ: CombSUM * number_of_rankers_that_retrieved_doc."""
    doc_scores: dict[str, float] = {}
    doc_count: dict[str, int] = {}
    doc_info: dict[str, ScoredDoc] = {}

    for ranker_name, results in results_dict.items():
        raw_scores = [d.score for d in results]
        norm_scores = _min_max_normalize(raw_scores)

        for doc, ns in zip(results, norm_scores):
            doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0.0) + ns
            doc_count[doc.doc_id] = doc_count.get(doc.doc_id, 0) + 1
            if doc.doc_id not in doc_info:
                doc_info[doc.doc_id] = doc

    # Multiply by count
    final_scores = {did: doc_scores[did] * doc_count[did] for did in doc_scores}
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        ScoredDoc(
            doc_id=doc_id,
            score=score,
            rank=rank + 1,
            title=doc_info[doc_id].title,
            text=doc_info[doc_id].text,
        )
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]
