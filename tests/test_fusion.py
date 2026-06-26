"""Tests for the fusion methods."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.schemas import ScoredDoc
from services.retrieval.fusion import fuse_results


def _make_docs(ids_scores: list[tuple[str, float]]) -> list[ScoredDoc]:
    return [
        ScoredDoc(doc_id=did, score=score, rank=i + 1, title="", text="")
        for i, (did, score) in enumerate(ids_scores)
    ]


def test_rrf_fusion():
    list1 = _make_docs([("a", 1.0), ("b", 0.8), ("c", 0.6)])
    list2 = _make_docs([("b", 1.0), ("c", 0.9), ("d", 0.7)])

    results = fuse_results({"r1": list1, "r2": list2}, method="rrf", top_k=3)
    # "b" appears in both lists at high rank, should be top
    assert results[0].doc_id == "b"
    assert len(results) == 3


def test_weighted_fusion():
    list1 = _make_docs([("a", 10.0), ("b", 5.0)])
    list2 = _make_docs([("b", 10.0), ("a", 2.0)])

    results = fuse_results(
        {"r1": list1, "r2": list2},
        method="weighted",
        weights={"r1": 0.3, "r2": 0.7},
        top_k=2,
    )
    # With higher weight on r2, "b" should win
    assert results[0].doc_id == "b"


def test_combmnz_fusion():
    list1 = _make_docs([("a", 1.0), ("b", 0.5)])
    list2 = _make_docs([("a", 0.8), ("c", 0.9)])

    results = fuse_results({"r1": list1, "r2": list2}, method="combmnz", top_k=3)
    # "a" appears in both, so multiplied by 2
    assert results[0].doc_id == "a"
