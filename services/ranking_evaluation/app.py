"""Ranking & Evaluation service — fusion ranking + IR metrics computation."""

from fastapi import FastAPI
from common.schemas import (
    RankRequest, RankResponse, EvalRequest, EvalResponse, HealthResponse
)
from common.logging import get_logger

logger = get_logger("ranking_evaluation")
app = FastAPI(title="Ranking & Evaluation Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="ranking_evaluation")


@app.post("/rank", response_model=RankResponse)
async def rank(request: RankRequest):
    from services.retrieval.fusion import fuse_results
    results = fuse_results(
        results_dict=request.results_lists,
        method=request.fusion_method,
        weights=request.fusion_weights,
        top_k=request.top_k,
    )
    return RankResponse(results=results)


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    from services.ranking_evaluation.metrics import run_evaluation
    metrics = run_evaluation(
        dataset=request.dataset,
        repr_type=request.repr_type,
        top_k=request.top_k,
        bm25_k1=request.bm25_k1,
        bm25_b=request.bm25_b,
        hybrid_mode=request.hybrid_mode,
        fusion_method=request.fusion_method,
        fusion_weights=request.fusion_weights,
        num_queries=request.num_queries,
    )
    return EvalResponse(
        dataset=request.dataset,
        repr_type=request.repr_type,
        mode=request.mode,
        metrics=metrics,
    )
