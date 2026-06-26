"""Retrieval service — searches indexes using various representations."""

from fastapi import FastAPI
from common.schemas import RetrievalRequest, RetrievalResponse, HealthResponse, ScoredDoc
from common.logging import get_logger
from services.retrieval.factory import get_retriever

logger = get_logger("retrieval")
app = FastAPI(title="Retrieval Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="retrieval")


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    if request.repr_type == "hybrid":
        results = _hybrid_retrieve(request)
    else:
        retriever = get_retriever(request.dataset, request.repr_type)
        results = retriever.search(
            query_tokens=request.query_tokens,
            query_text=request.query_text,
            top_k=request.top_k,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
        )
    return RetrievalResponse(results=results)


def _hybrid_retrieve(request: RetrievalRequest) -> list[ScoredDoc]:
    """Run hybrid retrieval (serial or parallel)."""
    if request.hybrid_mode == "serial":
        # Stage 1: BM25 retrieve top-N candidates
        bm25_retriever = get_retriever(request.dataset, "bm25")
        candidates = bm25_retriever.search(
            query_tokens=request.query_tokens,
            query_text=request.query_text,
            top_k=100,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
        )
        # Stage 2: Dense rerank those candidates
        dense_retriever = get_retriever(request.dataset, "dense")
        reranked = dense_retriever.rerank(
            query_text=request.query_text,
            candidates=candidates,
            top_k=request.top_k,
        )
        return reranked
    else:
        # Parallel: run multiple retrievers, return combined (fusion in ranking service)
        results_dict = {}
        for repr_name in ["bm25", "dense"]:
            retriever = get_retriever(request.dataset, repr_name)
            results_dict[repr_name] = retriever.search(
                query_tokens=request.query_tokens,
                query_text=request.query_text,
                top_k=request.top_k * 5,
                bm25_k1=request.bm25_k1,
                bm25_b=request.bm25_b,
            )

        # Apply fusion
        from services.retrieval.fusion import fuse_results
        fused = fuse_results(
            results_dict=results_dict,
            method=request.fusion_method,
            weights=request.fusion_weights,
            top_k=request.top_k,
        )
        return fused
