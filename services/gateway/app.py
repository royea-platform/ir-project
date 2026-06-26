"""API Gateway — orchestrates the search pipeline across services."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from common.schemas import (
    SearchRequest, SearchResponse, EvalRequest, EvalResponse, HealthResponse,
    ScoredDoc, DistributedSearchRequest, DistributedSearchResponse,
)
from common.clients import (
    preprocessing_client, retrieval_client, ranking_client, refinement_client,
    features_client,
)
from common.config import settings, DATASET_REGISTRY
from common import doc_db
from common.logging import get_logger

logger = get_logger("gateway")
app = FastAPI(title="IR Search Engine Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="gateway")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Orchestrate full search pipeline."""
    query = request.query
    refined_query = None

    logger.info(
        f"SEARCH start | dataset={request.dataset} repr={request.repr_type} "
        f"mode={request.mode} top_k={request.top_k} query={request.query!r}"
    )

    # Step 1: Query refinement (only in basic+extra mode)
    if request.mode == "basic+extra":
        logger.info("[1/3] query_refinement → calling refinement service")
        try:
            refine_resp = await refinement_client.post("/refine", {
                "query": query,
                "dataset": request.dataset,
                "use_spelling": True,
                "use_synonyms": True,
            })
            refined_query = refine_resp["refined_query"]
            query = refined_query
            logger.info(
                f"[1/3] query_refinement done | applied={refine_resp.get('applied')} "
                f"{request.query!r} → {refined_query!r}"
            )
        except Exception as e:
            logger.warning(f"[1/3] query refinement failed: {e}")
    else:
        logger.info("[1/3] query_refinement SKIPPED (mode != basic+extra)")

    # Step 2: Preprocess query
    logger.info("[2/3] preprocessing → calling preprocessing service")
    try:
        preprocess_resp = await preprocessing_client.post("/preprocess", {
            "text": query,
            "language": "en",
        })
        query_tokens = preprocess_resp["tokens"]
        logger.info(f"[2/3] preprocessing done | tokens={query_tokens}")
    except Exception as e:
        logger.error(f"[2/3] preprocessing failed: {e}")
        query_tokens = query.lower().split()
        logger.warning(f"[2/3] fallback tokens={query_tokens}")

    # Step 3: Retrieve
    logger.info(f"[3/3] retrieval → calling retrieval service (repr={request.repr_type})")
    try:
        retrieve_resp = await retrieval_client.post("/retrieve", {
            "dataset": request.dataset,
            "query_tokens": query_tokens,
            "query_text": query,
            "repr_type": request.repr_type,
            "top_k": request.top_k,
            "bm25_k1": request.bm25_k1,
            "bm25_b": request.bm25_b,
            "hybrid_mode": request.hybrid_mode,
            "fusion_method": request.fusion_method,
            "fusion_weights": request.fusion_weights,
        })
        results = retrieve_resp["results"]
        logger.info(f"[3/3] retrieval done | {len(results)} results")
    except Exception as e:
        logger.error(f"[3/3] retrieval failed: {e}")
        results = []

    logger.info(f"SEARCH done | returning {len(results)} results")
    return SearchResponse(
        query=request.query,
        refined_query=refined_query,
        dataset=request.dataset,
        repr_type=request.repr_type,
        results=results,
    )


@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    """Forward evaluation request to ranking service."""
    try:
        resp = await ranking_client.post("/evaluate", request.model_dump())
        return EvalResponse(**resp)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return EvalResponse(
            dataset=request.dataset,
            repr_type=request.repr_type,
            mode=request.mode,
            metrics={"error": -1.0},
        )


@app.get("/doc/{dataset}/{doc_id}")
async def get_document(dataset: str, doc_id: str):
    """Read the ORIGINAL document by ID from the MongoDB raw doc store."""
    import pickle

    if dataset not in DATASET_REGISTRY:
        return {"error": f"Unknown dataset: {dataset}"}

    doc = doc_db.fetch_one(dataset, doc_id)
    if not doc:
        return {"error": f"Document {doc_id} not found in {dataset}"}

    # 1-based position in corpus, for display
    doc_number = None
    doc_ids_path = settings.index_dir / dataset / "doc_ids.pkl"
    if doc_ids_path.exists():
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)
        try:
            doc_number = doc_ids.index(doc_id) + 1
        except ValueError:
            doc_number = None

    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "text": doc["text"],
        "doc_number": doc_number,
        "total_docs": doc_db.count(dataset),
        "dataset": dataset,
    }


@app.post("/search_distributed", response_model=DistributedSearchResponse)
async def search_distributed(request: DistributedSearchRequest):
    """#14 Distributed (federated) IR — fan a query out to every dataset shard,
    min-max normalize each shard's scores so they are comparable, then merge."""
    logger.info(f"DISTRIBUTED search | datasets={request.datasets} query={request.query!r}")

    merged: list[ScoredDoc] = []
    per_dataset: dict[str, int] = {}

    for ds in request.datasets:
        if ds not in DATASET_REGISTRY:
            logger.warning(f"skipping unknown dataset {ds!r}")
            continue
        sub = SearchRequest(
            dataset=ds, query=request.query, mode=request.mode,
            repr_type=request.repr_type, top_k=request.top_k,
            bm25_k1=request.bm25_k1, bm25_b=request.bm25_b,
            hybrid_mode=request.hybrid_mode, fusion_method=request.fusion_method,
            fusion_weights=request.fusion_weights,
        )
        resp = await search(sub)
        docs = resp.results
        if not docs:
            continue
        scores = [d.score for d in docs]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for d in docs:
            merged.append(d.model_copy(update={
                "dataset": ds,
                "score": (d.score - lo) / span,   # normalized 0..1 for cross-shard merge
            }))

    merged.sort(key=lambda d: d.score, reverse=True)
    merged = merged[:request.top_k]
    for rank, d in enumerate(merged):
        d.rank = rank + 1
        per_dataset[d.dataset] = per_dataset.get(d.dataset, 0) + 1

    logger.info(f"DISTRIBUTED done | {len(merged)} merged results, split={per_dataset}")
    return DistributedSearchResponse(
        query=request.query,
        datasets=request.datasets,
        repr_type=request.repr_type,
        results=merged,
        per_dataset=per_dataset,
    )


@app.get("/clusters/{dataset}")
async def clusters(dataset: str):
    """#15/#17 — proxy the features service: clusters + detected topics."""
    return await features_client.get(f"/clusters/{dataset}")


@app.get("/cluster/{dataset}/{cluster_id}")
async def cluster_docs(dataset: str, cluster_id: int, limit: int = 10):
    """#15 — sample documents from one cluster."""
    return await features_client.get(f"/cluster/{dataset}/{cluster_id}", params={"limit": limit})


@app.post("/compare")
async def compare(request: SearchRequest):
    """Run all 4 retrieval methods on the same query (each independently testable)."""
    methods = ["tfidf", "bm25", "dense", "hybrid"]
    out = {}
    for method in methods:
        req = request.model_copy(update={"repr_type": method})
        resp = await search(req)
        out[method] = {
            "results": [d.model_dump() for d in resp.results],
            "refined_query": resp.refined_query,
        }
    return {"query": request.query, "dataset": request.dataset, "methods": out}
