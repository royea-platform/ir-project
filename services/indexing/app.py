"""Indexing service — builds and persists TF-IDF, BM25, FAISS indexes."""

from fastapi import FastAPI, BackgroundTasks
from common.schemas import HealthResponse
from common.logging import get_logger

logger = get_logger("indexing")
app = FastAPI(title="Indexing Service", version="1.0.0")

# Track build status
_build_status: dict[str, str] = {}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="indexing")


@app.get("/status/{dataset}")
async def status(dataset: str):
    return {"dataset": dataset, "status": _build_status.get(dataset, "not_started")}


@app.post("/build/{dataset}")
async def build_index(dataset: str, background_tasks: BackgroundTasks):
    from services.indexing.builder import build_all_indexes
    _build_status[dataset] = "building"
    background_tasks.add_task(_run_build, dataset)
    return {"dataset": dataset, "status": "started"}


async def _run_build(dataset: str):
    try:
        from services.indexing.builder import build_all_indexes
        build_all_indexes(dataset)
        _build_status[dataset] = "complete"
        logger.info(f"Index build complete for {dataset}")
    except Exception as e:
        _build_status[dataset] = f"error: {e}"
        logger.error(f"Index build failed for {dataset}: {e}")
