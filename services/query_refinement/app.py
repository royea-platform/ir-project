"""Query refinement service — spelling correction, synonym expansion, PRF."""

from fastapi import FastAPI
from common.schemas import RefineRequest, RefineResponse, HealthResponse
from common.logging import get_logger

logger = get_logger("query_refinement")
app = FastAPI(title="Query Refinement Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="query_refinement")


@app.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    refined = request.query
    applied = []

    if request.use_spelling:
        from services.query_refinement.spelling import correct_spelling
        corrected = correct_spelling(refined)
        if corrected != refined:
            refined = corrected
            applied.append("spelling_correction")

    if request.use_synonyms:
        from services.query_refinement.expansion import expand_synonyms
        expanded = expand_synonyms(refined)
        if expanded != refined:
            refined = expanded
            applied.append("synonym_expansion")

    return RefineResponse(refined_query=refined, applied=applied)
