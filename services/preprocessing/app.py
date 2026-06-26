"""Preprocessing service — text normalization, tokenization, stemming/lemmatization."""

from fastapi import FastAPI
from common.schemas import PreprocessRequest, PreprocessResponse, HealthResponse
from services.preprocessing.pipeline import preprocess_text

app = FastAPI(title="Preprocessing Service", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(service="preprocessing")


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: PreprocessRequest):
    tokens = preprocess_text(
        text=request.text,
        language=request.language,
        use_stemming=request.use_stemming,
        use_lemmatization=request.use_lemmatization,
        remove_stopwords=request.remove_stopwords,
    )
    return PreprocessResponse(tokens=tokens, normalized=" ".join(tokens))
