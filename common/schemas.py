"""Pydantic schemas shared across all services."""

from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    text: str
    language: str = "en"
    use_stemming: bool = True
    use_lemmatization: bool = False
    remove_stopwords: bool = True


class PreprocessResponse(BaseModel):
    tokens: list[str]
    normalized: str


class SearchRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name: 'quora' or 'touche'")
    query: str
    mode: str = Field("basic", description="'basic' or 'basic+extra'")
    repr_type: str = Field("tfidf", description="tfidf | bm25 | dense | hybrid")
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_mode: str = Field("parallel", description="'serial' or 'parallel'")
    fusion_method: str = Field("rrf", description="'rrf' or 'weighted' or 'combmnz'")
    fusion_weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})


class ScoredDoc(BaseModel):
    doc_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""
    dataset: str = ""  # source dataset — set by federated (distributed) search


class SearchResponse(BaseModel):
    query: str
    refined_query: str | None = None
    dataset: str
    repr_type: str
    results: list[ScoredDoc]
    metrics: dict[str, float] | None = None


class RetrievalRequest(BaseModel):
    dataset: str
    query_tokens: list[str]
    query_text: str = ""
    repr_type: str = "tfidf"
    top_k: int = 100
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_mode: str = "parallel"
    fusion_method: str = "rrf"
    fusion_weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})


class RetrievalResponse(BaseModel):
    results: list[ScoredDoc]


class RankRequest(BaseModel):
    results_lists: dict[str, list[ScoredDoc]]
    fusion_method: str = "rrf"
    fusion_weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})
    top_k: int = 10


class RankResponse(BaseModel):
    results: list[ScoredDoc]


class EvalRequest(BaseModel):
    dataset: str
    repr_type: str = "tfidf"
    mode: str = "basic"
    top_k: int = 100
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_mode: str = "parallel"
    fusion_method: str = "rrf"
    fusion_weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})
    num_queries: int = 50


class EvalResponse(BaseModel):
    dataset: str
    repr_type: str
    mode: str
    metrics: dict[str, float]


class RefineRequest(BaseModel):
    query: str
    dataset: str = "quora"
    history: list[str] = Field(default_factory=list)
    use_spelling: bool = True
    use_synonyms: bool = True
    use_prf: bool = False


class RefineResponse(BaseModel):
    refined_query: str
    applied: list[str]


class HealthResponse(BaseModel):
    service: str
    status: str = "ok"


# ── Bonus feature schemas ─────────────────────────────────────────────────────

# #15 Document clustering + #17 Topic detection
class ClusterInfo(BaseModel):
    cluster_id: int
    topic: list[str]          # top TF-IDF terms = detected topic label
    size: int                 # number of docs in cluster


class ClustersResponse(BaseModel):
    dataset: str
    n_clusters: int
    clusters: list[ClusterInfo]


class ClusterDocsResponse(BaseModel):
    dataset: str
    cluster_id: int
    topic: list[str]
    docs: list[ScoredDoc]


class DocClusterResponse(BaseModel):
    dataset: str
    doc_id: str
    cluster_id: int
    topic: list[str]


# #14 Distributed (federated) information retrieval
class DistributedSearchRequest(BaseModel):
    query: str
    datasets: list[str] = Field(default_factory=lambda: ["quora", "touche"])
    mode: str = "basic"
    repr_type: str = "bm25"
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_mode: str = "parallel"
    fusion_method: str = "rrf"
    fusion_weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 0.5, "dense": 0.5})


class DistributedSearchResponse(BaseModel):
    query: str
    datasets: list[str]
    repr_type: str
    results: list[ScoredDoc]            # merged, score-normalized across shards
    per_dataset: dict[str, int]         # how many of the merged results came from each shard
