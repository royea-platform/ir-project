# IR Search Engine

SOA-based Information Retrieval system for university IR 2026 project. Searches two datasets (BEIR/Quora + BEIR/Touché) using multiple retrieval models.

## Quick Start

```bash
# 1. Create venv + install deps
make install

# 2. Download & verify datasets
make download-data

# 3a. Start MongoDB + ingest original documents (raw doc store)
make ingest

# 3b. Build indexes (TF-IDF, BM25, FAISS embeddings — offline, slow)
make build-index

# 4. Run all services (gateway :8000, UI :8501)
make run
```

Open **http://localhost:8501** for the web UI.

## Architecture (SOA)

True service-oriented architecture: each component runs as its own FastAPI
process and communicates over **HTTP**. The gateway (`:8000`) is the single
entrypoint; the Streamlit UI talks only to the gateway, which orchestrates the
rest. `python app.py` launches every service; `python app.py <name>` runs one.

| Service | Port | Description |
|---------|------|-------------|
| `gateway` | 8000 | Orchestrates pipeline; serves `/search`, `/compare`, `/evaluate`, `/doc` |
| `preprocessing` | 8001 | Text normalization, tokenization, stemming |
| `indexing` | 8002 | Builds TF-IDF, BM25, FAISS indexes + Mongo ingest (offline) |
| `retrieval` | 8003 | Searches indexes (TF-IDF/BM25/Dense/Hybrid) |
| `ranking_evaluation` | 8004 | Fusion ranking + IR metrics (MAP, nDCG) |
| `query_refinement` | 8005 | Spelling, synonyms, PRF |
| `features` | 8006 | Document clusters + detected topics (bonus #15/#17) |
| `frontend` | 8501 | Streamlit web UI (talks to gateway only) |

### Offline vs Online

- **Offline**: dataset download, preprocessing, TF-IDF/BM25/FAISS index build,
  Mongo ingest of original docs. Run once, persisted to disk + MongoDB.
- **Online**: query → refine → preprocess → retrieve top-k IDs → **read original
  documents from MongoDB by ID** → return. No training/indexing at query time.

### Raw document store (MongoDB)

MongoDB runs via Docker Compose (`docker-compose.yml`, service `mongo`, image
`mongo:7`, port 27017) with a named volume `mongo_data` so ingested documents
persist across container restarts. `make mongo-up` / `make mongo-down` wrap
`docker compose up -d mongo` / `docker compose down`. Requires Docker + the
Compose plugin.

Original (uncleaned) documents are stored in MongoDB during the offline phase,
one collection per dataset (`_id` = corpus doc_id). At query time the retrieval
service maps the top-k doc_ids back to their **original** text via MongoDB — the
stemmed index text is never returned to the user.

```
User Query → refine → preprocess → retrieval model → Top-10 doc IDs
          → read ORIGINAL docs from MongoDB by ID → display
```

Embeddings are built from **original** text; TF-IDF and BM25 from cleaned/stemmed text.

## Evaluation

```bash
make eval   # Runs evaluation across all datasets × representations → reports/
```

## Testing

```bash
make test
```

## Datasets

| Dataset | Docs | Test Queries | Relevance | Domain |
|---------|------|-------------|-----------|--------|
| BEIR/Quora | ~523K | 10,000 | Binary | Duplicate question retrieval |
| BEIR/Touché | ~382K | 49 | Graded (0/1/2) | Argument retrieval |

## Bonus Features

Three of the assignment's optional features (#10–#19):

- **#14 Distributed (federated) IR** — `POST /search_distributed` fans a query
  across both dataset shards, min-max normalizes per-shard scores, and merges
  into one ranked list (UI: *Federated Search* tab).
- **#15 Document clustering** — MiniBatchKMeans over document embeddings groups
  the corpus. Built offline: `make build-clusters` → `data/indexes/<ds>/clusters.npy`.
- **#17 Topic detection** — each cluster is labelled by its top mean-TF-IDF
  terms (UI: *Clusters & Topics* tab; `GET /clusters/{dataset}`).

Plus enhancements to the core models: dense (FAISS) retrieval, hybrid
serial/parallel fusion (RRF/Weighted/CombMNZ), and query refinement.

```bash
make build-clusters   # after make build-index — builds clusters + topics
```

## Representations

- **TF-IDF (VSM)** — cosine similarity
- **BM25** — adjustable k1, b parameters
- **Dense (Embeddings)** — sentence-transformers + FAISS
- **Hybrid Serial** — BM25 retrieve → Dense rerank
- **Hybrid Parallel** — BM25 + Dense → RRF/Weighted/CombMNZ fusion

## Project Structure

```
irapp/
├── common/          # Shared config, schemas, clients, logging
├── services/
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── ranking_evaluation/
│   ├── query_refinement/
│   └── gateway/
├── frontend/        # Streamlit app
├── scripts/         # Download, build, evaluate scripts
├── tests/           # Unit + integration tests
├── reports/         # Evaluation CSVs + figures
└── data/            # Raw datasets + built indexes (gitignored)
```
