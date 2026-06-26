.PHONY: venv install run download-data build-index build-clusters ingest rebuild-embeddings mongo-up mongo-down eval test clean

venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m nltk.downloader stopwords wordnet punkt punkt_tab

download-data:
	PYTHONPATH=. .venv/bin/python scripts/download_datasets.py

mongo-up:
	docker compose up -d mongo

mongo-down:
	docker compose down

# Stream original documents into MongoDB (raw doc store)
ingest: mongo-up
	PYTHONPATH=. .venv/bin/python scripts/ingest_mongo.py

# Rebuild ONLY the FAISS embedding index on original text (TF-IDF/BM25 untouched)
rebuild-embeddings:
	PYTHONPATH=. .venv/bin/python scripts/rebuild_embeddings.py

build-index: mongo-up
	PYTHONPATH=. .venv/bin/python scripts/build_all_indexes.py

# Build document clusters + topic labels (bonus #15/#17). Needs build-index first.
build-clusters:
	PYTHONPATH=. .venv/bin/python scripts/build_clusters.py

eval:
	PYTHONPATH=. .venv/bin/python scripts/run_evaluation.py

test:
	PYTHONPATH=. .venv/bin/pytest tests/ -v

run: mongo-up
	PYTHONPATH=. .venv/bin/python app.py

clean:
	rm -rf data/indexes/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
