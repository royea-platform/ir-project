"""MongoDB raw-document store.

Raw (original, uncleaned) documents are persisted here during the offline
indexing phase. At query time the retrieval service maps top-k doc_ids back to
their original text by reading from this database — never from the cleaned/
stemmed index. One collection per dataset; `_id` is the corpus doc_id.
"""

from functools import lru_cache

from pymongo import MongoClient, UpdateOne

from common.config import settings
from common.logging import get_logger

logger = get_logger("doc_db")


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)


def get_collection(dataset: str):
    """Return the Mongo collection holding raw docs for `dataset`."""
    return _client()[settings.mongo_db][dataset]


def store_docs(dataset: str, docs: list[dict], batch_size: int = 5000) -> int:
    """Upsert raw documents. Each dict: {doc_id, title, text}. Returns count.

    No logging here — callers that stream batches log their own running total.
    """
    coll = get_collection(dataset)
    total = 0
    ops: list[UpdateOne] = []
    for d in docs:
        ops.append(UpdateOne(
            {"_id": d["doc_id"]},
            {"$set": {"title": d.get("title", ""), "text": d["text"]}},
            upsert=True,
        ))
        if len(ops) >= batch_size:
            coll.bulk_write(ops, ordered=False)
            total += len(ops)
            ops = []
    if ops:
        coll.bulk_write(ops, ordered=False)
        total += len(ops)
    return total


def fetch_docs(dataset: str, doc_ids: list[str]) -> dict[str, dict]:
    """Read raw docs by id. Returns {doc_id: {title, text}} preserving originals."""
    coll = get_collection(dataset)
    cursor = coll.find({"_id": {"$in": doc_ids}})
    return {
        doc["_id"]: {"title": doc.get("title", ""), "text": doc.get("text", "")}
        for doc in cursor
    }


def fetch_one(dataset: str, doc_id: str) -> dict | None:
    """Read a single raw doc by id, or None."""
    doc = get_collection(dataset).find_one({"_id": doc_id})
    if not doc:
        return None
    return {"doc_id": doc["_id"], "title": doc.get("title", ""), "text": doc.get("text", "")}


def count(dataset: str) -> int:
    return get_collection(dataset).estimated_document_count()
