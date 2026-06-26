"""Typed HTTP clients for inter-service communication."""

import time

import httpx
from common.config import settings
from common.logging import get_logger

logger = get_logger("client")


def _summarize(data: dict | None) -> str:
    """Compact one-line preview of a payload for logs (truncated)."""
    if not data:
        return "-"
    parts = []
    for k, v in data.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return " ".join(parts)


class ServiceClient:
    """Base client for calling other services."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        # Human-friendly service name derived from the base URL.
        self.name = base_url.rstrip("/").rsplit(":", 1)[-1]

    async def post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        logger.info(f"→ POST {url}  [{self.name}]  {_summarize(data)}")
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=data)
                resp.raise_for_status()
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                logger.error(f"✗ POST {url}  [{self.name}]  failed in {dt:.0f}ms: {e}")
                raise
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"← POST {url}  [{self.name}]  {resp.status_code} in {dt:.0f}ms")
            return resp.json()

    async def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        logger.info(f"→ GET  {url}  [{self.name}]  {_summarize(params)}")
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                logger.error(f"✗ GET  {url}  [{self.name}]  failed in {dt:.0f}ms: {e}")
                raise
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"← GET  {url}  [{self.name}]  {resp.status_code} in {dt:.0f}ms")
            return resp.json()


preprocessing_client = ServiceClient(settings.preprocessing_url)
indexing_client = ServiceClient(settings.indexing_url, timeout=600.0)
retrieval_client = ServiceClient(settings.retrieval_url)
ranking_client = ServiceClient(settings.ranking_eval_url)
refinement_client = ServiceClient(settings.query_refinement_url)
features_client = ServiceClient(settings.features_url)
