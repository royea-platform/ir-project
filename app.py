"""
IR Search Engine — Service-Oriented launcher.

Boots every component as its own process (true SOA): six FastAPI microservices
that communicate over HTTP, plus the Streamlit UI. The gateway (:8000) is the
single entrypoint the frontend talks to; it orchestrates the others.

    python app.py            # start everything
    python app.py gateway    # start a single service by name

Each service can also be run directly, e.g.:
    uvicorn services.retrieval.app:app --port 8003
"""

import os
import sys
import signal
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logging import get_logger

logger = get_logger("launcher")

# name -> (uvicorn import path, port)
SERVICES = {
    "gateway":          ("services.gateway.app:app", 8000),
    "preprocessing":    ("services.preprocessing.app:app", 8001),
    "indexing":         ("services.indexing.app:app", 8002),
    "retrieval":        ("services.retrieval.app:app", 8003),
    "ranking":          ("services.ranking_evaluation.app:app", 8004),
    "query_refinement": ("services.query_refinement.app:app", 8005),
    "features":         ("services.features.app:app", 8006),
}


def _uvicorn(target: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", target,
         "--host", "0.0.0.0", "--port", str(port), "--log-level", "warning"],
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))},
    )


def _streamlit() -> subprocess.Popen:
    frontend = os.path.join(os.path.dirname(__file__), "frontend", "streamlit_app.py")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", frontend,
         "--server.port", "8501", "--server.address", "0.0.0.0",
         "--server.headless", "true"],
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))},
    )


def run_all():
    procs: list[subprocess.Popen] = []
    logger.info("Starting IR Search Engine (SOA) ...")
    for name, (target, port) in SERVICES.items():
        logger.info(f"  → {name:<16} http://localhost:{port}")
        procs.append(_uvicorn(target, port))
        time.sleep(0.5)

    logger.info("  → frontend         http://localhost:8501")
    procs.append(_streamlit())

    def shutdown(*_):
        logger.info("Shutting down all services...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("All services up. Gateway: http://localhost:8000  UI: http://localhost:8501")
    logger.info("Press Ctrl+C to stop.")
    # Block until any process exits, then tear everything down.
    while True:
        for p in procs:
            if p.poll() is not None:
                logger.error("A service exited; shutting down.")
                shutdown()
        time.sleep(1)


def run_one(name: str):
    if name not in SERVICES:
        logger.error(f"Unknown service '{name}'. Options: {', '.join(SERVICES)}")
        sys.exit(1)
    target, port = SERVICES[name]
    logger.info(f"Starting {name} on :{port}")
    _uvicorn(target, port).wait()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
    else:
        run_all()
