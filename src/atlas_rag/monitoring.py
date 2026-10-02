"""Lightweight structured observability for local and container deployments."""

import json
import logging
from collections import Counter
from threading import Lock
from time import time


logger = logging.getLogger("atlas_rag")


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.requests = Counter()
        self.latencies_ms: list[float] = []

    def record(self, kind: str, latency_ms: float, success: bool = True) -> None:
        with self._lock:
            self.requests[f"{kind}.total"] += 1
            self.requests[f"{kind}.success" if success else f"{kind}.error"] += 1
            self.latencies_ms.append(latency_ms)
            self.latencies_ms = self.latencies_ms[-1000:]

    def snapshot(self) -> dict:
        with self._lock:
            values = list(self.latencies_ms)
            avg = sum(values) / len(values) if values else 0.0
            return {
                "requests": dict(self.requests),
                "latency_ms": {
                    "count": len(values),
                    "average": round(avg, 2),
                    "max": round(max(values), 2) if values else 0.0,
                },
            }


metrics = Metrics()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, "timestamp": time(), **fields}
    logger.info(json.dumps(payload, default=str))
