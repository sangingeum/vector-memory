"""Ollama embedding client for the vector-memory server.

Exposes :func:`embed` (single text) and :func:`embed_many` (batch). Both try
the modern ``client.embed()`` API first and fall back to the legacy
``client.embeddings()`` API for older Ollama servers.
"""

from __future__ import annotations

import os
from typing import Any

import ollama

_DEFAULTS: dict[str, str] = {
    "ollama_url": "http://192.168.1.103:11434",
    "embed_model": "qwen3-embedding:8b",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", _DEFAULTS["ollama_url"])
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULTS["embed_model"])
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))

# Single shared Ollama client; tests replace this attribute.
ollama_client: Any = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)


class ModelNotFoundError(Exception):
    """The configured model is missing from the Ollama server."""


def _retry(op, description: str):
    """Retry transient failures with exponential backoff (max 3 attempts)."""
    import time

    delays = (1.0, 2.0)
    last: Exception | None = None
    for attempt in range(3):
        try:
            return op()
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message and "model" in message:
                raise ModelNotFoundError(
                    f"model {EMBED_MODEL!r} not found on Ollama; run: ollama pull {EMBED_MODEL}"
                ) from exc
            if isinstance(exc, (ConnectionError, TimeoutError)):
                last = exc
            else:
                # ollama wraps HTTP/connection errors; retry those too.
                if any(word in message for word in ("connect", "timeout", "timed out", "refused")):
                    last = exc
                else:
                    raise
        if attempt < len(delays):
            time.sleep(delays[attempt])
    raise last  # type: ignore[misc]


def embed(text: str) -> list[float]:
    """Return the embedding vector for *text*.

    Tries the modern ``client.embed()`` API first and falls back to the
    legacy ``client.embeddings()`` for older ollama servers. Transient
    failures retry with backoff; a missing model raises ModelNotFoundError.
    An AttributeError on the modern API (old server) falls back to legacy.
    """
    try:
        result = _retry(lambda: ollama_client.embed(model=EMBED_MODEL, input=text), "embed")
        return result["embeddings"][0]
    except (AttributeError, KeyError, TypeError):
        res = _retry(lambda: ollama_client.embeddings(model=EMBED_MODEL, prompt=text), "embed-legacy")
        return res["embedding"]


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single Ollama call (EMBED_BATCH-sized batches)."""
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        result = _retry(lambda chunk=chunk: ollama_client.embed(model=EMBED_MODEL, input=chunk), "embed-many")
        try:
            out.extend(result["embeddings"])
        except (AttributeError, KeyError, TypeError):
            # Legacy fallback: one call per text.
            out.extend(
                _retry(lambda t=t: ollama_client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"], "embed-one")
                for t in chunk
            )
    return out


# Back-compat alias used by earlier revisions and the live smoke test.
embed_batch = embed_many
