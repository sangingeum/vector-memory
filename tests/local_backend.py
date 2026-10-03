"""Integration-style fixtures over a real (local-mode) Qdrant.

`local_qdrant` gives a genuine QdrantClient(":memory:") instance patched into
`vector_memory.store.qdrant`, so payload/filter/scroll semantics are the real
ones. The embedder is a deterministic hash-based fake (stable 64-dim).
"""

from __future__ import annotations

import hashlib

import pytest
from qdrant_client import QdrantClient

from vector_memory import embedding as _emb
from vector_memory import store as _store


class HashEmbedder:
    """Deterministic offline embedder with the real Ollama client API shape."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        out: list[float] = []
        seed = digest
        while len(out) < self.dim:
            seed = hashlib.sha256(seed).digest()
            for i in range(0, len(seed), 2):
                if len(out) >= self.dim:
                    break
                out.append((int.from_bytes(seed[i : i + 2], "big") / 65535.0) - 0.5)
        norm = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / norm for x in out]

    def embed(self, model=None, input=None, **kw):  # noqa: A002 - API shape
        items: list[str] = list(input) if isinstance(input, list) else [str(input)]
        self.calls.append(items)
        return {"embeddings": [self._vec(t) for t in items]}

    def embeddings(self, model=None, prompt=None, **kw):  # noqa: A002
        return {"embedding": self._vec(str(prompt))}


@pytest.fixture()
def local_qdrant(monkeypatch):
    """Fresh local-mode Qdrant + hash embedder patched into the modules."""
    client = QdrantClient(":memory:")
    embedder = HashEmbedder()
    monkeypatch.setattr(_store, "qdrant", client)
    monkeypatch.setattr(_emb, "ollama_client", embedder)
    yield client
    client.close()
