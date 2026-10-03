"""``doctor`` — health check of the configured backends and collection (VM-18).

One line per check (``ok|warn|fail <name>: <detail>``), exit 1 on any fail;
``--json`` emits the same checks as an array. Read-only: no writes anywhere.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from . import payload as _payload
from . import store as _store
from .embedding import EMBED_MODEL, OLLAMA_URL


def _default_collection() -> str:
    """Read the configured collection at call time (tests may patch it)."""
    return _store.COLLECTION_NAME


def run_checks() -> list[dict[str, str]]:
    """Run all health checks; each result is {status, name, detail}."""
    checks: list[dict[str, str]] = []

    def add(status: str, name: str, detail: str) -> None:
        checks.append({"status": status, "name": name, "detail": detail})

    # Ollama
    try:
        from . import embedding as _emb

        models = _emb.ollama_client.list().models
        names = [m.model for m in models]
        if any(n == EMBED_MODEL for n in names):
            add("ok", "ollama-model", f"{EMBED_MODEL} present at {OLLAMA_URL}")
        else:
            add("warn", "ollama-model",
                f"{EMBED_MODEL} NOT in model list at {OLLAMA_URL} "
                f"(run: ollama pull {EMBED_MODEL})")
    except Exception as exc:
        add("fail", "ollama-reachable", f"{OLLAMA_URL}: {exc}")

    # Qdrant
    qdrant_url = _store.QDRANT_URL
    try:
        server_version = _store.qdrant.info().version
        client_version = importlib.metadata.version("qdrant-client")
        add("ok", "qdrant-reachable",
            f"{qdrant_url} server {server_version}, client {client_version}")
        major_server = server_version.split(".")[:2]
        major_client = client_version.split(".")[:2]
        if major_server != major_client:
            add("warn", "qdrant-compat",
                f"client {client_version} vs server {server_version} minor mismatch")
    except Exception as exc:
        add("fail", "qdrant-reachable", f"{qdrant_url}: {exc}")
        return checks

    # Default collection
    default_collection = _default_collection()
    try:
        if _store.qdrant.collection_exists(default_collection):
            info = _store.qdrant.get_collection(default_collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                vectors = vectors.get("")
            dim = int(vectors.size) if vectors is not None else -1
            add("ok", "collection-exists",
                f"{default_collection!r} ({info.points_count or 0} points, dim {dim})")
            # Legacy-points report
            legacy = _count_legacy_points(default_collection)
            if legacy == 0:
                add("ok", "legacy-points", "all points carry system fields")
            else:
                add("warn", "legacy-points",
                    f"{legacy} point(s) without system fields — run: "
                    f"vector-memory migrate --collection {default_collection} --dry-run")
        else:
            add("warn", "collection-exists",
                f"{default_collection!r} does not exist yet (created on first save)")
    except Exception as exc:
        add("fail", "collection-exists", str(exc))

    return checks


def _count_legacy_points(collection_name: str) -> int:
    """Points missing ``_created_ts`` (legacy), counted via scroll scan."""
    legacy = 0
    offset = None
    while True:
        batch, offset = _store.qdrant.scroll(
            collection_name=collection_name, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not batch:
            break
        legacy += sum(1 for p in batch if _payload.CREATED_TS not in (p.payload or {}))
        if offset is None:
            break
    return legacy


def format_text(checks: list[dict[str, str]]) -> str:
    return "\n".join(f"{c['status']} {c['name']}: {c['detail']}" for c in checks)


def has_failures(checks: list[dict[str, Any]]) -> bool:
    return any(c["status"] == "fail" for c in checks)
