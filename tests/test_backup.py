"""Backup export/import tests (VM-17)."""

from __future__ import annotations

import json

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_bk"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb
    from vector_memory import store as st

    embedder = HashEmbedder(dim=64)
    monkeypatch.setattr(emb, "ollama_client", embedder)
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    return core


def test_export_import_roundtrip(env):
    from vector_memory import backup

    env.save_memory("backup doc one", {"project": "p", "type": "decision"}, collection=COLL)
    env.save_memory("backup doc two", {"project": "p"}, collection=COLL)

    exported = backup.export_memories(COLL)
    lines = [json.loads(l) for l in exported.splitlines()]
    assert len(lines) == 2
    assert lines[0]["text"].startswith("backup doc")
    assert lines[0]["metadata"]["project"] == "p"
    assert "_created_ts" in lines[0]["metadata"]

    # Import into a fresh collection: re-embeds (no vectors in the export).
    out = backup.import_memories(exported, "vm_bk2", reembed=True)
    assert "Imported 2" in out
    from vector_memory import store as st

    pts, _ = st.qdrant.scroll("vm_bk2", limit=10, with_payload=True)
    assert len(pts) == 2
    assert all(p.payload["_embed_model"] == "qwen3-embedding:8b" for p in pts)


def test_import_on_conflict_skip(env):
    from vector_memory import backup

    env.save_memory("conflict doc", {"project": "p"}, collection=COLL)
    exported = backup.export_memories(COLL)
    out = backup.import_memories(exported, COLL)  # same collection, skip default
    assert "Imported 0" in out and "skipped 1" in out


def test_import_invalid_line(env):
    from vector_memory import backup
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="not valid JSON"):
        backup.import_memories("{bad json\n", "vm_bk3")
