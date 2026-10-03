"""Metadata-only patch/update tests: no embedding calls, vector untouched."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb

# Fixture registration: pytest resolves `local_qdrant` from this module's
# attribute; importing by name would shadow the fixture parameter (F811).
local_qdrant = _lb.local_qdrant
HashEmbedder = _lb.HashEmbedder


def test_patch_merges_keys_without_embedding(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import embedding as emb

    embedder = HashEmbedder()
    monkeypatch.setattr(emb, "ollama_client", embedder)
    out = core.save_memory("patch me", {"source": "a"}, collection="vm_patch")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    before_vector = local_qdrant.retrieve("vm_patch", ids=[pid], with_vectors=True)[0].vector
    calls_before = len(embedder.calls)

    core.patch_metadata(pid, {"tags": ["x"], "trust": "high"}, collection="vm_patch")
    point = local_qdrant.retrieve("vm_patch", ids=[pid], with_payload=True, with_vectors=True)[0]
    assert point.payload["source"] == "a"  # merge, not replace
    assert point.payload["tags"] == ["x"]
    assert point.payload["trust"] == "high"
    assert point.vector == before_vector
    assert len(embedder.calls) == calls_before  # no embedding call


def test_patch_unset_removes_keys(local_qdrant):
    from vector_memory import core

    out = core.save_memory("unset me", {"source": "a", "keep": 1}, collection="vm_patch")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    core.patch_metadata(pid, unset=["source"], collection="vm_patch")
    point = local_qdrant.retrieve("vm_patch", ids=[pid], with_payload=True)[0]
    assert "source" not in point.payload
    assert point.payload["keep"] == 1


def test_patch_unset_missing_key_is_noop(local_qdrant):
    from vector_memory import core

    out = core.save_memory("noop patch", {"source": "a"}, collection="vm_patch")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    out2 = core.patch_metadata(pid, unset=["not-there"], collection="vm_patch")
    assert "Memory patched" in out2


def test_patch_unknown_point(local_qdrant):
    from vector_memory import core
    from vector_memory.errors import NotFoundError

    with pytest.raises(NotFoundError):
        core.patch_metadata("00000000-0000-0000-0000-000000000000",
                            {"tags": ["x"]}, collection="vm_patch")


def test_update_merge_metadata(local_qdrant):
    from vector_memory import core

    out = core.save_memory("merge doc", {"source": "a", "v": 1}, collection="vm_patch")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    core.update_memory(pid, None, {"v": 2, "src": "obj"}, collection="vm_patch",
                       merge_metadata=True)
    point = local_qdrant.retrieve("vm_patch", ids=[pid], with_payload=True)[0]
    assert point.payload["v"] == 2
    assert point.payload["source"] == "a"  # preserved by merge
    assert point.payload["src"] == "obj"


def test_update_replace_semantics_unchanged(local_qdrant):
    from vector_memory import core

    out = core.save_memory("replace doc", {"source": "a", "v": 1}, collection="vm_patch")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    core.update_memory(pid, None, {"v": 2}, collection="vm_patch")
    point = local_qdrant.retrieve("vm_patch", ids=[pid], with_payload=True)[0]
    assert point.payload["v"] == 2
    assert "source" not in point.payload  # default remains replace
