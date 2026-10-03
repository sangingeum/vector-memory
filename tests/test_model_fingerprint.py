"""Embedding-model fingerprint + reembed tests (model-fingerprint work item)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant


def test_write_records_model_and_mismatch_fails(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory.errors import ConfigError

    core.save_memory("model a doc", {}, collection="vm_fp")
    # Same dimension under a different model name -> ConfigError on next write.
    monkeypatch.setattr("vector_memory.embedding.EMBED_MODEL", "other-model:1b")
    with pytest.raises(ConfigError, match="was built with"):
        core.save_memory("model b doc", {}, collection="vm_fp")
    with pytest.raises(ConfigError, match="was built with"):
        core.search_memory("anything", collection="vm_fp")


def test_force_mismatch_override(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory import reembed as fp

    core.save_memory("model a doc", {}, collection="vm_fp_override")
    monkeypatch.setattr("vector_memory.embedding.EMBED_MODEL", "other-model:1b")
    monkeypatch.setattr(fp, "force_model_mismatch", lambda: True)
    out = core.save_memory("forced write", {}, collection="vm_fp_override")
    assert "Memory saved" in out  # unsafe override documented


def test_legacy_points_without_model_ok(local_qdrant):
    from qdrant_client.models import PointStruct

    from vector_memory import core

    local_qdrant.create_collection("vm_fp_legacy", vectors_config={"size": 64, "distance": "Cosine"})
    local_qdrant.upsert("vm_fp_legacy", points=[
        PointStruct(id="11111111-1111-1111-1111-111111111111",
                    vector=[0.5] * 64, payload={"text": "no model recorded"}),
    ], wait=True)
    out = core.search_memory("no model recorded", collection="vm_fp_legacy")
    assert "no model recorded" in out


def test_reembed_copies_and_never_touches_source(local_qdrant):
    from vector_memory import core
    from vector_memory.reembed import reembed_collection

    core.save_memory("alpha doc", {"project": "p"}, collection="vm_re_src")
    core.save_memory("beta doc", {"project": "p"}, collection="vm_re_src")
    src_payloads = {p.id: dict(p.payload) for p in _scroll(local_qdrant, "vm_re_src")}

    summary = reembed_collection("vm_re_src", "vm_re_dst", "qwen3-embedding:8b")
    assert summary["copied"] == 2
    assert summary["target_count"] == 2

    dst_payloads = {p.id: dict(p.payload) for p in _scroll(local_qdrant, "vm_re_dst")}
    assert set(dst_payloads) == set(src_payloads)  # same IDs
    for pid, old in src_payloads.items():
        assert dst_payloads[pid]["text"] == old["text"]
        assert dst_payloads[pid]["project"] == old.get("project")
    # Source untouched (same payloads as before the reembed call).
    assert {p.id: dict(p.payload) for p in _scroll(local_qdrant, "vm_re_src")} == src_payloads


def test_reembed_resume_skips_existing(local_qdrant):
    from vector_memory import core
    from vector_memory.reembed import reembed_collection

    core.save_memory("r1", {}, collection="vm_re_r")
    core.save_memory("r2", {}, collection="vm_re_r")
    reembed_collection("vm_re_r", "vm_re_r2", "qwen3-embedding:8b")
    summary = reembed_collection("vm_re_r", "vm_re_r2", "qwen3-embedding:8b", resume=True)
    assert summary["copied"] == 0
    assert summary["skipped"] == 2


def test_reembed_same_collection_rejected(local_qdrant):
    from vector_memory.errors import ConfigError
    from vector_memory.reembed import reembed_collection

    with pytest.raises(ConfigError, match="must differ"):
        reembed_collection("vm_x", "vm_x", "qwen3-embedding:8b")


def _scroll(client, name):
    points, offset = [], None
    while True:
        batch, offset = client.scroll(name, limit=256, offset=offset,
                                      with_payload=True, with_vectors=False)
        if not batch:
            break
        points.extend(batch)
        if offset is None:
            break
    return points
