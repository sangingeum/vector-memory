"""Duplicate handling tests (exact idempotency + near-duplicate reporting)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant


def _pid(out: str) -> str:
    return out.split("ID: ")[1].split(",")[0].strip()


def test_exact_duplicate_is_idempotent(local_qdrant):
    from vector_memory import core

    a = core.save_memory("Postgres runs on port 5432 for staging", {"project": "p"},
                         collection="vm_dup")
    b = core.save_memory("Postgres runs on port 5432 for staging", {"project": "p", "v": 2},
                         collection="vm_dup")
    ida, idb = _pid(a), _pid(b.split("\n")[0])
    assert ida == idb  # same normalized text -> same point
    assert "duplicate of existing; updated" in b
    pts, _ = local_qdrant.scroll("vm_dup", limit=10, with_payload=True)
    assert len(pts) == 1
    assert pts[0].payload["v"] == 2  # metadata merged/refreshed


def test_whitespace_normalization_still_idempotent(local_qdrant):
    from vector_memory import core

    a = core.save_memory("hello   world", {}, collection="vm_dup2")
    b = core.save_memory("hello world", {}, collection="vm_dup2")
    assert _pid(a) == _pid(b.split("\n")[0])


def test_allow_duplicate_forces_new_point(local_qdrant):
    from vector_memory import core

    a = core.save_memory("same text", {}, collection="vm_dup3")
    b = core.save_memory("same text", {}, collection="vm_dup3", allow_duplicate=True)
    assert _pid(a) != _pid(b)


def test_near_duplicate_reported_not_merged(local_qdrant, monkeypatch):
    from vector_memory import core

    monkeypatch.setenv("VM_DEDUPE_THRESHOLD", "-1.0")  # hash embedder: report everything
    a = core.save_memory("first fact", {}, collection="vm_near")
    out = core.save_memory("second unrelated fact", {}, collection="vm_near")
    assert "similar" in out
    assert _pid(a) in out  # candidate line carries the existing id
    assert "does not imply equivalence" in out


def test_on_similar_skip_returns_existing_without_write(local_qdrant, monkeypatch):
    from vector_memory import core

    monkeypatch.setenv("VM_DEDUPE_THRESHOLD", "-1.0")
    monkeypatch.setenv("VM_ON_SIMILAR", "skip")
    a = core.save_memory("keep me", {}, collection="vm_skip")
    out = core.save_memory("totally different", {}, collection="vm_skip")
    existing_id = _pid(a)
    assert existing_id in out  # returns the existing id
    pts, _ = local_qdrant.scroll("vm_skip", limit=10, with_payload=True)
    assert len(pts) == 1  # nothing new written


def test_on_similar_error_raises(local_qdrant, monkeypatch):
    from vector_memory import core
    from vector_memory.errors import ConflictError

    monkeypatch.setenv("VM_DEDUPE_THRESHOLD", "-1.0")
    core.save_memory("existing fact", {}, collection="vm_err")
    with pytest.raises(ConflictError, match="near-duplicate"):
        core.save_memory("another fact", {}, collection="vm_err", on_similar="error")


def test_threshold_configurable(local_qdrant, monkeypatch):
    from vector_memory import dedupe

    monkeypatch.setenv("VM_DEDUPE_THRESHOLD", "0.5")
    assert dedupe.dedupe_threshold() == 0.5
    monkeypatch.delenv("VM_DEDUPE_THRESHOLD")
    assert dedupe.dedupe_threshold() == dedupe.DEDUPE_THRESHOLD == 0.985
