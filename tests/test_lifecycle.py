"""Supersede/archive lifecycle tests (supersede work item)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant


def _pid(out: str) -> str:
    return out.split("ID: ")[1].split(",")[0].strip()


def test_supersede_hides_old_from_default_search(local_qdrant):
    from vector_memory import core

    old = core.save_memory("we use Postgres for staging", {"project": "p"},
                           collection="vm_life")
    old_id = _pid(old)
    new = core.save_memory("we migrated staging to MySQL",
                           {"project": "p"}, collection="vm_life",
                           supersedes=[old_id])
    assert "Memory saved" in new
    new_id = _pid(new)

    default = core.search_memory("staging database", collection="vm_life")
    assert "MySQL" in default and "Postgres for staging" not in default

    with_inactive = core.search_memory("staging database", collection="vm_life",
                                       include_inactive=True)
    assert "MySQL" in with_inactive and "Postgres for staging" in with_inactive
    assert f"status=superseded -> {new_id}" in with_inactive

    # New point records what it replaced.
    pts = {p.id: p.payload for p in _scroll(local_qdrant, "vm_life")}
    assert pts[new_id]["_supersedes"] == [old_id]
    assert pts[old_id]["_status"] == "superseded"
    assert pts[old_id]["_superseded_by"] == new_id


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


def test_supersede_unknown_id_fails_before_write(local_qdrant):
    from vector_memory import core
    from vector_memory.errors import NotFoundError

    with pytest.raises(NotFoundError, match="supersede target"):
        core.save_memory("new fact", {}, collection="vm_life",
                         supersedes=["00000000-0000-0000-0000-000000000000"])
    pts = _scroll(local_qdrant, "vm_life")
    assert not pts  # nothing written


def test_archive_and_unarchive(local_qdrant):
    from vector_memory import core

    out = core.save_memory("archive me", {}, collection="vm_life")
    pid = _pid(out)
    core.set_status([pid], "archived", collection="vm_life")
    assert "archive me" not in core.search_memory("archive me", collection="vm_life")
    assert "archive me" in core.search_memory("archive me", collection="vm_life",
                                              include_inactive=True)
    core.set_status([pid], "active", collection="vm_life")
    assert "archive me" in core.search_memory("archive me", collection="vm_life")
    # Unarchive clears the marker (missing = active).
    payload = _scroll(local_qdrant, "vm_life")[0].payload
    assert "_status" not in payload


def test_legacy_points_still_searchable(local_qdrant):
    """Points without _status remain visible to the default (active) search."""
    from qdrant_client.models import PointStruct

    from vector_memory import core

    local_qdrant.create_collection("vm_legacy", vectors_config={"size": 64, "distance": "Cosine"})
    local_qdrant.upsert("vm_legacy", points=[
        PointStruct(id="11111111-1111-1111-1111-111111111111",
                    vector=[0.5] * 64, payload={"text": "old point no status"}),
    ], wait=True)
    out = core.search_memory("old point no status", collection="vm_legacy")
    assert "old point no status" in out


def test_save_supersedes_itself_rejected(local_qdrant):
    from vector_memory import core
    from vector_memory.errors import NotFoundError

    with pytest.raises(NotFoundError):
        core.save_memory("self", {}, collection="vm_life",
                         supersedes=["00000000-0000-0000-0000-000000000000"])
