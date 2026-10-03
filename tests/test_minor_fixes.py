"""Minor-fix tests: B-10, B-14, B-17."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_minor"


@pytest.fixture()
def env(local_qdrant, monkeypatch):
    from vector_memory import embedding as emb
    from vector_memory import store as st

    monkeypatch.setattr(emb, "ollama_client", HashEmbedder(dim=64))
    monkeypatch.setattr(st, "qdrant", local_qdrant)
    return st


def test_b10_non_string_metadata_raises(env):
    from vector_memory.errors import ArgumentError
    from vector_memory.validation import parse_metadata_strict

    with pytest.raises(ArgumentError, match="must be a JSON object"):
        parse_metadata_strict(42)
    with pytest.raises(ArgumentError):
        parse_metadata_strict([1])


def test_b14_migrate_missing_collection_not_found_error():
    from vector_memory.errors import NotFoundError
    from vector_memory.migrate import migrate_collection

    with pytest.raises(NotFoundError, match="does not exist"):
        migrate_collection("vm_no_such_collection")


def test_b14_delete_collection_missing_not_found_error(env):
    from vector_memory.bulk import delete_collection
    from vector_memory.errors import NotFoundError

    with pytest.raises(NotFoundError, match="does not exist"):
        delete_collection("vm_no_such_collection", confirm="vm_no_such_collection")


def test_b17_unarchive_clears_superseded_by(env):
    """Chosen behavior: unarchive of a superseded point CLEARS the stale
    _superseded_by link (documented in README) so no dangling link remains."""
    from qdrant_client.models import PointStruct

    from vector_memory import core
    from vector_memory import store as st

    st.qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    st.qdrant.upsert(COLL, points=[PointStruct(
        id="aaaaaaaa-1111-4111-8111-111111111111", vector=[0.5] * 64,
        payload={"text": "superseded then unarchived", "_status": "superseded",
                 "_superseded_by": "bbbbbbbb-2222-4222-8222-222222222222"},
    )], wait=True)
    core.set_status(["aaaaaaaa-1111-4111-8111-111111111111"], "active", collection=COLL)
    point = st.qdrant.retrieve(COLL, ids=["aaaaaaaa-1111-4111-8111-111111111111"],
                               with_payload=True)[0]
    assert "_status" not in point.payload
    assert "_superseded_by" not in point.payload


def test_b17_unarchive_of_active_point_noop_keys(env):
    from qdrant_client.models import PointStruct

    from vector_memory import core
    from vector_memory import store as st

    st.qdrant.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    st.qdrant.upsert(COLL, points=[PointStruct(
        id="cccccccc-3333-4333-8333-333333333333", vector=[0.5] * 64,
        payload={"text": "plain active point"},
    )], wait=True)
    out = core.set_status(["cccccccc-3333-4333-8333-333333333333"], "active", collection=COLL)
    assert "Status set to active" in out
    point = st.qdrant.retrieve(COLL, ids=["cccccccc-3333-4333-8333-333333333333"],
                               with_payload=True)[0]
    assert "_status" not in point.payload
