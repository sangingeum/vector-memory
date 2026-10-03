"""Safe bulk deletion tests (delete-by-filter, delete-collection)."""

from __future__ import annotations

import pytest

from tests import local_backend as _lb
from tests.local_backend import HashEmbedder

local_qdrant = _lb.local_qdrant

COLL = "vm_bulk"


@pytest.fixture()
def seeded(local_qdrant, monkeypatch):
    from vector_memory import core

    monkeypatch.setattr("vector_memory.embedding.ollama_client", HashEmbedder())
    monkeypatch.setattr("vector_memory.store.qdrant", local_qdrant)
    for i in range(6):
        core.save_memory(f"bulk doc {i}", {"project": f"p{i % 2}"}, collection=COLL)
    return local_qdrant


def _count(client, name):
    return client.count(collection_name=name, exact=True).count


def test_dry_run_default_deletes_nothing(seeded):
    from vector_memory import bulk

    out = bulk.delete_by_filter({"project": "p0"}, COLL)
    assert "would delete 3" in out
    assert _count(seeded, COLL) == 6  # untouched


def test_empty_filter_refused(seeded):
    from vector_memory import bulk
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="refusing to delete the whole collection"):
        bulk.delete_by_filter(None, COLL)


def test_real_deletion_needs_both_flags(seeded):
    from vector_memory import bulk

    out = bulk.delete_by_filter({"project": "p0"}, COLL, no_dry_run=True, yes=False)
    assert "would delete" in out  # --yes missing -> still dry-run
    out2 = bulk.delete_by_filter({"project": "p0"}, COLL, no_dry_run=True, yes=True)
    assert "Deleted 3" in out2
    assert _count(seeded, COLL) == 3


def test_delete_collection_confirmation_mismatch(seeded):
    from vector_memory import bulk
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="confirmation mismatch"):
        bulk.delete_collection("other-coll", confirm="wrong")


def test_delete_collection_default_refused(seeded, monkeypatch):
    from vector_memory import bulk
    from vector_memory.errors import ArgumentError

    monkeypatch.setattr("vector_memory.store.COLLECTION_NAME", COLL)
    with pytest.raises(ArgumentError, match="default collection"):
        bulk.delete_collection(COLL, confirm=COLL)


def test_delete_multi_ids(seeded):
    from vector_memory import core

    pts, _ = seeded.scroll(COLL, limit=2, with_payload=True)
    ids = [p.id for p in pts]
    out = core.delete_many(ids + ["22222222-2222-2222-2222-222222222222"], COLL)
    assert "Deleted 2" in out and "not found" in out
    assert _count(seeded, COLL) == 4
