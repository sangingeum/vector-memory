"""System fields + migration tests (system-fields work item).

Uses the local-mode Qdrant fixture; legacy points must stay readable and
searchable after migration.
"""

from __future__ import annotations

from qdrant_client.models import PointStruct

# Fixture registration: pytest resolves `local_qdrant` from this module's
# attribute; importing it by name would shadow the fixture parameter (F811).
from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant
from vector_memory import payload as pl


def _collect(client, name):
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


def test_new_save_carries_system_fields(local_qdrant):
    from vector_memory import core

    core.save_memory("hello world", {}, collection="vm_sys")
    (point,) = _collect(local_qdrant, "vm_sys")
    p = point.payload
    assert pl.CREATED_TS in p and pl.UPDATED_TS in p
    assert pl.CREATED_AT in p and pl.UPDATED_AT in p
    assert pl.CONTENT_HASH in p
    assert p[pl.EMBED_MODEL] == "qwen3-embedding:8b"
    assert pl.STATUS not in p  # missing = active


def test_update_bumps_updated_ts_and_hash(local_qdrant):
    from vector_memory import core

    out = core.save_memory("first text", {}, collection="vm_sys")
    pid = out.split("ID: ")[1].split(",")[0].strip()
    created = _collect(local_qdrant, "vm_sys")[0].payload[pl.CREATED_TS]
    core.update_memory(pid, "second text", collection="vm_sys")
    (point,) = _collect(local_qdrant, "vm_sys")
    assert point.payload[pl.UPDATED_TS] >= created
    assert point.payload[pl.CONTENT_HASH] == pl.content_hash("second text")
    assert point.payload[pl.CREATED_TS] == created  # created preserved


def test_active_filter_must_not_form(local_qdrant):
    """must_not form: legacy points without _status still match."""
    f = pl.active_filter()
    must_not = f.must_not
    assert isinstance(must_not, list) and len(must_not) == 1
    assert not f.must


def _seed_legacy(client, name, legacy: dict[str, dict]):
    client.create_collection(name, vectors_config={"size": 64, "distance": "Cosine"})
    client.upsert(
        name,
        points=[
            PointStruct(id=pid, vector=[0.1] * 64, payload=pay)
            for pid, pay in legacy.items()
        ],
        wait=True,
    )


def test_migrate_fixture(local_qdrant):
    from vector_memory.migrate import migrate_collection

    legacy = {
        "11111111-1111-1111-1111-111111111111": {"text": "legacy one", "project": "p"},
        "22222222-2222-2222-2222-222222222222": {"text": "legacy two"},
    }
    _seed_legacy(local_qdrant, "vm_mig", legacy)
    before = {p.id: dict(p.payload) for p in _collect(local_qdrant, "vm_mig")}

    summary = migrate_collection("vm_mig", assume_model="qwen3-embedding:8b")
    assert summary["scanned"] == 2 and summary["migrated"] == 2

    after = {p.id: dict(p.payload) for p in _collect(local_qdrant, "vm_mig")}
    for pid, old in before.items():
        assert after[pid]["text"] == old["text"]
        for k, v in old.items():
            if not k.startswith("_"):
                assert after[pid][k] == v
        assert after[pid][pl.CREATED_ESTIMATED] is True
        assert after[pid][pl.EMBED_MODEL] == "qwen3-embedding:8b"
        assert after[pid][pl.CONTENT_HASH] == pl.content_hash(old["text"])

    # Re-run is a no-op (resumable).
    summary2 = migrate_collection("vm_mig", assume_model="qwen3-embedding:8b")
    assert summary2["migrated"] == 0


def test_migrate_dry_run_touches_nothing(local_qdrant):
    from typer.testing import CliRunner

    from vector_memory.cli import app as cli_app

    _seed_legacy(local_qdrant, "vm_dry",
                 {"33333333-3333-3333-3333-333333333333": {"text": "x"}})
    before = {p.id: dict(p.payload) for p in _collect(local_qdrant, "vm_dry")}
    result = CliRunner().invoke(cli_app, ["migrate", "--collection", "vm_dry", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run: 1/1" in result.output
    after = {p.id: dict(p.payload) for p in _collect(local_qdrant, "vm_dry")}
    assert after == before
