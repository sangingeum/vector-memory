"""Filter DSL v2 tests: compile + evaluate against a pure-Python reference."""

from __future__ import annotations

import random

import pytest

from tests import local_backend as _lb

local_qdrant = _lb.local_qdrant

COLL = "vm_dsl"


def _reference_eval(payload: dict, expr: dict) -> bool:
    """Pure-Python evaluator mirroring the DSL (ground truth for property test)."""
    for key, value in expr.items():
        if key == "$not":
            if _reference_eval(payload, value):
                return False
            continue
        if key == "$or":
            if not any(_reference_eval(payload, b) for b in value):
                return False
            continue
        if isinstance(value, dict):
            pv = payload.get(key)
            if pv is None or not isinstance(pv, (int, float)):
                return False
            for op, operand in value.items():
                ok = {
                    "gt": pv > operand, "gte": pv >= operand,
                    "lt": pv < operand, "lte": pv <= operand,
                }[op]
                if not ok:
                    return False
            continue
        if isinstance(value, list):
            pv = payload.get(key)
            if not (isinstance(pv, list) and any(v in pv for v in value)):
                return False
            continue
        if payload.get(key) != value:
            return False
    return True


@pytest.fixture(scope="module")
def dsl_collection():
    from qdrant_client import QdrantClient

    from tests.local_backend import HashEmbedder
    from vector_memory import embedding, store

    client = QdrantClient(":memory:")
    store.qdrant = client
    embedding.ollama_client = HashEmbedder()
    from qdrant_client.models import PointStruct

    client.create_collection(COLL, vectors_config={"size": 64, "distance": "Cosine"})
    payloads = [
        {"project": "a", "type": "decision", "tags": ["urgent"], "importance": 3, "n": 10},
        {"project": "a", "type": "fact", "tags": ["arch"], "importance": 5, "n": 20},
        {"project": "b", "type": "decision", "tags": ["urgent", "arch"], "importance": 1, "n": 30},
        {"project": "b", "type": "note", "tags": [], "importance": 4, "n": 40},
        {"project": "c", "type": "fact", "tags": ["arch"], "importance": 2, "n": 50},
    ]
    client.upsert(COLL, points=[
        PointStruct(id=f"11111111-1111-1111-1111-{i:012d}", vector=[0.1 + i * 0.01] * 64,
                    payload=dict(p, text=f"doc {i}"))
        for i, p in enumerate(payloads)
    ], wait=True)
    return payloads


def _search(expr: dict) -> set[str]:
    from vector_memory import core

    out = core.search_memory("doc", limit=10, filter=expr, collection=COLL)
    return {line.split("ID: ")[1].split("]")[0] for line in out.splitlines()
            if "ID: " in line}


def test_scalar_list_semantics(dsl_collection):
    assert _search({"project": "a"}) == {
        "11111111-1111-1111-1111-000000000000",
        "11111111-1111-1111-1111-000000000001",
    }
    assert len(_search({"tags": ["urgent"]})) == 2  # any-of


def test_range_operator(dsl_collection):
    got = _search({"importance": {"gte": 4}})
    assert got == {"11111111-1111-1111-1111-000000000001",
                   "11111111-1111-1111-1111-000000000003"}


def test_negation(dsl_collection):
    got = _search({"$not": {"type": "decision"}})
    assert "11111111-1111-1111-1111-000000000000" not in got
    assert "11111111-1111-1111-1111-000000000004" in got


def test_disjunction(dsl_collection):
    got = _search({"$or": [{"project": "a"}, {"tags": ["arch"]}]})
    assert "11111111-1111-1111-1111-000000000004" in got
    assert "11111111-1111-1111-1111-000000000003" not in got  # project b, tags urgent+arch? b has arch!
    # b's decision has tags [urgent, arch] — it MUST match $or via tags.
    assert "11111111-1111-1111-1111-000000000002" in got


def test_unknown_operator_rejected(dsl_collection):
    from vector_memory import core
    from vector_memory.errors import ArgumentError

    with pytest.raises(ArgumentError, match="unknown range operator"):
        core.search_memory("doc", filter={"n": {"between": [1, 5]}}, collection=COLL)


def test_range_on_non_numeric_rejected(dsl_collection):
    from vector_memory.errors import ArgumentError
    from vector_memory.filters import compile_filter

    with pytest.raises(ArgumentError, match="numeric"):
        compile_filter({"project": {"gte": "alpha"}})


@pytest.mark.parametrize("seed", range(12))
def test_property_dsl_matches_reference(dsl_collection, seed):
    """Random expressions: Qdrant result == pure-Python reference evaluator."""
    rng = random.Random(seed)
    fields = {
        "project": ["a", "b", "c"],
        "type": ["decision", "fact", "note"],
        "tags": [["urgent"], ["arch"], ["urgent", "arch"], []],
        "importance": {"gte": [1, 3, 4]},
        "n": {"gt": [15, 35]},
    }
    expr: dict = {}
    for key in rng.sample(list(fields), rng.randint(1, 3)):
        value = fields[key]
        if isinstance(value, dict):
            op = rng.choice(list(value))
            expr[key] = {op: rng.choice(value[op])}
        elif key == "tags":
            expr[key] = rng.choice(value) or ["none"]
        else:
            expr[key] = rng.choice(value)
    got = _search(expr)
    expected = {
        f"11111111-1111-1111-1111-{i:012d}"
        for i, p in enumerate(dsl_collection)
        if _reference_eval(p, expr)
    }
    assert got == expected
