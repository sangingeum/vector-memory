"""Strict validation table tests (VM-01).

Each invalid input must raise ArgumentError, produce no Qdrant write, and
render as a single ``ArgumentError: ...`` line. Lenient mode reproduces the
legacy warn-and-continue behavior.
"""

from __future__ import annotations

import pytest

from tests import local_backend as _lb  # registers the local_qdrant fixture
from vector_memory import errors

# The fixture is resolved by pytest from the module attribute; keep an alias
# so ruff does not treat the fixture name as an unused/param redefinition.
local_qdrant = _lb.local_qdrant
from vector_memory.validation import (
    MAX_TEXT_CHARS,
    content_hash,
    normalize_text,
    validate_metadata,
    validate_text,
)


def test_normalize_text_form():
    assert normalize_text("  Hello   world ") == "Hello world"
    assert normalize_text("ＡＢＣ") == "ＡＢＣ"  # NFC keeps width, just composes


def test_content_hash_stable_and_different():
    a = content_hash("Postgres runs on port 5432")
    b = content_hash("Postgres runs\non port 5432")
    c = content_hash("Postgres runs on port 5433")
    assert a == b
    assert a != c


def test_empty_text_rejected():
    with pytest.raises(errors.ArgumentError, match="text is empty"):
        validate_text("   ")


def test_non_string_text_rejected():
    with pytest.raises(errors.ArgumentError, match="must be a string"):
        validate_text(123)  # type: ignore[arg-type]


def test_text_too_long_rejected():
    with pytest.raises(errors.ArgumentError, match="too long"):
        validate_text("x" * (MAX_TEXT_CHARS + 1))


@pytest.mark.parametrize(
    "meta, message",
    [
        ("{not json", "metadata is not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ({"Bad-Key": "v"}, "key 'Bad-Key' is invalid"),
        ({"9bad": "v"}, "key '9bad' is invalid"),
        ({"_internal": "v"}, "key '_internal' is invalid"),
        ({"nested": {"a": 1}}, "scalar or a list of scalars"),
        ({"tags": ["ok", 5]}, "tags.*list of strings"),
        ({"project": 5}, "project.*string"),
        ({"importance": "high"}, "importance.*number"),
        ({"summary": "x" * 201}, "key 'summary' too long"),
    ],
)
def test_invalid_metadata_no_write(meta, message, local_qdrant):
    """Strict mode: error raised, nothing written to the collection."""
    from vector_memory import core

    with pytest.raises(errors.ArgumentError, match=message):
        core.save_memory("some text", meta, collection="vm_strict_test")
    if local_qdrant.collection_exists("vm_strict_test"):
        info = local_qdrant.get_collection("vm_strict_test")
        assert (info.points_count or 0) == 0


def test_valid_metadata_accepted(local_qdrant):
    from vector_memory import core

    out = core.save_memory(
        "some text",
        {"project": "p", "tags": ["a"], "importance": 3, "trust": "low"},
        collection="vm_strict_test",
    )
    assert "Memory saved" in out


def test_unknown_type_warns_but_saves(local_qdrant, recwarn):
    from vector_memory import core

    out = core.save_memory("some text", {"type": "gate-verdict"},
                           collection="vm_strict_test")
    assert "Memory saved" in out
    assert any("unknown type 'gate-verdict'" in str(w.message) for w in recwarn.list)


def test_key_length_limit():
    with pytest.raises(errors.ArgumentError, match="key .* is invalid"):
        validate_metadata({"a" * 65: "v"})


def test_max_key_length_ok():
    validate_metadata({"a" * 64: "v"})
