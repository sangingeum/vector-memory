"""Filter DSL v2: JSON filter -> Qdrant Filter (documented, validated).

Grammar (see docs/filter-dsl.md):
- scalar            -> exact match
- list              -> any-of (MatchAny)
- {"op": {...}}     -> range with gt/gte/lt/lte on a numeric field
- {"$not": <expr>}  -> negation (must_not)
- {"$or": [expr..]} -> disjunction (should, min_should 1)
Top-level keys AND together. Unknown operators, range operators on
non-numeric values, and non-object expressions raise ArgumentError in strict
mode (the lenient escape warns and drops the filter, legacy behavior).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    MinShould,
    Range,
)

from .errors import ArgumentError
from .validation import lenient

RANGE_OPS = ("gt", "gte", "lt", "lte")
_LOGICAL_OPS = ("$not", "$or")
# Convenience-flag keys that compile to the same filter object.
_CONVENIENCE_KEYS = ("project", "type", "tags", "source")
_SINCE_RE = re.compile(r"^(\d+)([dh])$")


def parse_relative_time(value: str, *, future: bool = False) -> float:
    """`7d|24h|2026-09-01` -> epoch seconds."""
    m = _SINCE_RE.match(value.strip())
    if m:
        amount, unit = int(m.group(1)), m.group(2)
        seconds = amount * 86400 if unit == "d" else amount * 3600
        now = time.time()
        return now - seconds if not future else now + seconds
    try:
        import datetime

        return datetime.datetime.fromisoformat(value).timestamp()
    except ValueError as exc:
        raise ArgumentError(
            f"time value {value!r} is invalid (use 7d, 24h, or an ISO date)"
        ) from exc


def _field_condition(key: str, value: Any) -> FieldCondition:
    if isinstance(value, list):
        return FieldCondition(key=key, match=MatchAny(any=value))
    return FieldCondition(key=key, match=MatchValue(value=value))


def _range_condition(key: str, spec: dict[str, Any]) -> FieldCondition:
    for op, operand in spec.items():
        if op not in RANGE_OPS:
            raise ArgumentError(f"unknown range operator {op!r} (known: {', '.join(RANGE_OPS)})")
        if not isinstance(operand, (int, float)):
            raise ArgumentError(
                f"range operator {op!r} on field {key!r} needs a numeric value "
                f"(got {operand!r})"
            )
    kwargs = {op: spec[op] for op in RANGE_OPS if op in spec}
    return FieldCondition(key=key, range=Range(**kwargs))


def _sub_expr_to_filter(expr: dict[str, Any]) -> Filter:
    """A nested expression (used under $not / $or) -> Filter."""
    sub = compile_filter(expr)
    return sub if sub is not None else Filter()


def compile_filter(parsed: dict[str, Any]) -> Filter | None:
    """Compile an already-parsed filter object; None means "no filter"."""
    if not isinstance(parsed, dict) or not parsed:
        return None
    must: list[Any] = []
    must_not: list[Any] = []
    should: list[Any] = []
    for key, value in parsed.items():
        if key == "$not":
            if not isinstance(value, dict):
                raise ArgumentError("$not expects an object expression")
            must_not.append(_sub_expr_to_filter(value))
        elif key == "$or":
            if not isinstance(value, list) or not value:
                raise ArgumentError("$or expects a non-empty list of expressions")
            for branch in value:
                if not isinstance(branch, dict):
                    raise ArgumentError("$or branches must be objects")
                should.append(_sub_expr_to_filter(branch))
        elif isinstance(value, dict):
            must.append(_range_condition(key, value))
        else:
            must.append(_field_condition(key, value))
    if not (must or must_not or should):
        return None
    return Filter(
        must=must or None,
        must_not=must_not or None,
        should=should or None,
        min_should=None if not should else MinShould(conditions=[], min_count=1),
    )


def build_filter(filter_json: str | dict[str, Any] | None) -> tuple[Filter | None, str | None]:
    """Parse + validate + compile a filter (CLI/MCP entry).

    Returns ``(filter, warning)``: strict mode raises ArgumentError on invalid
    input; lenient mode returns ``(None, warning)`` (search unfiltered).
    """
    from .store import WARN_INVALID_FILTER

    if filter_json is None:
        return None, None
    if isinstance(filter_json, dict):
        parsed: Any = filter_json
    elif isinstance(filter_json, str) and filter_json.strip():
        try:
            parsed = json.loads(filter_json)
        except json.JSONDecodeError as exc:
            if lenient():
                return None, WARN_INVALID_FILTER
            raise ArgumentError(f"filter is not valid JSON (position {exc.pos})") from exc
    else:
        return None, None
    try:
        return compile_filter(parsed), None
    except ArgumentError:
        if lenient():
            return None, WARN_INVALID_FILTER
        raise


def merge_convenience(
    flt: Filter | None,
    *,
    project: str = "",
    type: str = "",
    tag: list[str] | None = None,
    source: str = "",
    since: str = "",
    before: str = "",
) -> Filter | None:
    """Compile convenience flags into the same filter (AND of everything)."""
    extra: dict[str, Any] = {}
    if project:
        extra["project"] = project
    if type:
        extra["type"] = type
    if tag:
        extra["tags"] = list(tag)
    if source:
        extra["source"] = source
    if since:
        extra["_created_ts"] = {"gte": parse_relative_time(since)}
    if before:
        extra["_created_ts"] = {"lt": parse_relative_time(before, future=False)}
    if since and before:
        extra["_created_ts"] = {
            "gte": parse_relative_time(since),
            "lt": parse_relative_time(before),
        }
    if not extra:
        return flt
    convenience = compile_filter(extra)
    if flt is None:
        return convenience
    return Filter(
        must=[*(flt.must or []), *(convenience.must or [])],
        must_not=[*(flt.must_not or []), *(convenience.must_not or [])],
        should=[*(flt.should or []), *(convenience.should or [])],
    )
