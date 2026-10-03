"""System payload fields and migration helpers (system-fields work item).

System fields are prefixed ``_`` and never settable by callers (strict
metadata validation rejects the ``_`` prefix). Legacy points without them
stay valid; filters that exclude inactive memories use ``must_not`` so legacy
points (no ``_status``) still match.
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
)

from .validation import content_hash

# Field name constants (documented in README §payload).
CREATED_TS = "_created_ts"
UPDATED_TS = "_updated_ts"
CREATED_AT = "_created_at"
UPDATED_AT = "_updated_at"
CONTENT_HASH = "_content_hash"
EMBED_MODEL = "_embed_model"
STATUS = "_status"
SUPERSEDES = "_supersedes"
SUPERSEDED_BY = "_superseded_by"
AGENT = "_agent"
CREATED_ESTIMATED = "_created_estimated"
GROUP_ID = "_group_id"
CHUNK_INDEX = "_chunk_index"

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_ARCHIVED = "archived"
KNOWN_STATUSES = (STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_ARCHIVED)

# Payload indexes created/ensured on every collection open (idempotent).
# Values are qdrant payload-schema strings accepted by both local and remote
# clients ("keyword"/"float").
PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("project", "keyword"),
    ("type", "keyword"),
    ("tags", "keyword"),
    ("source", "keyword"),
    (STATUS, "keyword"),
    (CREATED_TS, "float"),
    (UPDATED_TS, "float"),
)


def iso_now(ts: float | None = None) -> str:
    """UTC ISO-8601 rendering of an epoch timestamp."""
    import datetime

    moment = ts if ts is not None else time.time()
    return (
        datetime.datetime.fromtimestamp(moment, datetime.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def system_fields_for_new_text(text: str, embed_model: str, agent_id: str = "") -> dict[str, Any]:
    """System fields stamped on a fresh save."""
    now = time.time()
    fields: dict[str, Any] = {
        CREATED_TS: now,
        UPDATED_TS: now,
        CREATED_AT: iso_now(now),
        UPDATED_AT: iso_now(now),
        CONTENT_HASH: content_hash(text),
        EMBED_MODEL: embed_model,
    }
    if agent_id:
        fields[AGENT] = agent_id
    return fields


def system_fields_for_update(
    old_payload: dict[str, Any], new_text: str | None
) -> dict[str, Any]:
    """System fields stamped on an in-place update (bumps ``_updated_*``)."""
    now = time.time()
    fields: dict[str, Any] = {
        UPDATED_TS: now,
        UPDATED_AT: iso_now(now),
    }
    if new_text is not None:
        fields[CONTENT_HASH] = content_hash(new_text)
        if old_payload.get(CREATED_TS) is None:
            fields[CREATED_TS] = now
            fields[CREATED_AT] = iso_now(now)
            fields[CREATED_ESTIMATED] = True
    return fields


def active_filter(extra: Filter | None = None) -> Filter:
    """Filter matching memories that are not superseded/archived.

    Written as ``must_not _status in [superseded, archived]`` so legacy
    points without a ``_status`` field still match.
    """
    inactive = FieldCondition(
        key=STATUS, match=MatchAny(any=[STATUS_SUPERSEDED, STATUS_ARCHIVED])
    )
    if extra is not None:
        extra_must_not = list(extra.must_not or [])
        return Filter(
            must=list(extra.must or []),
            must_not=[inactive, *extra_must_not],
            should=list(extra.should or []),
        )
    return Filter(must_not=[inactive])


def ensure_payload_indexes(client: Any, collection_name: str) -> list[str]:
    """Create payload indexes idempotently; returns field names requested.

    "already exists" and unsupported-schema responses count as satisfied
    (concurrent creation and re-opened collections are normal).
    """
    created: list[str] = []
    for field, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
            created.append(field)
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" in message:
                continue
            continue
    return created
