"""Safe bulk deletion (delete-by-filter dry-run default, delete-collection)."""

from __future__ import annotations

from typing import Any

from . import store as _store
from .browse import _resolve, _scroll_all
from .errors import ArgumentError
from .filters import build_filter, merge_convenience


def delete_by_filter(
    filter: str | dict[str, Any], collection: str = "",
    project: str = "", *,
    no_dry_run: bool = False, yes: bool = False,
) -> str:
    """Delete points matching the filter. DRY-RUN by default: prints what
    would be deleted (up to 10 samples). Real deletion needs --no-dry-run --yes.
    An empty/unspecified filter is rejected.
    """
    name = _resolve(collection)
    flt, _ = build_filter(filter)
    flt = merge_convenience(flt, project=project)
    if flt is None:
        raise ArgumentError(
            "refusing to delete the whole collection; use delete-collection")
    points = _scroll_all(name, flt)
    if not points:
        return f"Nothing to delete in {name!r} (0 matching points)."
    sample = "\n".join(
        f"  {p.id} {(p.payload or {}).get('text', '')[:80]!r}"
        for p in points[:10]
    )
    if not (no_dry_run and yes):
        return (
            f"would delete {len(points)} points in {name!r} (dry-run; pass "
            f"--no-dry-run --yes to really delete):\n{sample}"
        )
    _store.qdrant.delete(
        collection_name=name,
        points_selector=[p.id for p in points],
        wait=True,
    )
    return f"Deleted {len(points)} points from {name!r}"


def delete_collection(name: str, confirm: str = "", default_collection: str = "") -> str:
    """Delete a WHOLE collection; refuses mismatches and the default collection."""
    resolved_default = default_collection or _store.COLLECTION_NAME
    if name != confirm:
        raise ArgumentError(
            f"confirmation mismatch: pass --confirm {name} to delete {name!r}")
    if name == resolved_default:
        raise ArgumentError(
            f"refusing to delete the configured default collection {name!r}; "
            "pass --i-know-this-is-default if you are certain")
    if not _store.qdrant.collection_exists(name):
        raise KeyError(f"collection {name!r} does not exist")
    _store.qdrant.delete_collection(name)
    return f"Deleted collection {name!r}"
