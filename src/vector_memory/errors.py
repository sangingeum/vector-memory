"""Error taxonomy for vector-memory (output-contract work item).

CLI failures print exactly one stderr line ``<ErrorType>: description`` and
exit 1; MCP tools return ``isError`` with the same text. Backend errors
(Ollama/Qdrant unreachable, embedding call failed) are grouped as
:class:`BackendError`.
"""

from __future__ import annotations

VALID_ERROR_TYPES: tuple[str, ...] = (
    "ArgumentError",
    "ConfigError",
    "NotFoundError",
    "ConflictError",
    "SensitiveContentError",
    "BackendError",
    "InternalError",
)


class VectorMemoryError(Exception):
    """Base class; ``error_type`` names the taxonomy bucket."""

    error_type = "InternalError"


class ArgumentError(VectorMemoryError):
    """Invalid caller input (bad JSON, bad metadata key, missing text)."""

    error_type = "ArgumentError"


class ConfigError(VectorMemoryError):
    """Environment/configuration problem (model missing, dimension mismatch)."""

    error_type = "ConfigError"


class NotFoundError(VectorMemoryError):
    """Referenced point or collection does not exist."""

    error_type = "NotFoundError"


class ConflictError(VectorMemoryError):
    """Refused due to conflicting existing state (near-duplicate --on-similar error)."""

    error_type = "ConflictError"


class SensitiveContentError(VectorMemoryError):
    """Text matched a high-confidence secret rule."""

    error_type = "SensitiveContentError"


class BackendError(VectorMemoryError):
    """Ollama or Qdrant unreachable / failed."""

    error_type = "BackendError"


def format_error(exc: BaseException) -> str:
    """Render one stderr line: ``<ErrorType>: description``."""
    if isinstance(exc, VectorMemoryError):
        return f"{exc.error_type}: {exc}"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return f"BackendError: {exc}"
    return f"InternalError: {exc}"
