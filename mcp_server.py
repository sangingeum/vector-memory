"""Back-compat shim: re-export the package API under the old module name.

KEPT (deliberate, advisory finding 4): old `python mcp_server.py` invocations
and the live smoke test still work after the rename. The real entry points
are `vector-memory` (CLI) and `vector-memory-mcp` (MCP stdio server).

The implementation moved into the ``vector_memory`` package (embedding.py /
store.py / core.py / server.py). Importing ``mcp_server`` still works for the
repo-root entry point and the live smoke test.
"""

from vector_memory import *  # noqa: F401,F403
from vector_memory import (  # noqa: F401  (explicit for tools)
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    delete_memory,
    list_collections,
    main,
    mcp,
    qdrant,
    save_memories,
    save_memory,
    search_memory,
    update_memory,
)

if __name__ == "__main__":
    main()
