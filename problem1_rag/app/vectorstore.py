"""
Vector store: ChromaDB running in embedded/local persistent mode.

Why Chroma over pgvector/Qdrant/LanceDB/FAISS/sqlite-vec (see README 1.4 for
the full write-up): it needs zero always-on server process (unlike a managed
pod or a self-hosted Postgres/Qdrant instance you'd have to keep running),
persists to a folder on disk, supports metadata filtering out of the box, and
has a trivial embedded Python API — which matters a lot for a "prove it's a
credible low-cost alternative" exercise where the whole point is: no idle
compute bill for a lightly-queried index.
"""
from __future__ import annotations
import chromadb
from app.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Delete and recreate the collection. Used only by eval tooling / manual reset."""
    client = get_client()
    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()
