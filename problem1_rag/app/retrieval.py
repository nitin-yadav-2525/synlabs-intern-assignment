"""
Top-k retrieval, with an optional metadata filter (Chroma `where` clause).
Supported filter keys today: source, doc_type (see ingest.py metadata).
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any

from app.config import settings
from app.embeddings import embed_query
from app.vectorstore import get_collection


def retrieve(query: str, k: int | None = None, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    k = k or settings.DEFAULT_TOP_K
    collection = get_collection()
    q_vec = embed_query(query)

    kwargs = {"query_embeddings": [q_vec], "n_results": k}
    if where:
        kwargs["where"] = where

    result = collection.query(**kwargs)

    hits = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append({
            "chunk_id": cid,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "similarity": 1 - dist,  # cosine space: similarity = 1 - distance
        })
    return hits
