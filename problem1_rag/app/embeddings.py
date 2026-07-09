"""
Embedding backends.

- "sentence-transformers": real semantic embeddings (all-MiniLM-L6-v2, 384-dim).
  Requires internet the FIRST time (downloads the model from HuggingFace) then
  runs fully offline/local. This is the backend you should use for real grading.

- "hash": a deterministic bag-of-words hashing embedding. No downloads, no
  internet, no GPU. Used only so the pipeline is runnable/testable in
  network-restricted environments (e.g. CI, sandboxes). It gives much weaker
  retrieval quality — do not use this for your actual evaluation numbers.

Model + dimensionality are recorded here because the assignment explicitly
asks you to state them.
"""
from __future__ import annotations
import hashlib
import re
from typing import List
import numpy as np

from app.config import settings

_st_model = None  # lazy singleton for sentence-transformers


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _st_model


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic offline embedding: hashed bag-of-words, L2-normalized."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using the configured backend."""
    if settings.EMBEDDING_BACKEND == "hash":
        return [_hash_embed(t, settings.EMBEDDING_DIM) for t in texts]

    model = _get_st_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def embedding_info() -> dict:
    return {
        "backend": settings.EMBEDDING_BACKEND,
        "model": settings.EMBEDDING_MODEL if settings.EMBEDDING_BACKEND == "sentence-transformers" else "hash-bow",
        "dimensionality": settings.EMBEDDING_DIM,
    }
