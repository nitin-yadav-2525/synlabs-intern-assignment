"""
Ingestion pipeline:
  read PDF/HTML/MD -> clean text -> chunk (size + overlap) -> embed -> upsert

Idempotency: each chunk's id is a SHA-256 hash of (source_path + chunk_text).
Re-ingesting the same corpus produces the same ids, so Chroma's `upsert`
overwrites in place instead of creating duplicate vectors. This is verified
in eval/idempotency_check.py.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import settings
from app.embeddings import embed_texts
from app.vectorstore import get_collection


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    doc_type: str
    chunk_index: int


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _read_md_or_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in (".html", ".htm"):
        return _read_html(path)
    if ext in (".md", ".txt"):
        return _read_md_or_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple, predictable fixed-window character chunking with overlap.
    Defaults: chunk_size=800 chars, overlap=120 chars (see app/config.py).
    Chosen so a chunk is ~1-2 paragraphs — enough context for the LLM to
    ground an answer, small enough that retrieval stays precise (see
    README "Design Decisions" for what happened at other settings).
    """
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size]
        if piece.strip():
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_id(source: str, text: str) -> str:
    h = hashlib.sha256(f"{source}::{text}".encode("utf-8")).hexdigest()
    return h[:32]


def build_chunks_for_file(path: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    raw = load_document(path)
    pieces = chunk_text(raw, chunk_size, overlap)
    doc_type = path.suffix.lower().lstrip(".")
    chunks = []
    for i, piece in enumerate(pieces):
        cid = chunk_id(str(path), piece)
        chunks.append(Chunk(id=cid, text=piece, source=str(path.name), doc_type=doc_type, chunk_index=i))
    return chunks


def ingest_corpus(corpus_dir: str | None = None, chunk_size: int | None = None, overlap: int | None = None) -> dict:
    corpus_dir = Path(corpus_dir or settings.CORPUS_DIR)
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    all_chunks: List[Chunk] = []
    files = [p for p in corpus_dir.rglob("*") if p.suffix.lower() in (".pdf", ".html", ".htm", ".md", ".txt")]
    for path in files:
        all_chunks.extend(build_chunks_for_file(path, chunk_size, overlap))

    if not all_chunks:
        return {"files_seen": len(files), "chunks_upserted": 0}

    collection = get_collection()
    vectors = embed_texts([c.text for c in all_chunks])
    collection.upsert(
        ids=[c.id for c in all_chunks],
        embeddings=vectors,
        documents=[c.text for c in all_chunks],
        metadatas=[
            {"source": c.source, "doc_type": c.doc_type, "chunk_index": c.chunk_index}
            for c in all_chunks
        ],
    )
    return {
        "files_seen": len(files),
        "chunks_upserted": len(all_chunks),
        "total_vectors_in_store": collection.count(),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(ingest_corpus(), indent=2))
