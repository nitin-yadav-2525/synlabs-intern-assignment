"""
HTTP service.

Run with:  uvicorn app.main:app --reload --port 8000

Endpoints:
  POST /ingest            -> re-ingest the corpus (idempotent)
  POST /query              -> {"question": "...", "k": 5, "where": {"doc_type": "md"}}
  GET  /health

Every query is logged (one JSON line per request) to logs/query_log.jsonl
with latency, chunk_count, and token usage, per the assignment requirement.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.ingest import ingest_corpus
from app.retrieval import retrieve
from app.llm import answer_question
from app.embeddings import embedding_info

app = FastAPI(title="Cost-Efficient RAG Service")

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "query_log.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None
    where: Optional[Dict[str, Any]] = None


def _log_query(record: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "embedding": embedding_info()}


@app.post("/ingest")
def ingest():
    return ingest_corpus()


@app.post("/query")
def query(req: QueryRequest):
    t0 = time.time()
    hits = retrieve(req.question, k=req.k, where=req.where)
    result = answer_question(req.question, hits)
    total_latency_ms = round((time.time() - t0) * 1000, 1)

    record = {
        "question": req.question,
        "k": req.k or settings.DEFAULT_TOP_K,
        "chunk_count": result["chunk_count"],
        "used_llm": result["used_llm"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "retrieval_latency_ms": round(total_latency_ms - result["latency_ms"], 1),
        "generation_latency_ms": result["latency_ms"],
        "total_latency_ms": total_latency_ms,
    }
    _log_query(record)
    logger.info("query served: %s", record)

    return {
        "answer": result["answer"],
        "sources": [
            {"chunk_id": h["chunk_id"], "source": h["metadata"].get("source"), "similarity": round(h["similarity"], 4)}
            for h in hits
        ],
        "metrics": record,
    }
