"""
Grounded answer generation.

- Builds a prompt that only contains retrieved chunks (numbered, so the
  model can cite [1], [2]...).
- If no chunk clears a minimum similarity threshold, we short-circuit and
  return "no relevant context" WITHOUT calling the LLM at all — this is the
  cheapest and most reliable way to avoid hallucination on out-of-corpus
  questions (no prompt-based promise can fully stop a model from answering
  anyway, so we don't rely on the prompt alone).
- The prompt additionally instructs the model to say it doesn't know if the
  retrieved chunks don't actually answer the question, as a second layer.
"""
from __future__ import annotations
import time
from typing import List, Dict, Any

from groq import Groq

from app.config import settings

MIN_SIMILARITY = 0.15  # below this, we treat retrieval as "nothing relevant"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


NO_CONTEXT_MESSAGE = "I don't have relevant context in the corpus to answer this question."

SYSTEM_PROMPT = (
    "You are a QA assistant. You will be given numbered context chunks and a "
    "question. Answer ONLY using the given chunks. Every factual sentence in "
    "your answer must end with a citation like [1] or [2] pointing to the "
    "chunk(s) it came from. If the chunks do not contain the answer, reply "
    "exactly with: " + NO_CONTEXT_MESSAGE
)


def _build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] (source: {c['metadata'].get('source')}) {c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Context chunks:\n{context_block}\n\nQuestion: {question}\n\nAnswer with citations:"


def answer_question(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns dict with answer text, token usage, latency_ms, and whether
    retrieval was judged empty (no LLM call made in that case)."""
    t0 = time.time()

    if not chunks or all(c["similarity"] < MIN_SIMILARITY for c in chunks):
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "used_llm": False,
            "chunk_count": len(chunks),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    client = _get_client()
    prompt = _build_prompt(question, chunks)
    resp = client.chat.completions.create(
        model=settings.GENERATOR_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or ""

    return {
        "answer": text.strip(),
        "used_llm": True,
        "chunk_count": len(chunks),
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
