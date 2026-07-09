"""
Lightweight LLM-as-judge for the RAG answer-quality layer (faithfulness +
relevance), scored 1-5. This is intentionally simple; Problem 2 builds the
full bias-aware judging pipeline. Using a judge here at all is optional per
the assignment ("LLM-as-judge is fine here") — we use it as one signal
alongside token-overlap F1.
"""
from __future__ import annotations
import json
import re
from groq import Groq
from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


JUDGE_PROMPT = """You are grading a RAG system's answer.

Question: {question}
Retrieved context chunks:
{context}

Model's answer: {answer}

Score two things from 1 (worst) to 5 (best):
- faithfulness: is every claim in the answer actually supported by the context chunks? (5 = fully grounded, 1 = fabricated / not supported)
- relevance: does the answer actually address the question asked? (5 = fully addresses it, 1 = off-topic)

Respond ONLY with JSON: {{"faithfulness": <int>, "relevance": <int>, "rationale": "<one sentence>"}}
"""


def judge_answer(question: str, context_chunks: list[str], answer: str) -> dict:
    client = _get_client()
    context = "\n---\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    resp = client.chat.completions.create(
        model=settings.GENERATOR_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"faithfulness": None, "relevance": None, "rationale": "PARSE_ERROR", "raw": text}
    try:
        parsed = json.loads(match.group(0))
        return parsed
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "rationale": "PARSE_ERROR", "raw": text}
