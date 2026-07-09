"""
IR metrics (Recall@k/Hit Rate, MRR, nDCG@k, context precision) and a simple
token-overlap EM/F1 for answer strings. All computed from scratch (no
external eval library) so the numbers are auditable.

Note on EM/F1: our gold_answer values are short human paraphrases, not
exact strings copied from the corpus, so raw Exact Match will almost always
be 0. We report token-level F1 as the more meaningful of the two and are
explicit about this limitation in the README (a dataset with literal gold
spans, e.g. SQuAD-style, would make EM meaningful).
"""
from __future__ import annotations
import math
import re
from typing import List, Optional


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def hit_rate_at_k(retrieved_sources: List[str], gold_source: Optional[str]) -> Optional[int]:
    if gold_source is None:
        return None
    return int(gold_source in retrieved_sources)


def reciprocal_rank(retrieved_sources: List[str], gold_source: Optional[str]) -> Optional[float]:
    if gold_source is None:
        return None
    for i, s in enumerate(retrieved_sources):
        if s == gold_source:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_sources: List[str], gold_source: Optional[str]) -> Optional[float]:
    if gold_source is None:
        return None
    # Binary relevance against a SINGLE relevant document. If that document's
    # chunks appear more than once in retrieved_sources (common in a small
    # corpus), only the first occurrence counts as relevant -- otherwise DCG
    # can accumulate gain from repeated "hits" of the same document and push
    # the score above the nDCG@k <= 1 bound, since IDCG assumes exactly one
    # relevant item total (matching the single-gold-source setup used here).
    dcg = 0.0
    matched = False
    for i, s in enumerate(retrieved_sources):
        rel = 1.0 if (s == gold_source and not matched) else 0.0
        if rel:
            matched = True
        dcg += rel / math.log2(i + 2)
    idcg = 1.0 / math.log2(2)  # best case: the single relevant doc at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved_sources: List[str], gold_source: Optional[str]) -> Optional[float]:
    """Fraction of retrieved chunks that are actually relevant (here: match
    the single gold source; extend to multi-relevant-chunk sets if you label
    more than one relevant chunk per question)."""
    if gold_source is None or not retrieved_sources:
        return None
    relevant = sum(1 for s in retrieved_sources if s == gold_source)
    return relevant / len(retrieved_sources)


def exact_match(pred: str, gold: Optional[str]) -> Optional[int]:
    if gold is None:
        return None
    return int(" ".join(_tokenize(pred)) == " ".join(_tokenize(gold)))


def f1_score(pred: str, gold: Optional[str]) -> Optional[float]:
    if gold is None:
        return None
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def no_hallucination_correct(used_llm: bool, in_corpus: bool) -> int:
    """Correct behavior: use the LLM only when the question is in-corpus."""
    return int(used_llm == in_corpus)
