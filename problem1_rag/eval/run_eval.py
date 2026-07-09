"""
Run with:  python -m eval.run_eval
Reads eval/questions.json, runs each question through retrieval + generation,
computes retrieval metrics, answer metrics (F1 + LLM-judge faithfulness/
relevance), and latency percentiles. Writes eval/results/results.json.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve
from app.llm import answer_question
from eval import metrics as m
from eval.judge_answer import judge_answer

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results" / "results.json"

TOP_K = 5
USE_LLM_JUDGE = True  # set False to skip judge calls (faster, no API cost)


def run():
    questions = json.loads(QUESTIONS_PATH.read_text())
    per_question = []
    latencies = []

    for q in questions:
        t0 = time.time()
        hits = retrieve(q["question"], k=TOP_K)
        retrieval_ms = (time.time() - t0) * 1000
        result = answer_question(q["question"], hits)
        total_ms = retrieval_ms + result["latency_ms"]
        latencies.append(total_ms)

        retrieved_sources = [h["metadata"].get("source") for h in hits]

        row = {
            "id": q["id"],
            "question": q["question"],
            "gold_source": q["gold_source"],
            "in_corpus": q["in_corpus"],
            "retrieved_sources": retrieved_sources,
            "answer": result["answer"],
            "used_llm": result["used_llm"],
            "chunk_count": result["chunk_count"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "retrieval_latency_ms": round(retrieval_ms, 1),
            "generation_latency_ms": result["latency_ms"],
            "total_latency_ms": round(total_ms, 1),
            "hit_rate": m.hit_rate_at_k(retrieved_sources, q["gold_source"]),
            "reciprocal_rank": m.reciprocal_rank(retrieved_sources, q["gold_source"]),
            "ndcg": m.ndcg_at_k(retrieved_sources, q["gold_source"]),
            "context_precision": m.context_precision(retrieved_sources, q["gold_source"]),
            "f1": m.f1_score(result["answer"], q["gold_answer"]),
            "exact_match": m.exact_match(result["answer"], q["gold_answer"]),
            "no_hallucination_correct": m.no_hallucination_correct(result["used_llm"], q["in_corpus"]),
        }

        if USE_LLM_JUDGE and q["in_corpus"]:
            verdict = judge_answer(q["question"], [h["text"] for h in hits], result["answer"])
            row["judge_faithfulness"] = verdict.get("faithfulness")
            row["judge_relevance"] = verdict.get("relevance")
        else:
            row["judge_faithfulness"] = None
            row["judge_relevance"] = None

        per_question.append(row)
        print(f"[{q['id']}] hit_rate={row['hit_rate']} f1={row['f1']} latency_ms={row['total_latency_ms']:.0f}")

    in_corpus_rows = [r for r in per_question if r["in_corpus"]]

    def avg(key, rows=in_corpus_rows):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted)//2]
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
    p95 = latencies_sorted[p95_idx]

    summary = {
        "k": TOP_K,
        "n_questions": len(questions),
        "n_in_corpus": len(in_corpus_rows),
        "n_out_of_corpus": len(questions) - len(in_corpus_rows),
        "retrieval": {
            "hit_rate_at_k": avg("hit_rate"),
            "mrr": avg("reciprocal_rank"),
            "ndcg_at_k": avg("ndcg"),
            "context_precision": avg("context_precision"),
        },
        "answer_quality": {
            "f1": avg("f1"),
            "exact_match": avg("exact_match"),
            "judge_faithfulness_avg_1to5": avg("judge_faithfulness"),
            "judge_relevance_avg_1to5": avg("judge_relevance"),
        },
        "no_hallucination_accuracy_all_questions": round(
            sum(r["no_hallucination_correct"] for r in per_question) / len(per_question), 4
        ),
        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "mean": round(statistics.mean(latencies), 1),
        },
        "tokens": {
            "total_input_tokens": sum(r["input_tokens"] for r in per_question),
            "total_output_tokens": sum(r["output_tokens"] for r in per_question),
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
