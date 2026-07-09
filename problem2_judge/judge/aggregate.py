"""
Aggregates per-case Verdicts into a suite-level report.
"""
from __future__ import annotations
from typing import List, Dict, Any
from judge.config import settings
from judge.schema import RUBRIC_CRITERIA


def aggregate_pointwise(rows: List[Dict[str, Any]]) -> dict:
    """rows: list of {"id":..., "verdict": Verdict.to_dict(), ...}"""
    valid = [r for r in rows if not r["verdict"]["parse_error"]]
    parse_error_rate = round(1 - len(valid) / len(rows), 4) if rows else None

    overall_scores = [r["verdict"]["overall_score"] for r in valid if r["verdict"]["overall_score"] is not None]
    mean_overall = round(sum(overall_scores) / len(overall_scores), 3) if overall_scores else None
    pass_rate = (
        round(sum(1 for s in overall_scores if s >= settings.PASS_THRESHOLD) / len(overall_scores), 4)
        if overall_scores else None
    )

    per_criterion_means = {}
    for crit in RUBRIC_CRITERIA:
        scores = []
        for r in valid:
            for c in r["verdict"]["criteria"]:
                if c["name"] == crit and c["score"] is not None:
                    scores.append(c["score"])
        per_criterion_means[crit] = round(sum(scores) / len(scores), 3) if scores else None

    return {
        "n_cases": len(rows),
        "n_valid": len(valid),
        "parse_error_rate": parse_error_rate,
        "mean_overall_score": mean_overall,
        "pass_threshold": settings.PASS_THRESHOLD,
        "pass_rate": pass_rate,
        "per_criterion_means": per_criterion_means,
    }


def compare_configs(report_a: dict, name_a: str, report_b: dict, name_b: str) -> dict:
    """Declares a winner between two pointwise suite reports (e.g. prompt v1
    vs v2). Winner = higher mean_overall_score, provided pass_rate agrees;
    if they disagree, flag as inconclusive rather than force a call."""
    score_a = report_a.get("mean_overall_score")
    score_b = report_b.get("mean_overall_score")
    pass_a = report_a.get("pass_rate")
    pass_b = report_b.get("pass_rate")

    if score_a is None or score_b is None:
        winner = "inconclusive"
    elif score_a == score_b:
        winner = "tie"
    else:
        by_score = name_a if score_a > score_b else name_b
        by_pass = None
        if pass_a is not None and pass_b is not None and pass_a != pass_b:
            by_pass = name_a if pass_a > pass_b else name_b
        winner = by_score if (by_pass is None or by_pass == by_score) else "inconclusive (score and pass_rate disagree)"

    return {
        name_a: {"mean_overall_score": score_a, "pass_rate": pass_a},
        name_b: {"mean_overall_score": score_b, "pass_rate": pass_b},
        "winner": winner,
    }


def aggregate_pairwise_win_rate(results: List[dict]) -> dict:
    """results: list of dicts with a 'consensus_winner' key (A/B/tie/no_confident_winner)."""
    n = len(results)
    wins_a = sum(1 for r in results if r["consensus_winner"] == "A")
    wins_b = sum(1 for r in results if r["consensus_winner"] == "B")
    ties = sum(1 for r in results if r["consensus_winner"] in ("tie", "no_confident_winner"))
    flips = sum(1 for r in results if r.get("flipped"))
    return {
        "n_pairs": n,
        "a_win_rate": round(wins_a / n, 4) if n else None,
        "b_win_rate": round(wins_b / n, 4) if n else None,
        "tie_or_no_confident_rate": round(ties / n, 4) if n else None,
        "position_bias_flip_rate": round(flips / n, 4) if n else None,
    }
