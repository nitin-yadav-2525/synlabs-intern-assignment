"""
Judge validation, implemented from scratch (no external stats library) so
the numbers are auditable:

  - Cohen's kappa: agreement between judge pass/fail and human pass/fail
    labels, corrected for chance agreement.
  - Test-retest consistency: run the judge N times on the SAME case and see
    how often the pass/fail verdict flips.
  - Adversarial probes: run the judge on hand-built verbose-but-wrong and
    terse-but-correct cases and report whether it was fooled.
"""
from __future__ import annotations
from typing import List, Tuple
from judge.config import settings
from judge.core import pointwise_score


def cohens_kappa(judge_labels: List[bool], human_labels: List[bool]) -> float:
    """Both lists are pass/fail booleans for the same N cases, same order."""
    n = len(judge_labels)
    assert n == len(human_labels) and n > 0

    po = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n

    p_judge_pass = sum(judge_labels) / n
    p_human_pass = sum(human_labels) / n
    pe = (p_judge_pass * p_human_pass) + ((1 - p_judge_pass) * (1 - p_human_pass))

    if pe == 1:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def test_retest_consistency(case: dict, n_runs: int = 5) -> dict:
    """Runs the SAME case through the judge n_runs times and reports how
    often the pass/fail verdict flips relative to the first run."""
    verdicts = []
    for i in range(n_runs):
        v = pointwise_score(
            case["input"], case["system_prompt"], case["model_output"],
            case.get("expected_output", ""), case.get("criteria", ""),
            tag=f"retest_run{i}",
        )
        verdicts.append(v)

    scores = [v.overall_score for v in verdicts if v.overall_score is not None]
    passes = [s >= settings.PASS_THRESHOLD for s in scores]
    if not passes:
        return {"n_runs": n_runs, "flip_rate": None, "scores": scores, "note": "all runs parse-errored"}

    baseline = passes[0]
    flips = sum(1 for p in passes[1:] if p != baseline)
    flip_rate = round(flips / max(len(passes) - 1, 1), 4)

    return {
        "n_runs": n_runs,
        "scores": scores,
        "pass_fail_sequence": passes,
        "flip_rate": flip_rate,
        "score_stdev": round(_stdev(scores), 4) if len(scores) > 1 else 0.0,
    }


def _stdev(vals: List[float]) -> float:
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return var ** 0.5


def run_adversarial_probes(probes: List[dict]) -> dict:
    """probes: list of {"id","type" ("verbose_wrong"|"terse_correct"), "input",
    "system_prompt", "model_output", "expected_output", "should_pass": bool}.
    Returns per-probe result + fooled rate."""
    results = []
    for p in probes:
        v = pointwise_score(
            p["input"], p["system_prompt"], p["model_output"],
            p.get("expected_output", ""), tag=f"adversarial_{p['id']}",
        )
        predicted_pass = (v.overall_score or 0) >= settings.PASS_THRESHOLD
        fooled = predicted_pass != p["should_pass"]
        results.append({
            "id": p["id"],
            "type": p["type"],
            "overall_score": v.overall_score,
            "predicted_pass": predicted_pass,
            "should_pass": p["should_pass"],
            "fooled": fooled,
        })

    fooled_count = sum(1 for r in results if r["fooled"])
    return {
        "n_probes": len(probes),
        "fooled_count": fooled_count,
        "fooled_rate": round(fooled_count / len(probes), 4) if probes else None,
        "results": results,
    }
