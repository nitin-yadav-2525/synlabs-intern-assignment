"""
Run with:  python run_suite.py
Full pipeline: loads suites/test_suite.json, judges every case (pointwise),
writes results/suite_report.json. Also runs:
  - position-bias check on a handful of pairs (results/position_bias.json)
  - verbosity probe (results/verbosity_probe.json)
  - judge validation: kappa vs human_pass labels, test-retest consistency,
    adversarial probes (results/judge_validation.json)

This is intentionally one script so a grader can run one command and get
every artifact the template's 2.3 section asks for.
"""
import json
from pathlib import Path

from judge.core import pointwise_score
from judge.aggregate import aggregate_pointwise
from judge.bias import run_pairwise_both_orders, probe_verbosity
from judge.validate import cohens_kappa, test_retest_consistency, run_adversarial_probes
from judge.client import get_call_stats, reset_call_stats

SUITE_PATH = Path("suites/test_suite.json")
PROBES_PATH = Path("suites/adversarial_probes.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_main_suite():
    cases = json.loads(SUITE_PATH.read_text())
    rows = []
    for case in cases:
        verdict = pointwise_score(
            case["input"], case["system_prompt"], case["model_output"],
            case.get("expected_output", ""), tag=f"suite_{case['id']}",
        )
        rows.append({"id": case["id"], "verdict": verdict.to_dict(), "human_pass": case.get("human_pass")})
        print(f"[{case['id']}] overall={verdict.overall_score} parse_error={verdict.parse_error}")

    report = aggregate_pointwise(rows)
    report["per_case"] = rows
    (RESULTS_DIR / "suite_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== SUITE REPORT ===")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    return cases, rows, report


def run_position_bias_check(cases):
    """Pairs up consecutive cases with the same input template (c01/c02,
    c03/c04, ...) as A-vs-B comparisons and checks the flip rate."""
    results = []
    for i in range(0, min(len(cases), 10), 2):
        a, b = cases[i], cases[i + 1]
        r = run_pairwise_both_orders(a["input"], a["system_prompt"], a["model_output"], b["model_output"])
        r["pair"] = f"{a['id']}_vs_{b['id']}"
        results.append(r)
        print(f"[{r['pair']}] forward={r['forward_winner']} backward_norm={r['backward_winner_normalized']} flipped={r['flipped']}")

    flip_rate = round(sum(1 for r in results if r["flipped"]) / len(results), 4) if results else None
    out = {"flip_rate": flip_rate, "pairs": results}
    (RESULTS_DIR / "position_bias.json").write_text(json.dumps(out, indent=2))
    print(f"\nPosition-bias flip rate: {flip_rate}")
    return out


def run_verbosity_check(cases):
    case = cases[6]  # c07: photosynthesis explanation, good short example
    result = probe_verbosity(case["input"], case["system_prompt"], case["model_output"], case.get("expected_output", ""))
    (RESULTS_DIR / "verbosity_probe.json").write_text(json.dumps(result, indent=2))
    print(f"\nVerbosity probe: original={result['original_score']} padded={result['padded_score']} "
          f"bias_detected={result['verbosity_bias_detected']}")
    return result


def run_validation(cases, rows):
    # Cohen's kappa: judge pass/fail vs human_pass labels
    from judge.config import settings
    judge_pass = []
    human_pass = []
    for r in rows:
        if r["verdict"]["overall_score"] is None:
            continue
        judge_pass.append(r["verdict"]["overall_score"] >= settings.PASS_THRESHOLD)
        human_pass.append(r["human_pass"])
    kappa = cohens_kappa(judge_pass, human_pass) if judge_pass else None
    print(f"\nCohen's kappa (judge vs human pass/fail): {kappa}")

    # Test-retest consistency on one case
    retest = test_retest_consistency(cases[0], n_runs=5)
    print(f"Test-retest flip rate on case {cases[0]['id']}: {retest['flip_rate']}")

    # Adversarial probes
    probes = json.loads(PROBES_PATH.read_text())
    adv = run_adversarial_probes(probes)
    print(f"Adversarial probe fooled rate: {adv['fooled_rate']}")

    out = {"cohens_kappa": kappa, "test_retest": retest, "adversarial_probes": adv}
    (RESULTS_DIR / "judge_validation.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    reset_call_stats()
    cases, rows, report = run_main_suite()
    run_position_bias_check(cases)
    run_verbosity_check(cases)
    run_validation(cases, rows)
    print("\n=== JUDGE CALL/TOKEN STATS ===")
    print(json.dumps(get_call_stats(), indent=2))
