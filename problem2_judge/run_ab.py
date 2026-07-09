# """
# Run with:  python run_ab.py
# Compares two configs (e.g. two system prompts, "concise" vs "detailed") on
# the same inputs from suites/test_suite.json:
#   1. generate an output from each config for every input
#   2. judge each config's outputs pointwise -> suite report per config
#   3. also run a direct pairwise comparison per input (with position-bias
#      order swap) for a second, more sensitive signal
#   4. declare a winner

# Edit CONFIG_A / CONFIG_B below to compare whatever you're actually testing
# (two prompt versions, or two GENERATOR_MODEL values via provider/model
# overrides).
# """
# import json
# from pathlib import Path

# from judge.generator import generate
# from judge.core import pointwise_score
# from judge.aggregate import aggregate_pointwise, compare_configs, aggregate_pairwise_win_rate
# from judge.bias import run_pairwise_both_orders

# SUITE_PATH = Path("suites/test_suite.json")
# RESULTS_DIR = Path("results")
# RESULTS_DIR.mkdir(exist_ok=True)

# CONFIG_A = {
#     "name": "concise_prompt",
#     "system_prompt": "Answer the user's question as briefly and directly as possible. No preamble.",
# }
# CONFIG_B = {
#     "name": "explained_prompt",
#     "system_prompt": "Answer the user's question and briefly explain your reasoning in 1-2 sentences.",
# }


# def run_config(cases, config):
#     rows = []
#     outputs = {}
#     for case in cases:
#         output = generate(case["input"], config["system_prompt"], tag=f"gen_{config['name']}_{case['id']}")
#         outputs[case["id"]] = output
#         verdict = pointwise_score(case["input"], config["system_prompt"], output,
#                                    case.get("expected_output", ""), tag=f"judge_{config['name']}_{case['id']}")
#         rows.append({"id": case["id"], "verdict": verdict.to_dict()})
#         print(f"[{config['name']}][{case['id']}] overall={verdict.overall_score}")
#     report = aggregate_pointwise(rows)
#     return report, outputs


# def run():
#     cases = json.loads(SUITE_PATH.read_text())[:10]  # keep the A/B run fast; raise if you want the full suite

#     report_a, outputs_a = run_config(cases, CONFIG_A)
#     report_b, outputs_b = run_config(cases, CONFIG_B)

#     comparison = compare_configs(report_a, CONFIG_A["name"], report_b, CONFIG_B["name"])

#     pairwise_results = []
#     for case in cases:
#         r = run_pairwise_both_orders(case["input"], "See config-specific system prompts",
#                                       outputs_a[case["id"]], outputs_b[case["id"]])
#         r["case_id"] = case["id"]
#         pairwise_results.append(r)

#     pairwise_summary = aggregate_pairwise_win_rate(pairwise_results)

#     out = {
#         "config_a": CONFIG_A,
#         "config_b": CONFIG_B,
#         "pointwise_report_a": report_a,
#         "pointwise_report_b": report_b,
#         "pointwise_comparison": comparison,
#         "pairwise_summary": pairwise_summary,
#         "pairwise_pairs": pairwise_results,
#     }
#     (RESULTS_DIR / "ab_comparison.json").write_text(json.dumps(out, indent=2))

#     print("\n=== A/B COMPARISON ===")
#     print(json.dumps(comparison, indent=2))
#     print("\n=== PAIRWISE SUMMARY (with position-bias flip rate) ===")
#     print(json.dumps(pairwise_summary, indent=2))


# if __name__ == "__main__":
#     run()


"""
Run with:  python run_ab.py
Compares two configs (e.g. two system prompts, "concise" vs "detailed") on
the same inputs from suites/test_suite.json:
  1. generate an output from each config for every input
  2. judge each config's outputs pointwise -> suite report per config
  3. also run a direct pairwise comparison per input (with position-bias
     order swap) for a second, more sensitive signal
  4. declare a winner

Edit CONFIG_A / CONFIG_B below to compare whatever you're actually testing
(two prompt versions, or two GENERATOR_MODEL values via provider/model
overrides).
"""
import json
from pathlib import Path

from judge.generator import generate
from judge.core import pointwise_score
from judge.aggregate import aggregate_pointwise, compare_configs, aggregate_pairwise_win_rate
from judge.bias import run_pairwise_both_orders

SUITE_PATH = Path("suites/test_suite.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG_A = {
    "name": "concise_prompt",
    "system_prompt": "Answer the user's question as briefly and directly as possible. No preamble.",
}
CONFIG_B = {
    "name": "explained_prompt",
    "system_prompt": "Answer the user's question and briefly explain your reasoning in 1-2 sentences.",
}


def run_config(cases, config):
    rows = []
    outputs = {}
    for case in cases:
        output = generate(case["input"], config["system_prompt"], tag=f"gen_{config['name']}_{case['id']}")
        outputs[case["id"]] = output
        verdict = pointwise_score(case["input"], config["system_prompt"], output,
                                   case.get("expected_output", ""), tag=f"judge_{config['name']}_{case['id']}")
        rows.append({"id": case["id"], "verdict": verdict.to_dict()})
        print(f"[{config['name']}][{case['id']}] overall={verdict.overall_score}")
    report = aggregate_pointwise(rows)
    return report, outputs


def run():
    cases = json.loads(SUITE_PATH.read_text())[:2]  # keep the A/B run fast; raise if you want the full suite
    PAIRWISE_SAMPLE_SIZE = 1 # position-bias check is expensive (2 calls/pair); a small sample is enough evidence

    report_a, outputs_a = run_config(cases, CONFIG_A)
    report_b, outputs_b = run_config(cases, CONFIG_B)

    comparison = compare_configs(report_a, CONFIG_A["name"], report_b, CONFIG_B["name"])

    # Save the pointwise winner NOW, before the pairwise phase, so a
    # rate-limit failure below doesn't lose the (already-expensive) pointwise
    # results that already declared a winner.
    partial_out = {
        "config_a": CONFIG_A,
        "config_b": CONFIG_B,
        "pointwise_report_a": report_a,
        "pointwise_report_b": report_b,
        "pointwise_comparison": comparison,
        "pairwise_summary": None,
        "pairwise_pairs": [],
        "note": "pairwise phase not yet run or did not finish - see log above",
    }
    (RESULTS_DIR / "ab_comparison.json").write_text(json.dumps(partial_out, indent=2))
    print("\n=== A/B COMPARISON (pointwise) ===")
    print(json.dumps(comparison, indent=2))

    pairwise_results = []
    for case in cases[:PAIRWISE_SAMPLE_SIZE]:
        r = run_pairwise_both_orders(case["input"], "See config-specific system prompts",
                                      outputs_a[case["id"]], outputs_b[case["id"]])
        r["case_id"] = case["id"]
        pairwise_results.append(r)
        # re-save after every pair so progress is never lost to a later rate-limit failure
        partial_out["pairwise_pairs"] = pairwise_results
        partial_out["pairwise_summary"] = aggregate_pairwise_win_rate(pairwise_results)
        partial_out["note"] = f"pairwise phase: {len(pairwise_results)}/{PAIRWISE_SAMPLE_SIZE} pairs done"
        (RESULTS_DIR / "ab_comparison.json").write_text(json.dumps(partial_out, indent=2))

    print("\n=== PAIRWISE SUMMARY (with position-bias flip rate) ===")
    print(json.dumps(partial_out["pairwise_summary"], indent=2))


if __name__ == "__main__":
    run()