"""
Core judging calls: one pointwise score, one pairwise A-vs-B comparison.

Judging mode note (per assignment: "implement at least one, explain when
each fits"): we implement BOTH.
  - Pointwise is used for single-config suite evaluation (Problem 2.3.1) -
    it's cheaper (one call per case) and gives an absolute score you can
    threshold into a pass rate.
  - Pairwise is used for the A/B comparison (prompt v1 vs v2, or model A vs
    B) and for the position-bias check, because a direct head-to-head is a
    much more sensitive way to detect a real difference between two close
    configs than comparing two absolute pointwise scores that both hover
    around "pretty good".
"""
from __future__ import annotations
from judge.config import settings
from judge.prompts import build_pointwise_prompt, build_pairwise_prompt
from judge.schema import parse_verdict, Verdict
from judge.client import call_model


def pointwise_score(input_, system_prompt, model_output, expected_output="", criteria_notes="", tag="pointwise") -> Verdict:
    prompt = build_pointwise_prompt(input_, system_prompt, model_output, expected_output, criteria_notes)
    result = call_model(prompt, settings.JUDGE_PROVIDER, settings.JUDGE_MODEL, max_tokens=1500, tag=tag)
    return parse_verdict(result["text"])


def pairwise_compare(input_, system_prompt, output_a, output_b, expected_output="", tag="pairwise") -> dict:
    """Returns {"verdict": Verdict, "winner": "A"/"B"/"tie"}."""
    prompt = build_pairwise_prompt(input_, system_prompt, output_a, output_b, expected_output)
    result = call_model(prompt, settings.JUDGE_PROVIDER, settings.JUDGE_MODEL, max_tokens=1500, tag=tag)
    verdict = parse_verdict(result["text"])

    winner = "parse_error"
    if not verdict.parse_error:
        import json as _json
        import re as _re
        match = _re.search(r"\{.*\}", result["text"], _re.DOTALL)
        if match:
            try:
                data = _json.loads(match.group(0))
                winner = data.get("winner", "tie")
            except Exception:
                winner = "tie" if verdict.overall_score == 3 else ("A" if (verdict.overall_score or 3) > 3 else "B")
    return {"verdict": verdict, "winner": winner}
