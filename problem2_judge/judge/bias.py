"""
Bias mitigations that need to run extra judge calls (as opposed to the ones
baked directly into the prompt text in prompts.py: sycophancy grounding,
verbosity instruction, score-clustering anchors).

Implements:
  - Position bias: run_pairwise_both_orders() runs A-vs-B and B-vs-A and
    reports whether the winner flips (the "flip rate" the assignment asks
    for). Mitigation policy: if the two orders disagree, we record the case
    as "no confident winner" rather than picking one arbitrarily.
  - Verbosity bias: probe_verbosity() takes a real output, builds a padded
    version (same content + repeated filler that adds nothing new), and
    checks whether the judge scores the padded version higher. If it does,
    that's a verbosity-bias hit.
  - Self-enhancement bias: mitigated at the config level (JUDGE_MODEL vs
    GENERATOR_MODEL are independent, see judge/config.py) rather than at
    call time; there is nothing extra to "run" here beyond using a
    different model for the judge, which report.py records for every run.
"""
from __future__ import annotations
from judge.core import pairwise_compare


def run_pairwise_both_orders(input_, system_prompt, output_a, output_b, expected_output="") -> dict:
    forward = pairwise_compare(input_, system_prompt, output_a, output_b, expected_output, tag="pairwise_AB")
    backward = pairwise_compare(input_, system_prompt, output_b, output_a, expected_output, tag="pairwise_BA")

    # normalize backward's winner back into A/B/tie terms relative to the ORIGINAL a/b
    backward_winner_normalized = {"A": "B", "B": "A", "tie": "tie", "parse_error": "parse_error"}.get(
        backward["winner"], "parse_error"
    )

    flipped = (
        forward["winner"] in ("A", "B")
        and backward_winner_normalized in ("A", "B")
        and forward["winner"] != backward_winner_normalized
    )

    if flipped:
        consensus = "no_confident_winner"
    elif forward["winner"] == backward_winner_normalized:
        consensus = forward["winner"]
    else:
        consensus = "no_confident_winner"

    return {
        "forward_winner": forward["winner"],
        "backward_winner_raw": backward["winner"],
        "backward_winner_normalized": backward_winner_normalized,
        "flipped": flipped,
        "consensus_winner": consensus,
        "forward_verdict": forward["verdict"].to_dict(),
        "backward_verdict": backward["verdict"].to_dict(),
    }


PADDING_FILLER = (
    " To elaborate further and provide additional context, it is worth "
    "restating that this point is important and should be considered "
    "carefully, as it reflects a broader consideration relevant to the "
    "overall matter at hand."
)


def make_padded_variant(text: str, repeats: int = 2) -> str:
    """Adds filler that repeats the same non-information, to test whether
    the judge rewards length that adds nothing new."""
    return text + (PADDING_FILLER * repeats)


def probe_verbosity(input_, system_prompt, model_output, expected_output="") -> dict:
    from judge.core import pointwise_score

    original_verdict = pointwise_score(input_, system_prompt, model_output, expected_output, tag="verbosity_original")
    padded_output = make_padded_variant(model_output)
    padded_verdict = pointwise_score(input_, system_prompt, padded_output, expected_output, tag="verbosity_padded")

    orig_score = original_verdict.overall_score
    padded_score = padded_verdict.overall_score
    biased = (
        orig_score is not None and padded_score is not None and padded_score > orig_score
    )

    return {
        "original_score": orig_score,
        "padded_score": padded_score,
        "verbosity_bias_detected": biased,
        "original_verdict": original_verdict.to_dict(),
        "padded_verdict": padded_verdict.to_dict(),
    }
