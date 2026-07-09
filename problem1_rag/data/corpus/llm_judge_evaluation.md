# LLM-as-Judge Evaluation

When human review cannot scale to every model output, a strong LLM can be
used as an automated judge to score or compare outputs. This is called
LLM-as-judge evaluation.

## Judging modes

Pointwise scoring asks the judge to assign a score to a single output
against a rubric. Pairwise comparison asks the judge to pick the better of
two outputs (A vs B). Reference-based judging compares an output against a
known correct answer. Reference-free judging evaluates an output using only
the input and general criteria, without a gold answer.

## Known biases

Judges exhibit several documented biases:

- Position bias: in pairwise comparisons, judges tend to favor whichever
  answer is shown first (or second), regardless of quality.
- Verbosity bias: judges tend to favor longer answers even when the extra
  length adds no new correct information.
- Self-enhancement bias: a judge tends to rate outputs from its own model
  family more favorably than outputs from other model families.
- Sycophancy: a judge may agree with a confident-sounding but incorrect
  answer rather than checking it against the source material.

## Mitigations

Position bias is mitigated by running each pairwise comparison in both
orders and checking whether the verdict flips. Verbosity bias is mitigated
by explicitly instructing the judge to penalize unsupported length and by
testing with padded-but-empty answers. Self-enhancement bias is mitigated by
using a judge from a different model family than the generator, or by using
an ensemble of judges from different families. Sycophancy is mitigated by
forcing the judge to ground each criterion in specific evidence from the
source material rather than giving a holistic impression.

## Validating a judge

A judge's reliability can be checked by measuring agreement with human-
labeled examples (for example using Cohen's kappa), by measuring test-retest
consistency (does the same input produce the same verdict on repeated
runs?), and by running an adversarial probe set containing answers that are
verbose-but-wrong or terse-but-correct to see whether the judge is fooled by
surface features.
