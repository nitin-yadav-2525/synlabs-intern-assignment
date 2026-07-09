# Problem 2 — LLM-as-Judge Evaluation Pipeline

Turns `{ input, system_prompt, model_output, expected_output?, criteria? }`
into a structured quality verdict, with concrete bias detection and
mitigation baked into the prompts and the pipeline code.

## 1. Prerequisites

- Python 3.10+, any OS
- Groq API key (judge + generator both default to Groq-hosted models —
  Llama for the generator, Gemma for the judge; see note on self-enhancement
  bias below for using a second provider)

## 2. Setup

```bash
cd problem2_judge
python -m venv venv
venv\Scripts\activate        # Windows. Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # Windows. Mac/Linux: cp .env.example .env
# edit .env, paste your GROQ_API_KEY (and OPENAI_API_KEY if you have one)
```

### Environment variables
`GROQ_API_KEY`, `OPENAI_API_KEY` (optional), `GENERATOR_PROVIDER`,
`GENERATOR_MODEL`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `PASS_THRESHOLD`,
`LOG_DIR`. Judge and generator are configured through **separate** env vars
specifically so they can point at different model families.

## 3. Run a suite → produce a report

```bash
python run_suite.py
```

This single command produces every artifact the submission template asks for:
- `results/suite_report.json` — pass rate, mean scores per criterion, parse-error rate
- `results/position_bias.json` — A/B-order-swap flip rate
- `results/verbosity_probe.json` — padded-vs-original score comparison
- `results/judge_validation.json` — Cohen's kappa vs human labels, test-retest
  consistency, adversarial-probe fooled rate
- `logs/judge_log.jsonl` — every raw judge prompt + raw response (auditable/replayable)

## 4. Run an A/B comparison

```bash
python run_ab.py
```

Compares two system-prompt configs (`concise_prompt` vs `explained_prompt`,
edit `CONFIG_A`/`CONFIG_B` at the top of `run_ab.py` to compare whatever
you're actually testing — two prompt versions, or two `GENERATOR_MODEL`
values). Produces `results/ab_comparison.json` with a declared winner from
both a pointwise suite-report comparison and a pairwise head-to-head with
position-bias order swap.

## Judging mode(s) used + when each fits

**Pointwise** (`judge/core.py: pointwise_score`) — used for single-config
suite grading (`run_suite.py`). Cheaper (one call per case), gives an
absolute score you can threshold into a pass rate, and is the only mode that
makes sense when there's nothing to compare against yet.

**Pairwise A-vs-B** (`judge/core.py: pairwise_compare`) — used for the A/B
comparison and the position-bias check (`run_ab.py`, `judge/bias.py`). A
direct head-to-head is far more sensitive than comparing two absolute
pointwise averages when both configs are already "pretty good" — small,
real differences show up much more clearly as a win rate than as a
0.1-point gap in a 1-5 average.

## Bias handling — what's implemented where

| Bias | Where it's mitigated |
|---|---|
| Position (A/B order) | `judge/bias.py: run_pairwise_both_orders` — runs every pair forward AND backward, reports `flipped` per pair and an aggregate flip rate in `results/position_bias.json` |
| Verbosity / length | Baked into the prompt (`judge/prompts.py: VERBOSITY_INSTRUCTION`) + measured directly by `judge/bias.py: probe_verbosity`, which pads a real answer with content-free filler and checks whether the score goes up |
| Self-enhancement | `judge/config.py` — `JUDGE_MODEL`/`JUDGE_PROVIDER` are configured fully independently of `GENERATOR_MODEL`/`GENERATOR_PROVIDER`. By default the generator uses Groq's Llama family and the judge uses Groq's Gemma family (different model families, same provider); point `JUDGE_PROVIDER=openai` at `gpt-4o-mini` for an even more independent cross-vendor judge — see the note in `judge/config.py` docstring |
| Sycophancy / style | Baked into the prompt (`judge/prompts.py: GROUNDING_INSTRUCTION`) — every per-criterion rationale must point to specific evidence, not a holistic impression; the confidently-wrong adversarial probes in `suites/adversarial_probes.json` test whether this actually works |
| Score clustering | Few-shot calibration anchors in every prompt (`judge/prompts.py: FEW_SHOT_ANCHORS`) pin down what a 1 / 3 / 5 concretely look like |

## Judge validation (at least one implemented — we did all three)

- **Agreement with human labels**: `judge/validate.py: cohens_kappa`, computed
  against the `human_pass` field already present in every `suites/test_suite.json`
  case.
- **Test-retest consistency**: `judge/validate.py: test_retest_consistency`
  runs the same case through the judge 5x and reports the pass/fail flip rate.
- **Adversarial probe set**: `judge/validate.py: run_adversarial_probes` runs
  `suites/adversarial_probes.json` (verbose-but-wrong / terse-but-correct
  pairs) and reports the fooled rate.

---

## Design Decisions & Trade-offs (draft — verify against YOUR actual run's
numbers and put it in your own words before pasting into the submission doc)

**Which judging mode did you pick, and why does it fit this case?**
Both: pointwise for absolute per-config quality (needed for a pass rate and
per-criterion breakdown), pairwise for the A/B comparison and position-bias
measurement specifically, because bias measurement requires a same-input
head-to-head — a pointwise score alone can't reveal an order effect.

**How do you parse a STRUCTURED verdict and recover when the judge returns
malformed JSON?** Three layers in `judge/schema.py: parse_verdict` —
straight `json.loads`, then a regex-extracted `{...}` block re-parsed, then
(if both fail) a `Verdict(parse_error=True)` with no scores rather than a
guessed value. Callers exclude `parse_error` cases from every average and
report the parse-error rate as its own number (`aggregate_pointwise`), so a
spike in malformed responses is visible instead of silently averaged away.

**Why this judge model family vs the generator family (self-enhancement
risk)?** [State what you actually ran with — if you only had one provider
key, say so plainly and note that a cross-family run (e.g. GPT-4o-mini
judging Claude output, or vice versa) is the recommended real mitigation,
per the config-level independence built into `judge/config.py`.]

**Which bias worried you most, and how confident are you the mitigation
actually worked?** [Fill in from your own `results/position_bias.json` and
`results/verbosity_probe.json` numbers — e.g. "position bias flip rate was
X% across N pairs, which tells me..." Be honest if a mitigation only
partially worked; that's a better answer than claiming it's fully solved.]

**Would you let this judge gate a release? Under what guardrails — and
where would you keep a human in the loop?** A reasonable default: yes for
non-safety-critical regressions (e.g. blocking a prompt-version rollout if
pass rate drops materially), no as the sole signal for safety-relevant
criteria — route any case scoring low on `safety` to mandatory human review
rather than auto-blocking or auto-passing based on the judge alone, and
periodically re-run the human-label kappa check to catch judge drift over
time.
