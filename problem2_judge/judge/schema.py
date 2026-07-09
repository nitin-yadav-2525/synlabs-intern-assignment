"""
Structured verdict schema + robust parsing of the judge's raw text response.

Malformed-JSON handling strategy (three layers, cheapest first):
  1. Try json.loads on the raw response directly.
  2. Regex-extract the first {...} block and try again (handles the model
     wrapping JSON in prose or markdown fences).
  3. If both fail, return a Verdict with parse_error=True and score=None for
     every criterion rather than crashing or silently guessing a score -
     callers (aggregate.py) must exclude parse_error cases from averages and
     report the parse-failure rate separately.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

RUBRIC_CRITERIA = ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]


@dataclass
class CriterionScore:
    name: str
    score: Optional[int]  # 1-5
    rationale: str


@dataclass
class Verdict:
    criteria: List[CriterionScore] = field(default_factory=list)
    overall_score: Optional[float] = None
    parse_error: bool = False
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "criteria": [{"name": c.name, "score": c.score, "rationale": c.rationale} for c in self.criteria],
            "overall_score": self.overall_score,
            "parse_error": self.parse_error,
        }


def _extract_json_block(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_verdict(raw_text: str, criteria_names: List[str] = RUBRIC_CRITERIA) -> Verdict:
    data = _extract_json_block(raw_text)
    if data is None:
        return Verdict(parse_error=True, raw_response=raw_text)

    try:
        criteria = []
        scores_for_avg = []
        for name in criteria_names:
            entry = data.get("criteria", {}).get(name) if isinstance(data.get("criteria"), dict) else None
            if entry is None:
                # tolerate a flat structure too: {"correctness": 4, "correctness_rationale": "..."}
                score = data.get(name)
                rationale = data.get(f"{name}_rationale", "")
            else:
                score = entry.get("score")
                rationale = entry.get("rationale", "")
            score = int(score) if score is not None else None
            if score is not None:
                scores_for_avg.append(score)
            criteria.append(CriterionScore(name=name, score=score, rationale=rationale))

        overall = data.get("overall_score")
        if overall is None:
            overall = round(sum(scores_for_avg) / len(scores_for_avg), 2) if scores_for_avg else None
        else:
            overall = float(overall)

        return Verdict(criteria=criteria, overall_score=overall, parse_error=False, raw_response=raw_text)
    except (ValueError, TypeError, AttributeError):
        return Verdict(parse_error=True, raw_response=raw_text)
