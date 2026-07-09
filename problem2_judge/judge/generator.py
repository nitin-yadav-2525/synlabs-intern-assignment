"""
Generates a model_output for a given (input, system_prompt) pair, using
whatever GENERATOR_PROVIDER/GENERATOR_MODEL is configured. Used by run_ab.py
to actually produce two sets of outputs (prompt v1 vs v2, or model A vs B)
before handing them to the judge.
"""
from __future__ import annotations
from judge.config import settings
from judge.client import call_model


def generate(input_: str, system_prompt: str, provider: str | None = None, model: str | None = None, tag="generate") -> str:
    result = call_model(
        prompt=input_,
        provider=provider or settings.GENERATOR_PROVIDER,
        model=model or settings.GENERATOR_MODEL,
        system=system_prompt,
        max_tokens=500,
        tag=tag,
    )
    return result["text"]
