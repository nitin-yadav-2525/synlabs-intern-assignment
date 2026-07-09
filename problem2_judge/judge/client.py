# """
# Thin wrapper that calls a judge (or generator) model and logs every raw
# prompt + raw response to logs/judge_log.jsonl (requirement: "Log every judge
# prompt + raw response - auditable/replayable"). Also tracks token/call counts
# so the suite report can report judge cost.
# """
# from __future__ import annotations
# import json
# import time
# from pathlib import Path
# from typing import Optional

# from judge.config import settings

# LOG_PATH = Path(settings.LOG_DIR) / "judge_log.jsonl"
# LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

# _groq_client = None
# _openai_client = None

# _call_counter = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


# def _get_groq():
#     global _groq_client
#     if _groq_client is None:
#         from groq import Groq
#         _groq_client = Groq(api_key=settings.GROQ_API_KEY)
#     return _groq_client


# def _get_openai():
#     global _openai_client
#     if _openai_client is None:
#         import openai
#         _openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
#     return _openai_client


# def call_model(prompt: str, provider: str, model: str, system: Optional[str] = None,
#                max_tokens: int = 600, tag: str = "judge") -> dict:
#     """Returns dict with text, input_tokens, output_tokens, latency_ms. Logs every call."""
#     t0 = time.time()

#     if provider == "groq":
#         client = _get_groq()
#         messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
#         resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
#         text = resp.choices[0].message.content
#         input_tokens = resp.usage.prompt_tokens
#         output_tokens = resp.usage.completion_tokens

#     elif provider == "openai":
#         client = _get_openai()
#         messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
#         resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
#         text = resp.choices[0].message.content
#         input_tokens = resp.usage.prompt_tokens
#         output_tokens = resp.usage.completion_tokens

#     else:
#         raise ValueError(f"Unknown provider: {provider}")

#     latency_ms = round((time.time() - t0) * 1000, 1)

#     _call_counter["calls"] += 1
#     _call_counter["input_tokens"] += input_tokens
#     _call_counter["output_tokens"] += output_tokens

#     record = {
#         "tag": tag,
#         "provider": provider,
#         "model": model,
#         "prompt": prompt,
#         "response": text,
#         "input_tokens": input_tokens,
#         "output_tokens": output_tokens,
#         "latency_ms": latency_ms,
#         "timestamp": time.time(),
#     }
#     with open(LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record) + "\n")

#     return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "latency_ms": latency_ms}


# def get_call_stats() -> dict:
#     return dict(_call_counter)


# def reset_call_stats():
#     _call_counter.update({"calls": 0, "input_tokens": 0, "output_tokens": 0})




# """
# Thin wrapper that calls a judge (or generator) model and logs every raw
# prompt + raw response to logs/judge_log.jsonl (requirement: "Log every judge
# prompt + raw response - auditable/replayable"). Also tracks token/call counts
# so the suite report can report judge cost.
# """
# from __future__ import annotations
# import json
# import random
# import re
# import time
# from pathlib import Path
# from typing import Optional

# from judge.config import settings

# LOG_PATH = Path(settings.LOG_DIR) / "judge_log.jsonl"
# LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

# _groq_client = None
# _openai_client = None

# _call_counter = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


# def _get_groq():
#     global _groq_client
#     if _groq_client is None:
#         from groq import Groq
#         _groq_client = Groq(api_key=settings.GROQ_API_KEY)
#     return _groq_client


# def _get_openai():
#     global _openai_client
#     if _openai_client is None:
#         import openai
#         _openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
#     return _openai_client


# def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
#     """Groq/OpenAI rate-limit errors often say e.g. 'Please try again in 2.149s'
#     in the message body. Pull that out if present; otherwise return None and
#     let the caller fall back to exponential backoff."""
#     msg = str(exc)
#     match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
#     if match:
#         return float(match.group(1))
#     return None


# def _is_rate_limit_error(exc: Exception) -> bool:
#     status = getattr(exc, "status_code", None)
#     if status == 429:
#         return True
#     return "rate_limit" in str(exc).lower() or "429" in str(exc)


# def call_model(prompt: str, provider: str, model: str, system: Optional[str] = None,
#                max_tokens: int = 600, tag: str = "judge", max_retries: int = 5) -> dict:
#     """Returns dict with text, input_tokens, output_tokens, latency_ms. Logs every call.
#     Automatically retries with backoff on rate-limit (429) errors — this is
#     normal on Groq's free tier once several calls run back-to-back (e.g.
#     run_ab.py), it's not a bug, just a shared-capacity limit resetting every
#     minute."""
#     t0 = time.time()

#     attempt = 0
#     while True:
#         try:
#             if provider == "groq":
#                 client = _get_groq()
#                 messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
#                 resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
#                 text = resp.choices[0].message.content
#                 input_tokens = resp.usage.prompt_tokens
#                 output_tokens = resp.usage.completion_tokens

#             elif provider == "openai":
#                 client = _get_openai()
#                 messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
#                 resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
#                 text = resp.choices[0].message.content
#                 input_tokens = resp.usage.prompt_tokens
#                 output_tokens = resp.usage.completion_tokens

#             else:
#                 raise ValueError(f"Unknown provider: {provider}")

#             break  # success, exit retry loop

#         except Exception as exc:
#             if not _is_rate_limit_error(exc) or attempt >= max_retries:
#                 raise
#             wait = _extract_retry_after_seconds(exc)
#             if wait is None:
#                 wait = (2 ** attempt) + random.uniform(0, 1)  # exponential backoff + jitter
#             wait = min(wait, 60) + 0.5  # small safety margin, cap at ~60s
#             print(f"[rate limit] {tag}: attempt {attempt + 1}/{max_retries}, waiting {wait:.1f}s before retry...")
#             time.sleep(wait)
#             attempt += 1

#     latency_ms = round((time.time() - t0) * 1000, 1)

#     _call_counter["calls"] += 1
#     _call_counter["input_tokens"] += input_tokens
#     _call_counter["output_tokens"] += output_tokens

#     record = {
#         "tag": tag,
#         "provider": provider,
#         "model": model,
#         "prompt": prompt,
#         "response": text,
#         "input_tokens": input_tokens,
#         "output_tokens": output_tokens,
#         "latency_ms": latency_ms,
#         "timestamp": time.time(),
#     }
#     with open(LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record) + "\n")

#     return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "latency_ms": latency_ms}


# def get_call_stats() -> dict:
#     return dict(_call_counter)


# def reset_call_stats():
#     _call_counter.update({"calls": 0, "input_tokens": 0, "output_tokens": 0})





"""
Thin wrapper that calls a judge (or generator) model and logs every raw
prompt + raw response to logs/judge_log.jsonl (requirement: "Log every judge
prompt + raw response - auditable/replayable"). Also tracks token/call counts
so the suite report can report judge cost.
"""
from __future__ import annotations
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

from judge.config import settings

LOG_PATH = Path(settings.LOG_DIR) / "judge_log.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

_groq_client = None
_openai_client = None

_call_counter = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Groq rate-limit errors say either 'try again in 2.149s' (per-minute
    limit) or 'try again in 5m38.688s' (daily token limit, TPD). Parse both."""
    msg = str(exc)
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if match:
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return "rate_limit" in str(exc).lower() or "429" in str(exc)


def call_model(prompt: str, provider: str, model: str, system: Optional[str] = None,
               max_tokens: int = 600, tag: str = "judge", max_retries: int = 5) -> dict:
    """Returns dict with text, input_tokens, output_tokens, latency_ms. Logs every call.
    Automatically retries with backoff on rate-limit (429) errors — this is
    normal on Groq's free tier once several calls run back-to-back (e.g.
    run_ab.py), it's not a bug, just a shared-capacity limit resetting every
    minute."""
    t0 = time.time()

    attempt = 0
    while True:
        try:
            if provider == "groq":
                client = _get_groq()
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                text = resp.choices[0].message.content
                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens

            elif provider == "openai":
                client = _get_openai()
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                text = resp.choices[0].message.content
                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens

            else:
                raise ValueError(f"Unknown provider: {provider}")

            break  # success, exit retry loop

        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            real_wait = _extract_retry_after_seconds(exc)
            if real_wait is not None:
                # Groq told us exactly how long to wait (per-minute limit or
                # daily token quota) - honor it, don't clip to some small cap.
                wait = real_wait + 1.0  # small safety margin
            else:
                # No explicit wait given - fall back to capped exponential backoff.
                wait = min((2 ** attempt) + random.uniform(0, 1), 30)
            mins = int(wait // 60)
            secs = wait % 60
            eta = f"{mins}m{secs:.0f}s" if mins else f"{secs:.1f}s"
            print(f"[rate limit] {tag}: attempt {attempt + 1}/{max_retries}, waiting {eta} before retry...")
            time.sleep(wait)
            attempt += 1

    latency_ms = round((time.time() - t0) * 1000, 1)

    _call_counter["calls"] += 1
    _call_counter["input_tokens"] += input_tokens
    _call_counter["output_tokens"] += output_tokens

    record = {
        "tag": tag,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "response": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "timestamp": time.time(),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens, "latency_ms": latency_ms}


def get_call_stats() -> dict:
    return dict(_call_counter)


def reset_call_stats():
    _call_counter.update({"calls": 0, "input_tokens": 0, "output_tokens": 0})