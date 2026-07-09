"""
Config for the judging pipeline. Judge and generator models are configured
INDEPENDENTLY via separate env vars (requirement 2.2 "configured
independently") specifically so self-enhancement bias can be tested: point
GENERATOR_MODEL at one model family and JUDGE_MODEL at a different one.

Limitation (documented, not hidden): this sandbox/repo only has an Anthropic
API key configured by default, so out of the box GENERATOR_MODEL and
JUDGE_MODEL are both Claude models (different tiers, not different
families). If you have an OpenAI or other provider key, set
JUDGE_PROVIDER=openai and JUDGE_MODEL=gpt-4o-mini (see judge/client.py) to
get a true cross-family judge, which is the recommended real setup for
self-enhancement bias mitigation.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Defaults use Groq (FREE, https://console.groq.com/keys) for both roles,
    # but with two DIFFERENT model families (Llama vs Gemma) so the
    # self-enhancement-bias mitigation is real even on a $0 budget.
    GENERATOR_PROVIDER: str = os.getenv("GENERATOR_PROVIDER", "groq")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")

    JUDGE_PROVIDER: str = os.getenv("JUDGE_PROVIDER", "groq")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

    PASS_THRESHOLD: float = float(os.getenv("PASS_THRESHOLD", "3.5"))  # overall score out of 5
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")


settings = Settings()