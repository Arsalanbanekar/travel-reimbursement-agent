"""
Central runtime configuration.

Every path and model name used by the project is resolved here once, so that
modules never depend on the current working directory and the model can be
swapped from the environment without touching code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# --- Paths -----------------------------------------------------------------
# Resolved from this file, not the cwd, so `pytest`, `streamlit run` and
# `python -m ...` all behave the same regardless of where they are launched.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VECTOR_DB_PATH = PROJECT_ROOT / "faiss_index"
POLICY_PATH = DATA_DIR / "travel_policy.md"


# --- Models ----------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq retired the Llama chat models, so the earlier default
# (llama-3.3-70b-versatile) no longer resolves. gpt-oss-120b is the strongest
# remaining option with native tool calling; gpt-oss-20b is the lighter
# fallback if the free tier rate-limits the larger model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# --- Agent runtime ---------------------------------------------------------

# Hard ceiling on tool-calling iterations, so a confused model cannot loop
# forever against the Groq free tier.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))

# How long to stop trying the primary model after it rate-limits, so a
# single 429 does not cost a failed attempt on every later call.
PRIMARY_COOLDOWN_SECONDS = float(os.getenv("PRIMARY_COOLDOWN_SECONDS", "300"))

LLM_TEMPERATURE = 0.0


def require_api_key() -> str:
    """
    Return the Groq API key, or raise a readable error.

    Called at agent construction rather than at import time so that the rule
    engine and tests stay usable without any credentials.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or set it in Streamlit secrets when deploying."
        )

    return GROQ_API_KEY
