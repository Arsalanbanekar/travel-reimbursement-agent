"""
Central runtime configuration.

Every path and model name used by the project is resolved here once, so that
modules never depend on the current working directory and the model can be
swapped from the environment without touching code.
"""

import os
import sys
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


def _setting(name: str, default: str = "") -> str:
    """
    Read a setting from the environment, then from Streamlit secrets.

    Locally the value comes from .env. On Streamlit Cloud there is no .env —
    values are entered in the dashboard and surface through st.secrets, which
    is not guaranteed to be mirrored into os.environ. Checking both means the
    same code runs in either place.
    """

    value = os.getenv(name)
    if value:
        return value

    # Only consult Streamlit if it is already running; importing it here would
    # be wrong for the CLI and the tests.
    streamlit = sys.modules.get("streamlit")
    if streamlit is not None:
        try:
            return str(streamlit.secrets[name])
        except Exception:
            pass

    return default


# --- Models ----------------------------------------------------------------

GROQ_API_KEY = _setting("GROQ_API_KEY")

# Groq retired the Llama chat models, so the earlier default
# (llama-3.3-70b-versatile) no longer resolves. gpt-oss-120b is the strongest
# remaining option with native tool calling; gpt-oss-20b is the lighter
# fallback if the free tier rate-limits the larger model.
GROQ_MODEL = _setting("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_FALLBACK_MODEL = _setting("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = _setting(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# --- Agent runtime ---------------------------------------------------------

# Hard ceiling on tool-calling iterations, so a confused model cannot loop
# forever against the Groq free tier.
MAX_AGENT_STEPS = int(_setting("MAX_AGENT_STEPS", "8"))

# How long to stop trying the primary model after it rate-limits, so a
# single 429 does not cost a failed attempt on every later call.
PRIMARY_COOLDOWN_SECONDS = float(_setting("PRIMARY_COOLDOWN_SECONDS", "300"))

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
            "key locally, or add it under Settings > Secrets on Streamlit Cloud."
        )

    return GROQ_API_KEY
