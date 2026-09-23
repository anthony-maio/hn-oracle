"""Paths, pinned model, prices, and rate limits shared by every subcommand."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=value lines from .env; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()

DATA = ROOT / "data"
PUBLISH = ROOT / "publish"
QUESTIONS_PATH = ROOT / "questions.json"

QUESTIONS = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
MODEL = QUESTIONS["_meta"]["model"]
SCHEMA_VERSION = QUESTIONS["_meta"]["schema_version"]

# Evaluation must never float on an alias.
assert MODEL != "jev-latest", "Pin a concrete Jev version in questions.json during evaluation"

API_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# USD per 1M input tokens. Verify against your TypeSafe pricing page before projecting.
PRICE_PER_M_INPUT = 0.042

# Public account-tier figures. Override on the CLI with your own tier.
RATE_LIMIT_RPM = 1200
RATE_LIMIT_TPS = 250_000
CONTEXT_WINDOW = 32_000

# Pilot sample window and full-run eligibility window (at least 5 years of hindsight).
SAMPLE_YEARS = (2009, 2019)
FULL_RUN_YEARS = (2006, 2020)
TEXT_LEN = (80, 4000)
N_UNIFORM = 600
N_ENRICHED = 400
ENRICHED_POOL = 20_000

# Fallback when data/eligible.json has not been produced yet.
ARCHIVE_COMMENTS_FALLBACK = 41_317_357

# OpenRouter baselines. Prices and capabilities checked against GET /api/v1/models on 2026-09-23;
# `pilot.py models` refreshes them. Rough cost per 1,000 comments assumes ~450 input tokens and
# ~25 output tokens (+~400 when the model reasons).
#
# Free models (`:free`) share 20 requests/min and 50 requests/day (1,000/day once you have bought
# $10 of credits, ever). Paid models have no daily cap and much higher per-minute limits.
OPENROUTER_FREE_RPM = 18
OPENROUTER_PAID_RPM = 300
BASELINE_PRESETS = {
    "cheap": "deepseek/deepseek-v4-flash",                     # $0.07/$0.14 per M, ~$0.04 per 1k comments
    "frontier": "anthropic/claude-sonnet-5",                   # $2/$10 per M, ~$5 per 1k comments
    "free_cheap": "liquid/lfm-2.5-2.6b:free",                  # smallest free model, native structured output
    "free_frontier": "nvidia/nemotron-3-ultra-550b-a55b:free",  # largest free model (550B MoE)
}
SWEEPS = {
    # every free model with native structured output
    "free": [
        "nex-agi/nex-n2.5-pro:free",
        "nex-agi/nex-n2.5-mini:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3.8-27b:free",
        "dots-studio/dots-3-note-preview:free",
        "liquid/lfm-2.5-2.6b:free",
    ],
    # low-cost paid models with native structured output, under ~$0.35 per 1k comments
    "cheap": [
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5-nano",
        "openai/gpt-6-luna",
        "qwen/qwen3.8-flash",
        "mistralai/mistral-small-24b-instruct-2501",
        "nvidia/nemotron-3-nano-30b-a3b",
        "mistralai/mistral-nemo",
    ],
}
STAGE_C_MODEL = "anthropic/claude-sonnet-5"  # plus the web plugin: Exa, ~$0.007 per request

HF_PARQUET_GLOB = "hf://datasets/nikhilambhure00/hacker-news/data/*/*.parquet"

LABEL_FIELDS_B = ["is_checkable", "is_sarcastic", "is_conditional", "direction", "domain",
                  "horizon", "stance_vs_thread", "certainty", "specificity"]
LABEL_COLUMNS = ["id", "stratum", "is_prediction", *LABEL_FIELDS_B, "subject_text", "labeler", "notes"]


def ensure_data() -> Path:
    DATA.mkdir(exist_ok=True)
    return DATA
