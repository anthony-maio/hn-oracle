"""Comment text cleanup, the prediction-hint regex, and horizon arithmetic done in code."""
from __future__ import annotations

import html
import re

# Used twice: to build the enriched stratum, and as the regex baseline on the uniform stratum.
PREDICTION_HINT = re.compile(
    r"\b(will|won't|going to|in \d+ years|by 20\d\d|mark my words|i bet|"
    r"eventually|never going|within (a|the next) (decade|year)|dead in|the future of)\b",
    re.I,
)


def clean_html(s: str | None) -> str:
    s = re.sub(r"<p>", "\n\n", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def regex_hint(text: str) -> bool:
    return bool(PREDICTION_HINT.search(text or ""))


# Horizon buckets -> years added to the posted year. Open-ended buckets have no resolution year.
HORIZON_YEARS = {"under_1y": 1, "1_to_5y": 5, "5_to_10y": 10, "over_10y": None, "unstated": None}


def resolution_year(posted_year: int, horizon: str) -> int | None:
    add = HORIZON_YEARS.get(horizon)
    return None if add is None else int(posted_year) + add


def gradable(posted_year: int, horizon: str, now_year: int) -> bool:
    ry = resolution_year(posted_year, horizon)
    return ry is not None and ry <= now_year
