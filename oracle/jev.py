"""HTTP plumbing: Jev client, rate limiting, retries, and a resumable parallel JSONL runner.

Jev contract (docs.typesafe.ai/api):
  POST https://api.typesafe.ai/v1/systemone, header `Authorization: Bearer <key>`
  body {model, state, questions}; response {model, answers{...}, usage{input_tokens, output_tokens}}
  Noul -> answers[k]["noul"] in [0, 1]; Choice -> "choice", "probabilities", "confidence";
  Score -> "score" (probability-weighted), "legend", "probabilities", "confidence".
  Retry on 429 (rate limited) and 529 (overloaded).
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from . import config as C

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class RateLimiter:
    """Evenly spaced request starts: `rpm` requests per minute across all threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.next_t = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            self.next_t = max(now, self.next_t) + self.interval
        if wait > 0:
            time.sleep(wait)


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class StopRun(RuntimeError):
    """Every remaining request would fail the same way. Stop the run; progress is kept."""


class DailyLimit(StopRun):
    """Quota exhausted for the day (OpenRouter free models). Resume tomorrow."""


class AccountBlocked(StopRun):
    """The account's own settings (OpenRouter guardrails / privacy) exclude every endpoint for this model."""


def is_daily_limit(status: int, body: str) -> bool:
    b = body.lower()
    return status == 429 and ("per-day" in b or "per day" in b or "daily" in b)


def post_json(client, url: str, body: dict, headers: dict, limiter: RateLimiter | None = None,
              retries: int = 6, timeout: float = 60.0) -> tuple[dict, float, int]:
    """POST with backoff. Returns (json, latency_ms of the successful attempt, attempts used)."""
    last = None
    for attempt in range(retries):
        if limiter:
            limiter.acquire()
        t0 = time.perf_counter()
        try:
            r = client.post(url, json=body, headers=headers, timeout=timeout)
        except Exception as e:  # transport errors, timeouts
            last = e
            time.sleep(min(2 ** attempt, 30) + random.random())
            continue
        dt = (time.perf_counter() - t0) * 1000
        if is_daily_limit(r.status_code, r.text):
            raise DailyLimit(r.text[:300])
        if r.status_code == 404 and ("guardrail" in r.text or "data policy" in r.text):
            raise AccountBlocked(
                "OpenRouter account settings exclude every endpoint for this model. Allow it at "
                "https://openrouter.ai/workspaces/default/guardrails (and, for :free models, allow "
                "free-model training at https://openrouter.ai/settings/privacy). Detail: " + r.text[:400])
        if r.status_code in RETRY_STATUS:
            last = ApiError(r.status_code, r.text)
            ra = r.headers.get("retry-after")
            delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30)
            time.sleep(delay + random.random())
            continue
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text)  # 401/422 etc: retrying will not help
        return r.json(), dt, attempt + 1
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def jev_headers() -> dict:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def jev_body(state, questions: dict) -> dict:
    return {"model": C.MODEL, "state": state, "questions": strip_comments(questions)}


def strip_comments(questions: dict) -> dict:
    return {k: v for k, v in questions.items() if not k.startswith("_")}


def call_jev(client, state, questions: dict, limiter: RateLimiter | None = None) -> dict:
    body = jev_body(state, questions)
    resp, dt, attempts = post_json(client, C.API_URL, body, jev_headers(), limiter)
    if resp.get("model") and resp["model"] != C.MODEL:
        print(f"warning: requested {C.MODEL}, response says {resp['model']}", file=sys.stderr)
    return {"latency_ms": dt, "attempts": attempts, "request": body, "response": resp}


# ---------------------------------------------------------------- resumable runner


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass  # a torn last line from an interrupted run
    return keys


def run_units(units: Iterable[dict], fn: Callable[[dict], dict], out_path: Path,
              workers: int = 8, label: str = "") -> dict:
    """Run fn(unit) for every unit whose `key` is not already in out_path. Appends one JSON line each.

    Failures go to <out>.errors.jsonl and are retried on the next invocation.
    """
    units = list(units)
    seen = done_keys(out_path)
    todo = [u for u in units if u["key"] not in seen]
    print(f"{label}: {len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return {"ran": 0, "failed": 0}
    lock = threading.Lock()
    err_path = out_path.with_suffix(".errors.jsonl")
    ok = failed = 0
    stop = threading.Event()

    def guarded(u):
        if stop.is_set():
            raise StopRun("skipped after the run was stopped")
        return fn(u)

    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(guarded, u): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                rec = fut.result()
            except StopRun as e:
                if not stop.is_set():
                    stop.set()
                    why = "daily request limit reached; rerun tomorrow" if isinstance(e, DailyLimit) else str(e)
                    print(f"stopping: {why}")
                    print("progress is saved; rerun the same command to continue")
                continue
            except Exception as e:
                failed += 1
                with lock, err_path.open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"key": u["key"], "error": repr(e), "ts": now_iso()}) + "\n")
                continue
            rec = {"key": u["key"], "ts": now_iso(), **rec}
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            ok += 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} (ok {ok}, failed {failed})")
    if stop.is_set():
        return {"ran": ok, "failed": failed, "stopped": True}
    if failed:
        print(f"{failed} failures logged to {err_path}; re-run the same command to retry them")
    return {"ran": ok, "failed": failed}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
