"""Jev runs: stage A single, stage A packed, nonce robustness, stage B. Plus loaders for their JSONL."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import pandas as pd

from . import config as C
from .jev import RateLimiter, call_jev, jev_body, run_units
from .sample import load_sample


def comment_state(row: dict) -> dict:
    # Year is context only. No question asks Jev to reason about dates.
    return {
        "story_title": row.get("story_title") or "",
        "parent_comment": row.get("parent_excerpt") or "(top-level comment)",
        "comment": row["text"],
        "posted_year": int(row["year"]),
    }


def cids_for(n: int) -> list[str]:
    return [f"c{j + 1:02d}" for j in range(n)]


def packed_questions(cids: list[str]) -> dict:
    tmpl = C.QUESTIONS["stage_a_packed_template"]["is_prediction_{id}"]
    return {f"is_prediction_{cid}": json.loads(json.dumps(tmpl).replace("{id}", cid)) for cid in cids}


def stage_b_questions(with_subject: bool) -> dict:
    q = copy.deepcopy(C.QUESTIONS["stage_b"])
    if with_subject:
        q |= {k: v for k, v in C.QUESTIONS["stage_b_subject_choice"].items() if not k.startswith("_")}
    return q


def make_packs(rows: list[dict], pack: int, seed: int) -> list[list[dict]]:
    """Deterministic shuffle so packs mix strata and threads, and resume reproduces the same packs."""
    rows = sorted(rows, key=lambda r: int(r["id"]))
    random.Random(seed).shuffle(rows)
    return [rows[i:i + pack] for i in range(0, len(rows), pack)]


def raw_path(mode: str, pack: int | None, tag: str) -> Path:
    name = f"a_packed{pack}" if mode == "a_packed" else mode
    return C.DATA / f"raw_{name}_{tag}.jsonl"


def build_units(args, sample: pd.DataFrame) -> list[dict]:
    rows = sample.to_dict("records")
    meta = {"mode": args.mode, "schema_version": C.SCHEMA_VERSION, "model": C.MODEL}

    if args.mode == "a_single":
        q = C.QUESTIONS["stage_a_single"]
        return [{"key": str(r["id"]), "ids": [int(r["id"])], "state": comment_state(r), "q": q, **meta}
                for r in rows]

    if args.mode == "a_packed":
        units = []
        for chunk in make_packs(rows, args.pack, args.seed):
            cids = cids_for(len(chunk))
            units.append({
                "key": "-".join(str(r["id"]) for r in chunk), "ids": [int(r["id"]) for r in chunk],
                "cids": cids, "state": {c: comment_state(r) for c, r in zip(cids, chunk)},
                "q": packed_questions(cids), "pack": args.pack, **meta,
            })
        return units

    if args.mode == "a_nonce":
        # Robustness: same comment, random nonce added to the state. Answers should barely move.
        q = C.QUESTIONS["stage_a_single"]
        picked = random.Random(args.seed).sample(rows, min(args.n_nonce, len(rows)))
        return [{"key": f"{r['id']}:rep{k}", "ids": [int(r["id"])], "rep": k,
                 "state": comment_state(r) | {"request_nonce": os.urandom(8).hex()}, "q": q, **meta}
                for r in picked for k in range(args.reps)]

    if args.mode == "b":
        if args.gate is None or not args.stage_a_file:
            raise SystemExit("stage B needs --gate and --stage-a-file (use the gate `score` recommends)")
        probs = load_stage_a(args.stage_a_file)
        keep = {i for i, p in probs.items() if p >= args.gate}
        if args.also_labeled_positives:
            lab = pd.read_csv(args.also_labeled_positives, dtype=str)
            keep |= set(lab.loc[lab.is_prediction.str.strip() == "1", "id"].astype(int))
        q = stage_b_questions(args.with_subject)
        return [{"key": str(r["id"]), "ids": [int(r["id"])], "state": comment_state(r), "q": q,
                 "gate": args.gate, "gated_in": probs.get(int(r["id"]), 0) >= args.gate, **meta}
                for r in rows if int(r["id"]) in keep]

    raise SystemExit(f"unknown mode {args.mode}")


def estimate_tokens(obj) -> int:
    """Rough chars/4 estimate, for the pre-flight context-window check only. Billing uses usage.input_tokens."""
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def cmd_run(args) -> None:
    C.ensure_data()
    sample = load_sample()
    if args.limit:
        sample = sample.head(args.limit)
    units = build_units(args, sample)
    out = raw_path(args.mode, args.pack if args.mode == "a_packed" else None, args.tag)

    worst = max(units, key=lambda u: estimate_tokens(u["state"]) + estimate_tokens(u["q"]))
    est = estimate_tokens(worst["state"]) + estimate_tokens(worst["q"])
    print(f"largest request ~{est} tokens (rough) vs {C.CONTEXT_WINDOW} window")
    if est > C.CONTEXT_WINDOW * 0.9:
        print("warning: largest request is near the context window; consider a smaller --pack")

    if args.dry_run:
        print(json.dumps(jev_body(units[0]["state"], units[0]["q"]), indent=2, ensure_ascii=False))
        print(f"{len(units)} units would be written to {out}")
        return

    import httpx

    limiter = RateLimiter(args.rpm)
    with httpx.Client(http2=False, limits=httpx.Limits(max_connections=args.workers * 2)) as client:
        def fn(u):
            res = call_jev(client, u["state"], u["q"], limiter)
            extra = {k: u[k] for k in ("ids", "cids", "rep", "mode", "pack", "gate", "gated_in",
                                       "schema_version", "model") if k in u}
            return {**extra, **res}

        run_units(units, fn, out, workers=args.workers, label=out.name)
    print(f"-> {out}")


# ---------------------------------------------------------------- loaders


def iter_jsonl(path) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def noul_p(answer: dict) -> float:
    return float(answer["noul"])


def load_stage_a(path) -> dict[int, float]:
    """id -> P(is_prediction). Works for single, packed, and baseline files ({"id", "p"} records)."""
    probs: dict[int, float] = {}
    for rec in iter_jsonl(path):
        if "items" in rec:  # LLM baseline, possibly packed
            for it in rec["items"]:
                probs[int(it["id"])] = float(it["p"])
            continue
        if "p" in rec and "response" not in rec:  # regex baseline
            probs[int(rec["id"])] = float(rec["p"])
            continue
        ans = rec["response"]["answers"]
        if rec.get("cids"):
            for iid, cid in zip(rec["ids"], rec["cids"]):
                probs[int(iid)] = noul_p(ans[f"is_prediction_{cid}"])
        else:
            probs[int(rec["ids"][0])] = noul_p(ans["is_prediction"])
    return probs


def load_nonce(path) -> pd.DataFrame:
    rows = [{"id": int(r["ids"][0]), "rep": int(r["rep"]), "p": noul_p(r["response"]["answers"]["is_prediction"])}
            for r in iter_jsonl(path)]
    return pd.DataFrame(rows, columns=["id", "rep", "p"])


def load_stage_b(path) -> dict[int, dict]:
    return {int(r["ids"][0]): r["response"]["answers"] for r in iter_jsonl(path)}


def usage_summary(path) -> dict:
    """Tokens and latency per request and per comment, from logged usage."""
    import numpy as np

    recs = [r for r in iter_jsonl(path) if "response" in r]
    if not recs:
        return {}
    toks = np.array([r["response"].get("usage", {}).get("input_tokens", 0) for r in recs], dtype=float)
    lat = np.array([r["latency_ms"] for r in recs], dtype=float)
    n_comments = sum(len(r["ids"]) for r in recs)
    return {
        "file": Path(path).name, "requests": len(recs), "comments": n_comments,
        "comments_per_request": n_comments / len(recs),
        "input_tokens_total": int(toks.sum()),
        "input_tokens_per_request": float(toks.mean()),
        "input_tokens_per_comment": float(toks.sum() / n_comments),
        "input_tokens_max_request": int(toks.max()),
        "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
        "retried_requests": int(sum(1 for r in recs if r.get("attempts", 1) > 1)),
    }
