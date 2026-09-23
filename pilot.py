"""HN Oracle, 1,000-comment Jev pilot. See PILOT_PLAN.md and README.md.

Subcommands, in pilot order:
  preflight  Check dataset columns/types, Jev auth, response shape, and packed-16 token size.
  count      Exact count of full-run-eligible comments (2006-2020) with DuckDB.
  sample     Draw the stratified 1,000-comment sample, fetch thread context, write label sheets.
  label      Blind terminal labeler (pass a: is_prediction; pass b: stage B fields).
  run        Jev runs: a_single, a_packed, a_nonce, b. Resumable; logs request, response, latency.
  models     List OpenRouter :free models and whether they support structured output.
  baseline   Regex baseline, or OpenRouter LLM baselines (free models by default, optional packing).
  stage-c    Frontier LLM with web search grades the top checkable predictions (demo pages only).
  score      All metrics, pre-registered gates G1-G7, go/no-go, full-run projection.
  publish    Copy public artifacts to publish/ with usernames removed.

Env: TYPESAFE_API_KEY (Jev), OPENROUTER_API_KEY (baselines, stage C).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from oracle import config as C


def lazy(module: str, fn: str):
    """Import subcommand modules on use, so `label` works without duckdb/httpx installed, etc."""
    return lambda a: getattr(importlib.import_module(module), fn)(a)


def cmd_preflight(args) -> None:
    from oracle.sample import check_schema

    ok = True
    if not args.skip_dataset:
        s = check_schema(args.parquet_glob)
        print(f"dataset: {s['source']}")
        for col, typ in s["columns"].items():
            print(f"  {col:<12} {typ}")
        print(f"  type counts (2010 files): {s['type_counts']}")
        if s["problems"]:
            ok = False
            print("  PROBLEMS: " + "; ".join(s["problems"]))
        else:
            print("  schema matches the dataset card assumptions")

    if args.call:
        import httpx

        from oracle.jev import call_jev
        from oracle.runs import cids_for, estimate_tokens, packed_questions

        state = {"story_title": "Ask HN: Where is the web going?", "parent_comment": "(top-level comment)",
                 "comment": "Mark my words: in five years nobody will be writing jQuery for new projects.",
                 "posted_year": 2013}
        with httpx.Client() as client:
            res = call_jev(client, state, C.QUESTIONS["stage_a_single"])
            print("single call ok:", json.dumps(res["response"], indent=2)[:1500])
            print(f"  latency {res['latency_ms']:.0f} ms")
            noul = res["response"]["answers"]["is_prediction"].get("noul")
            if not isinstance(noul, (int, float)) or not 0 <= noul <= 1:
                ok = False
                print(f"  PROBLEM: answers.is_prediction.noul is {noul!r}, expected a probability")

            # Worst case packed state: 16 comments near the 4,000-char cap.
            long = {**state, "comment": ("This will happen eventually. " * 140)[:4000],
                    "parent_comment": "x " * 300}
            cids = cids_for(16)
            packed_state = {c: long for c in cids}
            print(f"packed-16 worst case, rough estimate {estimate_tokens(packed_state) + estimate_tokens(packed_questions(cids))} tokens")
            res = call_jev(client, packed_state, packed_questions(cids))
            used = res["response"].get("usage", {}).get("input_tokens")
            print(f"  packed-16 worst case accepted: input_tokens={used} (window {C.CONTEXT_WINDOW})")
    print("preflight OK" if ok else "preflight found problems")
    sys.exit(0 if ok else 1)


def cmd_count(args) -> None:
    from oracle.sample import count_eligible

    print(json.dumps(count_eligible(args.parquet_glob), indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--parquet-glob", default=C.HF_PARQUET_GLOB)
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--call", action="store_true", help="make two tiny Jev calls (single + packed-16 worst case)")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("count")
    p.add_argument("--parquet-glob", default=C.HF_PARQUET_GLOB)
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("sample")
    p.add_argument("--parquet-glob", default=C.HF_PARQUET_GLOB)
    p.add_argument("--workers", type=int, default=16, help="threads for HN Firebase context fetches")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lazy("oracle.sample", "cmd_sample"))

    p = sub.add_parser("label")
    p.add_argument("--labeler", required=True)
    p.add_argument("--pass", dest="pass_", choices=["a", "b"], default="a")
    p.add_argument("--sheet", help="default data/labels_blank.csv; second labeler uses data/labels_blank_second.csv")
    p.add_argument("--out", help="default data/labels_<labeler>.csv")
    p.set_defaults(func=lazy("oracle.labeling", "cmd_label"))

    p = sub.add_parser("run")
    p.add_argument("--mode", choices=["a_single", "a_packed", "a_nonce", "b"], required=True)
    p.add_argument("--pack", type=int, default=8)
    p.add_argument("--gate", type=float, help="stage B: P(is_prediction) threshold from `score`")
    p.add_argument("--stage-a-file", help="stage B: the stage A single-run JSONL")
    p.add_argument("--also-labeled-positives", metavar="LABELS_CSV",
                   help="stage B: also run labeled positives below the gate, so field accuracy is measured on all true predictions")
    p.add_argument("--with-subject", action="store_true", help="stage B: add the optional subject roster Choice")
    p.add_argument("--n-nonce", type=int, default=100)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--limit", type=int, help="first N sample rows only (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rpm", type=float, default=C.RATE_LIMIT_RPM * 0.8, help="client-side request cap")
    p.add_argument("--dry-run", action="store_true", help="print the first request body and exit")
    p.set_defaults(func=lazy("oracle.runs", "cmd_run"))

    p = sub.add_parser("models", help="list OpenRouter :free models and the cheapest paid ones")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=lazy("oracle.llm", "cmd_models"))

    p = sub.add_parser("baseline")
    p.add_argument("--kind", choices=["regex", "llm"], required=True)
    p.add_argument("--model", help="any OpenRouter slug; default: the preset for --name")
    p.add_argument("--name", help=f"presets {list(C.BASELINE_PRESETS)}; with --model, the file suffix")
    p.add_argument("--sweep", choices=list(C.SWEEPS), help="run every model in config.SWEEPS[free|cheap]")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                   help="for reasoning models: trade accuracy for cost and speed")
    p.add_argument("--pack", type=int, default=1,
                   help="comments per request (16 turns 1,000 comments into 63 requests for the free tier)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rpm", type=float, help="default 18 for :free models, 300 for paid")
    p.set_defaults(func=lazy("oracle.llm", "cmd_baseline"))

    p = sub.add_parser("stage-c")
    p.add_argument("--stage-b-file", required=True)
    p.add_argument("--labels", help="restrict to labeled positives")
    p.add_argument("--model", default=C.STAGE_C_MODEL, help="OpenRouter slug; the web plugin is added (Exa, ~$0.007/request)")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--now-year", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=lazy("oracle.llm", "cmd_stage_c"))

    p = sub.add_parser("score")
    p.add_argument("--labels", required=True, help="primary labels CSV")
    p.add_argument("--second-labels", help="second labeler CSV (200 overlap) for the human ceiling")
    p.add_argument("--single", required=True, help="stage A single-comment raw JSONL")
    p.add_argument("--packed", nargs="*", help="stage A packed raw JSONL files (N read from 'packedN' in the name)")
    p.add_argument("--nonce", help="a_nonce raw JSONL")
    p.add_argument("--stage-b", help="stage B raw JSONL")
    p.add_argument("--baseline-regex")
    p.add_argument("--baseline-cheap")
    p.add_argument("--baseline-frontier")
    p.add_argument("--baseline-extra", nargs="*", metavar="NAME=PATH", help="more baselines for the table (no gate)")
    p.add_argument("--gate", type=float, help="fix the gate threshold instead of choosing it from the labels")
    p.add_argument("--target-recall", type=float, default=0.90,
                   help="gate choice: highest threshold with this recall on all labeled comments (margin over G1's 0.85)")
    p.add_argument("--eligible", type=int, help="override the full-run comment count")
    p.add_argument("--price", type=float, default=C.PRICE_PER_M_INPUT)
    p.add_argument("--rpm", type=float, default=C.RATE_LIMIT_RPM)
    p.add_argument("--tps", type=float, default=C.RATE_LIMIT_TPS)
    p.add_argument("--out")
    p.set_defaults(func=lazy("oracle.score", "cmd_score"))

    p = sub.add_parser("publish")
    p.set_defaults(func=lazy("oracle.publish", "cmd_publish"))
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
