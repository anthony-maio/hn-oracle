"""Baselines (regex, OpenRouter LLMs) and the stage C hindsight spot check."""
from __future__ import annotations

import functools
import json
import os
from datetime import date

import pandas as pd

from . import config as C
from .jev import RateLimiter, post_json, run_units
from .metrics import score_level
from .runs import comment_state, iter_jsonl, load_stage_b
from .sample import load_sample
from .text import gradable, regex_hint, resolution_year

BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "boolean", "description": "true if the comment makes a prediction"},
        "probability": {"type": "number", "description": "probability from 0 to 1 that the answer is true"},
    },
    "required": ["answer", "probability"],
    "additionalProperties": False,
}

STAGE_C_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["came_true", "did_not_come_true", "partly", "too_early", "unresolvable"]},
        "summary": {"type": "string", "description": "two or three sentences on what actually happened"},
        "evidence_urls": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "evidence_urls"],
    "additionalProperties": False,
}


def openrouter_headers() -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "X-Title": "HN Oracle pilot"}


def baseline_prompt(state: dict) -> str:
    """The same state and the same yes/no question Jev gets, rendered as text."""
    q = C.QUESTIONS["stage_a_single"]["is_prediction"]
    return (
        "State (JSON):\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n\n"
        f"Question: {q['instructions']}\n"
        f"Answer true if: {q['criteria']['true']}\n"
        f"Answer false if: {q['criteria']['false']}\n\n"
        "Return JSON {\"answer\": true|false, \"probability\": 0..1} with your answer and a calibrated "
        "probability that the answer is true."
    )


def packed_baseline_prompt(states: dict[str, dict]) -> str:
    q = C.QUESTIONS["stage_a_single"]["is_prediction"]
    return (
        "State (JSON), one entry per comment ID:\n" + json.dumps(states, indent=2, ensure_ascii=False) + "\n\n"
        f"For EACH comment ID, answer: {q['instructions']}\n"
        f"Answer true if: {q['criteria']['true']}\n"
        f"Answer false if: {q['criteria']['false']}\n\n"
        "Return one JSON object keyed by comment ID, each value {\"answer\": true|false, \"probability\": 0..1}, "
        "where probability is your calibrated probability that the answer is true."
    )


def packed_schema(cids: list[str]) -> dict:
    return {"type": "object", "properties": {c: BASELINE_SCHEMA for c in cids},
            "required": cids, "additionalProperties": False}


def parse_content(resp: dict) -> dict:
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in model reply: {content[:200]!r}")
    return json.loads(content[start:end + 1])


# ---------------------------------------------------------------- OpenRouter model catalog

MODELS_CACHE = "openrouter_models.json"


def fetch_models(refresh: bool = False) -> dict[str, dict]:
    import httpx

    C.ensure_data()
    path = C.DATA / MODELS_CACHE
    if refresh or not path.exists():
        with httpx.Client() as client:
            r = client.get("https://openrouter.ai/api/v1/models", timeout=60)
            r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return {m["id"]: m for m in json.loads(path.read_text(encoding="utf-8"))["data"]}


@functools.lru_cache(maxsize=None)
def capabilities(model: str) -> frozenset[str]:
    try:
        m = fetch_models().get(model)
    except Exception:
        return frozenset()
    return frozenset((m or {}).get("supported_parameters") or [])


def with_output_format(body: dict, model: str, name: str, schema: dict) -> dict:
    """Strict JSON schema when the model supports it, JSON mode next, else prompt-only JSON (parsed leniently).

    Also drops `temperature` for models that don't accept it (e.g. Claude Sonnet 5), since
    require_parameters would otherwise route to no endpoint at all.
    """
    caps = capabilities(model)
    if caps and "temperature" not in caps:
        body.pop("temperature", None)
    if "structured_outputs" in caps:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        body["provider"] = {"require_parameters": True}
    elif "response_format" in caps:
        body["response_format"] = {"type": "json_object"}
    return body


def cmd_models(args) -> None:
    models = fetch_models(refresh=True)
    free = sorted((m for i, m in models.items() if i.endswith(":free")), key=lambda m: -(m.get("context_length") or 0))
    print(f"{len(models)} models, {len(free)} free (saved to data/{MODELS_CACHE})\n")
    print(f"{'model':<52} {'context':>9}  structured  json_mode")
    for m in free:
        sp = set(m.get("supported_parameters") or [])
        print(f"{m['id']:<52} {m.get('context_length') or 0:>9}  {'yes' if 'structured_outputs' in sp else '-':^10}  "
              f"{'yes' if 'response_format' in sp else '-':^9}")
    print("\nFree models share 20 req/min and 50 req/day (1,000/day after $10 of credits purchased, ever).")

    paid = []
    for i, m in models.items():
        sp = set(m.get("supported_parameters") or [])
        try:
            pin, pout = float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        if i.endswith((":free", ":batch")) or pin <= 0 or "structured_outputs" not in sp:
            continue
        reasoning = "reasoning" in sp
        per_1k = (450 * pin + (425 if reasoning else 25) * pout) / 1e6 * 1000
        paid.append((per_1k, i, pin, pout, reasoning))
    paid.sort()
    print(f"\nCheapest {args.top} paid models with structured output (rough $ per 1,000 comments, single request):\n")
    print(f"{'model':<52} {'in $/M':>7} {'out $/M':>8} {'$/1k':>7}  reasons")
    for per_1k, i, pin, pout, reasoning in paid[:args.top]:
        print(f"{i:<52} {pin:>7.3f} {pout:>8.3f} {per_1k:>7.3f}  {'yes' if reasoning else '-'}")


def cmd_baseline(args) -> None:
    C.ensure_data()
    sample = load_sample()
    if args.limit:
        sample = sample.head(args.limit)

    if args.kind == "regex":
        out = C.DATA / "baseline_regex.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in sample.to_dict("records"):
                fh.write(json.dumps({"key": str(r["id"]), "id": int(r["id"]), "name": "regex",
                                     "p": 1.0 if regex_hint(r["text"]) else 0.0}) + "\n")
        print(f"-> {out} (score it on the uniform stratum only; the enriched stratum was drawn with it)")
        return

    slug = lambda m: m.split("/")[-1].replace(":", "_")
    if args.sweep:
        models = [(f"{args.sweep}_{slug(m)}", m) for m in C.SWEEPS[args.sweep]]
    elif args.model:
        models = [(args.name or slug(args.model), args.model)]
    elif args.name in C.BASELINE_PRESETS:
        models = [(args.name, C.BASELINE_PRESETS[args.name])]
    else:
        raise SystemExit(f"pass --model <openrouter slug>, --name {'|'.join(C.BASELINE_PRESETS)}, or --sweep free|cheap")
    for name, model in models:
        run_llm_baseline(args, sample, name, model)


def run_llm_baseline(args, sample, name: str, model: str) -> None:
    import httpx

    from .runs import cids_for, make_packs

    out = C.DATA / f"baseline_{name}.jsonl"
    rpm = args.rpm if args.rpm is not None else (
        C.OPENROUTER_FREE_RPM if model.endswith(":free") else C.OPENROUTER_PAID_RPM)
    limiter = RateLimiter(rpm)
    headers = openrouter_headers()
    pack = max(1, args.pack)

    with httpx.Client() as client:
        def fn(u):
            if pack == 1:
                prompt, schema = baseline_prompt(u["states"]["c01"]), BASELINE_SCHEMA
            else:
                prompt, schema = packed_baseline_prompt(u["states"]), packed_schema(list(u["states"]))
            body = {"model": model, "temperature": 0, "max_tokens": 6000 + 120 * pack,  # reasoning models need room
                    "messages": [{"role": "system", "content": "You label Hacker News comments. Reply with JSON only."},
                                 {"role": "user", "content": prompt}]}
            with_output_format(body, model, "is_prediction", schema)
            if args.reasoning_effort and "reasoning" in capabilities(model):
                body["reasoning"] = {"effort": args.reasoning_effort, "exclude": True}
            resp, dt, attempts = post_json(client, C.OPENROUTER_URL, body, headers, limiter, timeout=300)
            parsed = parse_content(resp)
            per = {"c01": parsed} if pack == 1 else parsed
            items = [{"id": iid, "p": min(max(float(per[c]["probability"]), 0.0), 1.0), "answer": per[c]["answer"]}
                     for iid, c in zip(u["ids"], u["states"])]
            return {"ids": u["ids"], "items": items, "name": name, "model": model, "pack": pack,
                    "latency_ms": dt, "attempts": attempts, "request": body,
                    "llm_usage": resp.get("usage"), "llm_response": resp}

        units = []
        for chunk in make_packs(sample.to_dict("records"), pack, args.seed):
            cids = cids_for(len(chunk))
            units.append({"key": "-".join(str(r["id"]) for r in chunk), "ids": [int(r["id"]) for r in chunk],
                          "states": {c: comment_state(r) for c, r in zip(cids, chunk)}})
        print(f"{model}: {len(units)} requests at <= {rpm:.0f}/min"
              + (" (counts against your daily free-model quota)" if model.endswith(":free") else ""))
        run_units(units, fn, out, workers=args.workers, label=out.name)
    print(f"-> {out}")


# ---------------------------------------------------------------- stage C


def stage_c_candidates(stage_b_file, labels_csv: str | None, top: int, now_year: int) -> pd.DataFrame:
    """Top checkable, gradable predictions. Horizon -> resolution year and the hindsight test are code."""
    sample = load_sample().set_index("id")
    answers = load_stage_b(stage_b_file)
    positives = None
    if labels_csv:
        lab = pd.read_csv(labels_csv, dtype=str)
        positives = set(lab.loc[lab.is_prediction.str.strip() == "1", "id"].astype(int))
    n_spec = len(C.QUESTIONS["stage_b"]["specificity"]["criteria"])
    rows = []
    for iid, a in answers.items():
        if positives is not None and iid not in positives:
            continue
        if iid not in sample.index:
            continue
        s = sample.loc[iid]
        horizon = a["horizon"]["choice"]
        if not gradable(int(s.year), horizon, now_year):
            continue
        spec, _ = score_level(a["specificity"], n_spec)
        rows.append({"id": iid, "year": int(s.year), "horizon": horizon,
                     "resolution_year": resolution_year(int(s.year), horizon),
                     "p_checkable": float(a["is_checkable"]["noul"]),
                     "p_sarcastic": float(a["is_sarcastic"]["noul"]),
                     "specificity_level": spec, "domain": a["domain"]["choice"],
                     "direction": a["direction"]["choice"],
                     "story_title": s.story_title, "comment": s.text})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.p_sarcastic < 0.5]
    df["rank_key"] = df.p_checkable * (1 + df.specificity_level)
    return df.sort_values("rank_key", ascending=False).head(top)


def stage_c_prompt(row) -> str:
    return (
        f"In {row['year']}, a Hacker News commenter wrote the following, in a thread titled "
        f"\"{row['story_title']}\":\n\n\"\"\"\n{row['comment']}\n\"\"\"\n\n"
        f"The prediction's stated horizon resolves by {row['resolution_year']}. Using web search, decide "
        f"whether the prediction came true as of today ({date.today().isoformat()}). Judge the prediction as "
        "the commenter meant it, not a stronger or weaker version. Cite the sources you relied on."
    )


def cmd_stage_c(args) -> None:
    now_year = args.now_year or date.today().year
    cands = stage_c_candidates(args.stage_b_file, args.labels, args.top, now_year)
    if cands.empty:
        raise SystemExit("no gradable checkable predictions found")
    out = C.DATA / "stage_c.jsonl"
    headers = openrouter_headers()
    model = args.model

    import httpx

    with httpx.Client() as client:
        def fn(u):
            body = {"model": model, "temperature": 0, "max_tokens": 4000,
                    "plugins": [{"id": "web", "max_results": 5}],  # Exa for non-native models: ~$0.007/request
                    "messages": [{"role": "user", "content": stage_c_prompt(u["row"]) +
                                  '\n\nReply with JSON {"verdict": "came_true|did_not_come_true|partly|too_early|'
                                  'unresolvable", "summary": "...", "evidence_urls": ["..."]}.'}]}
            with_output_format(body, model, "hindsight", STAGE_C_SCHEMA)
            resp, dt, _ = post_json(client, C.OPENROUTER_URL, body, headers, None, timeout=300)
            ann = resp["choices"][0]["message"].get("annotations") or []
            cites = [a.get("url_citation", a).get("url") for a in ann if isinstance(a, dict)]
            return {"id": u["row"]["id"], "model": model, "verdict": parse_content(resp),
                    "citations": [c for c in cites if c], "latency_ms": dt, "request": body, "llm_response": resp}

        units = [{"key": str(r["id"]), "row": r} for r in cands.to_dict("records")]
        run_units(units, fn, out, workers=min(args.workers, 4), label=out.name)

    verdicts = {r["id"]: r for r in iter_jsonl(out)}
    sheet = cands.drop(columns=["rank_key"]).copy()
    sheet["llm_verdict"] = sheet.id.map(lambda i: verdicts.get(i, {}).get("verdict", {}).get("verdict", ""))
    sheet["llm_summary"] = sheet.id.map(lambda i: verdicts.get(i, {}).get("verdict", {}).get("summary", ""))
    sheet["llm_sources"] = sheet.id.map(lambda i: " ".join(
        verdicts.get(i, {}).get("verdict", {}).get("evidence_urls", []) + verdicts.get(i, {}).get("citations", [])))
    sheet["anthony_verdict"] = ""   # checked against sources by hand; demo pages only, never Jev scoring
    sheet["anthony_sources"] = ""
    sheet["notes"] = ""
    path = C.DATA / "stage_c_review.csv"
    sheet.to_csv(path, index=False)
    print(f"-> {out}\n-> {path} (fill anthony_verdict / anthony_sources by hand)")
