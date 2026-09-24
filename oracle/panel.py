"""Two-model labeling panel (plan amendment A1).

Human labels stay the answer key where the headline claims live: `is_prediction` on the uniform
stratum (G1-G3), a blind audit of the panel, and every enriched comment the panel does not agree on
unanimously. The panel labels the rest of the enriched stratum and the stage B fields (2-of-3
majority, median for scores).

Steps, in order:
  a       every panel model labels is_prediction on the enriched stratum
  sheet   build the human sheet: uniform + audit + disagreements, shuffled together
  (you)   python pilot.py label --labeler anthony --sheet data/labels_blank_human.csv
  b       every panel model labels the stage B fields on every positive
  merge   data/labels_final.csv (with provenance) and data/panel_audit.json
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .jev import RateLimiter, post_json, run_units
from .llm import openrouter_headers, parse_content, with_output_format
from .runs import comment_state, iter_jsonl
from .sample import load_sample

RULES = (C.ROOT / "LABELING_RULES.md").read_text(encoding="utf-8")
SYSTEM = ("You are labeling Hacker News comments for a research dataset. Follow the labeling rules exactly, "
          "including the edge cases. Reply with JSON only.\n\n" + RULES)
SB = C.QUESTIONS["stage_b"]
BOOL_B = [f for f in C.LABEL_FIELDS_B if SB[f]["type"] == "noul"]
CHOICE_B = [f for f in C.LABEL_FIELDS_B if SB[f]["type"] == "choice"]
SCORE_B = [f for f in C.LABEL_FIELDS_B if SB[f]["type"] == "score"]

SCHEMA_A = {"type": "object", "properties": {"is_prediction": {"type": "boolean"}},
            "required": ["is_prediction"], "additionalProperties": False}
SCHEMA_B = {
    "type": "object",
    "properties": {
        **{f: {"type": "boolean"} for f in BOOL_B},
        **{f: {"type": "string", "enum": list(SB[f]["criteria"])} for f in CHOICE_B},
        **{f: {"type": "integer", "enum": list(range(len(SB[f]["criteria"])))} for f in SCORE_B},
        "subject_text": {"type": "string"},
    },
    "required": [*C.LABEL_FIELDS_B, "subject_text"],
    "additionalProperties": False,
}


def slug(model: str) -> str:
    return model.split("/")[-1].replace(":", "_").replace(".", "_")


def path_for(step: str, model: str):
    return C.DATA / f"panel_{step}_{slug(model)}.jsonl"


def prompt_a(state: dict) -> str:
    return ("State (JSON):\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n\n"
            "Task: panel_a. Label `is_prediction` for the comment, following the rules above. "
            'Reply {"is_prediction": true|false}.')


def prompt_b(state: dict) -> str:
    lines = []
    for f in C.LABEL_FIELDS_B:
        q = SB[f]
        lines.append(f"- {f}: {q['instructions']}")
        crit = q.get("criteria")
        if isinstance(crit, dict):
            lines += [f"    {k}: {v}" for k, v in crit.items()]
        elif isinstance(crit, list):
            lines += [f"    {i}: {v}" for i, v in enumerate(crit)]
    return ("State (JSON):\n" + json.dumps(state, indent=2, ensure_ascii=False) + "\n\n"
            "Task: panel_b. This comment is a prediction. Label every field, following the rules above:\n"
            + "\n".join(lines) + "\n- subject_text: the main company, product or technology, in a few words\n\n"
            "Booleans are true/false, choices use the option names exactly, scores are integers.")


def run_panel(step: str, ids: list[int], args) -> None:
    import httpx

    sample = load_sample().set_index("id")
    headers = openrouter_headers()
    schema, make = (SCHEMA_A, prompt_a) if step == "a" else (SCHEMA_B, prompt_b)
    for model in C.LABEL_PANEL:
        limiter = RateLimiter(args.rpm)
        with httpx.Client() as client:
            def fn(u):
                body = {"model": model, "temperature": 0, "max_tokens": 6000,  # reasoning models need room
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": make(u["state"])}]}
                with_output_format(body, model, f"panel_{step}", schema)
                resp, dt, attempts = post_json(client, C.OPENROUTER_URL, body, headers, limiter, timeout=180)
                return {"id": u["id"], "model": model, "labels": parse_content(resp), "latency_ms": dt,
                        "attempts": attempts, "llm_usage": resp.get("usage"), "request": body}

            units = [{"key": str(i), "id": i, "state": comment_state({**sample.loc[i].to_dict(), "id": i})}
                     for i in ids]
            out = path_for(step, model)
            run_units(units, fn, out, workers=args.workers, label=out.name)


def load_panel(step: str) -> dict[str, dict[int, dict]]:
    out = {}
    for model in C.LABEL_PANEL:
        p = path_for(step, model)
        out[model] = {int(r["id"]): r["labels"] for r in iter_jsonl(p)} if p.exists() else {}
    return out


def consensus_a() -> pd.DataFrame:
    """Per enriched id: each model's is_prediction (m0..mN) and whether all of them agree."""
    panel = load_panel("a")
    ids = sorted(set().union(*(set(v) for v in panel.values())))
    cols = [f"m{k}" for k in range(len(C.LABEL_PANEL))]
    rows = []
    for i in ids:
        votes = [panel[m].get(i, {}).get("is_prediction") for m in C.LABEL_PANEL]
        unanimous = None not in votes and len(set(votes)) == 1
        rows.append({"id": i, **dict(zip(cols, votes)), "agree": unanimous,
                     "label": ("1" if votes[0] else "0") if unanimous else ""})
    return pd.DataFrame(rows, columns=["id", *cols, "agree", "label"])


def majority(values: list) -> str:
    """2-of-3 (strict majority) or ''."""
    best = max(set(values), key=values.count)
    return best if values.count(best) * 2 > len(values) else ""


def build_sheet(args) -> None:
    sample = load_sample()
    enriched = sample[sample.stratum == "enriched"]
    cons = consensus_a()
    missing = set(enriched.id) - set(cons.id)
    if missing:
        raise SystemExit(f"panel step a is missing {len(missing)} enriched comments; rerun `pilot.py panel a`")
    agreed = sorted(cons.loc[cons.agree, "id"])
    disagree = sorted(cons.loc[~cons.agree, "id"])
    audit = sorted(random.Random(args.seed).sample(agreed, min(args.audit, len(agreed))))
    human = [*sample.loc[sample.stratum == "uniform", "id"], *audit, *disagree]
    sheet = sample.set_index("id").loc[human, ["stratum"]].reset_index()
    for c in C.LABEL_COLUMNS[2:]:
        sheet[c] = ""
    sheet = sheet.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sheet.to_csv(C.DATA / "labels_blank_human.csv", index=False)
    plan = {"audit_ids": audit, "adjudicate_ids": disagree, "n_uniform": int((sample.stratum == "uniform").sum()),
            "n_enriched_unanimous": len(agreed), "panel": C.LABEL_PANEL, "seed": args.seed}
    (C.DATA / "panel_plan.json").write_text(json.dumps(plan, indent=2))
    print(f"human sheet: {len(sheet)} comments ({plan['n_uniform']} uniform, {len(audit)} audit, "
          f"{len(disagree)} panel splits), shuffled together -> data/labels_blank_human.csv")
    print("next: python pilot.py label --labeler anthony --sheet data/labels_blank_human.csv")


def final_is_prediction(human_csv) -> pd.DataFrame:
    sample = load_sample()
    plan = json.loads((C.DATA / "panel_plan.json").read_text())
    human = pd.read_csv(human_csv, dtype=str, keep_default_na=False)
    human = human[human.is_prediction.isin(["0", "1"])].assign(id=lambda d: d.id.astype(int)).set_index("id")
    cons = consensus_a().set_index("id")
    audit, adjud = set(plan["audit_ids"]), set(plan["adjudicate_ids"])
    rows = []
    for r in sample.itertuples():
        i = int(r.id)
        if i in human.index:
            src = ("human" if r.stratum == "uniform" else
                   "human_audit" if i in audit else "human_adjudicated" if i in adjud else "human")
            rows.append({"id": i, "stratum": r.stratum, "is_prediction": human.at[i, "is_prediction"], "source": src,
                         "notes": human.at[i, "notes"]})
        elif r.stratum == "enriched" and i in cons.index and cons.at[i, "agree"]:
            rows.append({"id": i, "stratum": r.stratum, "is_prediction": cons.at[i, "label"], "source": "panel",
                         "notes": ""})
        else:
            rows.append({"id": i, "stratum": r.stratum, "is_prediction": "", "source": "unlabeled", "notes": ""})
    return pd.DataFrame(rows)


def cmd_panel(args) -> None:
    C.ensure_data()
    sample = load_sample()
    if args.step == "a":
        run_panel("a", [int(i) for i in sample.loc[sample.stratum == "enriched", "id"]], args)
        cons = consensus_a()
        print(f"panel unanimous on enriched is_prediction: {cons.agree.mean():.3f} ({(~cons.agree).sum()} splits go to you)")
        print("next: python pilot.py panel sheet")
    elif args.step == "sheet":
        build_sheet(args)
    elif args.step == "b":
        final = final_is_prediction(args.labels)
        pos = [int(i) for i in final.loc[final.is_prediction == "1", "id"]]
        print(f"{len(pos)} positives get stage B panel labels")
        run_panel("b", pos, args)
        print("next: python pilot.py panel merge --labels", args.labels)
    elif args.step == "merge":
        merge(args)


def merge(args) -> None:
    from itertools import combinations

    from .score import agreement, field_kind, n_levels

    final = final_is_prediction(args.labels)
    panel_b = load_panel("b")
    for f in [*C.LABEL_FIELDS_B, "subject_text", "labeler"]:
        final[f] = ""
    votes: dict[str, dict[int, list]] = {f: {} for f in C.LABEL_FIELDS_B}
    for k in final.index:
        i = int(final.at[k, "id"])
        src = final.at[k, "source"]
        final.at[k, "labeler"] = "anthony" if src.startswith("human") else ("panel" if src == "panel" else "")
        answers = [a for a in (panel_b[m].get(i) for m in C.LABEL_PANEL) if a is not None]
        if final.at[k, "is_prediction"] != "1" or len(answers) < 2:  # one model may fail on a comment
            continue
        for f in C.LABEL_FIELDS_B:
            vals = [a.get(f) for a in answers]
            if None in vals:
                continue
            pair_ok = len(vals) == len(C.LABEL_PANEL)  # pairwise agreement stats only on complete votes
            if f in SCORE_B:
                vals = [float(v) for v in vals]
                final.at[k, f] = str(float(np.median(vals)))
            else:
                vals = [("1" if v else "0") for v in vals] if f in BOOL_B else list(vals)
                final.at[k, f] = majority(vals)  # no majority: left unlabeled for that field
            if pair_ok:
                votes[f][i] = vals
        final.at[k, "subject_text"] = answers[0].get("subject_text", "")
    final = final[C.LABEL_COLUMNS + ["source"]]
    final.to_csv(C.DATA / "labels_final.csv", index=False)

    # Audit: unanimous panel label vs your blind label, on a random sample of unanimous enriched comments
    plan = json.loads((C.DATA / "panel_plan.json").read_text())
    cons = consensus_a().set_index("id")
    fi = final.set_index("id")
    audit = [i for i in plan["audit_ids"] if fi.at[i, "is_prediction"] in ("0", "1")]
    p = np.array([int(cons.at[i, "label"]) for i in audit])
    h = np.array([int(fi.at[i, "is_prediction"]) for i in audit])
    err = int((p != h).sum())
    adj = [i for i in plan["adjudicate_ids"] if fi.at[i, "is_prediction"] in ("0", "1")]
    cols = [f"m{k}" for k in range(len(C.LABEL_PANEL))]
    sided = {m: int(sum(cons.at[i, c] is not None and bool(cons.at[i, c]) == (fi.at[i, "is_prediction"] == "1")
                        for i in adj)) for m, c in zip(C.LABEL_PANEL, cols)}
    both = cons.dropna(subset=cols)
    pairwise_a = {f"{C.LABEL_PANEL[x]} vs {C.LABEL_PANEL[y]}": float((both[cols[x]] == both[cols[y]]).mean())
                  for x, y in combinations(range(len(cols)), 2)} if len(both) else {}
    report = {
        "panel": C.LABEL_PANEL,
        "rule": "is_prediction: unanimous -> panel label, any split -> human. Stage B: strict majority, median for scores.",
        "provenance": final.groupby(["stratum", "source"]).size().rename("n").reset_index().to_dict("records"),
        "audit_is_prediction": {
            "n": len(audit), "errors": err, "error_rate": err / len(audit) if audit else None,
            "error_rate_ci95": list(wilson(err, len(audit))) if audit else None,
            "panel_false_positives": int(((p == 1) & (h == 0)).sum()),
            "panel_false_negatives": int(((p == 0) & (h == 1)).sum()),
            "kappa": M.cohen_kappa(p, h) if audit else None},
        "adjudicated": {"n": len(adj), "human_sided_with": sided},
        "panel_unanimity_is_prediction": {"n": int(len(both)), "rate": float(both.agree.mean()) if len(both) else None,
                                          "pairwise_agreement": pairwise_a},
        "stage_b_panel_agreement": {},
    }
    # Stage B reference ceiling: mean pairwise agreement between panel models, per field
    for f, per_id in votes.items():
        kind = field_kind(f)
        pairs = [agreement([v[x] for v in per_id.values()], [v[y] for v in per_id.values()], kind,
                           n_levels(f) if kind == "score" else None)
                 for x, y in combinations(range(len(C.LABEL_PANEL)), 2)] if per_id else []
        ags = [a["agreement"] for a in pairs if a.get("n")]
        report["stage_b_panel_agreement"][f] = {"n": len(per_id), "agreement": float(np.mean(ags)) if ags else None,
                                                "pairwise": [a.get("agreement") for a in pairs]}
    (C.DATA / "panel_audit.json").write_text(json.dumps(report, indent=2, default=float))
    au = report["audit_is_prediction"]
    print(f"-> data/labels_final.csv ({(final.is_prediction != '').sum()} labeled, "
          f"{(final.is_prediction == '1').sum()} positives)")
    print(f"-> data/panel_audit.json: audit error rate {au['error_rate']} on {au['n']} (95% CI {au['error_rate_ci95']})")
    print("next: python pilot.py score --labels data/labels_final.csv ...")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))
