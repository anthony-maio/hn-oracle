"""Scoring, pre-registered go/no-go gates, and the full-run projection.

Everything here reads labels and raw JSONL; it never calls an API.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .runs import load_nonce, load_stage_a, load_stage_b, usage_summary
from .sample import load_sample

# Pre-registered thresholds (PILOT_PLAN.md, "Go/no-go gates"). Do not edit after the first Jev call.
GATES = {
    "G1_recall_min": 0.85,
    "G2_ece_max": 0.08,
    "G3_gap_abs_max": 0.02,
    "G4_f1_drop_max": 0.03,
    "G4_brier_ratio_max": 1.10,
    "G5_regex_margin": 0.15,
    "G5_frontier_margin": 0.05,
    "G6_nonce_median_max": 0.05,
    "G7_cost_max_usd": 3000.0,
    "G7_days_max": 7.0,
    "stageB_ratio_min": 0.80,
    "subject_other_max": 0.40,
}
STAGE_B_TOKENS_ASSUMED = 1250  # plan's 1,000-1,500 estimate; only used when no stage B run exists


# ---------------------------------------------------------------- labels


def load_labels(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["id"] = df.id.astype(int)
    df = df[df.is_prediction.str.strip().isin(["0", "1"])].copy()
    df["y"] = df.is_prediction.astype(int)
    if "source" in df.columns:
        df = df[df.source != "unlabeled"]
    return df


def rel_paths(v):
    """Input paths relative to the repo, so the published report doesn't carry a local drive layout."""
    if isinstance(v, (list, tuple)):
        return [rel_paths(x) for x in v]
    if isinstance(v, (str, Path)):
        try:
            return Path(v).resolve().relative_to(C.ROOT).as_posix()
        except (ValueError, OSError):
            return str(v)
    return v


def gate_result(passed: bool | None, value, threshold, detail: str = "") -> dict:
    status = "not_evaluated" if passed is None else ("pass" if passed else "fail")
    return {"status": status, "value": value, "threshold": threshold, "detail": detail}


# ---------------------------------------------------------------- stage A


def join_probs(labels: pd.DataFrame, probs: dict[int, float]) -> pd.DataFrame:
    df = labels[labels.id.isin(probs.keys())][["id", "stratum", "y"]].copy()
    df["p"] = df.id.map(probs).astype(float)
    return df


def strata(df: pd.DataFrame):
    yield "uniform", df[df.stratum == "uniform"]
    yield "enriched", df[df.stratum == "enriched"]
    yield "all", df


def stage_a_block(df: pd.DataFrame, gate: float) -> dict:
    out = {}
    for name, part in strata(df):
        if part.empty:
            continue
        p, y = part.p.to_numpy(), part.y.to_numpy()
        out[name] = {
            "n": int(len(part)), "positives": int(y.sum()), "base_rate": float(y.mean()),
            "brier": M.brier(p, y), "brier_base_rate": M.brier(np.full_like(p, y.mean()), y),
            "ece": M.ece(p, y), **M.best_f1(p, y),
            "at_gate": M.prf(p, y, gate),
            "reliability": M.reliability(p, y),
        }
    return out


def heldout_gate_recall(single: pd.DataFrame, target: float, seed: int = 0) -> dict:
    """Two-fold check on gate selection: choose the gate on one half, measure uniform recall on the other."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(single), int)
    for _, idx in single.reset_index(drop=True).groupby(["stratum", "y"]).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        fold[idx[len(idx) // 2:]] = 1
    df = single.reset_index(drop=True).assign(fold=fold)
    out = []
    for f in (0, 1):
        tr, te = df[df.fold != f], df[(df.fold == f) & (df.stratum == "uniform")]
        if not tr.y.sum() or not te.y.sum():
            continue
        t = M.choose_gate(tr.p.to_numpy(), tr.y.to_numpy(), target)
        out.append({"gate": t, "uniform_recall": M.prf(te.p, te.y, t)["recall"]})
    return {"folds": out, "uniform_recall_mean": float(np.mean([o["uniform_recall"] for o in out])) if out else None}


def recalibration_block(u: pd.DataFrame, seed: int = 0) -> dict:
    p, y = u.p.to_numpy(), u.y.to_numpy()
    out = {"raw": {"ece": M.ece(p, y), "brier": M.brier(p, y), "prevalence": M.prevalence(p, y)}}
    if y.sum() < 2 or (1 - y).sum() < 2:
        return out
    for method in ("temperature", "isotonic"):
        q, params = M.cross_fit(p, y, method, seed)
        out[method] = {"ece": M.ece(q, y), "brier": M.brier(q, y), "prevalence": M.prevalence(q, y),
                       "fold_params": params, "reliability": M.reliability(q, y)}
    out["temperature_full_fit"] = M.fit_temperature(p, y)
    out["protocol"] = ("two-fold, label-stratified cross-fit on the uniform stratum: fit on one half, "
                       "report on the other, then swap, so every comment gets a held-out probability")
    return out


def packing_block(single: pd.DataFrame, packed: dict[int, pd.DataFrame], gate: float, seed: int = 0) -> dict:
    out = {}
    for n, pk in sorted(packed.items()):
        common = single.merge(pk[["id", "p"]], on="id", suffixes=("_single", "_packed"))
        ps, pp, y = common.p_single.to_numpy(), common.p_packed.to_numpy(), common.y.to_numpy()
        f1_s, f1_p = M.best_f1(ps, y)["best_f1"], M.best_f1(pp, y)["best_f1"]
        b_s, b_p = M.brier(ps, y), M.brier(pp, y)
        ci = M.bootstrap(lambda a, b, c: M.f1_curve(b, c).max() - M.f1_curve(a, c).max(),
                         ps, pp, y, n=1000, seed=seed)
        f1_ok = f1_s - f1_p <= GATES["G4_f1_drop_max"]
        brier_ok = b_p <= b_s * GATES["G4_brier_ratio_max"]
        u = common[common.stratum == "uniform"]
        out[str(n)] = {
            "n_common": int(len(common)), "n_missing": int(len(single) - len(common)),
            "best_f1_single": f1_s, "best_f1_packed": f1_p, "delta_f1": f1_p - f1_s, "delta_f1_ci95": ci,
            "brier_single": b_s, "brier_packed": b_p, "brier_ratio": b_p / b_s if b_s else None,
            "uniform_ece_packed": M.ece(u.p_packed, u.y) if len(u) else None,
            "uniform_recall_at_gate_packed": M.prf(u.p_packed, u.y, gate)["recall"] if len(u) else None,
            "median_abs_dp_vs_single": float(np.median(np.abs(pp - ps))),
            "passes": bool(f1_ok and brier_ok),
        }
    return out


def nonce_block(nonce: pd.DataFrame, single_probs: dict[int, float]) -> dict:
    diffs, ranges, vs_single = [], [], []
    for iid, g in nonce.groupby("id"):
        ps = g.sort_values("rep").p.to_numpy()
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                diffs.append(abs(ps[a] - ps[b]))
        ranges.append(ps.max() - ps.min())
        if iid in single_probs:
            vs_single.extend(np.abs(ps - single_probs[iid]).tolist())
    q = lambda xs, k: float(np.percentile(xs, k)) if len(xs) else None
    return {"n_comments": int(nonce.id.nunique()), "n_pairs": len(diffs),
            "abs_dp_median": q(diffs, 50), "abs_dp_p95": q(diffs, 95),
            "range_median": q(ranges, 50), "range_p95": q(ranges, 95),
            "abs_dp_vs_single_median": q(vs_single, 50), "abs_dp_vs_single_p95": q(vs_single, 95)}


def baselines_block(single: pd.DataFrame, files: dict[str, str], labels: pd.DataFrame) -> dict:
    out = {}
    for name, path in files.items():
        df = join_probs(labels, load_stage_a(path))
        if name == "regex":
            df = df[df.stratum == "uniform"]  # enriched was selected with the regex: circular
        common = df.merge(single[["id", "p"]], on="id", suffixes=("", "_jev"))
        if common.empty:
            continue
        p, pj, y = common.p.to_numpy(), common.p_jev.to_numpy(), common.y.to_numpy()
        entry = {"n": int(len(common)), "scope": "uniform" if name == "regex" else "all",
                 "baseline": {**M.best_f1(p, y), "f1_at_0.5": M.prf(p, y, 0.5)["f1"],
                              "brier": M.brier(p, y), "ece": M.ece(p, y)},
                 "jev_same_ids": {**M.best_f1(pj, y), "brier": M.brier(pj, y), "ece": M.ece(pj, y)}}
        if name == "regex":
            entry["baseline"]["best_f1"] = entry["baseline"]["f1_at_0.5"]  # binary output: one operating point
        entry["jev_minus_baseline_f1"] = entry["jev_same_ids"]["best_f1"] - entry["baseline"]["best_f1"]
        out[name] = entry
        part = common[common.stratum == "uniform"]
        if name != "regex" and part.y.sum():
            entry["uniform"] = {"baseline_best_f1": M.best_f1(part.p, part.y)["best_f1"],
                                "jev_best_f1": M.best_f1(part.p_jev, part.y)["best_f1"]}
    return out


# ---------------------------------------------------------------- humans and stage B


def field_kind(field: str) -> str:
    t = C.QUESTIONS["stage_b"][field]["type"]
    return {"noul": "bool", "choice": "choice", "score": "score"}[t]


def n_levels(field: str) -> int:
    return len(C.QUESTIONS["stage_b"][field]["criteria"])


def agreement(a: list, b: list, kind: str, levels: int | None = None) -> dict:
    if not a:
        return {"n": 0}
    if kind == "score":
        x, z = np.array(a, float), np.array(b, float)
        mae = float(np.mean(np.abs(x - z)))
        return {"n": len(a), "mae_levels": mae, "agreement": 1 - mae / (levels - 1),
                "exact": float(np.mean(np.round(x) == np.round(z))),
                "kappa_linear": M.cohen_kappa(np.round(x).astype(int), np.round(z).astype(int), "linear")}
    acc = float(np.mean([x == z for x, z in zip(a, b)]))
    return {"n": len(a), "accuracy": acc, "agreement": acc, "kappa": M.cohen_kappa(a, b)}


def human_block(primary: pd.DataFrame, second: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    m = primary.merge(second, on="id", suffixes=("_1", "_2"))
    out = {"is_prediction": {"n": int(len(m)),
                             "kappa": M.cohen_kappa(m.y_1.to_numpy(), m.y_2.to_numpy()),
                             "accuracy": float((m.y_1 == m.y_2).mean()) if len(m) else None}}
    both = m[(m.y_1 == 1) & (m.y_2 == 1)]
    for f in C.LABEL_FIELDS_B:
        ok = both[(both[f + "_1"] != "") & (both[f + "_2"] != "")]
        kind = field_kind(f)
        a, b = ok[f + "_1"].tolist(), ok[f + "_2"].tolist()
        out[f] = agreement([float(v) for v in a] if kind == "score" else a,
                           [float(v) for v in b] if kind == "score" else b, kind, n_levels(f) if kind == "score" else None)
    rows = []
    for _, r in m.iterrows():
        diff = ["is_prediction"] if r.y_1 != r.y_2 else []
        if r.y_1 == 1 and r.y_2 == 1:
            diff += [f for f in C.LABEL_FIELDS_B if r[f + "_1"] != r[f + "_2"] and r[f + "_1"] and r[f + "_2"]]
        if diff:
            rows.append({"id": r.id, "fields": " ".join(diff),
                         **{f"{f}_{s}": r[f"{f}_{s}"] for f in ["is_prediction", *C.LABEL_FIELDS_B] for s in (1, 2)},
                         "resolution": "", "notes": ""})
    return out, pd.DataFrame(rows)


def jev_field_value(answer: dict, field: str):
    kind = field_kind(field)
    if kind == "bool":
        return "1" if float(answer["noul"]) >= 0.5 else "0"
    if kind == "choice":
        return answer["choice"]
    return M.score_level(answer, n_levels(field))[0]


def stage_b_block(labels: pd.DataFrame, b_answers: dict[int, dict], human: dict | None) -> dict:
    pos = labels[(labels.y == 1) & labels.id.isin(b_answers.keys())]
    out = {"n_labeled_positives_with_stage_b": int(len(pos)), "fields": {}}
    rules = set()
    for f in C.LABEL_FIELDS_B:
        kind = field_kind(f)
        rows = pos[pos[f] != ""]
        a, b = [], []
        for _, r in rows.iterrows():
            ans = b_answers[r.id].get(f)
            if ans is None:
                continue
            if kind == "score":
                lvl, rule = M.score_level(ans, n_levels(f))
                rules.add(rule)
                a.append(float(r[f])); b.append(lvl)
            else:
                a.append(r[f]); b.append(jev_field_value(ans, f))
        ag = agreement(a, b, kind, n_levels(f) if kind == "score" else None)
        h = (human or {}).get(f, {})
        ratio = (ag.get("agreement") / h["agreement"]) if ag.get("n") and h.get("n") and h.get("agreement") else None
        ag["human_agreement"] = h.get("agreement")
        ag["ratio_to_human"] = ratio
        ag["status"] = ("not_evaluated" if ratio is None else
                        "pass" if ratio >= GATES["stageB_ratio_min"] else "fail_drop_or_mark_experimental")
        out["fields"][f] = ag
    out["score_level_rule"] = sorted(rules)
    subj = [a["subject"]["choice"] for i, a in b_answers.items() if "subject" in a and i in set(pos.id)]
    if subj:
        share = float(np.mean([s == "other" for s in subj]))
        out["subject_other_share"] = {"n": len(subj), "share": share,
                                      "roster_needs_buildout": share > GATES["subject_other_max"]}
    return out


# ---------------------------------------------------------------- economics


def eligible_count(override: int | None) -> tuple[int, str]:
    if override:
        return override, "cli"
    path = C.DATA / "eligible.json"
    if path.exists():
        e = json.loads(path.read_text())
        return int(e["n_len_filtered"]), "data/eligible.json (2006-2020, 80-4000 chars)"
    return C.ARCHIVE_COMMENTS_FALLBACK, "FALLBACK: whole archive; run `pilot.py count` for the exact eligible subset"


def project(n: int, a_usage: dict, pass_rate: float, b_tokens_per_comment: float,
            price: float, rpm: float, tps: float) -> dict:
    cpr = a_usage["comments_per_request"]
    req_a, req_b = n / cpr, n * pass_rate
    tok_a, tok_b = n * a_usage["input_tokens_per_comment"], n * pass_rate * b_tokens_per_comment
    requests, tokens = req_a + req_b, tok_a + tok_b
    h_req, h_tok = requests / rpm / 60, tokens / tps / 3600
    hours = max(h_req, h_tok)
    return {"comments": n, "stage_b_pass_rate": pass_rate, "stage_b_comments": int(n * pass_rate),
            "requests": int(requests), "input_tokens": int(tokens),
            "cost_usd": tokens / 1e6 * price, "cost_usd_stage_a": tok_a / 1e6 * price,
            "hours_request_limit": h_req, "hours_token_limit": h_tok, "hours": hours, "days": hours / 24,
            "binding_limit": "requests" if h_req >= h_tok else "tokens"}


# ---------------------------------------------------------------- report


def excerpt(s: str, n: int = 400) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def failure_cases(single: pd.DataFrame, gate: float, k: int = 10) -> dict:
    sample = load_sample().set_index("id")
    miss = single[(single.y == 1) & (single.p < gate)].sort_values("p").head(k)
    fp = single[(single.y == 0) & (single.p >= gate)].sort_values("p", ascending=False).head(k)
    fmt = lambda d: [{"id": int(r.id), "stratum": r.stratum, "p": float(r.p),
                      "story_title": sample.loc[r.id].story_title, "comment": excerpt(sample.loc[r.id].text)}
                     for r in d.itertuples()]
    return {"misses": fmt(miss), "false_positives": fmt(fp)}


def cmd_score(args) -> None:
    labels = load_labels(args.labels)
    report: dict = {"schema_version": C.SCHEMA_VERSION, "model": C.MODEL, "gates_config": GATES,
                    "inputs": {k: rel_paths(v) for k, v in vars(args).items() if k != "func"}}
    gates: dict = {}

    # Label provenance (plan amendment A1) and the stage B reference ceiling
    human = None
    audit_path = C.DATA / "panel_audit.json"
    if "source" in labels.columns:
        report["label_provenance"] = labels.groupby(["stratum", "source"]).size().rename("n").reset_index().to_dict("records")
        if audit_path.exists():
            pa = json.loads(audit_path.read_text())
            report["panel_audit"] = {k: pa[k] for k in ("panel", "rule", "audit_is_prediction", "adjudicated",
                                                        "panel_unanimity_is_prediction")}
            human = pa["stage_b_panel_agreement"]
            report["stage_b_reference"] = "mean pairwise agreement between labeling-panel models"
    if args.second_labels:
        human, dis = human_block(labels, load_labels(args.second_labels))
        report["human_agreement"] = human
        report["stage_b_reference"] = "human-human agreement (second labeler)"
        if len(dis):
            dis.to_csv(C.DATA / "disagreements.csv", index=False)
            report["human_agreement"]["disagreements_file"] = "data/disagreements.csv"

    # Stage A single
    single_probs = load_stage_a(args.single)
    single = join_probs(labels, single_probs)
    u = single[single.stratum == "uniform"]
    if args.gate is not None:
        gate, chosen_by = args.gate, "cli"
    else:
        gate = M.choose_gate(single.p.to_numpy(), single.y.to_numpy(), args.target_recall)
        chosen_by = f"all labeled comments: highest threshold with recall >= {args.target_recall}"
    report["gate"] = {"threshold": gate, "chosen_by": chosen_by,
                      "heldout_check": heldout_gate_recall(single, args.target_recall)}
    report["stage_a_single"] = stage_a_block(single, gate)

    ua = report["stage_a_single"].get("uniform", {})
    rec = ua.get("at_gate", {}).get("recall")
    gates["G1_recall"] = gate_result(None if rec is None or not ua.get("positives") else
                                     rec >= GATES["G1_recall_min"], rec, GATES["G1_recall_min"],
                                     f"uniform recall at gate {gate}; held-out (gate chosen on the other half): "
                                     f"{fmt(report['gate']['heldout_check'].get('uniform_recall_mean'))}")

    recal = recalibration_block(u) if len(u) else {}
    report["recalibration_uniform"] = recal
    if recal:
        methods = [m for m in ("temperature", "isotonic") if m in recal]
        best = min(methods, key=lambda m: recal[m]["ece"]) if methods else None
        best_ece = min([recal["raw"]["ece"]] + [recal[m]["ece"] for m in methods])
        gates["G2_calibration"] = gate_result(best_ece <= GATES["G2_ece_max"], best_ece, GATES["G2_ece_max"],
                                              f"raw ECE {recal['raw']['ece']:.4f}; best recalibrated: {best}")
        if best:
            pv = recal[best]["prevalence"]
            gates["G3_prevalence"] = gate_result(abs(pv["gap"]) <= GATES["G3_gap_abs_max"] and pv["ci_includes_zero"],
                                                 pv["gap"], GATES["G3_gap_abs_max"],
                                                 f"{best}-recalibrated gap, 95% CI [{pv['gap_ci95'][0]:+.3f}, {pv['gap_ci95'][1]:+.3f}]")
        report["prevalence_uniform_raw"] = recal["raw"]["prevalence"]

    # Packing
    packed_frames, packed_usage = {}, {}
    for path in args.packed or []:
        m = re.search(r"packed(\d+)", Path(path).name)
        n = int(m.group(1)) if m else len(packed_frames) + 2
        packed_frames[n] = join_probs(labels, load_stage_a(path))
        packed_usage[n] = usage_summary(path)
    if packed_frames:
        pk = packing_block(single, packed_frames, gate)
        report["packing"] = pk
        passing = [int(n) for n, v in pk.items() if v["passes"]]
        design_n = max(passing) if passing else 1
        gates["G4_packing"] = gate_result(bool(passing), design_n, "F1 drop <= 0.03 and Brier <= 1.10x single",
                                          "largest passing N goes into the full-run design" if passing else
                                          "no packed N passed: try --pack 4; if that fails, a full run needs a higher rate limit")
    else:
        design_n = 1

    # Nonce robustness
    if args.nonce:
        nb = nonce_block(load_nonce(args.nonce), single_probs)
        report["nonce_robustness"] = nb
        gates["G6_robustness"] = gate_result(nb["abs_dp_median"] is not None and
                                             nb["abs_dp_median"] <= GATES["G6_nonce_median_max"],
                                             nb["abs_dp_median"], GATES["G6_nonce_median_max"],
                                             f"p95 {fmt(nb['abs_dp_p95'])}")

    # Baselines
    bfiles = {k: v for k, v in (("regex", args.baseline_regex), ("cheap_llm", args.baseline_cheap),
                                ("frontier_llm", args.baseline_frontier)) if v}
    for spec in args.baseline_extra or []:
        name, _, path = spec.partition("=")
        bfiles[name] = path
    if bfiles:
        # Baselines are scored on human labels only: a model must not be graded on panel-written labels.
        human_labels = labels[labels.source.str.startswith("human")] if "source" in labels.columns else labels
        bl = baselines_block(single, bfiles, human_labels)
        report["baselines"] = bl
        g5_parts, ok = [], True
        if "regex" in bl:
            d = bl["regex"]["jev_minus_baseline_f1"]
            ok &= d >= GATES["G5_regex_margin"]
            g5_parts.append(f"vs regex (uniform) {d:+.3f}")
        if "frontier_llm" in bl:
            d = bl["frontier_llm"]["jev_minus_baseline_f1"]
            ok &= d >= -GATES["G5_frontier_margin"]
            g5_parts.append(f"vs frontier (all) {d:+.3f}")
        complete = "regex" in bl and "frontier_llm" in bl
        gates["G5_baselines"] = gate_result(ok if complete else (False if not ok else None), "; ".join(g5_parts),
                                            "regex +0.15, frontier -0.05",
                                            "" if complete else "needs both regex and frontier baselines")

    # Stage B
    b_usage = None
    if args.stage_b:
        report["stage_b"] = stage_b_block(labels, load_stage_b(args.stage_b), human)
        b_usage = usage_summary(args.stage_b)
        fields = report["stage_b"]["fields"]
        evaluated = {f: v["status"] for f, v in fields.items() if v["status"] != "not_evaluated"}
        gates["stageB_fields"] = gate_result(
            None if not evaluated else all(s == "pass" for s in evaluated.values()),
            {f: None if v.get("ratio_to_human") is None else round(v["ratio_to_human"], 3) for f, v in fields.items()},
            GATES["stageB_ratio_min"],
            "failing fields are dropped or marked experimental; this does not block Go")

    # Cost and speed
    n_elig, elig_src = eligible_count(args.eligible)
    usage = {"single": usage_summary(args.single), **{f"packed{n}": v for n, v in packed_usage.items()}}
    if b_usage:
        usage["stage_b"] = b_usage
    b_tpc = b_usage["input_tokens_per_comment"] if b_usage else STAGE_B_TOKENS_ASSUMED
    projections = {}
    for name, frame in [("single", single), *[(f"packed{n}", f) for n, f in sorted(packed_frames.items())]]:
        fu = frame[frame.stratum == "uniform"]
        pass_rate = float((fu.p >= gate).mean()) if len(fu) else float((frame.p >= gate).mean())
        projections[name] = project(n_elig, usage[name], pass_rate, b_tpc, args.price, args.rpm, args.tps)
    design = f"packed{design_n}" if design_n > 1 else "single"
    report["economics"] = {"eligible_comments": n_elig, "eligible_source": elig_src,
                           "price_per_m_input": args.price, "rpm": args.rpm, "tps": args.tps,
                           "stage_b_tokens_per_comment": b_tpc,
                           "stage_b_tokens_source": "measured" if b_usage else "ASSUMED (no stage B run)",
                           "usage": usage, "projections": projections, "design": design}
    pj = projections[design]
    gates["G7_economics"] = gate_result(pj["cost_usd"] < GATES["G7_cost_max_usd"] and pj["days"] < GATES["G7_days_max"],
                                        {"cost_usd": round(pj["cost_usd"], 2), "days": round(pj["days"], 2)},
                                        {"cost_usd": GATES["G7_cost_max_usd"], "days": GATES["G7_days_max"]},
                                        f"design {design}, binding limit: {pj['binding_limit']}")

    # Decision
    order = ("G1_recall", "G2_calibration", "G3_prevalence", "G4_packing", "G5_baselines", "G6_robustness",
             "G7_economics", "stageB_fields")
    gates = {g: gates.get(g, gate_result(None, None, None, "inputs not provided")) for g in order}
    core = [gates[g]["status"] for g in ("G1_recall", "G2_calibration", "G4_packing", "G7_economics")]
    decision = "go" if all(s == "pass" for s in core) else ("no_go" if "fail" in core else "incomplete")
    report["gates"] = gates
    report["decision"] = {"result": decision, "rule": "Go if G1, G2, G4 and G7 pass. G3, G5, G6 shape the public framing.",
                          "framing": {
                              "prevalence": "estimates" if gates["G3_prevalence"]["status"] == "pass" else
                                            "labeled counts, not estimates (honest section in the write-up)",
                          }}
    report["failure_cases"] = failure_cases(single, gate)

    out = Path(args.out) if args.out else C.DATA / "report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    md = out.with_suffix(".md")
    md.write_text(render_md(report), encoding="utf-8")
    print(render_gates(report))
    print(f"-> {out}\n-> {md}")


# ---------------------------------------------------------------- markdown


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_gates(r: dict) -> str:
    lines = [f"DECISION: {r['decision']['result'].upper()}", ""]
    for g, v in r["gates"].items():
        lines.append(f"  {g:<16} {v['status']:<14} value={v['value']}  threshold={v['threshold']}  {v['detail']}")
    return "\n".join(lines)


def render_md(r: dict) -> str:
    L = [f"# HN Oracle pilot report", "",
         f"Model `{r['model']}`, schema `{r['schema_version']}`. Gate threshold {r['gate']['threshold']} "
         f"({r['gate']['chosen_by']}).", "",
         f"**Decision: {r['decision']['result'].upper()}** ({r['decision']['rule']})", "",
         "## Gates", "", "| Gate | Status | Value | Threshold | Detail |", "|---|---|---|---|---|"]
    for g, v in r["gates"].items():
        L.append(f"| {g} | {v['status']} | {fmt(v['value'])} | {fmt(v['threshold'])} | {v['detail']} |")

    L += ["", "## Stage A (single comment per request)", "",
          "| Stratum | n | pos | base rate | Brier | Brier (base rate) | ECE | best F1 @t | P / R / F1 at gate | pass rate |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in r["stage_a_single"].items():
        g = v["at_gate"]
        L.append(f"| {s} | {v['n']} | {v['positives']} | {fmt(v['base_rate'])} | {fmt(v['brier'])} | "
                 f"{fmt(v['brier_base_rate'])} | {fmt(v['ece'])} | {fmt(v['best_f1'])} @{v['best_f1_threshold']} | "
                 f"{fmt(g['precision'])} / {fmt(g['recall'])} / {fmt(g['f1'])} | {fmt(g['pass_rate'])} |")
    if "uniform" in r["stage_a_single"]:
        L += ["", "Reliability, uniform stratum (raw):", "", "| bin | n | mean p | frac true |", "|---|---|---|---|"]
        L += [f"| {b['bin']} | {b['n']} | {fmt(b['mean_p'])} | {fmt(b['frac_true'])} |"
              for b in r["stage_a_single"]["uniform"]["reliability"]]

    rc = r.get("recalibration_uniform") or {}
    if rc:
        L += ["", "## Calibration and prevalence (uniform stratum)", "", rc.get("protocol", ""), "",
              "| | ECE | Brier | est. prevalence | labeled | gap | gap 95% CI |", "|---|---|---|---|---|---|---|"]
        for m in ("raw", "temperature", "isotonic"):
            if m in rc:
                pv = rc[m]["prevalence"]
                L.append(f"| {m} | {fmt(rc[m]['ece'])} | {fmt(rc[m]['brier'])} | {fmt(pv['estimated'])} | "
                         f"{fmt(pv['labeled'])} | {pv['gap']:+.3f} | [{pv['gap_ci95'][0]:+.3f}, {pv['gap_ci95'][1]:+.3f}] |")

    if r.get("packing"):
        L += ["", "## Packing penalty (all labeled comments)", "",
              "| N | best F1 single | best F1 packed | ΔF1 (95% CI) | Brier ratio | median abs(Δp) vs single | passes |",
              "|---|---|---|---|---|---|---|"]
        for n, v in r["packing"].items():
            L.append(f"| {n} | {fmt(v['best_f1_single'])} | {fmt(v['best_f1_packed'])} | {v['delta_f1']:+.3f} "
                     f"[{v['delta_f1_ci95'][0]:+.3f}, {v['delta_f1_ci95'][1]:+.3f}] | {fmt(v['brier_ratio'])} | "
                     f"{fmt(v['median_abs_dp_vs_single'])} | {v['passes']} |")

    if r.get("nonce_robustness"):
        v = r["nonce_robustness"]
        L += ["", "## Nonce robustness", "",
              f"{v['n_comments']} comments. Pairwise abs(Δp) across repeats: median {fmt(v['abs_dp_median'])}, "
              f"p95 {fmt(v['abs_dp_p95'])}. Against the single run: median {fmt(v['abs_dp_vs_single_median'])}, "
              f"p95 {fmt(v['abs_dp_vs_single_p95'])}."]

    if r.get("baselines"):
        L += ["", "## Baselines (same state, same question)", "",
              "| Baseline | scope | n | baseline F1 | Jev F1 (same ids) | Jev − baseline | baseline Brier | baseline ECE |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in r["baselines"].items():
            L.append(f"| {k} | {v['scope']} | {v['n']} | {fmt(v['baseline']['best_f1'])} | "
                     f"{fmt(v['jev_same_ids']['best_f1'])} | {v['jev_minus_baseline_f1']:+.3f} | "
                     f"{fmt(v['baseline']['brier'])} | {fmt(v['baseline']['ece'])} |")

    if r.get("panel_audit"):
        pa = r["panel_audit"]
        au = pa["audit_is_prediction"]
        L += ["", "## Labels: provenance and panel audit (amendment A1)", "",
              f"Panel: {', '.join(pa['panel'])}. {pa['rule']}", "",
              "| stratum | source | n |", "|---|---|---|"]
        L += [f"| {x['stratum']} | {x['source']} | {x['n']} |" for x in r.get("label_provenance", [])]
        ci = au.get("error_rate_ci95") or [None, None]
        L += ["", f"Blind audit of unanimous panel labels: {au['errors']} errors in {au['n']} "
                  f"(rate {fmt(au['error_rate'])}, 95% CI [{fmt(ci[0])}, {fmt(ci[1])}]; "
                  f"{au['panel_false_positives']} false positives, {au['panel_false_negatives']} false negatives). "
                  f"Panel splits you adjudicated: {pa['adjudicated']['n']}, sided with each model: {pa['adjudicated']['human_sided_with']}.",
              "", f"Stage B reference: {r.get('stage_b_reference', 'none')}."]

    if r.get("human_agreement"):
        h = r["human_agreement"]
        L += ["", "## Human agreement (second labeler)", "",
              f"is_prediction: n {h['is_prediction']['n']}, Cohen's kappa {fmt(h['is_prediction']['kappa'])}, "
              f"raw agreement {fmt(h['is_prediction']['accuracy'])}."]

    if r.get("stage_b"):
        b = r["stage_b"]
        L += ["", "## Stage B fields", "",
              f"{b['n_labeled_positives_with_stage_b']} labeled positives with stage B answers. "
              f"Score mapping rule: {', '.join(b['score_level_rule']) or '–'}.", "",
              "| Field | n | Jev agreement | human agreement | ratio | status |", "|---|---|---|---|---|---|"]
        for f, v in b["fields"].items():
            L.append(f"| {f} | {v.get('n', 0)} | {fmt(v.get('agreement'))} | {fmt(v.get('human_agreement'))} | "
                     f"{fmt(v.get('ratio_to_human'))} | {v['status']} |")
        if b.get("subject_other_share"):
            s = b["subject_other_share"]
            L.append(f"\nSubject roster: {fmt(s['share'])} of {s['n']} positives landed in `other` "
                     f"({'build out the roster' if s['roster_needs_buildout'] else 'ok'}).")

    ec = r["economics"]
    L += ["", "## Cost, speed, and full-run projection", "",
          f"Eligible comments: {ec['eligible_comments']:,} ({ec['eligible_source']}). Price ${ec['price_per_m_input']}/M input, "
          f"{ec['rpm']} req/min, {ec['tps']:,} tokens/s. Stage B tokens/comment: {fmt(ec['stage_b_tokens_per_comment'], 0)} "
          f"({ec['stage_b_tokens_source']}).", "",
          "| Run | requests | tokens/comment | comments/request | p50 ms | p95 ms |", "|---|---|---|---|---|---|"]
    for k, v in ec["usage"].items():
        if v:
            L.append(f"| {k} | {v['requests']} | {fmt(v['input_tokens_per_comment'], 0)} | {fmt(v['comments_per_request'], 1)} | "
                     f"{fmt(v['latency_ms_p50'], 0)} | {fmt(v['latency_ms_p95'], 0)} |")
    L += ["", "| Design | stage B pass rate | requests | input tokens | cost USD | days (request limit) | days (token limit) | binding |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in ec["projections"].items():
        mark = " **(design)**" if k == ec["design"] else ""
        L.append(f"| {k}{mark} | {fmt(v['stage_b_pass_rate'])} | {v['requests']:,} | {v['input_tokens']:,} | "
                 f"{v['cost_usd']:,.2f} | {v['hours_request_limit'] / 24:.2f} | {v['hours_token_limit'] / 24:.2f} | {v['binding_limit']} |")

    fc = r["failure_cases"]
    for title, key in (("Misses (labeled prediction, below gate)", "misses"),
                       ("False positives (labeled not a prediction, above gate)", "false_positives")):
        L += ["", f"## {title}", ""]
        for c in fc[key]:
            L.append(f"- **{c['id']}** ({c['stratum']}, p={c['p']:.3f}), *{c['story_title']}*: {c['comment']}")
    return "\n".join(L) + "\n"
