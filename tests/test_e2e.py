"""End to end on a synthetic archive with the dataset card's schema, and mocked Firebase + Jev + OpenRouter."""
import json
import random
import zlib

import httpx
import pandas as pd
import pytest

from oracle import config as C
from oracle.text import regex_hint
from pilot import build_parser

PRED = ["Mark my words, Rust will replace C++ in {n} years.", "Eventually everyone is going to run {x} on the server.",
        "This company will be dead in {n} years, I bet.", "By 2020 nobody will remember {x} at all, trust me."]
PLAIN = ["I tried {x} last week and the docs were decent, though setup took a while longer than expected.",
         "Does anyone know how {x} handles memory? I have been reading the source and it is confusing.",
         "We use {x} at work and it is fine for our scale, no complaints from the team so far really."]
THINGS = ["Go", "Rust", "Docker", "Kubernetes", "Bitcoin", "Node", "Haskell", "React"]


def make_archive(root):
    import duckdb

    rng = random.Random(0)
    rows, i = [], 1
    for year in range(2008, 2021):
        for k in range(2200):
            tmpl = rng.choice(PRED) if rng.random() < 0.25 else rng.choice(PLAIN)
            text = tmpl.format(n=rng.randint(2, 10), x=rng.choice(THINGS)) + " " + "Some more context here." * rng.randint(0, 3)
            if k % 40 == 0:
                text = "too short"
            rows.append({"id": i, "deleted": int(k % 97 == 0), "type": 2 if k % 50 else 1, "by": f"user{i % 300}",
                         "time": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} 12:00:00",
                         "text": text.replace("'", "&#x27;"), "dead": int(k % 89 == 0),
                         "parent": 5_000_000 + i, "title": ""})
            i += 1
    df = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("df", df)
    for year in range(2008, 2021):
        (root / "data" / str(year)).mkdir(parents=True)
        con.execute(f"""
            COPY (SELECT id::UINTEGER AS id, deleted::UTINYINT AS deleted, type::TINYINT AS "type", "by", time::TIMESTAMP AS "time",
                         text, dead::UTINYINT AS dead, parent::UINTEGER AS parent, title
                  FROM df WHERE year(time::TIMESTAMP) = {year})
            TO '{(root / "data" / str(year) / f"{year}-01.parquet").as_posix()}' (FORMAT parquet)""")
    return (root / "data" / "*" / "*.parquet").as_posix()


def fake_transport(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "firebaseio" in url:
        iid = int(url.rsplit("/", 1)[1].split(".")[0])
        if iid < 9_000_000:  # parent comment -> story
            return httpx.Response(200, json={"id": iid, "type": "comment", "parent": iid + 4_000_000,
                                             "text": f"Parent comment {iid} &amp; context"})
        return httpx.Response(200, json={"id": iid, "type": "story", "title": f"Story {iid}"})

    if url.endswith("/api/v1/models"):
        return httpx.Response(200, json={"data": [
            {"id": "test/strict:free", "supported_parameters": ["structured_outputs", "response_format"]},
            {"id": "test/plain:free", "supported_parameters": []}]})
    body = json.loads(request.content)
    if "openrouter" in url:
        text = body["messages"][-1]["content"]
        states, _ = json.JSONDecoder().raw_decode(text[text.index("{", text.index("State (JSON)")):])
        if "Task: panel_a" in text:
            hit = regex_hint(states["comment"])
            if body["model"] == "test/plain:free" and zlib.crc32(states["comment"].encode()) % 6 == 0:
                hit = not hit  # manufacture panel splits
            out = {"is_prediction": hit}
        elif "Task: panel_b" in text:
            out = {"is_checkable": True, "is_sarcastic": False, "is_conditional": False, "direction": "success",
                   "domain": "language_or_framework", "horizon": "1_to_5y", "stance_vs_thread": "agrees",
                   "certainty": 2 if body["model"] != "test/plain:free" else 3, "specificity": 1, "subject_text": "Rust"}
        else:
            answer = lambda s: {"answer": regex_hint(s["comment"]), "p_prediction": 0.85 if regex_hint(s["comment"]) else 0.1}
            out = {c: answer(v) for c, v in states.items()} if "c01" in states else answer(states)
        strict = body.get("response_format", {}).get("type") == "json_schema"
        assert strict == body["model"].startswith("test/strict")
        content = json.dumps(out) if strict else "Sure! Here you go:\n" + json.dumps(out)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {}})

    state, answers = body["state"], {}
    for key, q in body["questions"].items():
        if key.startswith("is_prediction"):
            cid = key.removeprefix("is_prediction").lstrip("_")
            s = state[cid] if cid else state
            noise = (zlib.crc32((s["comment"] + s.get("request_nonce", "")).encode()) % 100) / 1000
            answers[key] = {"noul": min(1.0, (0.75 if regex_hint(s["comment"]) else 0.04) + noise)}
        elif q["type"] == "noul":
            answers[key] = {"noul": 0.7}
        elif q["type"] == "choice":
            opts = list(q["criteria"])
            answers[key] = {"choice": opts[0], "probabilities": {o: 1 / len(opts) for o in opts}, "confidence": 0.5}
        else:
            answers[key] = {"score": 0.6, "legend": {str(k): k / 3 for k in range(4)}, "probabilities": {}, "confidence": 0.5}
    return httpx.Response(200, json={"model": body["model"], "answers": answers,
                                     "usage": {"input_tokens": len(request.content) // 4, "output_tokens": 1}})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA", tmp_path / "out")
    monkeypatch.setattr(C, "PUBLISH", tmp_path / "publish")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    real = httpx.Client

    class MockClient(real):
        def __init__(self, *a, **kw):
            kw.pop("http2", None)
            kw["transport"] = httpx.MockTransport(fake_transport)
            super().__init__(*a, **kw)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setattr("oracle.jev.time.sleep", lambda s: None)  # a mock bug should fail fast, not back off
    return make_archive(tmp_path)


def cli(*argv):
    a = build_parser().parse_args([str(x) for x in argv])
    a.func(a)


def write_labels(data):
    sample = pd.read_json(data / "sample.jsonl", lines=True).set_index("id")
    sheet = pd.read_csv(data / "labels_blank.csv", dtype=str, keep_default_na=False)
    rng = random.Random(3)
    for i in sheet.index:
        y = regex_hint(sample.loc[int(sheet.at[i, "id"])].text)
        if rng.random() < 0.05:
            y = not y
        sheet.at[i, "is_prediction"] = "1" if y else "0"
        if y:
            sheet.loc[i, C.LABEL_FIELDS_B] = ["1", "0", "0", "success", "language_or_framework", "1_to_5y",
                                              "agrees", "2", "2"]
    sheet["labeler"] = "anthony"
    sheet.to_csv(data / "labels_anthony.csv", index=False)
    second = pd.read_csv(data / "labels_blank_second.csv", dtype=str)[["id"]].merge(
        sheet.assign(id=sheet.id.astype(int)).astype({"id": str}), on="id")
    second.loc[second.index[:5], "is_prediction"] = "0"
    second["labeler"] = "second"
    second.to_csv(data / "labels_second.csv", index=False)


def test_pipeline(env, monkeypatch):
    glob = env
    data = C.DATA

    cli("count", "--parquet-glob", glob)
    elig = json.loads((data / "eligible.json").read_text())
    assert [r["year"] for r in elig["by_year"]] == list(range(2006, 2021))[2:]  # synthetic data starts in 2008
    assert elig["n_len_filtered"] < elig["n_comments"]

    cli("sample", "--parquet-glob", glob, "--workers", 4)
    s = pd.read_json(data / "sample.jsonl", lines=True)
    assert len(s) == 1000 and s.stratum.value_counts().to_dict() == {"uniform": 600, "enriched": 400}
    assert s.year.between(2009, 2019).all() and s.id.is_unique
    assert s.text.str.len().between(40, 4000).all()
    assert s.story_title.str.startswith("Story").all() and not s.text.str.contains("&#x27;").any()
    assert len(pd.read_csv(data / "labels_blank_second.csv")) == 200

    # Sampling is deterministic
    first = list(s.id)
    cli("sample", "--parquet-glob", glob, "--force")
    assert list(pd.read_json(data / "sample.jsonl", lines=True).id) == first

    # Labeler: y (+note), n, undo, n, then quit; pass b on the one positive
    answers = iter(["y", "edge case", "n", "", "u", "n", "", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    cli("label", "--labeler", "tester", "--pass", "a")
    lab = pd.read_csv(data / "labels_tester.csv", dtype=str, keep_default_na=False)
    assert list(lab.is_prediction[:3]) == ["1", "0", ""] and lab.notes[0] == "edge case"
    answers = iter(["y", "n", "n", "1", "3", "2", "1", "4", "3", "Rust", ""])
    cli("label", "--labeler", "tester", "--pass", "b")
    lab = pd.read_csv(data / "labels_tester.csv", dtype=str, keep_default_na=False)
    assert list(lab.loc[0, C.LABEL_FIELDS_B]) == ["1", "0", "0", "success", "product_adoption", "1_to_5y",
                                                  "agrees", "3", "2"]

    write_labels(data)

    cli("run", "--mode", "a_single", "--rpm", 0)
    cli("run", "--mode", "a_single", "--rpm", 0)  # resume: nothing re-sent
    assert sum(1 for _ in open(data / "raw_a_single_v1.jsonl")) == 1000
    cli("run", "--mode", "a_packed", "--pack", 8, "--rpm", 0)
    cli("run", "--mode", "a_packed", "--pack", 16, "--rpm", 0)
    packed = [json.loads(l) for l in open(data / "raw_a_packed16_v1.jsonl")]
    assert len(packed) == 63 and sorted(i for r in packed for i in r["ids"]) == sorted(first)
    cli("run", "--mode", "a_nonce", "--rpm", 0)
    assert sum(1 for _ in open(data / "raw_a_nonce_v1.jsonl")) == 300
    cli("run", "--mode", "b", "--gate", 0.5, "--stage-a-file", data / "raw_a_single_v1.jsonl",
        "--also-labeled-positives", data / "labels_anthony.csv", "--with-subject", "--rpm", 0)
    cli("baseline", "--kind", "regex")
    cli("baseline", "--kind", "llm", "--model", "test/strict:free", "--name", "frontier", "--rpm", 0)
    cli("baseline", "--kind", "llm", "--model", "test/plain:free", "--name", "packed", "--pack", 16, "--rpm", 0)
    assert sum(1 for _ in open(data / "baseline_packed.jsonl")) == 63

    cli("score", "--labels", data / "labels_anthony.csv", "--second-labels", data / "labels_second.csv",
        "--single", data / "raw_a_single_v1.jsonl",
        "--packed", data / "raw_a_packed8_v1.jsonl", data / "raw_a_packed16_v1.jsonl",
        "--nonce", data / "raw_a_nonce_v1.jsonl", "--stage-b", data / "raw_b_v1.jsonl",
        "--baseline-regex", data / "baseline_regex.jsonl", "--baseline-frontier", data / "baseline_frontier.jsonl",
        "--baseline-extra", f"free_packed16={data / 'baseline_packed.jsonl'}")
    r = json.loads((data / "report.json").read_text())
    assert set(r["gates"]) >= {"G1_recall", "G2_calibration", "G3_prevalence", "G4_packing", "G5_baselines",
                               "G6_robustness", "G7_economics", "stageB_fields"}
    assert r["decision"]["result"] in {"go", "no_go"}
    assert r["gates"]["G4_packing"]["status"] == "pass" and r["economics"]["design"] == "packed16"
    assert r["stage_a_single"]["uniform"]["n"] == 600
    assert r["baselines"]["free_packed16"]["n"] == 1000
    assert r["human_agreement"]["is_prediction"]["n"] == 200
    assert r["stage_b"]["score_level_rule"] == ["legend"]
    assert r["stage_b"]["subject_other_share"]["n"] > 0
    assert r["economics"]["eligible_source"].startswith("data/eligible.json")
    assert (data / "report.md").read_text(encoding="utf-8").startswith("# HN Oracle pilot report")

    # Plan amendment A1: three-model labeling panel, human answer key on uniform + audit + splits
    monkeypatch.setattr(C, "LABEL_PANEL", ["test/strict:free", "test/plain:free", "test/other:free"])
    cli("panel", "a", "--rpm", 0)
    cli("panel", "sheet")
    plan = json.loads((data / "panel_plan.json").read_text())
    human_sheet = pd.read_csv(data / "labels_blank_human.csv", dtype=str, keep_default_na=False)
    assert len(human_sheet) == 600 + 100 + len(plan["adjudicate_ids"]) and plan["adjudicate_ids"]
    assert set(plan["audit_ids"]).isdisjoint(plan["adjudicate_ids"])
    sample = pd.read_json(data / "sample.jsonl", lines=True).set_index("id")
    human_sheet["is_prediction"] = [("1" if regex_hint(sample.loc[int(i)].text) else "0") for i in human_sheet.id]
    human_sheet.loc[human_sheet.index[:3], "is_prediction"] = "1"  # a few audit disagreements are fine
    human_sheet.to_csv(data / "labels_human.csv", index=False)
    cli("panel", "b", "--labels", data / "labels_human.csv", "--rpm", 0)
    cli("panel", "merge", "--labels", data / "labels_human.csv")
    final = pd.read_csv(data / "labels_final.csv", dtype=str, keep_default_na=False)
    assert (final.is_prediction != "").all()
    assert set(final.loc[final.stratum == "uniform", "source"]) == {"human"}
    assert {"panel", "human_audit", "human_adjudicated"} <= set(final.source)
    pos = final[final.is_prediction == "1"]
    assert (pos.certainty == "2.0").all() and (pos.direction == "success").all()  # median / majority
    audit = json.loads((data / "panel_audit.json").read_text())
    assert audit["audit_is_prediction"]["n"] == 100
    assert audit["stage_b_panel_agreement"]["certainty"]["agreement"] < 1.0

    cli("score", "--labels", data / "labels_final.csv", "--single", data / "raw_a_single_v1.jsonl",
        "--stage-b", data / "raw_b_v1.jsonl", "--baseline-regex", data / "baseline_regex.jsonl",
        "--baseline-frontier", data / "baseline_frontier.jsonl", "--out", data / "report_panel.json")
    rp = json.loads((data / "report_panel.json").read_text())
    n_human = int(final.source.str.startswith("human").sum())
    assert rp["baselines"]["frontier_llm"]["n"] == n_human  # baselines never graded on panel labels
    assert rp["stage_b_reference"].startswith("mean pairwise")
    assert rp["panel_audit"]["audit_is_prediction"]["n"] == 100
    assert "provenance and panel audit" in (data / "report_panel.md").read_text(encoding="utf-8")

    cli("publish")
    pub = C.PUBLISH
    assert (pub / "MANIFEST.json").exists()
    assert "author" not in pd.read_json(pub / "sample.jsonl", lines=True).columns
    assert not (pub / "hn_items_cache.jsonl").exists()
