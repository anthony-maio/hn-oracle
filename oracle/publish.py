"""Copy publishable artifacts into publish/ with HN usernames removed, plus a checksum manifest."""
from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd

from . import config as C
from .sample import load_sample

DROP_KEYS = {"author", "by"}
JSONL_GLOBS = ["raw_*.jsonl", "baseline_*.jsonl", "stage_c.jsonl"]
COPY_AS_IS = ["report.json", "report.md", "eligible.json", "sample_meta.json", "disagreements.csv",
              "stage_c_review.csv"]


def scrub(obj):
    """Drop author fields at any depth. States sent to Jev never contain usernames; this is a backstop."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def has_author_field(obj) -> bool:
    if isinstance(obj, dict):
        return bool(DROP_KEYS & obj.keys()) or any(has_author_field(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_author_field(v) for v in obj)
    return False


def cmd_publish(args) -> None:
    out = C.PUBLISH
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    sample = load_sample()

    sample.drop(columns=["author"]).to_json(out / "sample.jsonl", orient="records", lines=True, force_ascii=False)
    for name in ("questions.json", "PILOT_PLAN.md", "LABELING_RULES.md"):
        if (C.ROOT / name).exists():
            shutil.copy(C.ROOT / name, out / name)
    for pattern in JSONL_GLOBS:
        for path in sorted(C.DATA.glob(pattern)):
            if path.name.endswith(".errors.jsonl"):
                continue
            with path.open(encoding="utf-8") as src, (out / path.name).open("w", encoding="utf-8") as dst:
                for line in src:
                    if line.strip():
                        dst.write(json.dumps(scrub(json.loads(line)), ensure_ascii=False) + "\n")
    for path in sorted(C.DATA.glob("labels_*.csv")):
        if path.name.startswith("labels_blank"):
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.drop(columns=[c for c in df.columns if c in DROP_KEYS]).to_csv(out / path.name, index=False)
    for name in COPY_AS_IS:
        if (C.DATA / name).exists():
            shutil.copy(C.DATA / name, out / name)

    # Final check: no published record or table still has an author field.
    leaks = []
    for f in out.iterdir():
        if f.suffix == ".jsonl":
            leaks += [f.name for line in f.open(encoding="utf-8") if line.strip() and has_author_field(json.loads(line))][:1]
        elif f.suffix == ".json":
            leaks += [f.name] if has_author_field(json.loads(f.read_text(encoding="utf-8"))) else []
        elif f.suffix == ".csv":
            leaks += [f.name] if DROP_KEYS & set(pd.read_csv(f, nrows=0).columns) else []
    if leaks:
        shutil.rmtree(out)
        raise SystemExit(f"username leak check failed, nothing published: {leaks[:5]}")

    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"published {len(manifest)} files to {out}")
