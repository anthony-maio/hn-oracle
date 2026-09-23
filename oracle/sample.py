"""Stratified 1,000-comment sample, exact full-run eligibility counts, and schema checks.

Dataset: nikhilambhure00/hacker-news (monthly Parquet). Per the dataset card:
  type int8 (2 = comment), "by" string, time TIMESTAMP (UTC), dead/deleted uint8 (0/1).
"""
from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import config as C
from .text import clean_html, regex_hint

EXPECTED_COLUMNS = {
    "id": "UINTEGER", "type": "TINYINT", "by": "VARCHAR", "time": "TIMESTAMP",
    "text": "VARCHAR", "parent": "UINTEGER", "dead": "UTINYINT", "deleted": "UTINYINT",
    "title": "VARCHAR",
}


def connect():
    import duckdb

    con = duckdb.connect()
    # `time` is TIMESTAMP WITH TIME ZONE; year boundaries and year() must be UTC, not the machine's zone.
    con.execute("SET TimeZone = 'UTC'")
    return con


def year_sources(src: str, years: tuple[int, int]) -> list[str]:
    """Only read the year folders we need when the glob follows the dataset's data/YYYY/ layout."""
    if "/data/*/" not in src:
        return [src]
    srcs = [src.replace("/data/*/", f"/data/{y}/") for y in range(years[0], years[1] + 1)]
    if "://" not in src:  # local mirror: skip year folders you did not download
        import glob

        srcs = [s for s in srcs if glob.glob(s)]
        if not srcs:
            raise SystemExit(f"no Parquet files under {src} for years {years[0]}-{years[1]}")
    return srcs


def _read(srcs: list[str]) -> str:
    return "read_parquet([" + ", ".join(f"'{s}'" for s in srcs) + "], union_by_name = true)"


def comment_filter(years: tuple[int, int], text_len: tuple[int, int] | None) -> str:
    # Range predicates on `time` (not year(time)) so Parquet row-group stats can prune.
    where = f"""
        type = 2
        AND coalesce(dead, 0) = 0
        AND coalesce(deleted, 0) = 0
        AND time >= TIMESTAMP '{years[0]}-01-01'
        AND time <  TIMESTAMP '{years[1] + 1}-01-01'
    """
    if text_len:
        where += f" AND length(text) BETWEEN {text_len[0]} AND {text_len[1]}"
    return where


def check_schema(src: str) -> dict:
    con = connect()
    one = year_sources(src, (2010, 2010))[0]
    rows = con.execute(f"DESCRIBE SELECT * FROM {_read([one])}").fetchall()
    actual = {r[0]: r[1] for r in rows}
    problems = []
    for col, typ in EXPECTED_COLUMNS.items():
        if col not in actual:
            problems.append(f"missing column {col}")
        elif not actual[col].startswith(typ.split("(")[0]):
            problems.append(f"{col}: expected {typ}, got {actual[col]}")
    type_counts = con.execute(f"SELECT type, count(*) FROM {_read([one])} GROUP BY 1 ORDER BY 1").fetchall()
    return {"source": one, "columns": actual, "problems": problems, "type_counts": type_counts}


def count_eligible(src: str) -> dict:
    """Exact count of comments with at least 5 years of hindsight, per year."""
    con = connect()
    y0, y1 = C.FULL_RUN_YEARS
    lo, hi = C.TEXT_LEN
    rows = con.execute(f"""
        SELECT year(time) AS y,
               count(*) AS n_comments,
               count(*) FILTER (WHERE length(text) BETWEEN {lo} AND {hi}) AS n_len_filtered
        FROM {_read(year_sources(src, C.FULL_RUN_YEARS))}
        WHERE {comment_filter(C.FULL_RUN_YEARS, None)}
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    by_year = [{"year": int(y), "n_comments": int(a), "n_len_filtered": int(b)} for y, a, b in rows]
    out = {
        "years": [y0, y1],
        "text_len": [lo, hi],
        "n_comments": sum(r["n_comments"] for r in by_year),
        "n_len_filtered": sum(r["n_len_filtered"] for r in by_year),
        "by_year": by_year,
        "note": "n_len_filtered matches the pilot's sample frame and is the default basis for projections.",
    }
    C.ensure_data()
    (C.DATA / "eligible.json").write_text(json.dumps(out, indent=2))
    return out


def draw_sample(src: str, seed_uniform: int = 42, seed_pool: int = 7) -> pd.DataFrame:
    """Deterministic draws by hash order: uniform over the filtered frame, stable across machines and threads.

    (DuckDB's USING SAMPLE applies before WHERE, so it cannot be used on the filtered frame directly.)
    """
    con = connect()
    base = f"""
        SELECT id, "by" AS author, time::TIMESTAMP AS time, text, parent  -- UTC (session zone), naive
        FROM {_read(year_sources(src, C.SAMPLE_YEARS))}
        WHERE {comment_filter(C.SAMPLE_YEARS, C.TEXT_LEN)}
    """
    uniform = con.execute(f"{base} ORDER BY hash(id, {seed_uniform}) LIMIT {C.N_UNIFORM}").df()
    pool = con.execute(f"{base} ORDER BY hash(id, {seed_pool}) LIMIT {C.ENRICHED_POOL}").df()
    return assemble_strata(uniform, pool, seed_pool)


def assemble_strata(uniform: pd.DataFrame, pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    pool = pool[~pool.id.isin(uniform.id)].copy()
    pool["clean"] = pool.text.map(clean_html)
    hits = pool[pool.clean.map(regex_hint).astype(bool)]
    if len(hits) < C.N_ENRICHED:
        raise RuntimeError(f"only {len(hits)} regex hits in the {len(pool)}-comment pool; need {C.N_ENRICHED}")
    enriched = hits.sample(C.N_ENRICHED, random_state=seed).drop(columns="clean")

    uniform = uniform.assign(stratum="uniform")
    enriched = enriched.assign(stratum="enriched")
    sample = pd.concat([uniform, enriched], ignore_index=True)
    sample["text"] = sample.text.map(clean_html)
    sample["time"] = pd.to_datetime(sample.time, utc=True)
    sample["year"] = sample.time.dt.year.astype(int)
    sample["time"] = sample.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    sample["id"] = sample.id.astype(int)
    sample["parent"] = sample.parent.astype(int)
    sample.attrs["pool_regex_hit_rate"] = len(hits) / max(len(pool), 1)
    return sample


# ---------------------------------------------------------------- thread context


class ItemCache:
    """Local cache of HN Firebase items so re-running `sample` does not re-walk every thread."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[int, dict] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                it = json.loads(line)
                self.items[int(it["id"])] = it
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def get(self, item_id: int, client) -> dict:
        if item_id in self.items:
            return self.items[item_id]
        for attempt in range(4):
            try:
                r = client.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=20)
                r.raise_for_status()
                raw = r.json() or {}
                break
            except Exception:
                if attempt == 3:
                    raise
        it = {k: raw.get(k) for k in ("type", "parent", "title", "text", "deleted", "dead")}
        it["id"] = item_id
        with self._lock:
            self.items[item_id] = it
            self._fh.write(json.dumps(it) + "\n")
            self._fh.flush()
        return it

    def close(self):
        self._fh.close()


def fetch_context(item_parent: int, cache: ItemCache, client) -> dict:
    """Walk up via the official HN Firebase API to the story title."""
    parent = cache.get(item_parent, client)
    excerpt = clean_html(parent.get("text"))[:600] if parent.get("type") == "comment" else ""
    node, hops = parent, 0
    while node.get("type") == "comment" and node.get("parent") and hops < 200:
        node, hops = cache.get(int(node["parent"]), client), hops + 1
    return {"story_id": node.get("id"), "story_title": clean_html(node.get("title")),
            "parent_excerpt": excerpt, "depth": hops + 1}


def add_context(sample: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    import httpx

    C.ensure_data()
    cache = ItemCache(C.DATA / "hn_items_cache.jsonl")
    try:
        with httpx.Client() as client, ThreadPoolExecutor(workers) as ex:
            ctx = list(ex.map(lambda p: fetch_context(int(p), cache, client), sample.parent))
    finally:
        cache.close()
    for k in ("story_id", "story_title", "parent_excerpt", "depth"):
        sample[k] = [c[k] for c in ctx]
    return sample


# ---------------------------------------------------------------- label sheets


def write_label_sheets(sample: pd.DataFrame, n_second: int = 200, seed: int = 1) -> None:
    sheet = sample[["id", "stratum"]].copy()
    for c in C.LABEL_COLUMNS[2:]:
        sheet[c] = ""
    sheet = sheet.sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet.to_csv(C.DATA / "labels_blank.csv", index=False)

    rng = random.Random(seed)
    second_ids = set(rng.sample(list(sheet.id), n_second))
    sheet[sheet.id.isin(second_ids)].to_csv(C.DATA / "labels_blank_second.csv", index=False)


def cmd_sample(args) -> None:
    C.ensure_data()
    out = C.DATA / "sample.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to redraw (this invalidates any labels)")
    sample = draw_sample(args.parquet_glob)
    sample = add_context(sample, workers=args.workers)
    sample.to_json(out, orient="records", lines=True, force_ascii=False)
    write_label_sheets(sample)
    meta = {
        "source": args.parquet_glob, "n": len(sample),
        "by_stratum": sample.stratum.value_counts().to_dict(),
        "years": list(C.SAMPLE_YEARS), "text_len": list(C.TEXT_LEN),
        "seeds": {"uniform_hash": 42, "pool_hash": 7, "enriched_draw": 7, "sheet_shuffle": 1},
        "pool_regex_hit_rate": sample.attrs.get("pool_regex_hit_rate"),
        "top_level_share": float((sample.parent_excerpt == "").mean()),
        "missing_story_title": int((sample.story_title == "").sum()),
    }
    (C.DATA / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"-> {out}\n-> {C.DATA / 'labels_blank.csv'}\n-> {C.DATA / 'labels_blank_second.csv'}")


def load_sample() -> pd.DataFrame:
    return pd.read_json(C.DATA / "sample.jsonl", lines=True, dtype={"id": int, "parent": int})
