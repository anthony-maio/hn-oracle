"""Blind terminal labeler. Never reads any Jev or baseline output, and never shows the stratum.

Pass a: is_prediction for every row in the sheet.
Pass b: stage B fields for every row labeled is_prediction = 1.
Saves after every answer; re-running resumes where you stopped.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd

from . import config as C
from .sample import load_sample

RULES = C.ROOT / "LABELING_RULES.md"
BOOL_FIELDS = ["is_checkable", "is_sarcastic", "is_conditional"]
CHOICE_FIELDS = ["direction", "domain", "horizon", "stance_vs_thread"]
SCORE_FIELDS = ["certainty", "specificity"]


class Quit(Exception):
    pass


def load_sheet(sheet: Path, out: Path, labeler: str) -> pd.DataFrame:
    if out.exists():
        df = pd.read_csv(out, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
        df["labeler"] = labeler
    for c in C.LABEL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[C.LABEL_COLUMNS]


def save(df: pd.DataFrame, out: Path) -> None:
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)


def wrap(s: str, indent: str = "  ") -> str:
    paras = (s or "").split("\n")
    return "\n".join(textwrap.fill(p, 100, initial_indent=indent, subsequent_indent=indent) if p.strip() else ""
                     for p in paras)


def show(row, i: int, n: int) -> None:
    print("\n" + "=" * 104)
    print(f"[{i}/{n}] id {row.name}   posted {row['year']}")
    print(f"STORY: {row['story_title'] or '(unknown)'}")
    if row["parent_excerpt"]:
        print("PARENT (excerpt):\n" + wrap(row["parent_excerpt"], "  | "))
    else:
        print("PARENT: (top-level comment)")
    print("COMMENT:\n" + wrap(row["text"]))
    print("-" * 104)


def ask(prompt: str, valid: dict[str, str]) -> str:
    while True:
        a = input(prompt).strip().lower()
        if a == "q":
            raise Quit
        if a == "r" and RULES.exists():
            print(RULES.read_text(encoding="utf-8"))
            continue
        if a in valid:
            return valid[a]
        print("  ? " + ", ".join(valid) + ", r = rules, q = quit")


def ask_choice(field: str, opts: list[str], descs: list[str]) -> str:
    print(f"  {field}:")
    for k, (o, d) in enumerate(zip(opts, descs), 1):
        print(f"    {k}. {o}: {d}")
    return ask(f"  {field} [1-{len(opts)}]: ", {str(k): o for k, o in enumerate(opts, 1)})


def pass_a(df: pd.DataFrame, sample: pd.DataFrame, out: Path) -> None:
    todo = [i for i in df.index if df.at[i, "is_prediction"] == ""]
    print(f"pass a: {len(df) - len(todo)}/{len(df)} done. y = prediction, n = not, s = skip, "
          "u = undo last, r = rules, q = quit")
    history: list[int] = []
    k = 0
    while k < len(todo):
        i = todo[k]
        show(sample.loc[int(df.at[i, "id"])], int((df.is_prediction != "").sum()) + 1, len(df))
        a = ask("is_prediction? [y/n/s/u]: ", {"y": "1", "n": "0", "s": "skip", "u": "undo"})
        if a == "undo":
            if history:
                last = history.pop()
                df.at[last, "is_prediction"] = ""
                save(df, out)
                todo.insert(k, last)
            continue
        if a != "skip":
            df.at[i, "is_prediction"] = a
            note = input("  note (enter to skip): ").strip()
            if note:
                df.at[i, "notes"] = note
            save(df, out)
            history.append(i)
        k += 1


def pass_b(df: pd.DataFrame, sample: pd.DataFrame, out: Path) -> None:
    sb = C.QUESTIONS["stage_b"]
    todo = [i for i in df.index if df.at[i, "is_prediction"] == "1"
            and any(df.at[i, f] == "" for f in C.LABEL_FIELDS_B)]
    print(f"pass b: {len(todo)} positives still need stage B fields. q quits (the current row is not saved)")
    for n, i in enumerate(todo, 1):
        show(sample.loc[int(df.at[i, "id"])], n, len(todo))
        vals = {}
        for f in BOOL_FIELDS:
            vals[f] = ask(f"  {f}? [y/n]: ", {"y": "1", "n": "0"})
        for f in CHOICE_FIELDS:
            crit = sb[f]["criteria"]
            vals[f] = ask_choice(f, list(crit), list(crit.values()))
        for f in SCORE_FIELDS:
            levels = sb[f]["criteria"]
            vals[f] = ask_choice(f, [str(k) for k in range(len(levels))], levels)
        vals["subject_text"] = input("  subject (free text, for building the roster; enter to skip): ").strip()
        note = input("  note (enter to skip): ").strip()
        for f, v in vals.items():
            df.at[i, f] = v
        if note:
            df.at[i, "notes"] = (df.at[i, "notes"] + " | " + note).strip(" |")
        save(df, out)


def cmd_label(args) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sheet = Path(args.sheet) if args.sheet else C.DATA / "labels_blank.csv"
    out = Path(args.out) if args.out else C.DATA / f"labels_{args.labeler}.csv"
    df = load_sheet(sheet, out, args.labeler)
    sample = load_sample().set_index("id")
    try:
        (pass_a if args.pass_ == "a" else pass_b)(df, sample, out)
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nstopped; progress saved")
    save(df, out)
    done = (df.is_prediction != "").sum()
    print(f"{out}: is_prediction {done}/{len(df)}, positives {(df.is_prediction == '1').sum()}")
