# Runbook: the 1,000-comment pilot

The day-by-day commands for [PILOT_PLAN.md](../PILOT_PLAN.md). One CLI, `pilot.py`, covers every step from sampling to the go/no-go report.

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# keys go in .env (gitignored) or the environment
TYPESAFE_API_KEY=...        # Jev
OPENROUTER_API_KEY=...      # LLM baselines and stage C only
```

## Run order

**Day 1: verify, count, sample**

```
python pilot.py preflight                      # dataset columns/types vs assumptions (no API calls)
python pilot.py preflight --skip-dataset --call  # Jev auth, answer shape, packed-16 worst case vs 32K window
python pilot.py count                          # exact 2006-2020 eligible comments -> data/eligible.json
python pilot.py sample                         # 600 uniform + 400 enriched, thread context, label sheets
```

`sample` and `count` read Parquet straight from Hugging Face. They only touch the year folders they need. If you download the years first (`hf download nikhilambhure00/hacker-news --repo-type dataset --include "data/20*/*"`), pass `--parquet-glob path/to/data/*/*.parquet`.

**Day 2: labels (plan amendment A1)**

```
python pilot.py panel a                        # 3-model panel labels is_prediction on the enriched 400 (~$0.50)
python pilot.py panel sheet                    # your blind sheet: uniform 600 + audit 100 + panel splits, shuffled
python pilot.py label --labeler anthony --sheet data/labels_blank_human.csv    # ~2.5 hours
python pilot.py panel b                        # panel labels stage B fields on every positive (~$1.50)
python pilot.py panel merge                    # data/labels_final.csv + data/panel_audit.json
```

Score everything with `--labels data/labels_final.csv`. It carries a `source` column (`human`, `human_audit`, `human_adjudicated`, `panel`), and `score` reports provenance and the audit error rate, and grades baselines on human labels only. The labeler never reads Jev or panel output and never shows the stratum. Press `r` at any prompt to see [LABELING_RULES.md](../LABELING_RULES.md), which the panel gets too. Progress saves after every answer.

A second human labeler is optional: `python pilot.py label --labeler second --sheet data/labels_blank_second.csv`, then `score --second-labels data/labels_second.csv`, and human-human agreement becomes the stage B reference.

**Day 3: runs 1-7**

```
python pilot.py run --mode a_single                 # run 1
python pilot.py run --mode a_packed --pack 8        # run 2
python pilot.py run --mode a_packed --pack 16       # run 3
python pilot.py run --mode a_nonce                  # run 4: 100 comments x 3 nonce repeats
python pilot.py score --labels data/labels_final.csv --single data/raw_a_single_v1.jsonl   # read off the gate
python pilot.py run --mode b --gate <t> --stage-a-file data/raw_a_single_v1.jsonl \
    --also-labeled-positives data/labels_final.csv --with-subject                        # run 5
python pilot.py baseline --kind regex                                                      # run 6
python pilot.py baseline --kind llm --model <cheap-openrouter-slug> --name cheap
python pilot.py baseline --kind llm --model <frontier-openrouter-slug> --name frontier
python pilot.py stage-c --stage-b-file data/raw_b_v1.jsonl --labels data/labels_final.csv \
    --model <frontier-openrouter-slug>                                                     # run 7
```

Every run appends to `data/raw_<mode>_<tag>.jsonl`, one line per request, holding the request, response, `usage`, latency, and attempt count. If a run is interrupted, rerun the same command: finished requests are skipped and failed ones are retried. Add `--dry-run` to print the first request body, or `--limit 20` for a smoke test with a different `--tag`.

**Day 4: score, decide, publish**

```
python pilot.py score --labels data/labels_final.csv \
    --single data/raw_a_single_v1.jsonl \
    --packed data/raw_a_packed8_v1.jsonl data/raw_a_packed16_v1.jsonl \
    --nonce data/raw_a_nonce_v1.jsonl --stage-b data/raw_b_v1.jsonl \
    --baseline-regex data/baseline_regex.jsonl \
    --baseline-cheap data/baseline_cheap.jsonl --baseline-frontier data/baseline_frontier.jsonl
python pilot.py publish
```

`score` writes `data/report.json` and `data/report.md`. Together they hold the gate table, the decision, per-stratum metrics, reliability tables, recalibration, packing penalty, nonce robustness, baselines, human agreement, stage B fields, cost and speed with a full-run projection, and 10 misses and 10 false positives with their text. It also writes `data/disagreements.csv` for adjudication. `publish` copies the public artifacts to `results/` (committed) without HN usernames, adds a SHA-256 manifest, and refuses to publish if any author field survives.

## Choices the plan left open

- **Sampling** uses `ORDER BY hash(id, seed)` over the filtered frame, not reservoir sampling. It gives the same draw on every machine and thread count. The kit's `USING SAMPLE` runs *before* `WHERE` in DuckDB, so it returned far fewer than 600 rows.
- **Time zone.** `time` is `TIMESTAMP WITH TIME ZONE`. The DuckDB session is pinned to UTC so that year boundaries don't shift with the machine's zone.
- **Gate threshold.** By default, the gate is the highest threshold with recall ≥ 0.90 on all labeled comments, which leaves margin over G1's 0.85. G1 is then measured on the uniform stratum. The report also includes a two-fold held-out recall, so you can see how optimistic that in-sample choice is. To fix the gate yourself, pass `--gate`.
- **Recalibration.** Temperature and isotonic scaling use a two-fold, label-stratified cross-fit on the uniform stratum: fit on one half, report on the other, then swap. G2 takes the best of raw, temperature and isotonic. G3 uses whichever recalibration had the lower ECE.
- **G4** compares best F1 and Brier on all labeled comments. The report includes a paired bootstrap CI for ΔF1.
- **G5** compares best F1 on the same comments: the regex on the uniform stratum only, and the frontier LLM on all labeled comments.
- **G7** projects stage A for every eligible comment (`data/eligible.json`, 80 to 4,000 characters, matching the sample frame) plus stage B at the uniform pass rate. Time is the larger of the request limit and the token limit. The design uses the largest N that passes G4.
- **Stage B** accuracy is measured on labeled positives. `--also-labeled-positives` makes sure predictions below the gate get stage B answers too. Score fields are compared in level units via the answer's `legend`. The report records which mapping rule was used.
- **Labels** have an extra free-text `subject_text` column, used to build out the subject roster if more than 40% of positives land in `other`.

## Baseline models and what they cost

`python pilot.py models` prints the live OpenRouter free list and the cheapest paid models with structured output. Presets live in `oracle/config.py`:

| Preset | Model | Rough cost for 1,000 comments |
|---|---|---|
| `--name cheap` | `openai/gpt-5-nano` | $0.03 |
| `--name frontier` | `anthropic/claude-sonnet-5` | $1.60 (measured) |
| `--name free_cheap` | `liquid/lfm-2.5-2.6b:free` | $0 |
| `--name free_frontier` | `nvidia/nemotron-3-ultra-550b-a55b:free` | $0 |
| `--sweep cheap` | 7 paid models under ~$0.35 each | ~$1 |
| `--sweep free` | 6 free models with native structured output | $0 |

Free models share 20 requests/min and 50 requests/day (1,000/day once you have bought $10 of OpenRouter credits, ever). With `--pack 16`, 1,000 comments is 63 requests. If a run hits the daily cap, it stops cleanly; rerun the same command the next day. For reasoning models, `--reasoning-effort low` trades some accuracy for cost and speed.

Stage C uses OpenRouter's web plugin. For models without native search that is Exa at about $0.007 per request, so the 30 predictions cost about $0.21 plus model tokens.

## Before the first call

- `PRICE_PER_M_INPUT` is $0.042 per million input tokens. The rate limits in [oracle/config.py](../oracle/config.py) are the public tier figures; `score --price/--rpm/--tps` override them.
- The auth header is confirmed as `Authorization: Bearer` by docs.typesafe.ai. `preflight --call` confirms it for your key.
- The model is pinned to `jev-1.13.0` in `questions.json`. The code refuses to run against `jev-latest`. If you reword any question, bump `schema_version`.

## Layout

```
pilot.py              CLI
oracle/config.py      paths, pinned model, price, rate limits
oracle/sample.py      DuckDB sampling, eligibility count, schema check, HN Firebase context
oracle/labeling.py    blind terminal labeler
oracle/panel.py       three-model labeling panel, audit, merge (amendment A1)
oracle/jev.py         Jev client, rate limiter, retries (429/529/5xx), resumable JSONL runner
oracle/runs.py        run modes and raw-file loaders
oracle/llm.py         regex and OpenRouter baselines (free and paid presets), model catalog, stage C
oracle/metrics.py     Brier, ECE, F1, bootstrap, temperature/isotonic, kappa
oracle/score.py       report, gates, projection
oracle/publish.py     public artifact export
tests/                unit tests and an end-to-end run on a synthetic archive with mocked APIs
```
