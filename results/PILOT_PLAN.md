# HN Oracle: 1,000-Comment Jev Pilot

Version: pilot-0.1, September 23, 2026
Model: `jev-1.13.0` (pinned; never `jev-latest` during evaluation)

## Question the pilot answers

Can Jev find predictions in old Hacker News comments, and produce calibrated probabilities, at a cost and speed that make a full-archive run worth doing?

The pilot answers this with a go/no-go decision against gates written down before any Jev call. It also produces the first public artifacts: the question schema, raw responses, labels, and a calibration report.

## Correction to the first cost estimate

The earlier estimate of "about $500, about 14 hours" left out two things.

1. Question text counts as billed input tokens. TypeSafe's own one-sentence Choice example costs 328 input tokens. A full 10-question stage B schema will probably cost 1,000 to 1,500 tokens per comment.
2. The binding limit is the request count, not tokens. At 1,200 requests per minute (20 per second), sending one comment per request across 41.3M comments takes about 24 days.

Two design changes follow from that, and the pilot tests both:

- **Two-stage cascade.** Stage A asks one short Noul ("is this a prediction?") of every comment. Stage B's full schema runs only on comments that pass the gate, probably 5 to 15% of the total.
- **Packing.** Stage A puts N comments into one state (IDs `c01` to `cNN`) with one Noul per ID. At N = 16 the request limit drops to about 36 hours. Packing could hurt quality: an independent benchmark found that 40-row batches failed a ranking gate that one row per request passed. So the packed and single-comment runs go head-to-head in the pilot.

The full run also only needs comments with at least 5 years of hindsight, so posted between 2006 and 2020. The pilot counts that subset exactly with DuckDB. The true full-run volume is smaller than 41.3M.

## Files

- `questions.json`: the Jev question schema (stage A single, stage A packed template, stage B, optional subject roster)
- `pilot.py`: `sample`, `run`, and `score` subcommands
- `data/`: sample, blank label sheet, raw responses (JSONL, request plus response plus latency), reports

## Sample design

1,000 comments posted 2009 to 2019, taken from the Hugging Face HN mirror (`nikhilambhure00/hacker-news`, monthly Parquet files). Filters: not dead, not deleted, 80 to 4,000 characters of text.

**Uniform stratum: 600 comments.** A plain random sample. This is the only stratum used for base-rate and prevalence estimates, and for testing the regex baseline, because nothing about how it was drawn favors predictions.

**Enriched stratum: 400 comments.** Drawn from a 20,000-comment pool, keeping comments that match a prediction-hint regex ("will", "going to", "in N years", "by 20XX", "mark my words", "eventually", and so on). Predictions are probably only 3 to 8% of random comments, so this stratum makes sure there are enough positives to test stage B and the high end of the probability scale.

**Context per comment:** the story title and a 600-character excerpt of the parent comment, fetched by walking up the official HN Firebase API. The posting year goes into the state as context only. No question asks Jev to reason about dates, because TypeSafe documents dates and numbers as known weak spots.

**Privacy:** usernames are kept locally for joins and removed from every published artifact. No per-person scoring happens in the pilot.

## Jev question schema

Everything is in `questions.json`. Summary:

### Stage A: gate (every comment)

- `is_prediction` (Noul, with true/false criteria): does the comment claim something will or won't happen later? Present-tense claims ("X is dead"), advice, questions, and jokes count as no.

The packed variant uses the same wording, with the comment ID in the instruction and criteria. It is kept short on purpose, because this text is billed once per comment.

### Stage B: detail (only comments that pass the gate)

- `is_checkable` (Noul): could someone decide years later whether it came true?
- `is_sarcastic` (Noul)
- `is_conditional` (Noul): does it depend on an explicit "if X then Y"?
- `direction` (Choice): success / failure / change / no_change
- `domain` (Choice, 10 options): language_or_framework, company_fate, product_adoption, ai_ml, crypto_finance, policy_law, infrastructure, society_work, science_space, other
- `horizon` (Choice): under_1y / 1_to_5y / 5_to_10y / over_10y / unstated
- `stance_vs_thread` (Choice): agrees / contrarian / unclear, relative to the parent and story shown in the state
- `certainty` (Score, 4 levels): openly speculative up to emphatic
- `specificity` (Score, 4 levels): vague direction up to measurable outcome with a time frame

**Done in code, never by Jev:**

- converting a horizon into a resolution year (posted year plus the horizon bucket)
- deciding whether enough time has passed to grade a prediction
- any arithmetic or date comparison

### Optional: subject roster

Jev can't pull out free-text entities, so the subject of a prediction is a Choice over a curated roster (up to 255 options). The pilot ships 15 seed options plus `other`. The share of positives landing in `other` is a pilot metric: if it's above 40%, the roster has to be built out from the labels before the full run.

### Wording rules

- Option names are stable enum values.
- Rewording any instruction or criterion bumps `schema_version`, and every threshold has to be re-fit.
- Scores from rubrics with different numbers of levels are never compared directly.
- A `certainty` score of 0.5 is never described as "50% likely".

## Labeling protocol

1. **Blind first.** Labels are finished before anyone looks at a Jev output. `labels_blank.csv` is shuffled and hides everything except `id` and `stratum`.
2. **Primary labeler:** Anthony labels `is_prediction` for all 1,000 (estimate: 4 to 6 hours), then the stage B fields for every positive.
3. **Second labeler:** a person labels a random 200 for `is_prediction` and all positives in that 200 for stage B. Cohen's kappa between the two people sets the ceiling. Jev isn't expected to agree with the labels more than two humans agree with each other.
4. **Edge-case rules, written before labeling starts:**
   - Repeating someone else's prediction without endorsing it: no.
   - "X is dying": no, unless a future state is claimed.
   - "You should switch to X": no.
   - Rhetorical forecast in a joke: yes for `is_prediction`, and `is_sarcastic` = true.
   - A conditional prediction still counts as yes.
5. **Disagreements:** settled after kappa is computed, and logged in `notes`.

## Runs

All runs log the full request, response, `usage.input_tokens`, and wall-clock latency to `data/raw_*.jsonl`.

1. `run --mode a_single`: stage A, one comment per request, all 1,000.
2. `run --mode a_packed --pack 8`: stage A, 8 per request, all 1,000.
3. `run --mode a_packed --pack 16`: stage A, 16 per request, all 1,000.
4. `run --mode a_nonce`: robustness check on 100 comments. Each gets 3 repeats with a random nonce added to the state; the answers should barely move. This is TypeSafe's own robustness method, described on Latent Space.
5. `run --mode b --gate <t>`: stage B on comments above the gate threshold picked from run 1.
6. **Baselines**, sent the same state and the same yes/no question:
   - The prediction-hint regex, on the uniform stratum only (the enriched stratum was selected with it, so testing there would be circular).
   - One cheap LLM and one frontier LLM, using OpenRouter's structured output with a stated probability.
7. **Stage C spot check:** a frontier LLM with web search grades the top 30 checkable predictions for hindsight. Anthony checks every verdict against sources. This is for sample demo pages only, not for scoring Jev.

## Metrics

All stage A metrics are reported separately for the uniform stratum, the enriched stratum, and both combined. Calibration claims come from the uniform stratum only.

- **Brier score**, next to a baseline that always predicts the base rate
- **ECE** (10 equal-width bins) and a reliability table
- **Best F1** and its threshold, plus recall and precision at the chosen gate
- **Prevalence gap:** the average Jev probability on the uniform stratum minus the labeled rate, with a 95% bootstrap interval. This tests the core idea that calibrated probabilities can be added up to measure something.
- **Recalibration:** fit temperature or isotonic scaling on half the uniform stratum and report on the other half. If recalibration is needed, that's a finding, not a failure.
- **Packing penalty:** change in F1 and Brier between single, packed-8, and packed-16
- **Nonce robustness:** median and p95 of |change in p| across repeats
- **Stage B:** per-field accuracy (Choice) and mean absolute error in level units (Score) against labels, shown as a ratio of human-human agreement
- **Cost and speed:** input tokens per comment, p50 and p95 latency, and the full run projected in dollars and hours under both the token and request rate limits

`pilot.py score` produces everything above for stage A. The stage B and baseline comparisons are short notebook cells that reuse the same functions.

## Go/no-go gates (pre-registered)

Written before any Jev call. Every gate is reported pass or fail, including the ones that fail.

- **G1, recall:** stage A recall of at least 0.85 on the uniform stratum at the chosen gate threshold. Missed predictions are gone for good; false positives are cheap, because stage B filters them out.
- **G2, calibration:** ECE of 0.08 or less on the uniform stratum, either raw or after recalibration on a held-out half.
- **G3, prevalence:** after recalibration, the prevalence gap is within ±2 percentage points and the 95% interval includes zero.
- **G4, packing:** packed-N F1 within 0.03 of single, and Brier no more than 10% worse. The largest N that passes goes into the full-run design. If N = 8 fails, try N = 4. If that fails too, a full run needs a higher rate limit from TypeSafe.
- **G5, baselines:** Jev F1 at least 0.15 above the regex, and no more than 0.05 below the frontier LLM.
- **G6, robustness:** median nonce |change in p| of 0.05 or less.
- **G7, economics:** projected full run (stage A for every eligible comment plus stage B for gated ones) costs less than $3,000 and finishes in under 7 days at current rate limits.
- **Stage B:** each field reaches at least 80% of human-human agreement. Any field that fails is dropped from the public product or marked experimental.

**Decision rule:**

- **Go** if G1, G2, G4, and G7 pass.
- G3, G5, and G6 decide how the result is framed publicly, not whether the project continues. A failed prevalence test becomes an honest section in the write-up, and the trend charts ship as labeled counts instead of estimates.

## Budget and timeline

- **Jev tokens:** well under $1 for every pilot run combined (about 1M to 3M input tokens).
- **LLM baselines and stage C:** roughly $10 to $30.
- **Main cost:** labeling time, about 8 hours total (about 2.5 hours after amendment A1).

Schedule:

- **Day 1:** confirm dataset columns and the API auth header, run `sample`, write the edge-case rules, start labeling.
- **Day 2:** finish blind labels, second labeler, compute kappa.
- **Day 3:** runs 1 to 7.
- **Day 4:** score, go/no-go, write-up.

## Items to verify before the first call

- The dataset's column names and type encoding (`pilot.py` assumes `id`, `type`, `by`, `time`, `text`, `parent`, `dead`, `deleted`, with comment = 2). Check them against the dataset card.
- The auth header. The public docs show the endpoint and body but not the header. Check it against the official SDK (`typesafe_sdk.TypeSafeClient`) or switch the runner to the SDK.
- Rate limits for your account tier. The public figures are 1,200 requests per minute and 250,000 tokens per second.
- Whether packed states with 16 comments fit in the 32K window for state plus longest question. They should, easily, at 16 × about 400 tokens.

## Pilot write-up (evidence-first)

Working title: "I asked a decision model to find every prediction on Hacker News. Here's how calibrated it was."

1. The Karpathy baseline and why the full archive was never affordable
2. The cascade and packing design, and why the request limit mattered more than token cost
3. Reliability tables, the prevalence test, and the packing penalty, with every gate reported pass or fail
4. Failure cases: 10 misses and 10 false positives with the actual comment text
5. Full-run projection and what launches next

Everything gets published: `questions.json`, raw responses, labels (usernames removed), and the report JSON.

## Amendment A1: labeling panel (September 23, 2026, before any labeling)

Recorded after the sample was drawn and a 20-comment Jev smoke test ran, and before any label was written or any labeled Jev result existed. The gates, thresholds, and metrics above are unchanged. What changes is who writes which labels, because about 8 hours of solo labeling was the bottleneck.

**Human labels, blind, remain the answer key for:**

- `is_prediction` on all 600 uniform comments. G1, G2 and G3 (recall, calibration, prevalence) are computed only on these.
- A blind audit of 100 random enriched comments the panel labeled unanimously. This measures the panel's error rate, which the write-up reports with a 95% Wilson interval.
- Every enriched comment the panel does not agree on unanimously.

These three sets are shuffled into one sheet, so the labeler can't tell uniform, audit, and split comments apart. The labeling tool still never shows model output.

**A three-model panel labels the rest:** `anthropic/claude-haiku-4.5`, `deepseek/deepseek-v4-flash`, `openai/gpt-5.6-luna`, three model families at temperature 0 (where supported), given the same labeling rules as the human (`LABELING_RULES.md`).

- `is_prediction` on enriched comments: the panel's label counts only when all three agree.
- Stage B fields on every positive: strict majority for yes/no and choice fields (no majority leaves the field unlabeled), median for scores.

**Consequences for scoring:**

- No panel model is a baseline. The cheap-LLM baseline moves from `deepseek/deepseek-v4-flash` to `openai/gpt-5-nano`.
- Baselines (G5) are scored only on human-labeled comments, so no model is graded against labels an LLM wrote.
- G4 (packing) still uses every labeled comment; single and packed runs are compared on the same labels.
- The stage B gate's reference ceiling becomes the mean pairwise agreement between panel models, since the stage B labels come from the panel. If a second human labeler is added, human-human agreement replaces it.
- The report states each label's provenance (`human`, `human_audit`, `human_adjudicated`, `panel`) and the audit error rate next to every result that depends on panel labels.

Human labeling drops from about 8 hours to about 2.5. Panel API cost is about $2.

**Observed when the panel ran, before any human label:** the three models agreed unanimously on 59.0% of the 400 enriched comments. They said "prediction" at very different rates (Haiku 26.8%, DeepSeek 34.2%, GPT-5.6 Luna 55.0%; pairwise agreement 68-80%), with Luna the odd one out in 82 of 164 splits. The unanimity rule stays as registered, so all 164 splits go to the human sheet, which becomes 864 comments (600 uniform, 100 audit, 164 splits). Model disagreement at this level says the task is ambiguous on regex-hinted comments, which is a reason to add a second human labeler if one is available.

**Human labeling, as done:** all 864 sheet comments were labeled blind in the terminal tool, then the labeler re-read the file in a text editor and added short `review:` notes to 19 of them. No model output was consulted during labeling or review.
