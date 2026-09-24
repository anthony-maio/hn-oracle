# Labeling rules (fixed before labeling starts)

Blind: labels are finished before anyone looks at a Jev output. The labeler never sees Jev, baseline or
panel outputs, and does not see the stratum. The three-model labeling panel (plan amendment A1) gets
this same file as its instructions.

## is_prediction

Yes if the comment claims something **will or won't happen later**: a technology, company, product,
market, law, or social trend.

| Case | Label |
|---|---|
| Repeating someone else's prediction without endorsing it | no |
| "X is dying" / "X is dead" | no, unless a future state is claimed ("X will be gone by 2015") |
| Advice: "You should switch to X" | no |
| Questions: "Will X survive?" | no |
| Rhetorical forecast inside a joke | **yes**, and `is_sarcastic` = yes |
| Conditional: "If Apple does X, Y will happen" | **yes**, and `is_conditional` = yes |
| Present-tense claims about now | no |

## Stage B fields (positives only)

- `is_checkable`: could a reasonable person decide years later whether it came true?
- `is_sarcastic`: meant as a joke or sarcastically rather than sincerely
- `is_conditional`: depends on an explicit "if X then Y"
- `direction`: success / failure / change / no_change
- `domain`: the 10 options in `questions.json`
- `horizon`: under_1y / 1_to_5y / 5_to_10y / over_10y / unstated. Label what the comment says; never compute a year.
- `stance_vs_thread`: agrees / contrarian / unclear, relative to the parent and story shown
- `certainty` 0-3: speculative, leaning, confident, emphatic
- `specificity` 0-3: vague direction, named subject + general outcome, + rough time frame, + measurable outcome with explicit time frame or number
- `subject_text` (free text, optional): the main company/product/technology, used to build out the subject roster

## Disagreements

Settled only after kappa is computed. `score` writes `data/disagreements.csv`; record the resolution
and reason there and in the `notes` column of the primary labels.
