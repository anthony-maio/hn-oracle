# HN Oracle pilot report

Model `jev-1.13.0`, schema `pilot-0.1`. Gate threshold 0.52 (all labeled comments: highest threshold with recall >= 0.9).

**Decision: GO** (Go if G1, G2, G4 and G7 pass. G3, G5, G6 shape the public framing.)

## Gates

| Gate | Status | Value | Threshold | Detail |
|---|---|---|---|---|
| G1_recall | pass | 0.920 | 0.850 | uniform recall at gate 0.52; held-out (gate chosen on the other half): 0.920 |
| G2_calibration | pass | 0.017 | 0.080 | raw ECE 0.1045; best recalibrated: isotonic |
| G3_prevalence | pass | 0.000 | 0.020 | isotonic-recalibrated gap, 95% CI [-0.015, +0.015] |
| G4_packing | pass | 8 | F1 drop <= 0.03 and Brier <= 1.10x single | largest passing N goes into the full-run design |
| G5_baselines | pass | vs regex (uniform) +0.289; vs frontier (all) +0.072 | regex +0.15, frontier -0.05 |  |
| G6_robustness | pass | 0.010 | 0.050 | p95 0.040 |
| G7_economics | pass | {'cost_usd': 406.82, 'days': 2.41} | {'cost_usd': 3000.0, 'days': 7.0} | design packed8, binding limit: requests |
| stageB_fields | pass | {'is_checkable': 1.027, 'is_sarcastic': 0.973, 'is_conditional': 0.976, 'direction': 1.062, 'domain': 1.178, 'horizon': 1.109, 'stance_vs_thread': 1.047, 'certainty': 1.068, 'specificity': 1.048} | 0.800 | failing fields are dropped or marked experimental; this does not block Go |

## Stage A (single comment per request)

| Stratum | n | pos | base rate | Brier | Brier (base rate) | ECE | best F1 @t | P / R / F1 at gate | pass rate |
|---|---|---|---|---|---|---|---|---|---|
| uniform | 600 | 50 | 0.083 | 0.051 | 0.076 | 0.104 | 0.765 @0.61 | 0.605 / 0.920 / 0.730 | 0.127 |
| enriched | 400 | 153 | 0.383 | 0.118 | 0.236 | 0.099 | 0.828 @0.5 | 0.758 / 0.902 / 0.824 | 0.455 |
| all | 1000 | 203 | 0.203 | 0.078 | 0.162 | 0.096 | 0.802 @0.5 | 0.713 / 0.906 / 0.798 | 0.258 |

Reliability, uniform stratum (raw):

| bin | n | mean p | frac true |
|---|---|---|---|
| 0.0-0.1 | 369 | 0.045 | 0.000 |
| 0.1-0.2 | 77 | 0.134 | 0.013 |
| 0.2-0.3 | 42 | 0.247 | 0.000 |
| 0.3-0.4 | 16 | 0.344 | 0.062 |
| 0.4-0.5 | 20 | 0.438 | 0.100 |
| 0.5-0.6 | 11 | 0.566 | 0.182 |
| 0.6-0.7 | 9 | 0.662 | 0.667 |
| 0.7-0.8 | 8 | 0.731 | 0.250 |
| 0.8-0.9 | 24 | 0.849 | 0.667 |
| 0.9-1.0 | 24 | 0.940 | 0.833 |

## Calibration and prevalence (uniform stratum)

two-fold, label-stratified cross-fit on the uniform stratum: fit on one half, report on the other, then swap, so every comment gets a held-out probability

| | ECE | Brier | est. prevalence | labeled | gap | gap 95% CI |
|---|---|---|---|---|---|---|
| raw | 0.104 | 0.051 | 0.188 | 0.083 | +0.104 | [+0.089, +0.121] |
| temperature | 0.072 | 0.047 | 0.155 | 0.083 | +0.072 | [+0.056, +0.089] |
| isotonic | 0.017 | 0.036 | 0.084 | 0.083 | +0.000 | [-0.015, +0.015] |

## Packing penalty (all labeled comments)

| N | best F1 single | best F1 packed | ΔF1 (95% CI) | Brier ratio | median abs(Δp) vs single | passes |
|---|---|---|---|---|---|---|
| 8 | 0.802 | 0.785 | -0.017 [-0.044, +0.010] | 0.922 | 0.020 | True |
| 16 | 0.802 | 0.771 | -0.031 [-0.055, -0.001] | 0.926 | 0.030 | False |

## Nonce robustness

100 comments. Pairwise abs(Δp) across repeats: median 0.010, p95 0.040. Against the single run: median 0.010, p95 0.050.

## Baselines (same state, same question)

| Baseline | scope | n | baseline F1 | Jev F1 (same ids) | Jev − baseline | baseline Brier | baseline ECE |
|---|---|---|---|---|---|---|---|
| regex | uniform | 600 | 0.476 | 0.765 | +0.289 | 0.128 | 0.128 |
| cheap_llm | all | 864 | 0.486 | 0.763 | +0.278 | 0.258 | 0.342 |
| frontier_llm | all | 864 | 0.691 | 0.763 | +0.072 | 0.098 | 0.067 |

## Labels: provenance and panel audit (amendment A1)

Panel: anthropic/claude-haiku-4.5, deepseek/deepseek-v4-flash, openai/gpt-5.6-luna. is_prediction: unanimous -> panel label, any split -> human. Stage B: strict majority, median for scores.

| stratum | source | n |
|---|---|---|
| enriched | human_adjudicated | 164 |
| enriched | human_audit | 100 |
| enriched | panel | 136 |
| uniform | human | 600 |

Blind audit of unanimous panel labels: 11 errors in 100 (rate 0.110, 95% CI [0.063, 0.186]; 5 false positives, 6 false negatives). Panel splits you adjudicated: 164, sided with each model: {'anthropic/claude-haiku-4.5': 91, 'deepseek/deepseek-v4-flash': 101, 'openai/gpt-5.6-luna': 82}.

Stage B reference: mean pairwise agreement between labeling-panel models.

## Stage B fields

203 labeled positives with stage B answers. Score mapping rule: legend.

| Field | n | Jev agreement | human agreement | ratio | status |
|---|---|---|---|---|---|
| is_checkable | 202 | 0.743 | 0.723 | 1.027 | pass |
| is_sarcastic | 203 | 0.941 | 0.967 | 0.973 | pass |
| is_conditional | 203 | 0.773 | 0.792 | 0.976 | pass |
| direction | 193 | 0.725 | 0.683 | 1.062 | pass |
| domain | 186 | 0.780 | 0.662 | 1.178 | pass |
| horizon | 199 | 0.854 | 0.771 | 1.109 | pass |
| stance_vs_thread | 198 | 0.803 | 0.767 | 1.047 | pass |
| certainty | 203 | 0.885 | 0.828 | 1.068 | pass |
| specificity | 203 | 0.891 | 0.850 | 1.048 | pass |

Subject roster: 0.680 of 203 positives landed in `other` (build out the roster).

## Cost, speed, and full-run projection

Eligible comments: 17,950,609 (data/eligible.json (2006-2020, 80-4000 chars)). Price $0.042/M input, 1200 req/min, 250,000 tokens/s. Stage B tokens/comment: 1732 (measured).

| Run | requests | tokens/comment | comments/request | p50 ms | p95 ms |
|---|---|---|---|---|---|
| single | 1000 | 567 | 1.0 | 180 | 241 |
| packed8 | 125 | 355 | 8.0 | 195 | 415 |
| packed16 | 63 | 339 | 15.9 | 218 | 649 |
| stage_b | 277 | 1732 | 1.0 | 212 | 354 |

| Design | stage B pass rate | requests | input tokens | cost USD | days (request limit) | days (token limit) | binding |
|---|---|---|---|---|---|---|---|
| single | 0.127 | 20,224,352 | 14,108,966,766 | 592.58 | 11.70 | 0.65 | requests |
| packed8 **(design)** | 0.107 | 4,158,557 | 9,686,253,598 | 406.82 | 2.41 | 0.45 | requests |
| packed16 | 0.107 | 3,045,619 | 9,399,115,656 | 394.76 | 1.76 | 0.44 | requests |

## Misses (labeled prediction, below gate)

- **13301263** (enriched, p=0.090), *$2T in Proceeds of Corruption Removed from China and Taken to US, AUS, CAN, NL*: QE is not resulting in enough growth in the Western world, because there are just not enough profitable businesses for investors to pour their cheap money into. The central bankers created QE as a way to add liquidity into the markets to allow banks to lend, to allow small businesses to grow, to help economies to generate growth. Unfortunately that money was lent to the big bankers and they realiz…
- **14773647** (uniform, p=0.130), *The Internet of Things – A Disaster*: The vast majority of issues will arise from software defects, not hardware defects. Exactly. The last thing I want in my house is a smoke alarm with software written by idiots like me.
- **16872195** (enriched, p=0.170), *I built a progressive web app and published it in three app stores*: Almost literally by definition, a webapp isn't going to be as good as a proper application. If one of the advantages is "front end guys don't have to learn another language" I'm starting to be of the opinion that that's their reticence to use the proper tools, rather than the inherent superiority of JS. People visit websites because many of them don't need to be apps. I don't need an app (webapp o…
- **6630740** (enriched, p=0.190), *April 5, 2007: "Show HN, Dropbox"*: Came here to post this. No, seriously, exactly this. That was my favorite quote out of the entire comments! FTP with curlftpfs and SVN/CVS isn't "trivial" to 99.9% of the population! It's the same reason that sites like SlashDot were filled with all sorts of reasons why the iPad would never work. Why would you use that?!? You could do the same thing with ______. Yes, you could. But you won't.
- **21317654** (enriched, p=0.220), *Google attempted to shut down a unionization meeting in Zurich*: I'd vote for everyone who has a reasonable plan of more scrutiny towards 40 hour work weeks. This may reduce salaries but oh God most of us make too much money anyhow. Let's have a life now! When are you going to enjoy life, after retirement? Would you wait with sex till after retirement, too, if that offered a salary increase?
- **19768461** (enriched, p=0.250), *Could ImGUI Be the Future of GUIs?*: Sorry if I misinterpreted your post but I don't think that changes my response much. You can absolutely use Win32 Forms and WPF (and probably Qt) with a game. They can be overlaid on top of a DirectX or OpenGL window (with a transparent background) -- I've done it before! I don't think it would have any of the downsides you mentioned, either, except that it wouldn't be GPU accelerated or actually …
- **8786009** (enriched, p=0.280), *Is Homo Economicus a Psychopath?*: > Note on Nietzche: In Thus Spake Zarathrusa, the "uberman" concept is (maybe) considered a positive that humanity should work towards in the absence of belief in God, and I think it can still be understood as such. The problem is that, in the absence of God, there is no absolute morality that is not the result of an arbitrary choice. When the "uberman" arises, he/she creates a new morality, which…
- **6340655** (enriched, p=0.310), *Has the time come to kill the Remember Me checkbox? (2009)*: but isn't what's being asked? If the box is checked by default, then public computers will have a whole list of email address and logins to steal
- **19329296** (enriched, p=0.330), *The $2M Urinal: Why Hard Work Doesn’t Cut It (2018)*: Not really. The lottery (unless cheating) should provide real randomness. ie: A president has the same chance as the next guy. If both a former president and the next guy do an art piece, I'm pretty certain the president will fetch a higher price for his art work. That's the network effect.
- **16034144** (uniform, p=0.330), *Call of Duty gaming community points to ‘swatting’ in Wichita police shooting*: > For example, while SWAT is gearing up, the operator could ask who is calling and how they learned of the issue. How exactly would that help? Swatting is by definition fraudulent callers claiming to have firsthand knowledge of an issue -- in this case, he pretended to be a hostage taker.[1] Better hoax detection training is a fine idea but swatters are determined adversaries. It's going to take m…

## False positives (labeled not a prediction, above gate)

- **10426384** (enriched, p=0.970), *Teen Who Hacked CIA Director’s Email Tells How He Did It*: Two things. 1) This kid just got at least one person fired from his job (though he may deserve it). 2) This kid WILL be caught and regret it the rest of his life.
- **417121** (uniform, p=0.950), *Math pastebin with LaTeX math equation rendering*: Superb! Just what I wanted some months backs and had given up on.. I am surely gonna send you visitors by the truckload in coming weeks.
- **2970135** (uniform, p=0.940), *Replacing a Development VPS with Linux on OSX*: > Not true for Rackspace. You're right. From what I hear, it may become a feature once they convert everything to openstack.
- **20560166** (enriched, p=0.910), *Megapack: Utility-Scale Energy Storage*: I sincerely hope that Tesla can increase cell production to keep up with demand for all their products. When you look at product lines like this it makes you realise the scale and breadth of Elon Musk's influence and even if only half of his endeavours prove to be as successful as they seem like they will be he will be written about extensively in history books.
- **8594594** (uniform, p=0.910), *Rosetta comet landing – live stream*: They received the confirmation that the lander separation was successful. EDIT: First lander data expected in about 2 hours from now.
- **7866115** (enriched, p=0.900), *Show HN: Dominus – Multiplayer browser strategy game made with Meteor*: Yeah, some things were messed up and the database was rolled back. It will hopefully be faster now.
- **6014644** (uniform, p=0.900), *Netflix dumps Exchange, other on-premise software in cloud-first strategy*: I don't think it's just about SSO. Exchange isn't cheap to license or keep operational. If you switch to exchange in cloud it doesn't reduce the costs that much. Hopefully soon Microsoft will realize the value some of their products (SQL Server and Exchange) isn't nearly what it used to be and that they need to adjust prices and licensing requirements.
- **19239216** (uniform, p=0.890), *Ask HN: Pros and cons of working at a startup in 2019?*: Plus the opportunity to get promoted and make significantly more in a couple years.
- **7487827** (uniform, p=0.890), *Tesla Adds Titanium Underbody Shield and Aluminum Deflector Plates to Model S*: I just had a thought that Elon Musk owning both Tesla and SpaceX, the next logical step would be an electric powered airplane or a chopper. It could be an airplane/chopper that would glide down to safety in case of a mechanical failure, or crash land on rough terrain without fear of catching fire. That would disrupt aviation industry like never before.
- **12064590** (uniform, p=0.880), *Silicon Valley-Driven Hype for Self-Driving Cars*: What surprises me is that last-mile and local transportation of non-living cargo hasn't gotten more attention. I can't wait to get cheap delivery from practically anywhere in town and nearly on demand.
