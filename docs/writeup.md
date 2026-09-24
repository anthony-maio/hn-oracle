# Hacker News has made about 1.5 million predictions. Finding them costs $400, not $30,000.

*Anthony Maio, September 2026. First draft; a revised version will run on my Substack. Pre-registered plan, code, raw responses and labels: [github.com/anthony-maio/hn-oracle](https://github.com/anthony-maio/hn-oracle).*

In 2015 a Hacker News commenter wrote: "Swift 3 will be ABI stable, so things will break less." Swift 3 shipped in September 2016 without ABI stability. It finally landed in Swift 5, in March 2019, two major versions late.

That comment is one of about 1.5 million like it. Between 2006 and 2020, HN users posted 17.95 million comments long enough to say something and old enough to have five years of hindsight behind them. I hand-labeled a random 600 of them, and 50 made a real claim about the future -- 8.3%, which puts the archive at somewhere between 1.1 and 1.9 million predictions about languages, companies, markets, laws and the shape of work, most of them now checkable.

Last December Andrej Karpathy graded one month of this. He pointed GPT-5.1 Thinking at the 930 front-page threads from December 2015 and had it judge them in hindsight, for about $60 ([write-up](https://karpathy.bearblog.dev/auto-grade-hn/)). I wanted all fifteen years. That's not a $60 job, and before spending real money I wanted to know whether a much cheaper model could do the first and largest step: reading 17.95 million comments and deciding which ones are predictions at all.

So I wrote the test down first. Seven pass/fail gates, fixed before the first API call, published whether they passed or not. The pilot ran on September 23. It cost $9.34 in API calls and one evening of labeling, and every gate passed.

## The expensive part is the calendar, not the bill

The model under test is [Jev](https://docs.typesafe.ai), TypeSafe's "System One" decision model. It doesn't write text. You hand it a state and typed questions -- yes/no, one-of-N, or a graded scale -- and it returns a probability for each. Input costs $0.042 per million tokens.

My first estimate for the full archive was $500 and 14 hours. It was wrong in the way that matters. Jev's public rate limit is 1,200 requests a minute. At one comment per request, 17.95 million comments is 17.95 million requests, and 17.95 million divided by 1,200 is 14,958 minutes: **10.4 days**, no matter how much money you have.

Two design choices follow. A two-stage cascade asks one cheap yes/no question of every comment and saves the ten detailed questions (checkable? horizon? how certain?) for the roughly 11% that pass the gate. And packing puts several comments in one request, each under its own ID, which divides the request count by the pack size if the answers survive it.

They mostly did. At 8 comments per request, F1 dropped 0.017, inside the pre-registered limit of 0.03. At 16 it dropped 0.031 and failed by a thousandth. Packing 8 also cut billed input tokens per comment from 567 to 355, since each request's fixed overhead is shared across the pack.

**Projected full run, both stages, packed 8: $407 and 2.4 days.**

## Jev tied Claude Sonnet 5, and my first run said it won

On the 864 comments I labeled myself, I ran the same question through Jev, Claude Sonnet 5, gpt-5-nano, and a regex that looks for "will", "going to", "mark my words" and friends.

| | F1 | Brier | cost per comment | latency (p50) |
|---|---|---|---|---|
| Jev | **0.76** | 0.082 | **$0.000024** | **180 ms** |
| Claude Sonnet 5 | 0.74 | **0.073** | $0.0017 | 1.8 s |
| gpt-5-nano | 0.62 | 0.185 | $0.0003 | 7.0 s |
| regex (random comments only) | 0.48 | 0.128 | free | -- |

The F1 gap between Jev and Sonnet is 0.027, with a 95% bootstrap interval of -0.016 to +0.069. That's a tie. Sonnet is better calibrated out of the box. What Jev does is match Sonnet's accuracy at 1/70th of the price and a tenth of the latency. Put Sonnet on the first stage for all 17.95 million comments, one per request, and the bill is about $30,000. Jev's whole projected run, both stages, is $407.

My first baseline run said something else. It showed Jev beating Sonnet by 0.07, a clean win outside the error bars, and I nearly shipped that.

The prompt asked each LLM for "a calibrated probability that the answer is true." Read that again the way a model might. Sonnet sometimes read it as "how sure are you of your answer": it would answer *not a prediction* and attach 0.97. Asked whether "Is there a separate submit page for Ask HN?" makes a prediction, it said no, with probability 0.97. Twenty-five of its 1,000 answers did that. gpt-5-nano did it 209 times.

The error only ever ran against the LLMs, so it inflated Jev's margin. I renamed the field `p_prediction`, spelled out that it means the chance the comment is a prediction, and reran both models. Zero contradictions the second time, and Jev's lead shrank to the tie above. The broken runs are published next to the fixed ones.

**A free-text probability is a prompt-engineering problem; Jev's typed yes/no answer comes back as a number that means one thing.**

## Jev's raw numbers count double

The idea that got me into this was that calibrated probabilities can be added up. If Jev says 0.3 for each of ten comments, you should expect about three predictions among them, and summing over a year of HN would tell you how often the site predicted anything, with no labels at all.

Raw Jev fails that test. Add up its probabilities across my 600 random comments and you get 113 predictions. I found 50. The shape is consistent, though: Jev overshoots in the middle of the range, and comments it scores between 0.2 and 0.5 are almost never predictions. Isotonic recalibration, fit on one half of the labels and tested on the other, fixes it. Calibration error drops from 0.105 to 0.017, and the recalibrated sum comes to 50.3.

So the prevalence gate passed, and every calibration claim in this project carries the words "after recalibration." For a full run that means labeling a random slice of each era and fitting the correction per era, not trusting the raw scores.

## What Jev gets wrong

The ten worst false positives, all on comments I labeled myself, fall into a few groups. Hopes: "Hopefully soon Microsoft will realize the value..." Personal plans: "I am surely gonna send you visitors by the truckload in coming weeks." Status updates: "First lander data expected in about 2 hours from now." Hearsay: "From what I hear, it may become a feature once they convert everything to openstack."

Jev scored each of those 0.88 or higher. They all have the grammar of a forecast and none of the substance. One was a judgment call on my side: "This kid WILL be caught and regret it the rest of his life" is a claim about the future, and I labeled it no because the question's criteria ask about a technology, company, product, market, law or social trend, and one hacker's fate is none of those. Jev read the question more loosely than I did.

The misses run the other way: real forecasts without forecast grammar. "Yes, you could. But you won't." "This may reduce salaries." "Only if we get very, very lucky." "It's going to take more than just asking a couple basic questions." Jev scored those between 0.19 and 0.33. The gate still caught 46 of the 50 predictions in the random sample, a recall of 0.92, but choosing the gate on one half of the data and testing on the other gave 1.00 and 0.84. The margin over the 0.85 bar is thin, and at 8 comments per request recall drops to 0.86 unless the gate is re-picked on packed output.

## Nobody agrees on what a prediction is

I couldn't label 1,000 comments plus nine detail fields each in one sitting, so I brought in a panel of three models from three families -- Claude Haiku 4.5, DeepSeek v4 Flash and GPT-5.6 Luna -- and wrote the change into the plan as a dated amendment before labeling started. I kept the parts that carry the headline claims: all 600 random comments, every comment the panel split on, and a blind audit of 100 of the panel's unanimous labels, all shuffled into one sheet so I couldn't tell which was which.

The panel's first result was that it didn't agree. On 400 comments flagged by the regex, the three models were unanimous only 59% of the time. GPT-5.6 Luna called 55% of them predictions; Claude Haiku 4.5 called 27%. When all three did agree, the audit found them wrong 11 times in 100.

That's the real ceiling on this whole project. "Is this a prediction?" sounds like a yes/no question, and on a good fraction of HN it isn't one. The next version of the pilot needs a second human labeler, so there's a human-to-human agreement number to hold Jev against.

## What Sonnet says came true

The last stage of the pilot is a demo: take the checkable predictions whose stated horizon has passed and have a frontier model with web search grade them. Twenty-four qualified. Sonnet 5 judged 8 came true, 7 didn't, 8 partly, and 1 couldn't be resolved.

Sonnet wrote the verdicts; I checked each one against its sources. A sample:

| Year | Prediction | Sonnet's verdict |
|---|---|---|
| 2015 | "Swift 3 will be ABI stable, so things will break less" | didn't come true: ABI stability shipped in Swift 5, 2019 |
| 2011 | Making HTML5 games with 2D canvas "is going to die before it even gets off the ground" | didn't come true |
| 2015 | Xiaomi won't spend its new $1.1bn coming to the US: "I doubt that... Apple will sue them if they enter a market with good IP protection" | came true: still no official US phone business |
| 2019 | "This is part of the long term plan for the T2 chip, they just need a few years before they can stop supporting the 'normal' computers" | came true: Apple finished moving Macs off Intel in 2023 |
| 2013 | Amazon Prime Air "per unit cost will fall below $2000 within few years" | didn't come true: no commercial drone deliveries until 2022 |

## What happens next

The pilot said go. Before the full run, two things from the results need fixing. The subject question -- a fixed list of companies and technologies Jev picks from, since it can't extract free text -- put 68% of predictions in `other`, so the list has to grow from the labels first. And the gate has to be re-picked on 8-per-request output, where recall sits one comment above the bar.

Then it's 2.4 days and about $400 to read fifteen years of Hacker News, find the million and a half comments that bet on the future, and hand the checkable ones to a model that can look up how they turned out. The Swift 3 commenter was wrong by two major versions. There are about 1.5 million more of those, and most of them have never been checked.

---

*Everything is published: the pre-registered plan and its amendment, every raw response from Jev, the baselines and the panel, my labels with 370 notes, and the full report with reliability tables and bootstrap intervals. HN usernames are removed. I'm an engineer with 20+ years as an IC, staff engineer turned AI safety researcher, and I'm looking for my next role in evals, applied ML infrastructure or research engineering: [making-minds.ai](https://making-minds.ai).*
