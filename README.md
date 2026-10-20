# World Cup 2026 Final Predictor

Machine Learning model predicting the result of the 2026 FIFA World Cup final
(**Spain vs Argentina**, 2026-07-19, East Rutherford), using the history of
~49,500 international matches since 1872 from the
[martj42/international_results](https://github.com/martj42/international_results)
dataset.

| | |
|---|---|
| **Pre-match forecast** (frozen 2026-07-16, tag [`v1.0-pre-final`](https://github.com/valeeeeeeeeeeeeee/2026-world-cup-finals-predictor/tree/v1.0-pre-final)) | Argentina champion, **51.1%** vs Spain 48.9% |
| **Actual result** (2026-07-19) | **Spain 1-0 Argentina** after extra time |

The forecast made before kickoff is kept separate from everything built
after the final: see [Pre-match forecast](#pre-match-forecast) and
[Post-match analysis](#post-match-analysis). The code on `main` is the
**revised** model; `git checkout v1.0-pre-final` restores the exact code and
outputs that produced the original forecast.

## Methodology (current, revised model)

Three components, combined into a final ensemble:

1. **Strength ratings: Elo + Glicko-2** (`src/elo.py`, `src/glicko2.py`) —
   the full match history is replayed in chronological order by two
   complementary systems: Elo following the
   [World Football Elo](https://www.eloratings.net/about) rules (K-factor by
   competition importance — 60 World Cup, 50 continental finals, 40 World
   Cup/continental qualifiers and Nations Leagues, 30 other tournaments,
   20 friendlies — goal-difference scaling, home-advantage bonus)
   and **Glicko-2** (Glickman), which besides the rating also estimates each
   team's **RD (rating deviation — uncertainty)** and **volatility**.
   Glicko-2 ignores goal margin and reacts differently to recent runs,
   giving the classifier a second strength signal partially independent
   of Elo.
2. **W/D/L classifier with 3-base stacking** (`src/features.py`,
   `src/combine.py`, `src/train_model.py`) — three base models:

   - multinomial logistic regression and **CatBoost**, both recency-weighted
     (6-year half-life), over leak-free features (Elo diff and level,
     Glicko-2 diff and RD gap, 10-match form, rest days, competition
     importance, neutral venue);
   - **W/D/L probabilities from the Poisson model** (score matrix built
     from expected goals) — a signal only weakly correlated with the two
     rating-driven classifiers, which is exactly the diversity a
     combination needs.

   A **logistic meta-model** combines the three outputs, receiving the
   **log-odds** of the 9 base probabilities + match context (neutral venue,
   competition importance). To train it without leakage, the base models
   are refit on expanding chronological windows (2-year blocks from 2000;
   Poisson on a rolling ~11-year window) and predict only the block
   *after* their training window — 25,513 out-of-sample meta-training
   examples (2000–2026).

   `train_model.py` evaluates four combiners on the temporal holdout
   (train < 2019, validate ≥ 2019) and deploys the one with the lowest
   competitive log-loss:

   | Candidate | Competitive log-loss |
   |---|---|
   | Logistic alone | 0.8438 |
   | CatBoost alone | 0.8437 |
   | Poisson W/D/L alone | 0.8732 |
   | Fixed 0.65/0.35 blend (semifinals baseline) | 0.8428 |
   | Average of the 3 bases | 0.8440 |
   | **Stacking v2 (deployed)** | **0.8391** |
   | Average of stacking + blend | 0.8402 |

   Naive baseline: 1.05. With the third Poisson base, the log-odds and the
   context features, stacking now **beats the fixed blend** (0.8391 vs
   0.8428) — the previous version, a meta-model over raw probabilities of
   two correlated bases, lost to the blend. Caveat: picking the best of 4
   combiners on the holdout is mild selection; the exact margin should be
   read with that grain of salt.
3. **Poisson goal model** (`src/poisson_model.py`) — attack/defense
   strengths via L2-regularized Poisson regression (independent Poisson
   goals, without the Dixon-Coles low-score correction), fit on a rolling
   11-year window of matches.

> **Added after the final.** The knockout adjustments below were developed
> once the result was known. The pre-match forecast instead averaged the
> stacking classifier with the Poisson Monte Carlo and had no draw
> recalibration.

Because a final cannot end in a draw, `src/predict_final.py` turns the
90-minute forecast into a title probability with two knockout adjustments
(`src/knockout.py`), both fit on historical knockouts by the backtest below
(only matches before the final):

- **Knockout draw recalibration**: knockouts are level after 90' more often
  than the all-matches classifier predicts (34.0% observed vs 24.9%), so
  the stacking draw odds are multiplied by **1.55** (maximum likelihood),
  with win/loss rescaled to keep their ratio.
- **Extra time + shootout**: a level tie goes to extra time where each side
  scores Poisson(φ · its 90' expected goals), with **φ = 0.242** fit on 105
  real extra times since 2005 (vs. the naive 30/90 = 0.333: extra time is
  ~27% less productive per minute); still level → a coin flip mildly tilted
  by the Elo gap.

Title = P(win in 90') + P(level after 90') × P(win | level). The Poisson
Monte Carlo (50,000 runs, same φ) is no longer averaged in — it was the
weakest method in the backtest — and is kept only as a reference number.
The scoreline matrix is the exact Poisson score distribution, rescaled to
the adjusted W/D/L.

## Pre-match forecast

Frozen at tag [`v1.0-pre-final`](https://github.com/valeeeeeeeeeeeeee/2026-world-cup-finals-predictor/tree/v1.0-pre-final)
(commit `7e18df4`, 2026-07-16, three days before the final). These are the
numbers published before kickoff, unchanged; the method at that point was
stacking v2 for the 90 minutes, averaged with the Poisson Monte Carlo for
the title, with draws resolved by an Elo-tilted shootout.

| Signal | Spain | Argentina |
|---|---|---|
| Elo (eve of the final) | 2308 | 2287 |
| Glicko-2 (RD) | 1889 (67) | 1923 (71) |
| Expected goals (Poisson) | 1.12 | 1.17 |
| 90 minutes (stacking v2) | 34.8% | 37.5% (draw 27.7%) |
| Title — Monte Carlo | 48.8% | 51.2% |
| Title — classifier | 48.9% | 51.1% |
| **Title — final ensemble** | **48.9%** | **51.1%** |

**Predicted champion: Argentina, by a narrow margin.** Elo and Glicko-2
disagreed about who was stronger (Elo gave Spain +21; Glicko-2, which ignores
goal margin and reacts faster to recent runs, gave Argentina +34); the
Poisson model saw a slight Argentine edge in attack. The meta-model, seeing
all three signals, leaned toward Argentina.
P(extra time) = 28.6%, P(penalties) = 15.4%.
Most likely 90-minute scores: 1-1 (13.3%), 0-1 (11.9%), 1-0 (11.4%).

Per-method view at the time (competitive-holdout log-loss, train < 2019,
validate ≥ 2019; title columns resolve the draw via the Elo-tilted
shootout):

| Method | Log-loss ↓ | Spain (90') | Draw | Argentina (90') | Title ESP | Title ARG |
|---|---|---|---|---|---|---|
| Logistic | 0.8438 | 34.4% | 31.8% | 33.8% | 50.6% | 49.4% |
| CatBoost | 0.8439 | 28.8% | 29.9% | 41.2% | 44.1% | 55.9% |
| Poisson W/D/L | 0.8728 | 34.5% | 28.5% | 37.1% | 49.0% | 51.0% |
| Fixed 0.65/0.35 blend | 0.8428 | 32.4% | 31.1% | 36.4% | 48.3% | 51.7% |
| Average of the 3 bases | 0.8440 | 32.6% | 30.1% | 37.4% | 47.9% | 52.1% |
| **Stacking v2 (deployed)** | **0.8391** | 34.8% | 27.7% | 37.5% | 48.9% | 51.1% |
| Average of stacking + blend | 0.8402 | 33.6% | 29.4% | 37.0% | 48.6% | 51.4% |
| Monte Carlo (Poisson simulation) | — | — | — | — | 48.8% | 51.2% |
| **Final ensemble (MC + stacking)** | — | — | — | — | **48.9%** | **51.1%** |

The full pre-match output is `results/final_prediction.json` at that tag
(`git show v1.0-pre-final:results/final_prediction.json`).

## Post-match analysis

Everything from here on was done **after** the final, with the result
known. It evaluates the pre-match forecast and documents a revised model;
the revision is not a forecast and should not be read as one.

### What happened

Spain won 1-0 after extra time (0-0 at 90', Ferran Torres 106').

| Pre-match claim | Outcome | Comment |
|---|---|---|
| Argentina champion, 51.1% | Spain won | A near coin flip that went the other way. Log-loss on this one match: 0.716 vs 0.693 for a 50/50 call. |
| P(extra time) = 28.6% | Extra time happened | Plausible, but the knockout backtest below shows level-after-90' was systematically underpredicted. |
| P(penalties) = 15.4% | No penalties | — |

A single match cannot validate or refute a probabilistic model; the
[knockout backtest](#knockout-backtest-srcbacktest_knockoutspy) over 327
historical knockouts is the meaningful check.

### Revised model (current code)

What the current code on `main` outputs for the same fixture, with the
knockout adjustments added after the final. Ratings differ slightly from
the pre-match run: it was re-run on a later snapshot of the dataset, and two
Elo K-factor fixes now follow the World Football Elo table: continental
qualifiers (Euro, AFCON, Asian Cup, Gold Cup qualification) get K = 40
instead of 50, and other tournaments (regional cups such as the CECAFA, Gulf
or AFF cups, invitationals) K = 30 instead of 40. The Poisson model is also
fit on the same rolling 11-year window as the Poisson base the meta-model
was trained on (it previously used a fixed 2015-01-01 start, which alone
moved the expected goals from 1.11–1.17 to 1.15–1.15).

| Signal | Spain | Argentina |
|---|---|---|
| Elo (eve of the final) | 2298 | 2279 |
| Glicko-2 (RD) | 1889 (67) | 1923 (71) |
| Expected goals (Poisson) | 1.15 | 1.15 |
| 90 minutes (stacking v2, raw) | 36.1% | 36.6% (draw 27.4%) |
| 90 minutes (knockout-adjusted) | 31.4% | 31.8% (draw 36.8%) |
| Win a tie level after 90' | 50.6% | 49.4% |
| *Reference: Monte Carlo title* | *50.2%* | *49.8%* |
| *Reference: previous ensemble (MC + stacking), re-run* | *50.1%* | *49.9%* |
| **Title probability** | **49.99%** | **50.01%** |

The revision mainly changes *how* a team wins (P(extra time) 28.6% → 36.8%,
P(penalties) 15.4% → 22.8%). After the K-factor and Poisson-window fixes
the title is a pure coin flip: Argentina stays nominally ahead by 0.01
points, which is far below the model's precision — read it as 50/50, not as
a call. Most likely 90-minute scores: 1-1 (17.2%), 0-0 (13.0%), 0-1 (10.2%),
1-0 (10.1%).

Full breakdown in `results/final_prediction.json`.

### Method comparison (revised model)

Competitive-holdout log-loss (train < 2019, validate ≥ 2019) and each
method's raw 90-minute prediction for the final. The title columns apply
the same knockout adjustment to every method (draw odds ×1.55, then a level
tie goes 50.6% / 49.4% via extra time + shootout):

| Method | Log-loss ↓ | Spain (90') | Draw | Argentina (90') | Title ESP | Title ARG |
|---|---|---|---|---|---|---|
| Logistic | 0.8438 | 34.2% | 31.8% | 34.1% | 50.3% | 49.7% |
| CatBoost | 0.8437 | 30.2% | 26.7% | 43.1% | 44.6% | 55.4% |
| Poisson W/D/L | 0.8732 | 35.8% | 28.3% | 35.8% | 50.2% | 49.8% |
| Fixed 0.65/0.35 blend | 0.8428 | 32.8% | 30.0% | 37.2% | 48.3% | 51.7% |
| Average of the 3 bases | 0.8440 | 33.4% | 28.9% | 37.7% | 48.4% | 51.6% |
| **Stacking v2 (deployed)** | **0.8391** | 36.1% | 27.4% | 36.6% | **50.0%** | **50.0%** |
| Average of stacking + blend | 0.8402 | 34.4% | 28.7% | 36.9% | 49.2% | 50.8% |

### Figures

Generated by `src/plots.py` from the artifacts in `results/`.

| | |
|---|---|
| ![Scorelines](results/figures/score_heatmap.png) | ![Outcome by method](results/figures/outcome_by_method.png) |
| **90-minute scoreline matrix** (exact Poisson shape, rescaled to the knockout-adjusted W/D/L); dashed diagonal = draw → extra time. | **Raw W/D/L by method** with the knockout-adjusted title odds. |
| ![Calibration](results/figures/calibration.png) | ![Log-loss CI](results/figures/logloss_ci.png) |
| **Calibration** of the deployed combiner on the competitive holdout (ECE ≈ 0.014–0.018 per class). | **Log-loss with 95% bootstrap CIs**: the absolute CIs overlap, but the *paired* difference shows stacking v2 ahead of every other candidate on the same matches, except stacking + blend (+0.0011, CI ≈ [0.000, +0.002], a near tie). |

![Rating history](results/figures/rating_history.png)

**Elo and Glicko-2 since 2018**, with the ±2 RD uncertainty band. The two systems disagree on the eve of the final.

### Knockout backtest (`src/backtest_knockouts.py`)

The W/D/L holdout cannot validate how a draw is turned into a winner, so
the title methods are replayed on real knockout matches (World Cup, Euro,
AFCON, Asian Cup), with every model refit before each edition's first
knockout match. The knockout adjustment's two parameters are re-estimated
before every edition from earlier knockouts only. Target: who advanced.
Paired bootstrap 95% CIs vs. the previously published ensemble.

| Method (who advances, 327 knockouts 2004–2026) | Log-loss ↓ | Δ vs ensemble [95% CI] |
|---|---|---|
| Classifier (stacking v2 + shootout tilt) | 0.5970 | −0.0052 [−0.0120, +0.0016] |
| Monte Carlo (Poisson + extra time + tilt) | 0.6113 | +0.0091 [+0.0024, +0.0160] |
| Ensemble (previous published method) | 0.6022 | — |
| Hybrid (stacking 90' + naive extra-time model) | 0.5964 | −0.0058 [−0.0121, +0.0007] |
| **Knockout-adjusted stacking (deployed)** | **0.5977** | **−0.0046 [−0.0094, +0.0002]** |
| Elo expected score only | 0.5959 | −0.0064 [−0.0216, +0.0091] |

![Knockout methods](results/figures/knockout_methods.png)

| Sub-model check | Observed | Before | After adjustment |
|---|---|---|---|
| P(level after 90'), 326 knockouts 2004+ | 34.0% | 24.9% (log-loss 0.656) | 33.7% (log-loss 0.641) |
| P(penalties \| extra time), 154 extra times 1986+ | 64.9% | 54.7% (log-loss 0.666) | 58.3% (log-loss 0.653) |

![Knockout sub-models](results/figures/knockout_submodels.png)

Findings:

- **The Monte Carlo dragged the ensemble down.** It is the weakest method
  and the only one clearly separated from the ensemble (its CI excludes 0);
  dropping it is the source of the ~0.005 gain over the previous ensemble,
  although that gain's CI now just touches 0.
- **Knockout draws were underpredicted** (34.0% level after 90' vs 24.9%);
  the draw recalibration fixes the level (33.7%), which makes P(extra
  time) and P(penalties) honest.
- **Extra time has ~27% fewer goals per minute** than the 90 minutes
  (φ = 0.242 vs 0.333), so more ties reach penalties; still slightly
  underpredicted (58.3% vs 64.9%).
- **The adjustment does not sharpen "who advances"**: deployed vs. the
  unadjusted hybrid is +0.0012 [−0.0019, +0.0041], i.e. a tie. Once level
  after 90' a tie is close to a coin flip (Elo tilt 0.678, extra-time model
  0.677, 50/50 0.693), so moving mass into the draw mostly changes *how*
  a team advances, not *who*. It is deployed for the calibrated
  extra-time/penalty probabilities at no cost in title accuracy.
- Plain Elo is as good as any method on this small sample; the CIs
  (±0.015) are too wide to separate them.

Full output: `results/knockout_backtest.json`, the row-level
`results/knockout_backtest.csv` and the deployed parameters in
`results/knockout_calibration.json`.

## Running it

Requires Python 3.13 (see `.python-version`); `requirements.txt` pins the
exact library versions `results/classifier.joblib` was saved with, and the
pickle may not load under other scikit-learn/CatBoost versions.

```bash
pip install -r requirements.txt
cd src
python train_model.py         # trains and saves the stacking classifier
python backtest_knockouts.py  # replays historical knockouts, fits the knockout adjustment (~3 min)
python predict_final.py       # runs the full pipeline, prints + saves the prediction
python plots.py               # renders the figures to results/figures/
```

`results.csv`, `shootouts.csv` and `goalscorers.csv` are downloaded
automatically on first run from a pinned commit of the data repository
(`DATA_COMMIT` in `src/data_io.py`) and checked against SHA-256 hashes, so
re-runs see exactly the same data (results already include the semifinals: Spain
2-0 France, Argentina 2-1 England; the cutoff keeps the final itself out of
every fit).

Tests (no dataset download needed; they run on synthetic data):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

They cover the Glicko-2 worked example from Glickman's paper, known Elo
K-factors and goal-difference multipliers, `poisson_wdl` normalization,
edge cases of `recalibrate_draw`/`title_prob`, and leakage: flipping a
match's result must not change that match's (or any earlier match's)
features.

To inspect or re-run the original pre-match forecast:

```bash
git checkout v1.0-pre-final
```

Re-running it today downloads a newer snapshot of the dataset, so ratings
can shift slightly; the committed `results/` at that tag are the published
numbers.

## Project layout

```
src/data_io.py         dataset download/loading
src/elo.py             Elo rating engine
src/glicko2.py         Glicko-2 rating engine (rating + RD + volatility)
src/features.py        leak-free chronological feature builder
src/combine.py         combiners (blends, stacking) and Poisson W/D/L
src/train_model.py     trains bases + meta, compares combiners, deploys the best
src/poisson_model.py   Dixon-Coles style Poisson goal model
src/predict_final.py   full pipeline -> final prediction
src/plots.py           figures (results/figures/*.png)
src/knockout.py        knockout draw recalibration + extra-time/shootout model
src/backtest_knockouts.py  title-method backtest on historical knockouts
tests/                 pytest suite (rating engines, probabilities, leakage)
results/               trained model + metrics + prediction (generated)
```

## Caveats

- International football is inherently noisy; ~61% accuracy on a 3-class
  problem is a solid edge over the baseline, not a crystal ball.
- Squad-level factors (injuries, suspensions, travel) are not modeled —
  only team-level Elo, recent form and scoring rates.
- The Elo shootout tilt (Elo gap / 1200) is a modeling assumption rather
  than a fitted parameter, but it agrees with the data: on 694 historical
  shootouts the maximum-likelihood divisor is ≈1640 (log-loss 0.6878 vs
  0.6885 for 1200 and 0.6931 for a fair coin).
- The Poisson goal model is an independent-Poisson (Maher) model with
  attack/defense strengths, no Dixon-Coles low-score correction and no time
  decay inside its 11-year window. Its hard window edge makes it sensitive:
  shifting the start by 6.5 months moved the final's expected goals by
  ~0.04 per side.
- Glicko-2 treats each match as a rating period and does not inflate RD
  while a team is idle, so RD mostly reflects how many matches a team has
  played, not how long ago.
- `results.csv` records knockout scores after extra time, so the W/D/L
  target is "result after extra time" for the ~0.3% of matches decided in
  extra time (146 known cases, 31 since 2019) — negligible for the
  classifier, and the knockout draw factor is fit on the true 90' state.

## Credits

Data: [martj42/international_results](https://github.com/martj42/international_results).
