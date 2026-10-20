# World Cup 2026 Final Predictor

Machine Learning model predicting the result of the 2026 FIFA World Cup final
(**Spain vs Argentina**, 2026-07-19, East Rutherford), using the history of
~49,500 international matches since 1872 from the
[martj42/international_results](https://github.com/martj42/international_results)
dataset.

## Methodology

Three components, combined into a final ensemble:

1. **Strength ratings: Elo + Glicko-2** (`src/elo.py`, `src/glicko2.py`) —
   the full match history is replayed in chronological order by two
   complementary systems: Elo following the
   [World Football Elo](https://www.eloratings.net/about) rules (K-factor by
   competition importance, goal-difference scaling, home-advantage bonus)
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
   | CatBoost alone | 0.8433 |
   | Poisson W/D/L alone | 0.8732 |
   | Fixed 0.65/0.35 blend (semifinals baseline) | 0.8427 |
   | Average of the 3 bases | 0.8439 |
   | **Stacking v2 (deployed)** | **0.8391** |
   | Average of stacking + blend | 0.8401 |

   Naive baseline: 1.05. With the third Poisson base, the log-odds and the
   context features, stacking now **beats the fixed blend** (0.8391 vs
   0.8427) — the previous version, a meta-model over raw probabilities of
   two correlated bases, lost to the blend. Caveat: picking the best of 4
   combiners on the holdout is mild selection; the exact margin should be
   read with that grain of salt.
3. **Poisson goal model** (`src/poisson_model.py`) — Dixon-Coles-style
   attack/defense strengths via L2-regularized Poisson regression
   (matches since 2015).

Because a final cannot end in a draw, `src/predict_final.py` resolves the
draw in two ways and averages them:

- **Monte Carlo (50,000 simulations)**: regulation goals ~ Poisson; a level
  match goes to extra time (rates rescaled to 30 minutes) and, if still
  level, penalties modeled as a coin flip mildly tilted by the Elo gap.
- **Classifier draw redistribution**: the 90-minute draw probability mass
  is reassigned to each side using the same shootout tilt.

## Result — Spain vs Argentina

| Signal | Spain | Argentina |
|---|---|---|
| Elo (eve of the final) | 2309 | 2288 |
| Glicko-2 (RD) | 1889 (67) | 1923 (71) |
| Expected goals (Poisson) | 1.11 | 1.17 |
| 90 minutes (stacking v2) | 34.3% | 38.1% (draw 27.7%) |
| Title — Monte Carlo | 48.7% | 51.3% |
| Title — classifier | 48.4% | 51.6% |
| **Title — final ensemble** | **48.6%** | **51.4%** |

**Predicted champion: Argentina, by a narrow margin.** Elo and Glicko-2
disagree about who is stronger (Elo gives Spain +21; Glicko-2, which ignores
goal margin and reacts faster to recent runs, gives Argentina +33); the
Poisson model sees a slight Argentine edge in attack. The meta-model, seeing
all three signals, leans toward Argentina.
P(extra time) = 28.6%, P(penalties) = 15.2%.
Most likely 90-minute scores: 1-1 (13.2%), 0-1 (11.9%), 1-0 (11.5%).

Full breakdown in `results/final_prediction.json`.

## Method comparison

Competitive-holdout log-loss (train < 2019, validate ≥ 2019) and each
method's prediction for the final (the title columns already resolve the
draw via the Elo-tilted shootout, 51%/49% in Spain's favor):

| Method | Log-loss ↓ | Spain (90') | Draw | Argentina (90') | Title ESP | Title ARG |
|---|---|---|---|---|---|---|
| Logistic | 0.8438 | 34.3% | 31.6% | 34.1% | 50.4% | 49.6% |
| CatBoost | 0.8433 | 30.3% | 26.2% | 43.5% | 43.6% | 56.4% |
| Poisson W/D/L | 0.8732 | 34.4% | 28.5% | 37.1% | 48.9% | 51.1% |
| Fixed 0.65/0.35 blend | 0.8427 | 32.9% | 29.7% | 37.4% | 48.0% | 52.0% |
| Average of the 3 bases | 0.8439 | 33.0% | 28.8% | 38.3% | 47.6% | 52.4% |
| **Stacking v2 (deployed)** | **0.8391** | 34.3% | 27.7% | 38.1% | 48.4% | 51.6% |
| Average of stacking + blend | 0.8401 | 33.6% | 28.7% | 37.7% | 48.2% | 51.8% |
| Monte Carlo (Poisson simulation) | — | — | — | — | 48.7% | 51.3% |
| **Final ensemble (MC + stacking)** | — | — | — | — | **48.6%** | **51.4%** |

## Figures

Generated by `src/plots.py` from the artifacts in `results/`.

| | |
|---|---|
| ![Scorelines](results/figures/score_heatmap.png) | ![Outcome by method](results/figures/outcome_by_method.png) |
| **90-minute scoreline matrix** (Monte Carlo); dashed diagonal = draw → extra time. | **W/D/L by method** with the implied title odds. |
| ![Calibration](results/figures/calibration.png) | ![Log-loss CI](results/figures/logloss_ci.png) |
| **Calibration** of the deployed combiner on the competitive holdout (ECE ≈ 0.014–0.015 per class). | **Log-loss with 95% bootstrap CIs**: the absolute CIs overlap, but the *paired* difference shows stacking v2 ahead of every other candidate on the same matches (vs. stacking + blend: +0.0011, CI barely above 0). |

![Rating history](results/figures/rating_history.png)

**Elo and Glicko-2 since 2018**, with the ±2 RD uncertainty band. The two systems disagree on the eve of the final.

## Knockout backtest (`src/backtest_knockouts.py`)

The W/D/L holdout cannot validate how a draw is turned into a winner, so
the title methods are replayed on real knockout matches (World Cup, Euro,
AFCON, Asian Cup), with every model refit before each edition's first
knockout match. Target: who advanced. Paired bootstrap 95% CIs vs. the
published ensemble.

| Method (who advances, 327 knockouts 2004–2026) | Log-loss ↓ | Δ vs ensemble [95% CI] |
|---|---|---|
| Classifier (stacking v2 + shootout tilt) | 0.5956 | −0.0059 [−0.0126, +0.0007] |
| Monte Carlo (Poisson + extra time + tilt) | 0.6112 | +0.0097 [+0.0029, +0.0165] |
| **Ensemble (published)** | 0.6015 | — |
| Hybrid (stacking 90' + MC extra-time model) | 0.5951 | −0.0065 [−0.0125, −0.0003] |
| Elo expected score only | 0.5946 | −0.0069 [−0.0224, +0.0089] |

Findings:

- **The Monte Carlo drags the ensemble down.** It is the weakest method,
  and averaging it in is measurably worse than using the classifier (or
  the hybrid) alone.
- **Knockout draws are underpredicted.** 34.0% of knockouts were level
  after 90', vs. 25.2% predicted by stacking and 27.2% by Poisson.
- **Extra time is tighter than modeled.** 64.9% of extra times went to
  penalties (n = 154, 1986+), vs. 54.7% implied by the "1/3 of the
  90-minute rates" assumption.
- **Once level after 90', it is close to a coin flip**: the Elo tilt and
  the Monte Carlo extra-time model tie (0.677 vs 0.676) and only slightly
  beat 50/50 (0.693).
- Plain Elo is as good as any method on this small sample; the CIs
  (±0.015) are too wide to separate them.

Full output: `results/knockout_backtest.json` and the row-level
`results/knockout_backtest.csv`.

## Running it

```bash
pip install -r requirements.txt
cd src
python train_model.py     # trains and saves the ensemble classifier
python predict_final.py   # runs the full pipeline, prints + saves the prediction
python plots.py           # renders the figures to results/figures/
python backtest_knockouts.py  # replays historical knockouts (~3 min)
```

`results.csv` is downloaded automatically on first run (it already includes
the semifinal results: Spain 2-0 France, Argentina 2-1 England).

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
src/backtest_knockouts.py  title-method backtest on historical knockouts
results/               trained model + metrics + prediction (generated)
```

## Caveats

- International football is inherently noisy; ~61% accuracy on a 3-class
  problem is a solid edge over the baseline, not a crystal ball.
- Squad-level factors (injuries, suspensions, travel) are not modeled —
  only team-level Elo, recent form and scoring rates.
- The Elo shootout tilt is a modeling assumption, not fit to historical
  penalty-shootout data.

## Credits

Data: [martj42/international_results](https://github.com/martj42/international_results).
