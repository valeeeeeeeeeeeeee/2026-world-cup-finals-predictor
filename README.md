# ⚽ World Cup 2026 Final Predictor

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
   *after* their training window — 25,456 out-of-sample meta-training
   examples (2000–2026).

   `train_model.py` evaluates four combiners on the temporal holdout
   (train < 2019, validate ≥ 2019) and deploys the one with the lowest
   competitive log-loss:

   | Candidate | Competitive log-loss |
   |---|---|
   | Logistic alone | 0.8438 |
   | CatBoost alone | 0.8439 |
   | Poisson W/D/L alone | 0.8728 |
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
| Elo (eve of the final) | 2308 | 2287 |
| Glicko-2 (RD) | 1889 (67) | 1923 (71) |
| Expected goals (Poisson) | 1.12 | 1.17 |
| 90 minutes (stacking v2) | 34.8% | 37.5% (draw 27.7%) |
| Title — Monte Carlo | 48.8% | 51.2% |
| Title — classifier | 48.9% | 51.1% |
| **Title — final ensemble** | **48.9%** | **51.1%** |

**Predicted champion: Argentina, by a narrow margin.** Elo and Glicko-2
disagree about who is stronger (Elo gives Spain +21; Glicko-2, which ignores
goal margin and reacts faster to recent runs, gives Argentina +34); the
Poisson model sees a slight Argentine edge in attack. The meta-model, seeing
all three signals, leans toward Argentina.
P(extra time) = 28.6%, P(penalties) = 15.4%.
Most likely 90-minute scores: 1-1 (13.3%), 0-1 (11.9%), 1-0 (11.4%).

Full breakdown in `results/final_prediction.json`.

## Method comparison

Competitive-holdout log-loss (train < 2019, validate ≥ 2019) and each
method's prediction for the final (the title columns already resolve the
draw via the Elo-tilted shootout, 51%/49% in Spain's favor):

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

## Running it

```bash
pip install -r requirements.txt
cd src
python train_model.py     # trains and saves the ensemble classifier
python predict_final.py   # runs the full pipeline, prints + saves the prediction
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

Data: [martj42/international_results](https://github.com/martj42/international_results) (CC0).
