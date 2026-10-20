"""
Prediction pipeline for the 2026 World Cup FINAL:

    Spain vs Argentina  (East Rutherford, 2026-07-19)

Same methodology as the semifinals model, retrained with both semifinal
results included (cutoff 2026-07-19 — no result from the final itself can
leak, it hasn't been played). Two independent signals are combined into a
single "probability of winning the title" per team:

  1. Poisson goal-model Monte Carlo simulation (50,000 runs). Regulation
     goals are drawn from each team's fitted attack/defense Poisson rates;
     drawn matches go to a lower-intensity extra time, and matches still
     level are resolved by a penalty shootout modeled as a mildly
     Elo-tilted coin flip.
  2. The trained Win/Draw/Loss stacking classifier's 90-minute
     probabilities (logistic meta-model over the logistic + GBM base
     outputs), with the drawn mass reassigned to each side using the same
     shootout-tilt logic (a final cannot end level).

The reported title probability is the average of the two estimates.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from combine import COMBINERS, combine_probs, poisson_wdl
from data_io import load_results, RESULTS_DIR
from features import build_dataset, fixture_features
from poisson_model import PoissonGoalModel

CUTOFF = "2026-07-19"
N_SIMS = 50_000
RNG = np.random.default_rng(42)
SCORE_GRID = 7  # 0..6 goals per side in the exported 90-minute score matrix

FINAL = {"home": "Spain", "away": "Argentina",
         "venue": "East Rutherford", "date": "2026-07-19"}


def shootout_win_prob(elo_diff_raw: float) -> float:
    """Penalty shootouts are close to a coin flip; skill still nudges it.
    Dampened logistic (divisor 3x the normal Elo scale of 400)."""
    return 1.0 / (1.0 + 10 ** (-elo_diff_raw / 1200.0))


def simulate_match(mu_home, mu_away, elo_diff_raw, n_sims=N_SIMS):
    home_reg = RNG.poisson(mu_home, size=n_sims)
    away_reg = RNG.poisson(mu_away, size=n_sims)

    home_wins = home_reg > away_reg
    away_wins = home_reg < away_reg
    draw_mask = home_reg == away_reg

    n_draw = int(draw_mask.sum())
    p_extra_time = n_draw / n_sims
    if n_draw:
        et_home = RNG.poisson(mu_home / 3.0, size=n_draw)  # 30 extra minutes
        et_away = RNG.poisson(mu_away / 3.0, size=n_draw)
        et_total_home = home_reg[draw_mask] + et_home
        et_total_away = away_reg[draw_mask] + et_away

        et_home_win = et_total_home > et_total_away
        et_away_win = et_total_home < et_total_away
        et_draw = et_total_home == et_total_away

        n_pens = int(et_draw.sum())
        pen_home_win = RNG.random(n_pens) < shootout_win_prob(elo_diff_raw)

        idx = np.where(draw_mask)[0]
        home_wins[idx[et_home_win]] = True
        away_wins[idx[et_away_win]] = True
        pen_idx = idx[et_draw]
        home_wins[pen_idx[pen_home_win]] = True
        away_wins[pen_idx[~pen_home_win]] = True
        p_penalties = n_pens / n_sims
    else:
        p_penalties = 0.0

    # Most likely 90-minute scorelines from the same simulated draws.
    scores = pd.Series(zip(home_reg.tolist(), away_reg.tolist())).value_counts(normalize=True)
    top_scores = [
        {"score": f"{h} x {a}", "prob": round(float(p), 4)}
        for (h, a), p in scores.head(5).items()
    ]

    score_matrix = np.zeros((SCORE_GRID, SCORE_GRID))
    in_grid = (home_reg < SCORE_GRID) & (away_reg < SCORE_GRID)
    np.add.at(score_matrix, (home_reg[in_grid], away_reg[in_grid]), 1.0 / n_sims)

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "p_home_title": float(home_wins.mean()),
        "p_away_title": float(away_wins.mean()),
        "p_extra_time": p_extra_time,
        "p_penalties": p_penalties,
        "top_scores": top_scores,
        "score_matrix": score_matrix,
    }


def _base_signals(clf_bundle, feats: dict, mu_home: float, mu_away: float):
    """The three aligned base signals (logistic, CatBoost, Poisson W/D/L) and
    the match context the meta-model expects."""
    classes = clf_bundle["classes"]
    X = pd.DataFrame([{k: feats[k] for k in clf_bundle["features"]}])

    def aligned(model):
        probs = model.predict_proba(X)[0]
        return np.array([[probs[list(model.classes_).index(c)] for c in classes]])

    lr_p = aligned(clf_bundle["members"]["logistic"])
    cb_p = aligned(clf_bundle["members"]["catboost"])
    pois_p = poisson_wdl(mu_home, mu_away)
    context = np.array([[feats["neutral"], feats["k_weight"]]], dtype=float)
    return lr_p, cb_p, pois_p, context


def ensemble_class_probs(clf_bundle, feats: dict, mu_home: float, mu_away: float) -> dict:
    """Apply the deployed combiner (see combine.py) to the three base signals."""
    out = combine_probs(clf_bundle["combiner"], *_base_signals(clf_bundle, feats, mu_home, mu_away),
                        clf_bundle["meta_model"])[0]
    return dict(zip(clf_bundle["classes"], out))


def method_breakdown(clf_bundle, feats: dict, mu_home: float, mu_away: float) -> dict:
    """90-minute W/D/L probabilities from every base model and combiner."""
    lr_p, cb_p, pois_p, context = _base_signals(clf_bundle, feats, mu_home, mu_away)
    probs = {"logistic": lr_p, "catboost": cb_p, "poisson_wdl": pois_p}
    for name in COMBINERS:
        probs[name] = combine_probs(name, lr_p, cb_p, pois_p, context, clf_bundle["meta_model"])
    keys = {"H": "home_win", "D": "draw", "A": "away_win"}
    return {name: {keys[c]: round(float(p[0][j]), 4) for j, c in enumerate(clf_bundle["classes"])}
            for name, p in probs.items()}


def rating_history(feat_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Pre-match Elo / Glicko-2 rating and RD of `team` at each of its matches."""
    rows = []
    for side in ("home", "away"):
        m = feat_df[feat_df[f"{side}_team"] == team]
        rows.append(pd.DataFrame({
            "date": m["date"], "team": team, "elo": m[f"{side}_elo"],
            "glicko": m[f"{side}_glicko"], "glicko_rd": m[f"{side}_glicko_rd"],
        }))
    return pd.concat(rows).sort_values("date")


def classifier_title_prob(clf_bundle, feats: dict, elo_diff_raw: float,
                          mu_home: float, mu_away: float):
    class_probs = ensemble_class_probs(clf_bundle, feats, mu_home, mu_away)
    p_home, p_draw, p_away = class_probs["H"], class_probs["D"], class_probs["A"]

    p_home_shootout = shootout_win_prob(elo_diff_raw)
    return {
        "p_home_90": p_home, "p_draw_90": p_draw, "p_away_90": p_away,
        "p_home_title": p_home + p_draw * p_home_shootout,
        "p_away_title": p_away + p_draw * (1 - p_home_shootout),
    }


def main():
    results = load_results()

    home, away = FINAL["home"], FINAL["away"]
    feat_df, elo, glicko, form = build_dataset(results, cutoff_date=CUTOFF)
    goal_model = PoissonGoalModel().fit(results, cutoff_date=CUTOFF)
    clf_bundle = joblib.load(RESULTS_DIR / "classifier.joblib")

    feats = fixture_features(elo, glicko, form, home, away, neutral=True, match_date=FINAL["date"])
    elo_diff_raw = feats["home_elo"] - feats["away_elo"]
    mu_home = goal_model.expected_goals(home, away, neutral=True)
    mu_away = goal_model.expected_goals(away, home, neutral=True)

    mc = simulate_match(mu_home, mu_away, elo_diff_raw)
    clf = classifier_title_prob(clf_bundle, feats, elo_diff_raw, mu_home, mu_away)

    title_home = (mc["p_home_title"] + clf["p_home_title"]) / 2.0
    title_away = 1.0 - title_home
    champion = home if title_home >= 0.5 else away

    report = {
        f"{home} vs {away}": {
            "venue": FINAL["venue"],
            "date": FINAL["date"],
            "elo": {home: round(feats["home_elo"], 1), away: round(feats["away_elo"], 1)},
            "glicko2": {
                home: {"rating": round(feats["home_glicko"], 1), "rd": round(feats["home_glicko_rd"], 1)},
                away: {"rating": round(feats["away_glicko"], 1), "rd": round(feats["away_glicko_rd"], 1)},
            },
            "poisson_expected_goals": {home: round(mc["mu_home"], 2), away: round(mc["mu_away"], 2)},
            "classifier_90min_prob": {
                "home_win": round(clf["p_home_90"], 4),
                "draw": round(clf["p_draw_90"], 4),
                "away_win": round(clf["p_away_90"], 4),
            },
            "monte_carlo_title_prob": {home: round(mc["p_home_title"], 4), away: round(mc["p_away_title"], 4)},
            "classifier_title_prob": {home: round(clf["p_home_title"], 4), away: round(clf["p_away_title"], 4)},
            "p_extra_time": round(mc["p_extra_time"], 4),
            "p_penalties": round(mc["p_penalties"], 4),
            "most_likely_scores_90min": mc["top_scores"],
            "score_matrix_90min": {
                "rows": f"{home} goals 0..{SCORE_GRID - 1}",
                "cols": f"{away} goals 0..{SCORE_GRID - 1}",
                "prob": np.round(mc["score_matrix"], 4).tolist(),
            },
            "method_breakdown_90min": method_breakdown(clf_bundle, feats, mu_home, mu_away),
            "final_title_probability": {home: round(title_home, 4), away: round(title_away, 4)},
            "predicted_champion": champion,
        }
    }

    print(f"=== WORLD CUP 2026 FINAL — {home} vs {away} ({FINAL['venue']}, {FINAL['date']}) ===")
    print(f"Elo:      {home} {feats['home_elo']:.0f}  |  {away} {feats['away_elo']:.0f}")
    print(f"Glicko-2: {home} {feats['home_glicko']:.0f} (RD {feats['home_glicko_rd']:.0f})"
          f"  |  {away} {feats['away_glicko']:.0f} (RD {feats['away_glicko_rd']:.0f})")
    print(f"Poisson expected goals: {home} {mc['mu_home']:.2f} - {mc['mu_away']:.2f} {away}")
    print(f"Classifier 90':  {home} {clf['p_home_90']:.1%}  |  draw {clf['p_draw_90']:.1%}  |  {away} {clf['p_away_90']:.1%}")
    print(f"Monte Carlo title prob:  {home} {mc['p_home_title']:.1%}  |  {away} {mc['p_away_title']:.1%}")
    print(f"Classifier title prob:   {home} {clf['p_home_title']:.1%}  |  {away} {clf['p_away_title']:.1%}")
    print(f"P(extra time) {mc['p_extra_time']:.1%}  |  P(penalties) {mc['p_penalties']:.1%}")
    print("Most likely 90' scores: " + ", ".join(f"{s['score']} ({s['prob']:.1%})" for s in mc["top_scores"]))
    print(f"\n>>> TITLE PREDICTION: {home} {title_home:.1%}  vs  {away} {title_away:.1%}  ->  {champion} champion")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "final_prediction.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    pd.concat([rating_history(feat_df, home), rating_history(feat_df, away)]).to_csv(
        RESULTS_DIR / "rating_history.csv", index=False, float_format="%.2f")
    print("Saved -> results/final_prediction.json")


if __name__ == "__main__":
    main()
