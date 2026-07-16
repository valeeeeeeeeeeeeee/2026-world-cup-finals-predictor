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
  2. The trained Win/Draw/Loss ensemble classifier's 90-minute
     probabilities, with the drawn mass reassigned to each side using the
     same shootout-tilt logic (a final cannot end level).

The reported title probability is the average of the two estimates.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from data_io import load_results, RESULTS_DIR
from features import build_dataset, fixture_features
from poisson_model import PoissonGoalModel

CUTOFF = "2026-07-19"
N_SIMS = 50_000
RNG = np.random.default_rng(42)

FINAL = {"home": "Spain", "away": "Argentina",
         "venue": "East Rutherford", "date": "2026-07-19"}


def shootout_win_prob(elo_diff_raw: float) -> float:
    """Penalty shootouts are close to a coin flip; skill still nudges it.
    Dampened logistic (divisor 3x the normal Elo scale of 400)."""
    return 1.0 / (1.0 + 10 ** (-elo_diff_raw / 1200.0))


def simulate_match(goal_model, home, away, elo_diff_raw, n_sims=N_SIMS):
    mu_home = goal_model.expected_goals(home, away, neutral=True)
    mu_away = goal_model.expected_goals(away, home, neutral=True)

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

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "p_home_title": float(home_wins.mean()),
        "p_away_title": float(away_wins.mean()),
        "p_extra_time": p_extra_time,
        "p_penalties": p_penalties,
        "top_scores": top_scores,
    }


def ensemble_class_probs(clf_bundle, feats: dict) -> dict:
    """Weighted blend of each member's predict_proba, aligned to class labels."""
    classes = clf_bundle["classes"]
    X = pd.DataFrame([{k: feats[k] for k in clf_bundle["features"]}])
    blended = np.zeros(len(classes))
    for member in clf_bundle["members"]:
        model = member["model"]
        probs = model.predict_proba(X)[0]
        aligned = np.array([probs[list(model.classes_).index(c)] for c in classes])
        blended += member["weight"] * aligned
    blended /= blended.sum()
    return dict(zip(classes, blended))


def classifier_title_prob(clf_bundle, feats: dict, elo_diff_raw: float):
    class_probs = ensemble_class_probs(clf_bundle, feats)
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
    _, elo, form = build_dataset(results, cutoff_date=CUTOFF)
    goal_model = PoissonGoalModel().fit(results, cutoff_date=CUTOFF)
    clf_bundle = joblib.load(RESULTS_DIR / "classifier.joblib")

    feats = fixture_features(elo, form, home, away, neutral=True, match_date=FINAL["date"])
    elo_diff_raw = feats["home_elo"] - feats["away_elo"]

    mc = simulate_match(goal_model, home, away, elo_diff_raw)
    clf = classifier_title_prob(clf_bundle, feats, elo_diff_raw)

    title_home = (mc["p_home_title"] + clf["p_home_title"]) / 2.0
    title_away = 1.0 - title_home
    champion = home if title_home >= 0.5 else away

    report = {
        f"{home} vs {away}": {
            "venue": FINAL["venue"],
            "date": FINAL["date"],
            "elo": {home: round(feats["home_elo"], 1), away: round(feats["away_elo"], 1)},
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
            "final_title_probability": {home: round(title_home, 4), away: round(title_away, 4)},
            "predicted_champion": champion,
        }
    }

    print(f"=== WORLD CUP 2026 FINAL — {home} vs {away} ({FINAL['venue']}, {FINAL['date']}) ===")
    print(f"Elo:  {home} {feats['home_elo']:.0f}  |  {away} {feats['away_elo']:.0f}")
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
    print("Saved -> results/final_prediction.json")


if __name__ == "__main__":
    main()
