"""
Prediction pipeline for the 2026 World Cup FINAL:

    Spain vs Argentina  (East Rutherford, 2026-07-19)

Same methodology as the semifinals model, retrained with both semifinal
results included (cutoff 2026-07-19 — no result from the final itself can
leak). The title probability is built in three steps:

  1. 90 minutes: the trained W/D/L stacking classifier (logistic
     meta-model over logistic, CatBoost and Poisson base outputs).
  2. Knockout draw recalibration (knockout.py): knockouts are level after
     90' more often than the all-matches classifier predicts, so the draw
     odds are scaled by a factor fit on historical knockouts.
  3. Drawn mass -> winner: exact extra-time model (each side scores
     Poisson(phi * mu) with phi fit on real extra times, not the naive 1/3),
     and a still-level tie goes to a mildly Elo-tilted shootout coin.

The parameters of steps 2-3 come from results/knockout_calibration.json,
written by backtest_knockouts.py (fit only on knockouts played before the
final). That backtest is also why the Monte Carlo is no longer averaged
into the headline number: it scored worse than the classifier on 327
historical knockouts. The simulation (50,000 runs, same extra-time factor)
is kept only as a reference title estimate. The scoreline matrix is the
exact Poisson score distribution (no sampling noise), rescaled so its
win/draw/loss regions match step 2.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from combine import CLASSES, COMBINERS, combine_probs, poisson_wdl
from data_io import load_results, RESULTS_DIR
from features import build_dataset, fixture_features
from knockout import NAIVE_ET_FACTOR, advance_given_level, recalibrate_draw, title_prob
from train_model import _fit_poisson

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


def simulate_match(mu_home, mu_away, elo_diff_raw, n_sims=N_SIMS,
                   et_goal_factor=NAIVE_ET_FACTOR):
    home_reg = RNG.poisson(mu_home, size=n_sims)
    away_reg = RNG.poisson(mu_away, size=n_sims)

    home_wins = home_reg > away_reg
    away_wins = home_reg < away_reg
    draw_mask = home_reg == away_reg

    n_draw = int(draw_mask.sum())
    p_extra_time = n_draw / n_sims
    if n_draw:
        et_home = RNG.poisson(mu_home * et_goal_factor, size=n_draw)  # 30 extra minutes
        et_away = RNG.poisson(mu_away * et_goal_factor, size=n_draw)
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

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "p_90": {"H": float((home_reg > away_reg).mean()), "D": p_extra_time,
                 "A": float((home_reg < away_reg).mean())},
        "p_home_title": float(home_wins.mean()),
        "p_away_title": float(away_wins.mean()),
        "p_extra_time": p_extra_time,
        "p_penalties": p_penalties,
    }


def _base_signals(clf_bundle, feats: dict, mu_home: float, mu_away: float, rho: float = 0.0):
    """The three aligned base signals (logistic, CatBoost, Poisson W/D/L) and
    the match context the meta-model expects."""
    classes = clf_bundle["classes"]
    X = pd.DataFrame([{k: feats[k] for k in clf_bundle["features"]}])

    def aligned(model):
        probs = model.predict_proba(X)[0]
        return np.array([[probs[list(model.classes_).index(c)] for c in classes]])

    lr_p = aligned(clf_bundle["members"]["logistic"])
    cb_p = aligned(clf_bundle["members"]["catboost"])
    pois_p = poisson_wdl(mu_home, mu_away, rho=rho)
    context = np.array([[feats["neutral"], feats["k_weight"]]], dtype=float)
    return lr_p, cb_p, pois_p, context


def ensemble_class_probs(clf_bundle, feats: dict, mu_home: float, mu_away: float,
                         rho: float = 0.0) -> dict:
    """Apply the deployed combiner (see combine.py) to the three base signals."""
    out = combine_probs(clf_bundle["combiner"],
                        *_base_signals(clf_bundle, feats, mu_home, mu_away, rho),
                        clf_bundle["meta_model"])[0]
    return dict(zip(clf_bundle["classes"], out))


def method_breakdown(clf_bundle, feats: dict, mu_home: float, mu_away: float,
                     rho: float = 0.0) -> dict:
    """90-minute W/D/L probabilities from every base model and combiner."""
    lr_p, cb_p, pois_p, context = _base_signals(clf_bundle, feats, mu_home, mu_away, rho)
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


def load_knockout_calibration() -> dict:
    path = RESULTS_DIR / "knockout_calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run backtest_knockouts.py first")
    with open(path) as f:
        return json.load(f)


def knockout_title_prob(clf_bundle, feats: dict, elo_diff_raw: float,
                        mu_home: float, mu_away: float, calib: dict, rho: float = 0.0) -> dict:
    """Steps 1-3 of the module docstring for one knockout fixture."""
    class_probs = ensemble_class_probs(clf_bundle, feats, mu_home, mu_away, rho)
    raw = np.array([[class_probs[c] for c in CLASSES]])
    adj = recalibrate_draw(raw, calib["draw_odds_factor"])
    p_shoot = shootout_win_prob(elo_diff_raw)
    cond, p_level_et = advance_given_level(mu_home, mu_away, p_shoot, calib["et_goal_factor"])
    title_home = float(title_prob(adj, cond)[0])
    return {
        "raw_90": dict(zip(CLASSES, raw[0])),
        "adj_90": dict(zip(CLASSES, adj[0])),
        "p_home_given_level": float(cond[0]),
        "p_pens_given_et": float(p_level_et[0]),
        "p_home_shootout": p_shoot,
        "p_home_title": title_home,
        # previous method (raw draw, tilt-only resolution), kept for reference
        "p_home_title_unadjusted": float(raw[0][CLASSES.index("H")]
                                         + raw[0][CLASSES.index("D")] * p_shoot),
    }


def rescale_score_matrix(score_matrix: np.ndarray, p_90: dict, target: dict) -> np.ndarray:
    """Keep the Poisson scoreline shape inside each outcome region (home win
    = below the diagonal, draw = diagonal, away win = above) but give the
    regions the recalibrated W/D/L masses."""
    n = score_matrix.shape[0]
    i, j = np.indices((n, n))
    scale = np.select([i > j, i == j], [target["H"] / p_90["H"], target["D"] / p_90["D"]],
                      target["A"] / p_90["A"])
    return score_matrix * scale


def main():
    results = load_results()

    home, away = FINAL["home"], FINAL["away"]
    feat_df, elo, glicko, form = build_dataset(results, cutoff_date=CUTOFF)
    # same rolling window as the Poisson base the meta-model was trained on
    goal_model = _fit_poisson(results, pd.Timestamp(CUTOFF))
    clf_bundle = joblib.load(RESULTS_DIR / "classifier.joblib")

    feats = fixture_features(elo, glicko, form, home, away, neutral=True, match_date=FINAL["date"])
    elo_diff_raw = feats["home_elo"] - feats["away_elo"]
    mu_home = goal_model.expected_goals(home, away, neutral=True)
    mu_away = goal_model.expected_goals(away, home, neutral=True)

    calib = load_knockout_calibration()
    ko = knockout_title_prob(clf_bundle, feats, elo_diff_raw, mu_home, mu_away, calib,
                             goal_model.rho)
    mc = simulate_match(mu_home, mu_away, elo_diff_raw, et_goal_factor=calib["et_goal_factor"])
    p_90 = dict(zip(CLASSES, goal_model.wdl(mu_home, mu_away)[0]))
    score_matrix = rescale_score_matrix(goal_model.score_matrix(mu_home, mu_away, SCORE_GRID),
                                        p_90, ko["adj_90"])
    top = sorted(((score_matrix[h, a], h, a) for h in range(SCORE_GRID) for a in range(SCORE_GRID)),
                 reverse=True)[:5]
    top_scores = [{"score": f"{h} x {a}", "prob": round(float(p), 4)} for p, h, a in top]

    title_home = ko["p_home_title"]
    title_away = 1.0 - title_home
    champion = home if title_home >= 0.5 else away
    p_extra_time = ko["adj_90"]["D"]
    p_penalties = p_extra_time * ko["p_pens_given_et"]

    def wdl(p):
        return {"home_win": round(p["H"], 4), "draw": round(p["D"], 4), "away_win": round(p["A"], 4)}

    report = {
        f"{home} vs {away}": {
            "venue": FINAL["venue"],
            "date": FINAL["date"],
            "elo": {home: round(feats["home_elo"], 1), away: round(feats["away_elo"], 1)},
            "glicko2": {
                home: {"rating": round(feats["home_glicko"], 1), "rd": round(feats["home_glicko_rd"], 1)},
                away: {"rating": round(feats["away_glicko"], 1), "rd": round(feats["away_glicko_rd"], 1)},
            },
            "poisson_expected_goals": {home: round(mu_home, 2), away: round(mu_away, 2)},
            "classifier_90min_prob": wdl(ko["raw_90"]),
            "knockout_adjusted_90min_prob": wdl(ko["adj_90"]),
            "knockout_adjustment": {
                "draw_odds_factor": round(calib["draw_odds_factor"], 4),
                "et_goal_factor": round(calib["et_goal_factor"], 4),
                "p_home_shootout": round(ko["p_home_shootout"], 4),
                "p_home_given_level_after_90": round(ko["p_home_given_level"], 4),
                "p_penalties_given_extra_time": round(ko["p_pens_given_et"], 4),
            },
            "p_extra_time": round(p_extra_time, 4),
            "p_penalties": round(p_penalties, 4),
            "most_likely_scores_90min": top_scores,
            "score_matrix_90min": {
                "rows": f"{home} goals 0..{SCORE_GRID - 1}",
                "cols": f"{away} goals 0..{SCORE_GRID - 1}",
                "prob": np.round(score_matrix, 4).tolist(),
            },
            "method_breakdown_90min": method_breakdown(clf_bundle, feats, mu_home, mu_away,
                                                       goal_model.rho),
            "reference_title_prob": {
                "monte_carlo": {home: round(mc["p_home_title"], 4), away: round(mc["p_away_title"], 4)},
                "classifier_unadjusted": {home: round(ko["p_home_title_unadjusted"], 4),
                                          away: round(1 - ko["p_home_title_unadjusted"], 4)},
            },
            "final_title_probability": {home: round(title_home, 4), away: round(title_away, 4)},
            "predicted_champion": champion,
        }
    }

    raw, adj = ko["raw_90"], ko["adj_90"]
    print(f"=== WORLD CUP 2026 FINAL — {home} vs {away} ({FINAL['venue']}, {FINAL['date']}) ===")
    print(f"Elo:      {home} {feats['home_elo']:.0f}  |  {away} {feats['away_elo']:.0f}")
    print(f"Glicko-2: {home} {feats['home_glicko']:.0f} (RD {feats['home_glicko_rd']:.0f})"
          f"  |  {away} {feats['away_glicko']:.0f} (RD {feats['away_glicko_rd']:.0f})")
    print(f"Poisson expected goals: {home} {mu_home:.2f} - {mu_away:.2f} {away}")
    print(f"Classifier 90':          {home} {raw['H']:.1%}  |  draw {raw['D']:.1%}  |  {away} {raw['A']:.1%}")
    print(f"Knockout-adjusted 90':   {home} {adj['H']:.1%}  |  draw {adj['D']:.1%}  |  {away} {adj['A']:.1%}"
          f"   (draw odds x{calib['draw_odds_factor']:.2f})")
    print(f"If level after 90': {home} advances {ko['p_home_given_level']:.1%}"
          f"   (extra-time goal factor {calib['et_goal_factor']:.3f}, shootout {ko['p_home_shootout']:.1%})")
    print(f"P(extra time) {p_extra_time:.1%}  |  P(penalties) {p_penalties:.1%}")
    print("Most likely 90' scores: " + ", ".join(f"{s['score']} ({s['prob']:.1%})" for s in top_scores))
    print(f"Reference only - Monte Carlo title {home} {mc['p_home_title']:.1%}  |  "
          f"unadjusted classifier {home} {ko['p_home_title_unadjusted']:.1%}")
    print(f"\n>>> TITLE PREDICTION: {home} {title_home:.1%}  vs  {away} {title_away:.1%}  ->  {champion} champion")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "final_prediction.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    pd.concat([rating_history(feat_df, home), rating_history(feat_df, away)]).to_csv(
        RESULTS_DIR / "rating_history.csv", index=False, float_format="%.2f")
    print("Saved -> results/final_prediction.json")


if __name__ == "__main__":
    main()
