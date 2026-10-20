"""
Backtest of the title-probability methods on historical KNOCKOUT matches.

The final's headline number averages two ways of turning a 90-minute
forecast into "probability of advancing" (see predict_final.py):

  - classifier : stacking v2 W/D/L, draw mass reassigned with the Elo-tilted
                 shootout coin (extra time is not modeled separately)
  - monte_carlo: Poisson goals for 90', extra time at 1/3 of the rates,
                 then the same shootout coin
  - ensemble   : average of the two (what predict_final.py publishes)

The W/D/L holdout cannot validate this step (it has no extra time or
penalties), so this script replays real knockout ties and scores each
method on who actually advanced. Two extra variants isolate where the
methods differ:

  - hybrid     : stacking v2 for the 90 minutes + the Monte Carlo's
                 extra-time model for the drawn mass (computed exactly)
  - calibrated : the production method (knockout.py): hybrid with the
                 knockout draw recalibration and the fitted extra-time goal
                 factor. Both parameters are re-estimated before every
                 edition from earlier knockouts only (expanding window), and
                 the final values -- fit on knockouts before the 2026 final
                 -- are saved to results/knockout_calibration.json for
                 predict_final.py.
  - elo_only   : plain Elo expected score (reference baseline)

Knockout matches: the last N matches of each edition of the World Cup,
Euro, Africa Cup of Nations and Asian Cup, with N taken from the known
tournament format (editions whose match count matches no format are
skipped, and the group stage must end before the first knockout date).
Extra time: a penalty shootout, or a goal after minute 90 in a match that
was level at 90 (goalscorers.csv records stoppage-time goals as 45/90, so
mislabeling is rare).

Leak-free: for each edition, every model is refit on data strictly before
its first knockout match (meta-model on the out-of-fold rows before that
date). Features (Elo, Glicko-2, form) are computed match by match, so they
include the group stage. Stacking needs out-of-fold meta rows, which start
in 2000, so the full-model test covers editions from 2004 on; the
extra-time-only tests need just Elo + Poisson and go back to 1986.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from combine import CLASSES, combine_probs
from data_io import load_goalscorers, load_results, load_shootouts, RESULTS_DIR
from elo import HOME_ADVANTAGE
from features import build_dataset
from knockout import (advance_given_level, fit_draw_odds_factor, fit_et_goal_factor,
                      recalibrate_draw, title_prob)
from predict_final import shootout_win_prob, simulate_match
from train_model import (CUTOFF, _catboost, _context, _fit, _fit_meta, _fit_poisson,
                         _logistic, _poisson_probs, _proba, build_oof_meta_dataset)

# tournament -> {matches in edition: knockout matches at the end of it}
TOURNAMENT_FORMATS = {
    "FIFA World Cup": {52: 16, 64: 16, 104: 32},
    "UEFA Euro": {31: 7, 51: 15},
    "African Cup of Nations": {32: 8, 52: 16},
    "AFC Asian Cup": {51: 15},
}
EDITION_GAP_DAYS = 60
FULL_MODEL_FROM = pd.Timestamp("2004-01-01")  # >= 4 years of OOF meta rows
ET_TEST_FROM = pd.Timestamp("1986-01-01")
ET_GOAL_FIT_FROM = pd.Timestamp("2005-01-01")  # after golden/silver goal was abolished
FEATURES_CUTOFF = "2026-07-20"  # include the 2026 final itself as a test match
N_BOOT = 5000
BOOT_RNG = np.random.default_rng(11)
KEY = ["date", "home_team", "away_team"]
METHODS = ["classifier", "monte_carlo", "ensemble", "hybrid", "calibrated", "elo_only"]


def knockout_matches(results: pd.DataFrame) -> pd.DataFrame:
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    parts, skipped = [], []
    for tournament, formats in TOURNAMENT_FORMATS.items():
        t = df[df["tournament"] == tournament].sort_values("date")
        edition_id = (t["date"].diff().dt.days > EDITION_GAP_DAYS).cumsum()
        for _, ed in t.groupby(edition_id):
            name = f"{tournament} {ed['date'].iloc[0].year}"
            n_ko = formats.get(len(ed))
            if n_ko is None:
                skipped.append(name)
                continue
            ko, group = ed.iloc[-n_ko:], ed.iloc[:-n_ko]
            if group["date"].max() >= ko["date"].min():
                skipped.append(name)
                continue
            parts.append(ko.assign(edition=name, ko_start=ko["date"].min()))
    return pd.concat(parts, ignore_index=True), skipped


def annotate_outcomes(ko: pd.DataFrame, shootouts: pd.DataFrame,
                      goals: pd.DataFrame) -> pd.DataFrame:
    """Adds: pens, went_et (NaN when goal data is incomplete), et_goals (goals
    scored in extra time, NaN unless known) and home_advanced."""
    so = shootouts.assign(date=pd.to_datetime(shootouts["date"]))[KEY + ["winner"]]
    ko = ko.merge(so, on=KEY, how="left")
    ko["pens"] = ko["winner"].notna()

    g = goals.assign(date=pd.to_datetime(goals["date"]))
    g = g[g["minute"].notna()].merge(ko[KEY], on=KEY)
    g["side"] = np.where(g["team"] == g["home_team"], "home", "away")
    reg = g[g["minute"] <= 90].pivot_table(index=KEY, columns="side", values="minute",
                                           aggfunc="count", fill_value=0)
    tot = g.pivot_table(index=KEY, columns="side", values="minute", aggfunc="count",
                        fill_value=0)
    late = g[g["minute"] > 90].groupby(KEY).size().rename("n_late")
    stats = (reg.reindex(columns=["home", "away"], fill_value=0).add_prefix("reg_")
             .join(tot.reindex(columns=["home", "away"], fill_value=0).add_prefix("tot_"),
                   how="outer")
             .join(late, how="outer").fillna(0).reset_index())
    # a match with no goal rows is complete if it ended 0-0
    stat_cols = ["reg_home", "reg_away", "tot_home", "tot_away", "n_late"]
    ko = ko.merge(stats, on=KEY, how="left")
    ko[stat_cols] = ko[stat_cols].fillna(0)

    complete = ((ko["tot_home"] == ko["home_score"]) & (ko["tot_away"] == ko["away_score"]))
    et_goal = (ko["n_late"] > 0) & (ko["reg_home"] == ko["reg_away"])
    ko["went_et"] = np.where(ko["pens"], 1.0, np.where(complete, et_goal.astype(float), np.nan))
    late_goals = ko["tot_home"] + ko["tot_away"] - ko["reg_home"] - ko["reg_away"]
    ko["et_goals"] = np.where(complete & (ko["went_et"] == 1), late_goals, np.nan)

    level = ko["home_score"] == ko["away_score"]
    ko["home_advanced"] = np.where(level, ko["winner"] == ko["home_team"],
                                   ko["home_score"] > ko["away_score"]).astype(float)
    # a level final score with no recorded shootout cannot be resolved
    return ko[~(level & ~ko["pens"])].reset_index(drop=True)


def predict_edition(ed: pd.DataFrame, results, feat_df, oof, full_model: bool) -> pd.DataFrame:
    cutoff = ed["ko_start"].iloc[0]
    rows = ed.drop(columns=["neutral"]).merge(feat_df, on=KEY, how="inner")
    pois = _fit_poisson(results, cutoff)
    is_home = 1.0 - rows["neutral"].to_numpy(dtype=float)
    mu_h = pois.expected_goals_batch(rows["home_team"], rows["away_team"], is_home)
    mu_a = pois.expected_goals_batch(rows["away_team"], rows["home_team"], np.zeros(len(rows)))
    elo_raw = (rows["home_elo"] - rows["away_elo"]).to_numpy()
    tilt = shootout_win_prob(elo_raw)
    cond, p_pens_given_et = advance_given_level(mu_h, mu_a, tilt)
    pois_p = pois.wdl(mu_h, mu_a)

    out = rows[KEY + ["edition", "tournament", "ko_start", "neutral", "pens", "went_et",
                      "et_goals", "home_advanced"]].copy()
    out["mu_home"], out["mu_away"] = mu_h, mu_a
    out["p_level90_poisson"] = pois_p[:, CLASSES.index("D")]
    out["p_pens_given_et"] = p_pens_given_et
    out["cond_tilt"] = tilt
    out["cond_mc"] = cond
    out["monte_carlo"] = [simulate_match(h, a, d)["p_home_title"]
                          for h, a, d in zip(mu_h, mu_a, elo_raw)]
    elo_adv = elo_raw + HOME_ADVANTAGE * is_home
    out["elo_only"] = 1.0 / (1.0 + 10 ** (-elo_adv / 400.0))

    if full_model:
        train = feat_df[feat_df["date"] < cutoff]
        lr = _fit(_logistic(), train, cutoff, is_pipeline=True)
        cb = _fit(_catboost(), train, cutoff, is_pipeline=False)
        meta = _fit_meta(oof, (oof["dates"] < cutoff).to_numpy(), cutoff)
        stack = combine_probs("stacking_v2", _proba(lr, rows), _proba(cb, rows),
                              _poisson_probs(pois, rows), _context(rows), meta)
        for j, c in enumerate(CLASSES):
            out[f"stack_{c}"] = stack[:, j]
        pH, pD = stack[:, CLASSES.index("H")], stack[:, CLASSES.index("D")]
        out["p_level90_stacking"] = pD
        out["classifier"] = pH + pD * tilt
        out["hybrid"] = pH + pD * cond
        out["ensemble"] = (out["classifier"] + out["monte_carlo"]) / 2.0
    return out


def fit_knockout_params(prior: pd.DataFrame) -> tuple[float, float, int, int]:
    """(draw odds factor, extra-time goal factor, n rows behind each) from
    knockouts already played."""
    d = prior[prior["went_et"].notna() & prior["p_level90_stacking"].notna()]
    e = prior[(prior["went_et"] == 1) & prior["et_goals"].notna()
              & (prior["date"] >= ET_GOAL_FIT_FROM)]
    lam = fit_draw_odds_factor(d["p_level90_stacking"], d["went_et"])
    phi = fit_et_goal_factor(e["et_goals"], e["mu_home"] + e["mu_away"])
    return lam, phi, len(d), len(e)


def add_calibrated(bt: pd.DataFrame) -> pd.DataFrame:
    """Production method with expanding-window parameters: every edition uses
    only knockouts played before its first knockout match."""
    bt = bt.copy()
    for col in ("draw_odds_factor", "et_goal_factor", "calibrated", "cond_calibrated",
                "p_level90_calibrated", "p_pens_given_et_calibrated"):
        bt[col] = np.nan
    for cutoff, idx in bt.groupby("ko_start").groups.items():
        lam, phi, _, _ = fit_knockout_params(bt[bt["date"] < cutoff])
        ed = bt.loc[idx]
        tilt = ed["cond_tilt"].to_numpy()
        cond, p_level = advance_given_level(ed["mu_home"].to_numpy(), ed["mu_away"].to_numpy(),
                                            tilt, phi)
        bt.loc[idx, ["draw_odds_factor", "et_goal_factor"]] = [lam, phi]
        bt.loc[idx, "cond_calibrated"] = cond
        bt.loc[idx, "p_pens_given_et_calibrated"] = p_level
        if ed["stack_D"].notna().all():
            probs = recalibrate_draw(ed[[f"stack_{c}" for c in CLASSES]].to_numpy(), lam)
            bt.loc[idx, "p_level90_calibrated"] = probs[:, CLASSES.index("D")]
            bt.loc[idx, "calibrated"] = title_prob(probs, cond)
    return bt


def _binary_ll(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _scores(df: pd.DataFrame, cols, target: str, reference: str) -> dict:
    y = df[target].to_numpy()
    idx = BOOT_RNG.integers(0, len(df), size=(N_BOOT, len(df)))
    ref_ll = _binary_ll(df[reference], y)
    out = {}
    for c in cols:
        ll = _binary_ll(df[c], y)
        diff = (ll - ref_ll)[idx].mean(axis=1)
        out[c] = {
            "log_loss": float(ll.mean()),
            "brier": float(((df[c] - y) ** 2).mean()),
            "accuracy": float(((df[c] > 0.5) == (y == 1)).mean()),
            "mean_pred": float(df[c].mean()),
            f"diff_vs_{reference}": float((ll - ref_ll).mean()),
            "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        }
    return {"n": int(len(df)), "observed_rate": float(y.mean()), "methods": out}


def main():
    results = load_results()
    ko, skipped = knockout_matches(results)
    ko = annotate_outcomes(ko, load_shootouts(), load_goalscorers())
    ko = ko[ko["ko_start"] >= ET_TEST_FROM]

    print("Building features and out-of-fold meta dataset...")
    feat_df, *_ = build_dataset(results, cutoff_date=FEATURES_CUTOFF)
    oof = build_oof_meta_dataset(results, feat_df, end=CUTOFF)

    preds = []
    for name, ed in ko.groupby("edition", sort=False):
        full = ed["ko_start"].iloc[0] >= FULL_MODEL_FROM
        print(f"  {name:32s} {len(ed):3d} knockout matches{'' if full else '  (ET tests only)'}")
        preds.append(predict_edition(ed, results, feat_df, oof, full_model=full))
    bt = add_calibrated(pd.concat(preds, ignore_index=True))

    # deployed parameters: every knockout before the 2026 final
    lam, phi, n_lam, n_phi = fit_knockout_params(bt[bt["date"] < CUTOFF])
    calibration = {
        "draw_odds_factor": lam, "n_knockouts_draw_fit": n_lam,
        "et_goal_factor": phi, "n_extra_times_goal_fit": n_phi,
        "naive_et_goal_factor": 1.0 / 3.0,
        "fit_on": f"knockouts before {CUTOFF.date()} (draw: 2004+, extra-time goals: "
                  f"{ET_GOAL_FIT_FROM.year}+)",
    }

    full = bt[bt["classifier"].notna()]
    et_known = bt[bt["went_et"].notna()]
    et = et_known[et_known["went_et"] == 1]

    report = {
        "editions_full_model": sorted(full["edition"].unique().tolist()),
        "editions_et_tests": sorted(et_known["edition"].unique().tolist()),
        "editions_skipped_unknown_format": skipped,
        "advance_all_knockouts": _scores(full, METHODS, "home_advanced", "ensemble"),
        "advance_et_matches_full_model": _scores(full[full["went_et"] == 1], METHODS,
                                                 "home_advanced", "ensemble"),
        "level_after_90": _scores(full[full["went_et"].notna()],
                                  ["p_level90_stacking", "p_level90_poisson",
                                   "p_level90_calibrated"],
                                  "went_et", "p_level90_stacking"),
        "advance_given_extra_time": _scores(et.assign(coin=0.5),
                                            ["cond_tilt", "cond_mc", "cond_calibrated", "coin"],
                                            "home_advanced", "cond_mc"),
        "penalties_given_extra_time": _scores(et, ["p_pens_given_et",
                                                   "p_pens_given_et_calibrated"],
                                              "pens", "p_pens_given_et"),
        "deployed_calibration": calibration,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    bt.to_csv(RESULTS_DIR / "knockout_backtest.csv", index=False, float_format="%.4f")
    with open(RESULTS_DIR / "knockout_backtest.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(RESULTS_DIR / "knockout_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    def show(title, block, ref):
        print(f"\n{title}  (n = {block['n']}, observed rate {block['observed_rate']:.1%})")
        for m, s in block["methods"].items():
            lo, hi = s["diff_ci95"]
            print(f"  {m:20s} log-loss {s['log_loss']:.4f}  brier {s['brier']:.4f}  "
                  f"acc {s['accuracy']:.1%}  mean pred {s['mean_pred']:.1%}  "
                  f"vs {ref}: {s[f'diff_vs_{ref}']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    show("Who advances - all knockout matches (2004+)", report["advance_all_knockouts"], "ensemble")
    show("Who advances - knockouts that went to extra time (2004+)",
         report["advance_et_matches_full_model"], "ensemble")
    show("P(level after 90') - knockouts 2004+", report["level_after_90"], "p_level90_stacking")
    show("Who advances GIVEN extra time (1986+)", report["advance_given_extra_time"], "cond_mc")
    show("P(penalties | extra time) (1986+)", report["penalties_given_extra_time"],
         "p_pens_given_et")
    print(f"\nDeployed: draw odds factor {lam:.3f} (n = {n_lam}), "
          f"extra-time goal factor {phi:.3f} (n = {n_phi}; naive 0.333)")
    print("Saved -> results/knockout_backtest.json, results/knockout_backtest.csv, "
          "results/knockout_calibration.json")


if __name__ == "__main__":
    main()
