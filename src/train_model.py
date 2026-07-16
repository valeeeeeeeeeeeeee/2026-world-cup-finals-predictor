"""
Trains the outcome classifier (Win / Draw / Loss for the home side), compares
several ways of COMBINING the base methods on a held-out time slice, and
deploys the combiner with the lowest competitive-holdout log-loss.

Base models (all leak-free, recency-weighted where applicable):

  a. **Multinomial logistic regression** on the expanded feature set
     (Elo diff/level, Glicko-2 diff and RD gap, 10-match form, rest days,
     competition importance, neutral venue).
  b. **CatBoost** gradient boosting on the same features.
  c. **Dixon-Coles Poisson** W/D/L probabilities, derived from each side's
     expected goals (attack/defense strengths on a rolling ~11-year window)
     via the full score matrix. This base is only weakly correlated with
     the two rating-driven classifiers — the diversity a combiner needs.

Combiners evaluated (see combine.py):

  - fixed_blend_lr_cb : 0.65*logistic + 0.35*catboost (semifinals baseline)
  - equal_blend_3     : average of the three bases
  - stacking_v2       : logistic meta-model over the LOG-ODDS of the nine
                        base probabilities + match context (neutral,
                        k_weight), trained on temporal out-of-fold
                        predictions (expanding 2-year blocks from 2000;
                        every meta-training row is a genuine out-of-sample
                        prediction from all three bases)
  - stack_blend_avg   : average of stacking_v2 and the fixed blend

The champion combiner on the competitive holdout is deployed. Note: picking
the best of four on the holdout is mild selection; the candidates are few
and structurally motivated, but the winning margin should be read with that
grain of salt.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combine import CLASSES, COMBINERS, combine_probs, meta_matrix, poisson_wdl
from data_io import load_results, RESULTS_DIR
from features import build_dataset, FEATURE_COLS
from poisson_model import PoissonGoalModel

SPLIT_DATE = pd.Timestamp("2019-01-01")  # train < this <= holdout
RECENCY_HALF_LIFE_YEARS = 6.0
FRIENDLY_K = 20  # matches with k_weight above this are competitive
CUTOFF = pd.Timestamp("2026-07-19")  # eve of the final: semifinals included

META_BLOCKS_START = pd.Timestamp("2000-01-01")  # ~128 years of history before the 1st block
META_BLOCK_YEARS = 2
POISSON_WINDOW_YEARS = 11  # rolling window length for historical Poisson fits


def recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    return 0.5 ** (((reference - dates).dt.days / 365.25) / RECENCY_HALF_LIFE_YEARS)


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _catboost():
    return CatBoostClassifier(
        loss_function="MultiClass", iterations=400, depth=3, learning_rate=0.04,
        l2_leaf_reg=2.0, min_data_in_leaf=60, random_seed=42,
        verbose=0, allow_writing_files=False,
    )


def _meta():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _fit(model, df, reference, is_pipeline):
    sw = recency_weights(df["date"], reference).values
    if is_pipeline:
        model.fit(df[FEATURE_COLS], df["outcome"], logisticregression__sample_weight=sw)
    else:
        model.fit(df[FEATURE_COLS], df["outcome"], sample_weight=sw)
    return model


def _proba(model, df):
    """predict_proba re-ordered into CLASSES order."""
    p = model.predict_proba(df[FEATURE_COLS])
    idx = [list(model.classes_).index(c) for c in CLASSES]
    return p[:, idx]


def _poisson_probs(goal_model: PoissonGoalModel, df: pd.DataFrame) -> np.ndarray:
    """W/D/L probabilities for each row of a feature frame."""
    is_home = 1.0 - df["neutral"].to_numpy(dtype=float)
    mu_home = goal_model.expected_goals_batch(df["home_team"], df["away_team"], is_home)
    mu_away = goal_model.expected_goals_batch(df["away_team"], df["home_team"],
                                              np.zeros(len(df)))
    return poisson_wdl(mu_home, mu_away)


def _fit_poisson(results: pd.DataFrame, cutoff: pd.Timestamp) -> PoissonGoalModel:
    window_start = cutoff - pd.DateOffset(years=POISSON_WINDOW_YEARS)
    return PoissonGoalModel().fit(results, cutoff_date=cutoff, window_start=window_start)


def _context(df: pd.DataFrame) -> np.ndarray:
    return df[["neutral", "k_weight"]].to_numpy(dtype=float)


def build_oof_meta_dataset(results: pd.DataFrame, feat_df: pd.DataFrame, end: pd.Timestamp):
    """Out-of-fold base-model predictions for meta-model training.

    Expanding chronological windows: for each META_BLOCK_YEARS block between
    META_BLOCKS_START and `end`, all three base models are fit on data
    strictly before the block and predict only that block — every
    meta-training example is a genuine out-of-sample prediction.
    """
    lr_parts, cb_parts, pois_parts, ctx_parts, y_parts, date_parts = [], [], [], [], [], []
    for block_start in pd.date_range(META_BLOCKS_START, end, freq=f"{META_BLOCK_YEARS}YS"):
        block_end = min(block_start + pd.DateOffset(years=META_BLOCK_YEARS), end)
        train = feat_df[feat_df["date"] < block_start]
        block = feat_df[(feat_df["date"] >= block_start) & (feat_df["date"] < block_end)]
        if train.empty or block.empty:
            continue
        lr = _fit(_logistic(), train, block_start, is_pipeline=True)
        cb = _fit(_catboost(), train, block_start, is_pipeline=False)
        pois = _fit_poisson(results, block_start)

        lr_parts.append(_proba(lr, block))
        cb_parts.append(_proba(cb, block))
        pois_parts.append(_poisson_probs(pois, block))
        ctx_parts.append(_context(block))
        y_parts.append(block["outcome"].to_numpy())
        date_parts.append(block["date"].to_numpy())
    return {
        "lr": np.vstack(lr_parts),
        "cb": np.vstack(cb_parts),
        "pois": np.vstack(pois_parts),
        "context": np.vstack(ctx_parts),
        "y": np.concatenate(y_parts),
        "dates": pd.Series(pd.to_datetime(np.concatenate(date_parts))),
    }


def _fit_meta(oof: dict, mask: np.ndarray, reference: pd.Timestamp):
    X = meta_matrix(oof["lr"][mask], oof["cb"][mask], oof["pois"][mask], oof["context"][mask])
    sw = recency_weights(oof["dates"][mask], reference).values
    return _meta().fit(X, oof["y"][mask], logisticregression__sample_weight=sw)


def _metrics_from_proba(y, proba):
    return {
        "accuracy": accuracy_score(y, np.array(CLASSES)[proba.argmax(1)]),
        "log_loss": log_loss(y, proba, labels=CLASSES),
        "n": int(len(y)),
    }


def main():
    results = load_results()
    feat_df, elo, glicko, form = build_dataset(results, cutoff_date=str(CUTOFF.date()))

    train_df = feat_df[feat_df["date"] < SPLIT_DATE]
    holdout = feat_df[feat_df["date"] >= SPLIT_DATE]

    print("Building out-of-fold meta dataset (expanding chronological windows)...")
    oof = build_oof_meta_dataset(results, feat_df, end=CUTOFF)
    print(f"  meta-training rows: {len(oof['y'])} "
          f"({oof['dates'].min().date()} .. {oof['dates'].max().date()})")

    # --- Evaluation: bases fit on < SPLIT_DATE; meta fit only on OOF rows
    # from < SPLIT_DATE. The holdout never touches any fit.
    lr = _fit(_logistic(), train_df, SPLIT_DATE, is_pipeline=True)
    cb = _fit(_catboost(), train_df, SPLIT_DATE, is_pipeline=False)
    pois = _fit_poisson(results, SPLIT_DATE)
    meta_eval = _fit_meta(oof, (oof["dates"] < SPLIT_DATE).to_numpy(), SPLIT_DATE)

    ho_lr, ho_cb = _proba(lr, holdout), _proba(cb, holdout)
    ho_pois, ho_ctx = _poisson_probs(pois, holdout), _context(holdout)
    comp_mask = (holdout["k_weight"] > FRIENDLY_K).to_numpy()

    def block(proba):
        return {
            "all_matches": _metrics_from_proba(holdout["outcome"], proba),
            "competitive": _metrics_from_proba(holdout["outcome"][comp_mask], proba[comp_mask]),
        }

    candidates = {
        "logistic": block(ho_lr),
        "catboost": block(ho_cb),
        "poisson_wdl": block(ho_pois),
    }
    for name in COMBINERS:
        candidates[name] = block(
            combine_probs(name, ho_lr, ho_cb, ho_pois, ho_ctx, meta_eval))

    deployed = min(COMBINERS, key=lambda n: candidates[n]["competitive"]["log_loss"])

    prior = train_df["outcome"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_ll = log_loss(holdout["outcome"], np.tile(prior, (len(holdout), 1)), labels=CLASSES)

    metrics = {
        "n_train": int(len(train_df)),
        "n_meta_train_oof": int(len(oof["y"])),
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "features": FEATURE_COLS,
        "meta_features": "log-odds of 9 base probs (logistic, catboost, poisson) + neutral + k_weight",
        "meta_oof_scheme": f"expanding windows, {META_BLOCK_YEARS}-year blocks from {META_BLOCKS_START.date()}",
        "naive_baseline_log_loss_all": naive_ll,
        "deployed": deployed,
        "candidates_holdout": candidates,
    }
    print("\nCompetitive holdout log-loss by candidate:")
    for name, cand in candidates.items():
        mark = "  <- deployed" if name == deployed else ""
        print(f"  {name:20s} {cand['competitive']['log_loss']:.4f}{mark}")

    # --- Deploy: refit all bases on ALL history up to the eve of the final;
    # meta on the full OOF dataset (recency-weighted to the cutoff).
    lr_full = _fit(_logistic(), feat_df, CUTOFF, is_pipeline=True)
    cb_full = _fit(_catboost(), feat_df, CUTOFF, is_pipeline=False)
    meta_full = _fit_meta(oof, np.ones(len(oof["y"]), dtype=bool), CUTOFF)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(
        {
            "combiner": deployed,
            "members": {"logistic": lr_full, "catboost": cb_full},
            "meta_model": meta_full,
            "features": FEATURE_COLS,
            "classes": CLASSES,
            "name": deployed,
        },
        RESULTS_DIR / "classifier.joblib",
    )
    print(f"\nDeployed combiner: {deployed} -> results/classifier.joblib")


if __name__ == "__main__":
    main()
