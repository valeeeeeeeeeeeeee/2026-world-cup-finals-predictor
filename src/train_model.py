"""
Trains the outcome classifier (Win / Draw / Loss for the home side) on the
full chronological feature set produced by features.py, evaluates the
candidates on a held-out time slice, and saves the deployed model + a
metrics report.

Methodology (evolved from the 2026_World_Cup_Semifinals model — the fixed
0.65/0.35 blend is replaced by **stacking**):

  1. **Recency weighting.** Training matches are weighted by an exponential
     decay on their age (6-year half-life), so current squad strength
     dominates and 40-year-old results barely count.
  2. **Expanded, leak-free features** (Elo diff *and* level, Glicko-2 diff
     and rating-deviation gap, goals-for / goals-against form, rest days,
     competition importance).
  3. **Stacking.** A multinomial logistic **meta-model** learns how to
     combine the two base models (logistic regression + CatBoost gradient
     boosting) from their probability outputs. To train it without leakage, base
     models are refit on expanding chronological windows (2-year blocks
     from 2000 onward) and predict only the block *after* their training
     window — classic out-of-fold stacking, adapted to time-ordered data:
     every meta-training example is a genuine out-of-sample prediction.
     The meta-model's 6 inputs are the two base models' class
     probabilities; its training rows are recency-weighted like everything
     else.

Metrics are reported on ALL validation matches and on the **competitive**
subset (real tournaments) — the domain a World Cup final lives in. The
fixed 0.65/0.35 blend is kept as a reported baseline for comparison.
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

from data_io import load_results, RESULTS_DIR
from features import build_dataset, FEATURE_COLS

CLASSES = ["A", "D", "H"]
SPLIT_DATE = pd.Timestamp("2019-01-01")  # train < this <= holdout
RECENCY_HALF_LIFE_YEARS = 6.0
FRIENDLY_K = 20  # matches with k_weight above this are competitive
BLEND_WEIGHTS = {"logistic": 0.65, "gradient_boosting": 0.35}  # baseline only
CUTOFF = pd.Timestamp("2026-07-19")  # eve of the final: semifinals included

META_BLOCKS_START = pd.Timestamp("2000-01-01")  # ~128 years of history before the 1st block
META_BLOCK_YEARS = 2


def recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    return 0.5 ** (((reference - dates).dt.days / 365.25) / RECENCY_HALF_LIFE_YEARS)


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _gbm():
    return CatBoostClassifier(
        loss_function="MultiClass", iterations=400, depth=3, learning_rate=0.04,
        l2_leaf_reg=2.0, min_data_in_leaf=60, random_seed=42,
        verbose=0, allow_writing_files=False,
    )


def _meta():
    return LogisticRegression(max_iter=3000, C=1.0)


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


def _align(model, p):
    """Single predict_proba matrix re-ordered into CLASSES order."""
    idx = [list(model.classes_).index(c) for c in CLASSES]
    return p[:, idx]


def build_oof_meta_dataset(feat_df: pd.DataFrame, end: pd.Timestamp):
    """Out-of-fold base-model predictions for meta-model training.

    Expanding chronological windows: for each META_BLOCK_YEARS block between
    META_BLOCKS_START and `end`, both base models are fit on everything
    strictly before the block (recency-weighted to the block start) and
    predict only that block. No block is ever predicted by a model that saw
    it — the meta-model trains exclusively on out-of-sample outputs.
    """
    X_parts, y_parts, date_parts = [], [], []
    for block_start in pd.date_range(META_BLOCKS_START, end, freq=f"{META_BLOCK_YEARS}YS"):
        block_end = min(block_start + pd.DateOffset(years=META_BLOCK_YEARS), end)
        train = feat_df[feat_df["date"] < block_start]
        block = feat_df[(feat_df["date"] >= block_start) & (feat_df["date"] < block_end)]
        if train.empty or block.empty:
            continue
        lr = _fit(_logistic(), train, block_start, is_pipeline=True)
        gb = _fit(_gbm(), train, block_start, is_pipeline=False)
        X_parts.append(np.hstack([_proba(lr, block), _proba(gb, block)]))
        y_parts.append(block["outcome"].to_numpy())
        date_parts.append(block["date"].to_numpy())
    return (
        np.vstack(X_parts),
        np.concatenate(y_parts),
        pd.Series(pd.to_datetime(np.concatenate(date_parts))),
    )


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
    holdout_comp = holdout[holdout["k_weight"] > FRIENDLY_K]

    # --- Out-of-fold meta dataset over the full pre-cutoff history.
    # Blocks before SPLIT_DATE are identical whether built to SPLIT_DATE or
    # to CUTOFF (each block only sees data before itself), so build once and
    # slice for the leak-free evaluation fit.
    print("Building out-of-fold meta dataset (expanding chronological windows)...")
    meta_X, meta_y, meta_dates = build_oof_meta_dataset(feat_df, end=CUTOFF)
    print(f"  meta-training rows: {len(meta_y)} "
          f"({meta_dates.min().date()} .. {meta_dates.max().date()})")

    # --- Evaluation: base models fit on < SPLIT_DATE; meta-model fit only on
    # OOF rows from < SPLIT_DATE. Holdout never touches either fit.
    lr = _fit(_logistic(), train_df, SPLIT_DATE, is_pipeline=True)
    gb = _fit(_gbm(), train_df, SPLIT_DATE, is_pipeline=False)

    eval_mask = (meta_dates < SPLIT_DATE).to_numpy()
    meta_eval = _meta().fit(
        meta_X[eval_mask], meta_y[eval_mask],
        sample_weight=recency_weights(meta_dates[eval_mask], SPLIT_DATE).values,
    )

    def blend_proba(df):
        return (BLEND_WEIGHTS["logistic"] * _proba(lr, df)
                + BLEND_WEIGHTS["gradient_boosting"] * _proba(gb, df))

    def stack_proba(df):
        X = np.hstack([_proba(lr, df), _proba(gb, df)])
        return _align(meta_eval, meta_eval.predict_proba(X))

    def block(model_proba_fn):
        return {
            "all_matches": _metrics_from_proba(holdout["outcome"], model_proba_fn(holdout)),
            "competitive": _metrics_from_proba(holdout_comp["outcome"], model_proba_fn(holdout_comp)),
        }

    candidates = {
        "logistic": block(lambda d: _proba(lr, d)),
        "gradient_boosting": block(lambda d: _proba(gb, d)),
        "fixed_blend_065_035": block(blend_proba),
        "stacking_logistic_meta": block(stack_proba),
    }

    prior = train_df["outcome"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_ll = log_loss(holdout["outcome"], np.tile(prior, (len(holdout), 1)), labels=CLASSES)

    metrics = {
        "n_train": int(len(train_df)),
        "n_meta_train_oof": int(len(meta_y)),
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "features": FEATURE_COLS,
        "meta_features": "P(A),P(D),P(H) from logistic + P(A),P(D),P(H) from GBM",
        "meta_oof_scheme": f"expanding windows, {META_BLOCK_YEARS}-year blocks from {META_BLOCKS_START.date()}",
        "naive_baseline_log_loss_all": naive_ll,
        "deployed": "stacking_logistic_meta",
        "candidates_holdout": candidates,
    }
    print("\nCompetitive holdout log-loss by candidate:")
    for name in ("logistic", "gradient_boosting", "fixed_blend_065_035", "stacking_logistic_meta"):
        ll = candidates[name]["competitive"]["log_loss"]
        mark = "  <- deployed" if name == "stacking_logistic_meta" else ""
        print(f"  {name:24s} {ll:.4f}{mark}")

    # --- Deploy: refit base models on ALL history up to the eve of the
    # final; refit the meta-model on the full OOF dataset (recency-weighted
    # to the cutoff).
    lr_full = _fit(_logistic(), feat_df, CUTOFF, is_pipeline=True)
    gb_full = _fit(_gbm(), feat_df, CUTOFF, is_pipeline=False)
    meta_full = _meta().fit(
        meta_X, meta_y, sample_weight=recency_weights(meta_dates, CUTOFF).values,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(
        {
            "members": [
                {"name": "logistic", "model": lr_full},
                {"name": "gradient_boosting", "model": gb_full},
            ],
            "meta_model": meta_full,
            "features": FEATURE_COLS,
            "classes": CLASSES,
            "name": "stacking_logistic_meta",
        },
        RESULTS_DIR / "classifier.joblib",
    )
    print("\nDeployed: stacking (logistic meta-model over logistic + GBM outputs)"
          " -> results/classifier.joblib")


if __name__ == "__main__":
    main()
