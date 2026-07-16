"""
Trains the outcome classifier (Win / Draw / Loss for the home side) on the
full chronological feature set produced by features.py, evaluates the
candidates on a held-out time slice, and saves the deployed model + a
metrics report.

Methodology (same as the 2026_World_Cup_Semifinals model, retrained with
both semifinal results included — cutoff moved to the eve of the final):

  1. **Recency weighting.** Training matches are weighted by an exponential
     decay on their age (6-year half-life), so current squad strength
     dominates and 40-year-old results barely count.
  2. **Expanded, leak-free features** (Elo diff *and* level, goals-for /
     goals-against form, rest days, competition importance).
  3. **Ensembling.** The deployed classifier is a fixed-weight blend of a
     multinomial logistic regression and a gradient-boosting model:

         P = 0.65 * logistic + 0.35 * gradient_boosting

     The weight is NOT tuned on the holdout; it is a fixed, round value
     validated (in the semifinals project) to beat logistic-alone across
     five independent time folds. Logistic is kept dominant because it
     extrapolates linearly to top-of-distribution finalists, where the
     tree model cannot.

Metrics are reported on ALL validation matches and on the **competitive**
subset (real tournaments) — the domain a World Cup final lives in.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
ENSEMBLE_WEIGHTS = {"logistic": 0.65, "gradient_boosting": 0.35}
CUTOFF = pd.Timestamp("2026-07-19")  # eve of the final: semifinals included


def recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    return 0.5 ** (((reference - dates).dt.days / 365.25) / RECENCY_HALF_LIFE_YEARS)


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=3, learning_rate=0.04,
        l2_regularization=2.0, min_samples_leaf=60, random_state=42,
    )


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


def _metrics_from_proba(y, proba):
    return {
        "accuracy": accuracy_score(y, np.array(CLASSES)[proba.argmax(1)]),
        "log_loss": log_loss(y, proba, labels=CLASSES),
        "n": int(len(y)),
    }


def main():
    results = load_results()
    feat_df, elo, form = build_dataset(results, cutoff_date=str(CUTOFF.date()))

    train_df = feat_df[feat_df["date"] < SPLIT_DATE]
    holdout = feat_df[feat_df["date"] >= SPLIT_DATE]
    holdout_comp = holdout[holdout["k_weight"] > FRIENDLY_K]

    lr = _fit(_logistic(), train_df, SPLIT_DATE, is_pipeline=True)
    gb = _fit(_gbm(), train_df, SPLIT_DATE, is_pipeline=False)

    def ensemble_proba(df):
        return (ENSEMBLE_WEIGHTS["logistic"] * _proba(lr, df)
                + ENSEMBLE_WEIGHTS["gradient_boosting"] * _proba(gb, df))

    def block(model_proba_fn):
        return {
            "all_matches": _metrics_from_proba(holdout["outcome"], model_proba_fn(holdout)),
            "competitive": _metrics_from_proba(holdout_comp["outcome"], model_proba_fn(holdout_comp)),
        }

    candidates = {
        "logistic": block(lambda d: _proba(lr, d)),
        "gradient_boosting": block(lambda d: _proba(gb, d)),
        "ensemble_logistic_gbm": block(ensemble_proba),
    }

    prior = train_df["outcome"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    naive_ll = log_loss(holdout["outcome"], np.tile(prior, (len(holdout), 1)), labels=CLASSES)

    metrics = {
        "n_train": int(len(train_df)),
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "features": FEATURE_COLS,
        "naive_baseline_log_loss_all": naive_ll,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "deployed": "ensemble_logistic_gbm",
        "candidates_holdout": candidates,
    }
    print("Competitive holdout log-loss by candidate:")
    for name in ("logistic", "gradient_boosting", "ensemble_logistic_gbm"):
        ll = candidates[name]["competitive"]["log_loss"]
        mark = "  <- deployed" if name == "ensemble_logistic_gbm" else ""
        print(f"  {name:24s} {ll:.4f}{mark}")

    # Deploy: refit both members on ALL history up to the eve of the final.
    lr_full = _fit(_logistic(), feat_df, CUTOFF, is_pipeline=True)
    gb_full = _fit(_gbm(), feat_df, CUTOFF, is_pipeline=False)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(
        {
            "members": [
                {"model": lr_full, "weight": ENSEMBLE_WEIGHTS["logistic"]},
                {"model": gb_full, "weight": ENSEMBLE_WEIGHTS["gradient_boosting"]},
            ],
            "features": FEATURE_COLS,
            "classes": CLASSES,
            "name": "ensemble_logistic_gbm",
        },
        RESULTS_DIR / "classifier.joblib",
    )
    print("\nDeployed: 0.65*logistic + 0.35*gradient_boosting -> results/classifier.joblib")


if __name__ == "__main__":
    main()
