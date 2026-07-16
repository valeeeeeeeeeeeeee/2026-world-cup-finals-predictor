"""
Shared combination utilities used by train_model.py (holdout evaluation +
meta-model training) and predict_final.py (applying the deployed bundle).

All probability arrays here are aligned to CLASSES = ["A", "D", "H"].

Base signals combined:
  - logistic regression class probabilities
  - CatBoost class probabilities
  - Dixon-Coles Poisson W/D/L probabilities (from expected goals)

Combiners:
  - fixed_blend_lr_cb : 0.65*logistic + 0.35*catboost (semifinals baseline)
  - equal_blend_3     : simple average of the three base signals
  - stacking_v2       : logistic meta-model over the LOG-ODDS of all nine
                        base probabilities + match context (neutral flag,
                        competition importance). Log-odds spread the
                        top-of-distribution probabilities the meta-model
                        must discriminate between; context lets it weigh
                        the bases differently per match type.
  - stack_blend_avg   : average of stacking_v2 and fixed_blend_lr_cb
                        (variance reduction over two decorrelated combiners)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

CLASSES = ["A", "D", "H"]
EPS = 1e-4
MAX_GOALS = 12
BLEND_WEIGHTS = {"logistic": 0.65, "catboost": 0.35}
COMBINERS = ("fixed_blend_lr_cb", "equal_blend_3", "stacking_v2", "stack_blend_avg")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def poisson_wdl(mu_home, mu_away, max_goals: int = MAX_GOALS) -> np.ndarray:
    """W/D/L probabilities from independent Poisson goal rates.
    Returns shape (n, 3) in CLASSES order [away win, draw, home win]."""
    mu_h = np.atleast_1d(np.asarray(mu_home, dtype=float))
    mu_a = np.atleast_1d(np.asarray(mu_away, dtype=float))
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k[None, :], mu_h[:, None])  # (n, G) home goal pmf
    pa = poisson.pmf(k[None, :], mu_a[:, None])
    M = ph[:, :, None] * pa[:, None, :]          # (n, G, G): home i, away j
    iH = np.tril_indices(max_goals + 1, -1)      # i > j  -> home win
    iA = np.triu_indices(max_goals + 1, 1)       # i < j  -> away win
    pH = M[:, iH[0], iH[1]].sum(axis=1)
    pA = M[:, iA[0], iA[1]].sum(axis=1)
    pD = M[:, k, k].sum(axis=1)
    total = pH + pD + pA                          # renormalize the tail mass
    return np.stack([pA / total, pD / total, pH / total], axis=1)


def meta_matrix(lr_p: np.ndarray, cb_p: np.ndarray, pois_p: np.ndarray,
                context: np.ndarray) -> np.ndarray:
    """Meta-model design matrix: log-odds of the 9 base probabilities +
    context columns (neutral, k_weight)."""
    return np.hstack([logit(np.hstack([lr_p, cb_p, pois_p])), context])


def _align(model, proba: np.ndarray) -> np.ndarray:
    idx = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, idx]


def combine_probs(name: str, lr_p: np.ndarray, cb_p: np.ndarray,
                  pois_p: np.ndarray, context: np.ndarray, meta=None) -> np.ndarray:
    """Apply a combiner to aligned base probability matrices (n, 3)."""
    if name == "fixed_blend_lr_cb":
        return BLEND_WEIGHTS["logistic"] * lr_p + BLEND_WEIGHTS["catboost"] * cb_p
    if name == "equal_blend_3":
        return (lr_p + cb_p + pois_p) / 3.0
    if name == "stacking_v2":
        X = meta_matrix(lr_p, cb_p, pois_p, context)
        return _align(meta, meta.predict_proba(X))
    if name == "stack_blend_avg":
        stack = combine_probs("stacking_v2", lr_p, cb_p, pois_p, context, meta)
        blend = combine_probs("fixed_blend_lr_cb", lr_p, cb_p, pois_p, context)
        return 0.5 * stack + 0.5 * blend
    raise ValueError(f"unknown combiner: {name}")
