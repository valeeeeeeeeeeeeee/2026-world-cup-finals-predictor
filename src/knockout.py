"""
Knockout-match adjustments, fit on historical knockouts by
backtest_knockouts.py and applied by predict_final.py.

The W/D/L models are trained on every kind of match, and two things about
knockouts differ systematically (see the README backtest section):

  1. Draw odds factor (lambda): more knockouts are level after 90' than the
     models predict. The draw probability is recalibrated on the odds
     scale,  odds(D)' = lambda * odds(D),  with win/loss rescaled to keep
     their ratio. lambda is the maximum-likelihood fit of "level after 90'"
     on leak-free knockout forecasts (a logistic model with logit(p) as
     offset and a single intercept, log lambda).
  2. Extra-time goal factor (phi): extra-time goals per team ~ Poisson(phi *
     mu), where mu is the side's 90-minute expected goals. The naive value
     is 1/3 (30 of 90 minutes); the MLE is total extra-time goals / total
     mu over extra times played after golden/silver goal was abolished.

P(advance | level after 90') then follows exactly: extra time under phi,
and a still-level tie goes to the Elo-tilted shootout coin.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from combine import CLASSES

NAIVE_ET_FACTOR = 1.0 / 3.0
MIN_FIT_ROWS = 30  # below this many knockouts, keep the uncalibrated values


def recalibrate_draw(probs: np.ndarray, draw_odds_factor: float) -> np.ndarray:
    """probs (n, 3) in CLASSES order -> same with the draw odds scaled."""
    probs = np.atleast_2d(probs)
    iD = CLASSES.index("D")
    pD = probs[:, iD]
    new_pD = draw_odds_factor * pD / (draw_odds_factor * pD + 1.0 - pD)
    out = probs * ((1.0 - new_pD) / (1.0 - pD))[:, None]
    out[:, iD] = new_pD
    return out


def fit_draw_odds_factor(p_draw, went_et) -> float:
    p = np.clip(np.asarray(p_draw, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(went_et, dtype=float)
    if len(y) < MIN_FIT_ROWS:
        return 1.0
    base = np.log(p / (1 - p))

    def nll(log_lam):
        z = base + log_lam
        return np.sum(np.logaddexp(0.0, z) - y * z)

    return float(np.exp(minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded").x))


def fit_et_goal_factor(et_goals, mu_total) -> float:
    et_goals = np.asarray(et_goals, dtype=float)
    if len(et_goals) < MIN_FIT_ROWS:
        return NAIVE_ET_FACTOR
    return float(et_goals.sum() / np.asarray(mu_total, dtype=float).sum())


def advance_given_level(mu_home, mu_away, p_home_shootout,
                        et_goal_factor: float = NAIVE_ET_FACTOR, max_goals: int = 10):
    """Exact P(home advances | level after 90') and P(penalties | extra time)."""
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k[None, :], np.atleast_1d(mu_home)[:, None] * et_goal_factor)
    pa = poisson.pmf(k[None, :], np.atleast_1d(mu_away)[:, None] * et_goal_factor)
    M = ph[:, :, None] * pa[:, None, :]
    p_win = (np.tril(np.ones((max_goals + 1,) * 2), -1)[None] * M).sum(axis=(1, 2))
    p_level = M[:, k, k].sum(axis=1)
    return p_win + p_level * np.asarray(p_home_shootout), p_level


def title_prob(probs_90: np.ndarray, p_home_given_level) -> np.ndarray:
    """P(home advances) from (already recalibrated) 90' W/D/L probabilities."""
    probs_90 = np.atleast_2d(probs_90)
    return (probs_90[:, CLASSES.index("H")]
            + probs_90[:, CLASSES.index("D")] * np.asarray(p_home_given_level))
