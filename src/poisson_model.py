"""
Attack/defense Poisson goal model (Maher), with optional recency weighting
and the Dixon-Coles low-score correction.

Each team gets an "attack" and "defense" strength learned via an L2
regularized Poisson regression (sklearn PoissonRegressor) on a
team/opponent/home-flag design matrix built from every match goal tally:

    goals_for(team, opponent, is_home) ~ Poisson(exp(b0 + attack[team]
                                                       + defense[opponent]
                                                       + home * is_home))

Fit on a recent window (train_model.py and predict_final.py use a rolling
20 years; the default below is 2015-present) with each match weighted by
0.5 ** (age / HALF_LIFE_YEARS), so the ratings reflect current squads and
old matches fade out smoothly instead of falling off a hard window edge. L2 regularization keeps thinly-sampled teams close to
average instead of overfitting on a handful of matches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

from combine import MAX_GOALS, poisson_score_matrix, poisson_wdl

WINDOW_START = "2015-01-01"
# Chosen on pre-2019 data only (yearly refits 2008-2018, each predicting the
# next 12 months): a 4-year half-life cut W/D/L log-loss from 0.9078 to 0.9012
# and removed the sensitivity to the window start; Dixon-Coles changed it by
# < 0.0005 (rho ~ -0.056), so it is implemented but off.
HALF_LIFE_YEARS = 4.0  # recency half-life of the match weights; None = equal weights
DIXON_COLES = False    # fit the low-score dependence parameter rho


def _long_format(results: pd.DataFrame, cutoff_date, window_start) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["date"] >= pd.Timestamp(window_start)) & (df["date"] < pd.Timestamp(cutoff_date))]

    home_rows = pd.DataFrame(
        {
            "date": df["date"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": np.where(df["neutral"], 0, 1),
            "goals": df["home_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "date": df["date"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals": df["away_score"],
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


class PoissonGoalModel:
    def __init__(self):
        self.encoder: OneHotEncoder | None = None
        self.model: PoissonRegressor | None = None
        self.teams: set[str] = set()
        self.rho = 0.0

    def fit(self, results: pd.DataFrame, cutoff_date, window_start=None,
            half_life_years=HALF_LIFE_YEARS, dixon_coles=DIXON_COLES) -> "PoissonGoalModel":
        """window_start defaults to WINDOW_START; pass an explicit value when
        fitting historical (out-of-fold) models so each fold sees a rolling
        recent window of comparable length instead of a fixed calendar date.
        half_life_years weights each match by 0.5 ** (age / half-life), so
        the window edge fades out instead of cutting off; dixon_coles fits
        rho (see combine.dixon_coles_tau) on the same weighted matches."""
        if window_start is None:
            window_start = WINDOW_START
        long_df = _long_format(results, cutoff_date, window_start)
        if half_life_years:
            age_years = (pd.Timestamp(cutoff_date) - long_df["date"]).dt.days / 365.25
            weights = 0.5 ** (age_years.to_numpy() / half_life_years)
        else:
            weights = np.ones(len(long_df))
        self.teams = set(long_df["team"]) | set(long_df["opponent"])

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        team_dummies = self.encoder.fit_transform(long_df[["team"]])
        opp_dummies = self.encoder.transform(
            long_df[["opponent"]].rename(columns={"opponent": "team"})
        )

        from scipy.sparse import hstack, csr_matrix

        home_col = csr_matrix(long_df[["is_home"]].to_numpy(dtype=float))
        X = hstack([team_dummies, opp_dummies, home_col]).tocsr()
        y = long_df["goals"].to_numpy(dtype=float)

        # alpha picked via a 2015-2023 train / 2024-2026 validation split on
        # mean Poisson deviance (see notes in README) - light regularization
        # mainly to stabilise thinly-sampled minor nations.
        self.model = PoissonRegressor(alpha=0.0005, max_iter=3000, tol=1e-8)
        self.model.fit(X, y, sample_weight=weights)
        self.rho = self._fit_rho(X, long_df["goals"].to_numpy(), weights) if dixon_coles else 0.0
        return self

    def _fit_rho(self, X, goals: np.ndarray, weights: np.ndarray) -> float:
        """Weighted MLE of the Dixon-Coles rho with the attack/defense rates
        held fixed (two-stage fit). _long_format stacks all home rows, then
        all away rows, so row m and row m + n_matches are the same match."""
        n = len(goals) // 2
        mu = self.model.predict(X)
        mu_h, mu_a, x, y, w = mu[:n], mu[n:], goals[:n], goals[n:], weights[:n]
        low = (x <= 1) & (y <= 1)  # tau only touches the four low scorelines
        mu_h, mu_a, x, y, w = mu_h[low], mu_a[low], x[low], y[low], w[low]

        def nll(rho):
            tau = np.select([(x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0)],
                            [1 - mu_h * mu_a * rho, 1 + mu_h * rho, 1 + mu_a * rho], 1 - rho)
            return -np.sum(w * np.log(np.maximum(tau, 1e-9)))

        return float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)

    def wdl(self, mu_home, mu_away) -> np.ndarray:
        """W/D/L probabilities (CLASSES order) for given expected goals."""
        return poisson_wdl(mu_home, mu_away, rho=self.rho)

    def score_matrix(self, mu_home: float, mu_away: float, n: int = MAX_GOALS + 1) -> np.ndarray:
        """(n, n) scoreline probabilities: rows home goals, cols away goals."""
        return poisson_score_matrix(mu_home, mu_away, n - 1, self.rho)[0]

    def expected_goals(self, team: str, opponent: str, neutral: bool) -> float:
        return float(self.expected_goals_batch([team], [opponent],
                                               [0.0 if neutral else 1.0])[0])

    def expected_goals_batch(self, teams, opponents, is_home) -> np.ndarray:
        """Vectorized expected goals; `is_home` is 1.0 for the scoring side
        playing at a declared home venue, else 0.0."""
        from scipy.sparse import hstack, csr_matrix

        team_vec = self.encoder.transform(pd.DataFrame({"team": list(teams)}))
        opp_vec = self.encoder.transform(pd.DataFrame({"team": list(opponents)}))
        home_val = csr_matrix(np.asarray(is_home, dtype=float).reshape(-1, 1))
        X = hstack([team_vec, opp_vec, home_val]).tocsr()
        return self.model.predict(X)


if __name__ == "__main__":
    from data_io import load_results

    goal_model = PoissonGoalModel().fit(load_results(), cutoff_date="2026-07-19")

    for home, away in [("Spain", "Argentina")]:
        mu_home = goal_model.expected_goals(home, away, neutral=True)
        mu_away = goal_model.expected_goals(away, home, neutral=True)
        print(f"{home} vs {away}: expected goals {mu_home:.2f} - {mu_away:.2f}")
