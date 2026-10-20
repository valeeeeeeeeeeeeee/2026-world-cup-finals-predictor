"""A match's features must not depend on that match's own result.

Builds a synthetic history, flips the score of one match, rebuilds, and
checks that this match's feature row (and every earlier one) is unchanged,
while later matches of the same teams do change (so the test has teeth)."""
import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLS, build_dataset
from poisson_model import PoissonGoalModel

RATING_COLS = ["home_elo", "away_elo", "home_glicko", "away_glicko",
               "home_glicko_rd", "away_glicko_rd"]
TEAMS = [f"T{i}" for i in range(8)]
TOURNAMENTS = ["Friendly", "FIFA World Cup", "UEFA Euro", "FIFA World Cup qualification"]


def _synthetic_results(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        rows.append({
            "date": pd.Timestamp("2016-01-01") + pd.Timedelta(days=7 * i),
            "home_team": home,
            "away_team": away,
            "home_score": int(rng.poisson(1.4)),
            "away_score": int(rng.poisson(1.1)),
            "tournament": TOURNAMENTS[i % len(TOURNAMENTS)],
            "neutral": bool(rng.random() < 0.3),
        })
    return pd.DataFrame(rows)


def _flip(results: pd.DataFrame, i: int) -> pd.DataFrame:
    """Same match, very different result (home win <-> big away win)."""
    out = results.copy()
    hs, as_ = out.loc[i, ["home_score", "away_score"]]
    out.loc[i, ["home_score", "away_score"]] = (0, 5) if hs >= as_ else (5, 0)
    return out


@pytest.mark.parametrize("i", [0, 1, 150, 299])
def test_features_ignore_own_result(i):
    results = _synthetic_results()
    base, *_ = build_dataset(results)
    flipped, *_ = build_dataset(_flip(results, i))

    cols = FEATURE_COLS + RATING_COLS
    pd.testing.assert_frame_equal(base.loc[:i, cols], flipped.loc[:i, cols])
    assert base.loc[i, "outcome"] != flipped.loc[i, "outcome"]

    later = base.index > i
    involved = later & (base["home_team"].isin(results.loc[i, ["home_team", "away_team"]])
                        | base["away_team"].isin(results.loc[i, ["home_team", "away_team"]]))
    if involved.any():
        assert not np.allclose(base.loc[involved, "elo_diff"], flipped.loc[involved, "elo_diff"])


def test_cutoff_excludes_the_cutoff_day():
    results = _synthetic_results()
    cutoff = results.loc[200, "date"]
    feats, elo, glicko, form = build_dataset(results, cutoff_date=str(cutoff.date()))
    assert feats["date"].max() < cutoff
    _, elo2, glicko2, _ = build_dataset(_flip(results, 200), cutoff_date=str(cutoff.date()))
    assert elo.ratings == elo2.ratings
    assert glicko.states == glicko2.states


def test_poisson_model_ignores_matches_on_or_after_cutoff():
    results = _synthetic_results()
    cutoff = results.loc[200, "date"]
    flipped = results
    for i in range(200, len(results)):
        flipped = _flip(flipped, i)

    a = PoissonGoalModel().fit(results, cutoff_date=cutoff, window_start="2000-01-01")
    b = PoissonGoalModel().fit(flipped, cutoff_date=cutoff, window_start="2000-01-01")
    h, w = results.loc[200, ["home_team", "away_team"]]
    assert a.expected_goals(h, w, neutral=True) == b.expected_goals(h, w, neutral=True)
