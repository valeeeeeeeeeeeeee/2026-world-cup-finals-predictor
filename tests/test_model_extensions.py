import numpy as np
import pandas as pd
import pytest

import glicko2
from combine import CLASSES, poisson_score_matrix, poisson_wdl
from glicko2 import INITIAL_RD, Glicko2Engine, TeamState, inflate_idle
from poisson_model import PoissonGoalModel


# --- Dixon-Coles -------------------------------------------------------------
@pytest.mark.parametrize("rho", [-0.15, -0.05, 0.05])
def test_dixon_coles_keeps_total_mass(rho):
    M0 = poisson_score_matrix(1.3, 0.9, rho=0.0)
    M = poisson_score_matrix(1.3, 0.9, rho=rho)
    assert M.sum() == pytest.approx(M0.sum(), abs=1e-12)
    np.testing.assert_allclose(M[0, 2:, :], M0[0, 2:, :])  # only the 2x2 corner moves


def test_negative_rho_raises_draws():
    iD = CLASSES.index("D")
    assert poisson_wdl(1.2, 1.1, rho=-0.1)[0, iD] > poisson_wdl(1.2, 1.1)[0, iD]


def test_zero_rho_is_independent_poisson():
    np.testing.assert_array_equal(poisson_wdl(1.4, 0.7, rho=0.0), poisson_wdl(1.4, 0.7))


# --- Poisson recency weights -------------------------------------------------
def _goals_history():
    """A scores 3 per match early on, 0 per match recently, vs. a fixed B."""
    dates = pd.date_range("2000-01-01", "2019-12-01", freq="MS")
    early = dates < "2015-01-01"
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "home_team": "A", "away_team": "B",
        "home_score": np.where(early, 3, 0), "away_score": 1, "neutral": True,
    })


def test_recency_weighting_tracks_recent_form():
    res = _goals_history()
    flat = PoissonGoalModel().fit(res, "2020-01-01", window_start="2000-01-01", half_life_years=None)
    recent = PoissonGoalModel().fit(res, "2020-01-01", window_start="2000-01-01", half_life_years=1.0)
    assert recent.expected_goals("A", "B", True) < flat.expected_goals("A", "B", True)


def test_dixon_coles_fit_is_off_by_default():
    m = PoissonGoalModel().fit(_goals_history(), "2020-01-01", window_start="2000-01-01")
    assert m.rho == 0.0


# --- Glicko-2 idle-time RD inflation -------------------------------------------
def test_idle_inflation_grows_with_time_and_is_capped():
    s = TeamState(rd=60.0, last_played=pd.Timestamp("2020-01-01"))
    rds = [inflate_idle(s, pd.Timestamp(d), period_days=30).rd
           for d in ("2020-01-01", "2020-07-01", "2022-01-01", "2300-01-01")]
    assert rds[0] == pytest.approx(60.0)
    assert rds[0] < rds[1] < rds[2] < rds[3] == INITIAL_RD
    # paper step 6 once per period: RD^2 grows by (173.7178 * vol)^2 per 30 days
    periods = (pd.Timestamp("2022-01-01") - pd.Timestamp("2020-01-01")).days / 30
    assert rds[2] == pytest.approx(np.sqrt(60.0 ** 2 + periods * (173.7178 * 0.06) ** 2))


def test_idle_inflation_off_and_unplayed_team_unchanged():
    s = TeamState(rd=60.0, last_played=pd.Timestamp("2020-01-01"))
    assert inflate_idle(s, pd.Timestamp("2025-01-01"), period_days=0).rd == 60.0
    assert inflate_idle(TeamState(), pd.Timestamp("2025-01-01"), period_days=30).rd == INITIAL_RD


def test_engine_rd_as_of_inflates_idle_team(monkeypatch):
    monkeypatch.setattr(glicko2, "IDLE_PERIOD_DAYS", 30)
    eng = Glicko2Engine()
    for d in pd.date_range("2020-01-01", periods=20, freq="MS"):
        eng.process_match("A", "B", 1, 0, True, d)
    last = pd.Timestamp("2021-08-01")
    assert eng.rd("A", last) == pytest.approx(eng.rd("A"))
    assert eng.rd("A", last + pd.DateOffset(years=2)) > eng.rd("A") + 5
