import pytest

from glicko2 import Glicko2Engine, TeamState, rate_period


def test_glickman_paper_example():
    # Worked example from Glickman's "Example of the Glicko-2 system"
    # (tau = 0.5): one player, three games in a single rating period.
    player = TeamState(rating=1500.0, rd=200.0, vol=0.06)
    games = [
        (TeamState(1400.0, 30.0), 1.0, 0.0, 0.0),
        (TeamState(1550.0, 100.0), 0.0, 0.0, 0.0),
        (TeamState(1700.0, 300.0), 0.0, 0.0, 0.0),
    ]
    new = rate_period(player, games)
    assert new.rating == pytest.approx(1464.06, abs=0.01)
    assert new.rd == pytest.approx(151.52, abs=0.01)
    assert new.vol == pytest.approx(0.05999, abs=1e-5)


def test_home_advantage_is_not_persisted():
    # A draw between equal teams at a neutral venue leaves ratings equal;
    # at home the bump only lowers the home side's expected score.
    neutral = Glicko2Engine()
    neutral.process_match("A", "B", 1, 1, neutral=True)
    assert neutral.rating("A") == pytest.approx(1500.0)
    assert neutral.rating("B") == pytest.approx(1500.0)

    home = Glicko2Engine()
    home.process_match("A", "B", 1, 1, neutral=False)
    assert home.rating("A") < 1500.0 < home.rating("B")
    assert home.rating("A") + home.rating("B") == pytest.approx(3000.0)


def test_rd_shrinks_after_a_match():
    engine = Glicko2Engine()
    engine.process_match("A", "B", 2, 0, neutral=True)
    assert engine.rd("A") < 350.0 and engine.rd("B") < 350.0
    assert engine.rating("A") > 1500.0 > engine.rating("B")
