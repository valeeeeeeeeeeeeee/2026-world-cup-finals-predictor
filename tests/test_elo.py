import pytest

from elo import EloEngine, goal_diff_multiplier, match_k_factor


@pytest.mark.parametrize("tournament, k", [
    ("FIFA World Cup", 60),
    ("Copa América", 50),
    ("UEFA Euro", 50),
    ("African Cup of Nations", 50),
    ("AFC Asian Cup", 50),
    ("Gold Cup", 50),
    ("Confederations Cup", 50),
    ("FIFA World Cup qualification", 40),
    ("UEFA Nations League", 40),
    ("Friendly", 20),
])
def test_k_factor(tournament, k):
    assert match_k_factor(tournament) == k


@pytest.mark.xfail(strict=True, reason=(
    "continental qualifiers hit the 'UEFA Euro'/'Gold Cup'/... keywords "
    "before the 'qualification' one and get K=50 instead of 40"))
@pytest.mark.parametrize("tournament", [
    "UEFA Euro qualification",
    "African Cup of Nations qualification",
    "AFC Asian Cup qualification",
    "Gold Cup qualification",
    "Copa América qualification",
])
def test_k_factor_continental_qualifiers(tournament):
    assert match_k_factor(tournament) == 40


@pytest.mark.parametrize("gd, mult", [
    (0, 1.0), (1, 1.0), (-1, 1.0),
    (2, 1.5), (-2, 1.5),
    (3, 1.75), (4, 1.875), (5, 2.0), (-6, 2.125),
])
def test_goal_diff_multiplier(gd, mult):
    assert goal_diff_multiplier(gd) == pytest.approx(mult)


def test_world_cup_3_0_between_equal_teams():
    # 60 * 1.75 * (1 - 0.5) = 52.5, zero-sum.
    engine = EloEngine()
    engine.process_match(None, "A", "B", 3, 0, "FIFA World Cup", neutral=True)
    assert engine.get("A") == pytest.approx(1552.5)
    assert engine.get("B") == pytest.approx(1447.5)


def test_home_draw_between_equal_teams():
    # Home side expected 1 / (1 + 10^-0.25) = 0.6401; 20 * (0.5 - 0.6401).
    engine = EloEngine()
    engine.process_match(None, "A", "B", 1, 1, "Friendly", neutral=False)
    assert engine.get("A") == pytest.approx(1500.0 - 2.80130, abs=1e-4)
    assert engine.get("B") == pytest.approx(1500.0 + 2.80130, abs=1e-4)
