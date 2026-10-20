import numpy as np
import pytest

from combine import CLASSES, poisson_wdl
from knockout import recalibrate_draw, title_prob

iA, iD, iH = (CLASSES.index(c) for c in "ADH")


@pytest.mark.parametrize("mu_home, mu_away", [
    (0.0, 0.0), (1.4, 1.1), (0.05, 4.0), (6.0, 6.0),
])
def test_poisson_wdl_sums_to_one(mu_home, mu_away):
    p = poisson_wdl(mu_home, mu_away)
    assert p.shape == (1, 3)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_poisson_wdl_vectorized_and_symmetric():
    mu_h = np.array([0.5, 1.2, 2.5, 1.0])
    mu_a = np.array([1.5, 1.2, 0.3, 1.0])
    p = poisson_wdl(mu_h, mu_a)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    # Swapping sides swaps home and away wins.
    q = poisson_wdl(mu_a, mu_h)
    np.testing.assert_allclose(p[:, iH], q[:, iA])
    np.testing.assert_allclose(p[:, iD], q[:, iD])
    # Equal rates -> equal win chances.
    assert p[1, iH] == pytest.approx(p[1, iA])


def test_poisson_wdl_no_goals_is_a_sure_draw():
    np.testing.assert_allclose(poisson_wdl(0.0, 0.0)[0], [0.0, 1.0, 0.0])


def _probs(a, d, h):
    p = np.zeros(3)
    p[[iA, iD, iH]] = [a, d, h]
    return p


def test_recalibrate_draw_identity():
    p = np.array([_probs(0.3, 0.25, 0.45), _probs(0.1, 0.2, 0.7)])
    np.testing.assert_allclose(recalibrate_draw(p, 1.0), p)


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.3, 1e6])
def test_recalibrate_draw_keeps_a_distribution(factor):
    p = _probs(0.3, 0.25, 0.45)
    out = recalibrate_draw(p, factor)[0]
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()
    assert out[iH] / out[iA] == pytest.approx(0.45 / 0.3)  # W/L ratio kept
    # odds(D)' = factor * odds(D)
    if factor > 0:
        assert out[iD] / (1 - out[iD]) == pytest.approx(factor * 0.25 / 0.75)


def test_recalibrate_draw_limits():
    p = _probs(0.3, 0.25, 0.45)
    assert recalibrate_draw(p, 0.0)[0, iD] == 0.0
    assert recalibrate_draw(p, 1e9)[0, iD] == pytest.approx(1.0)
    # A zero draw probability stays zero for any factor.
    np.testing.assert_allclose(recalibrate_draw(_probs(0.4, 0.0, 0.6), 2.0)[0],
                               _probs(0.4, 0.0, 0.6))


def test_recalibrate_draw_certain_draw_stays_finite():
    out = recalibrate_draw(_probs(0.0, 1.0, 0.0), 1.5)[0]
    np.testing.assert_array_equal(out, _probs(0.0, 1.0, 0.0))


def test_title_prob_limits():
    probs = np.array([
        _probs(0.0, 0.0, 1.0),  # sure home win in 90'
        _probs(1.0, 0.0, 0.0),  # sure away win in 90'
        _probs(0.0, 1.0, 0.0),  # sure draw -> decided by P(home | level)
        _probs(0.3, 0.3, 0.4),
        _probs(0.3, 0.3, 0.4),
    ])
    p_level = np.array([0.0, 1.0, 0.37, 1.0, 0.0])
    np.testing.assert_allclose(title_prob(probs, p_level), [1.0, 0.0, 0.37, 0.7, 0.4])


def test_title_prob_complements():
    # P(home advances) + P(away advances) = 1 when the tie-break is shared.
    p = _probs(0.25, 0.3, 0.45)
    swapped = _probs(0.45, 0.3, 0.25)
    assert title_prob(p, 0.6)[0] + title_prob(swapped, 0.4)[0] == pytest.approx(1.0)
