"""
Glicko-2 rating engine (Glickman, http://www.glicko.net/glicko/glicko2.pdf).

Complements Elo with two extra pieces of information Elo does not carry:

  - **RD (rating deviation)** — how uncertain the rating is (a team that
    rarely plays strong opposition has a high RD; its rating means less).
  - **volatility (sigma)** — how erratic the team's performances have been.

Implementation notes:
  - Each match is treated as its own rating period (the common streaming
    adaptation for continuous match feeds).
  - Draws count as score 0.5, wins 1.0, losses 0.0.
  - Home advantage is applied the same way the Elo engine applies it: as a
    rating bump inside the expected-score calculation only (never persisted
    into the stored rating), converted to the Glicko-2 internal scale.
  - Idle-time RD inflation (optional, IDLE_PERIOD_DAYS): with per-match
    periods, RD would only reflect how many matches a team has played. The
    paper's step 6 for a team that skips a period (phi' = sqrt(phi^2 +
    sigma^2)) is applied once per IDLE_PERIOD_DAYS since the team's last
    match, capped at the initial RD, before the next update and whenever
    the RD is read for a fixture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
INITIAL_VOL = 0.06
TAU = 0.5           # volatility constraint (0.3-1.2 typical; 0.5 is standard)
GLICKO2_SCALE = 173.7178
HOME_ADVANTAGE = 100.0  # rating points, expected-score calc only (mirrors elo.py)
EPS = 1e-6
# 30 days chosen on pre-2019 data: Glicko-2's own predictive log-loss on
# 2000-2018 matches 0.5623 (no inflation) -> 0.5593 (14 d: 0.5603, 90 d: 0.5599).
IDLE_PERIOD_DAYS = 30  # days of inactivity per extra rating period; None = no inflation


@dataclass
class TeamState:
    rating: float = INITIAL_RATING
    rd: float = INITIAL_RD
    vol: float = INITIAL_VOL
    last_played: object = None  # date of the last match (for idle-time RD inflation)


def inflate_idle(s: TeamState, date, period_days=None) -> TeamState:
    """RD after the idle time between s.last_played and `date`: one
    paper-step-6 inflation per `period_days` (fractional), capped at
    INITIAL_RD."""
    period_days = IDLE_PERIOD_DAYS if period_days is None else period_days
    if not period_days or s.last_played is None or date is None:
        return s
    periods = max((date - s.last_played).days, 0) / period_days
    phi = s.rd / GLICKO2_SCALE
    rd = GLICKO2_SCALE * math.sqrt(phi * phi + periods * s.vol * s.vol)
    return replace(s, rd=min(rd, INITIAL_RD))


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(phi: float, v: float, delta: float, sigma: float) -> float:
    """Illinois algorithm from the Glicko-2 paper (step 5)."""
    a = math.log(sigma * sigma)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    fA, fB = f(A), f(B)
    while abs(B - A) > EPS:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def rate_period(s: TeamState, games) -> TeamState:
    """Glicko-2 update of `s` over one rating period (paper steps 2-8).
    `games` is a list of (opponent TeamState, score, adv, opp_adv), where
    `adv`/`opp_adv` are expected-score-only rating bumps (home advantage)."""
    phi = s.rd / GLICKO2_SCALE
    v_inv = 0.0
    score_sum = 0.0
    for opp, score, adv, opp_adv in games:
        mu = (s.rating + adv - INITIAL_RATING) / GLICKO2_SCALE
        mu_j = (opp.rating + opp_adv - INITIAL_RATING) / GLICKO2_SCALE
        phi_j = opp.rd / GLICKO2_SCALE
        E = _expected(mu, mu_j, phi_j)
        g_j = _g(phi_j)
        v_inv += g_j * g_j * E * (1.0 - E)
        score_sum += g_j * (score - E)
    v = 1.0 / v_inv
    delta = v * score_sum

    sigma_new = _new_volatility(phi, v, delta, s.vol)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_base = (s.rating - INITIAL_RATING) / GLICKO2_SCALE  # persist without the adv bump
    mu_new = mu_base + phi_new * phi_new * score_sum

    return TeamState(
        rating=INITIAL_RATING + GLICKO2_SCALE * mu_new,
        rd=GLICKO2_SCALE * phi_new,
        vol=sigma_new,
    )


@dataclass
class Glicko2Engine:
    states: dict[str, TeamState] = field(default_factory=dict)

    def _state(self, team: str) -> TeamState:
        if team not in self.states:
            self.states[team] = TeamState()
        return self.states[team]

    def rating(self, team: str) -> float:
        return self._state(team).rating

    def rd(self, team: str, as_of=None) -> float:
        """RD now, or as of a later date (idle-time inflation applied)."""
        return inflate_idle(self._state(team), as_of).rd

    def _update_one(self, s: TeamState, opp: TeamState, score: float,
                    adv: float, opp_adv: float) -> TeamState:
        """One-sided Glicko-2 update against a single opponent.
        `adv`/`opp_adv` are expected-score-only rating bumps (home advantage)."""
        return rate_period(s, [(opp, score, adv, opp_adv)])

    def process_match(self, home_team: str, away_team: str,
                      home_score: float, away_score: float, neutral: bool,
                      date=None) -> None:
        home = inflate_idle(self._state(home_team), date)
        away = inflate_idle(self._state(away_team), date)
        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        adv = 0.0 if neutral else HOME_ADVANTAGE
        new_home = self._update_one(home, away, s_home, adv, 0.0)
        new_away = self._update_one(away, home, 1.0 - s_home, 0.0, adv)
        self.states[home_team] = replace(new_home, last_played=date)
        self.states[away_team] = replace(new_away, last_played=date)


if __name__ == "__main__":
    import pandas as pd

    from data_io import load_results

    df = load_results()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"] < pd.Timestamp("2026-07-19")].sort_values("date")

    engine = Glicko2Engine()
    for row in df.itertuples(index=False):
        engine.process_match(row.home_team, row.away_team,
                             row.home_score, row.away_score, bool(row.neutral), row.date)

    top = sorted(engine.states.items(), key=lambda kv: -kv[1].rating)[:10]
    print("Top 10 Glicko-2 on the eve of the final (2026-07-19):")
    for i, (team, st) in enumerate(top, 1):
        print(f"{i:2d}. {team:20s} {st.rating:7.1f}  (RD {st.rd:5.1f}, vol {st.vol:.4f})")
