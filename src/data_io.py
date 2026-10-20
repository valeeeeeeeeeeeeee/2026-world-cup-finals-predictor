"""
Data loading helper.

Loads the martj42/international_results CSVs from data/ (paths resolved
from the repo root, so scripts work whether run from the root or from
src/), downloading the latest snapshot of any file that is missing:
results.csv (already includes both 2026 semifinal results: Spain 2-0
France, Argentina 2-1 England), shootouts.csv and goalscorers.csv (the
last two are only used by backtest_knockouts.py).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "results.csv"
RESULTS_DIR = ROOT / "results"
BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
DATA_URL = f"{BASE_URL}/results.csv"


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {BASE_URL}/{name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
    return pd.read_csv(path)


def load_results() -> pd.DataFrame:
    return _load_csv("results.csv")


def load_shootouts() -> pd.DataFrame:
    """Penalty shootouts: date, home_team, away_team, winner, first_shooter."""
    return _load_csv("shootouts.csv")


def load_goalscorers() -> pd.DataFrame:
    """Goal-level data: date, home_team, away_team, team, scorer, minute, ..."""
    return _load_csv("goalscorers.csv")
