"""
Data loading helper.

Loads the martj42/international_results CSVs from data/ (paths resolved
from the repo root, so scripts work whether run from the root or from
src/), downloading any file that is missing from a pinned commit of the
data repository and checking its SHA-256, so every run sees the same data:
results.csv (already includes both 2026 semifinal results: Spain 2-0
France, Argentina 2-1 England), shootouts.csv and goalscorers.csv (the
last two are only used by backtest_knockouts.py).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "results.csv"
RESULTS_DIR = ROOT / "results"
# martj42/international_results master as of 2026-10-10. Bump the commit and
# the hashes together to move to a newer snapshot (results will change).
DATA_COMMIT = "d5bd6de012c87e2f325de602ae77fe2f9ab61ef3"
BASE_URL = f"https://raw.githubusercontent.com/martj42/international_results/{DATA_COMMIT}"
DATA_URL = f"{BASE_URL}/results.csv"
SHA256 = {
    "results.csv": "5dcefd1aa0483979ea8378cef5be448d2005609d39945ea162378561a30a6767",
    "shootouts.csv": "072b445adf5183fbe981087857e5b64c8f17abfdbf257b1822ceacb4a37919fa",
    "goalscorers.csv": "e02b25cfcad3fca9af79a5cf075c33a6875416d6d80e5ab42514535cb0d639fe",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {BASE_URL}/{name} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        if _sha256(tmp) != SHA256[name]:
            tmp.unlink()
            raise RuntimeError(f"{name} from {BASE_URL} does not match the pinned SHA-256")
        tmp.replace(path)
    if _sha256(path) != SHA256[name]:
        raise RuntimeError(
            f"{path} does not match the pinned SHA-256 (snapshot {DATA_COMMIT[:7]}); "
            f"delete it to re-download the pinned version")
    return pd.read_csv(path)


def load_results() -> pd.DataFrame:
    return _load_csv("results.csv")


def load_shootouts() -> pd.DataFrame:
    """Penalty shootouts: date, home_team, away_team, winner, first_shooter."""
    return _load_csv("shootouts.csv")


def load_goalscorers() -> pd.DataFrame:
    """Goal-level data: date, home_team, away_team, team, scorer, minute, ..."""
    return _load_csv("goalscorers.csv")
