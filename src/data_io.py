"""
Data loading helper.

Loads data/results.csv (path resolved from the repo root, so scripts work
whether run from the root or from src/). If the file is missing, downloads
the latest snapshot from martj42/international_results — which already
includes both 2026 semifinal results (Spain 2-0 France, Argentina 2-1
England).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "results.csv"
RESULTS_DIR = ROOT / "results"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def load_results() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)
