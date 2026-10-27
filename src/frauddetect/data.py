"""Download and load the Bao, Ke, Li, Yu & Zhang (2020) fraud dataset.

The dataset is published by the authors at https://github.com/JarFraud/FraudDetection
without an explicit license, so this repo does not redistribute it. It is
downloaded at run time from a pinned commit and checked against a SHA-256 hash.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
COMMIT = "2d10bc6c7ce18f5dea051a90d823546907abaa84"
FILE = "data_FraudDetection_JAR2020.csv"
URL = f"https://raw.githubusercontent.com/JarFraud/FraudDetection/{COMMIT}/{FILE}"
SHA256 = "ed4cefd9db2a23d0a13911d9f482fc755d3dcf6c9be0bc051af95ddcae5b286f"

# The 28 raw Compustat items used by Bao et al. (2020), in file order.
RAW28 = [
    "act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
    "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
    "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f",
]
# The 14 Dechow et al. (2011)-style ratios shipped with the dataset.
RATIO14 = [
    "dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs", "ch_cm",
    "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / FILE
    if force or not path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    digest = _sha256(path)
    if digest != SHA256:
        raise RuntimeError(
            f"{path} has SHA-256 {digest}, expected {SHA256}. "
            "The upstream file changed; refusing to continue."
        )
    return path


def load() -> pd.DataFrame:
    """Return the full panel (fiscal years 1990-2014) with float32 features."""
    path = download()
    dtypes = {c: np.float32 for c in RAW28 + RATIO14}
    df = pd.read_csv(path, dtype=dtypes)
    df["fyear"] = df["fyear"].astype(int)
    df["gvkey"] = df["gvkey"].astype(int)
    df["misstate"] = df["misstate"].astype(int)
    return df
