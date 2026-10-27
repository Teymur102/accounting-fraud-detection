"""Beneish (1999) M-score from the 28 raw Compustat items in the Bao et al. data.

M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
    - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

Approximations forced by the data (documented in the README):
  * SGAI: SG&A (xsga) is not in the dataset, so SGAI is set to its neutral value 1.
  * AQI and DEPI: only gross PP&E (ppegt) is available, not net PP&E (ppent).
  * Prior-year values come from the same panel (gvkey, fyear-1). When the prior
    year is missing or a ratio is undefined, the component takes its neutral
    value (1 for the indices, 0 for TATA), a common practitioner convention.
TATA uses Beneish's original balance-sheet accrual definition, which needs no
cash-flow statement:
  TATA = [(dCA - dCash) - (dCL - dCurrent debt - dTaxes payable) - Dep] / TA
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COEF = {
    "const": -4.84, "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
    "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327,
}
COMPONENTS = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
NEUTRAL = {c: 1.0 for c in COMPONENTS} | {"TATA": 0.0}
# Beneish's published cut-off for the 8-variable model.
THRESHOLD = -1.78


def m_score(DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA):
    """The Beneish (1999) 8-variable M-score. Works on scalars or arrays."""
    return (
        COEF["const"] + COEF["DSRI"] * DSRI + COEF["GMI"] * GMI + COEF["AQI"] * AQI
        + COEF["SGI"] * SGI + COEF["DEPI"] * DEPI + COEF["SGAI"] * SGAI
        + COEF["TATA"] * TATA + COEF["LVGI"] * LVGI
    )


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def components(df: pd.DataFrame, winsorize: bool = True) -> pd.DataFrame:
    """Compute the eight Beneish components for every firm-year in ``df``.

    ``df`` must contain gvkey, fyear and the raw items. Lags are looked up by
    (gvkey, fyear - 1) inside ``df``. Returns a frame indexed like ``df`` with
    the components, a ``lag_available`` flag and ``m_score``.
    Winsorization (1st/99th percentile) is cross-sectional within each fiscal
    year and uses features only, never labels.
    """
    cols = ["act", "che", "lct", "dlc", "txp", "dp", "at", "rect", "sale",
            "cogs", "ppegt", "dltt"]
    cur = df[["gvkey", "fyear"] + cols].astype({c: float for c in cols})
    lag = cur.copy()
    lag["fyear"] = lag["fyear"] + 1
    m = cur.merge(lag, on=["gvkey", "fyear"], how="left", suffixes=("", "_l"))
    m.index = df.index

    v = {c: m[c].to_numpy(dtype=float) for c in m.columns if c not in ("gvkey", "fyear")}
    has_lag = m["at_l"].notna().to_numpy()

    dsri = _safe_div(_safe_div(v["rect"], v["sale"]), _safe_div(v["rect_l"], v["sale_l"]))

    gm = _safe_div(v["sale"] - v["cogs"], v["sale"])
    gm_l = _safe_div(v["sale_l"] - v["cogs_l"], v["sale_l"])
    gmi = _safe_div(gm_l, gm)

    soft = 1 - _safe_div(v["act"] + v["ppegt"], v["at"])
    soft_l = 1 - _safe_div(v["act_l"] + v["ppegt_l"], v["at_l"])
    # Gross PP&E can push the "soft asset" share below zero; treat as undefined.
    soft[soft <= 0] = np.nan
    soft_l[soft_l <= 0] = np.nan
    aqi = _safe_div(soft, soft_l)

    sgi = _safe_div(v["sale"], v["sale_l"])

    dep_rate = _safe_div(v["dp"], v["dp"] + v["ppegt"])
    dep_rate_l = _safe_div(v["dp_l"], v["dp_l"] + v["ppegt_l"])
    depi = _safe_div(dep_rate_l, dep_rate)

    lev = _safe_div(v["lct"] + v["dltt"], v["at"])
    lev_l = _safe_div(v["lct_l"] + v["dltt_l"], v["at_l"])
    lvgi = _safe_div(lev, lev_l)

    accr = ((v["act"] - v["act_l"]) - (v["che"] - v["che_l"])
            - ((v["lct"] - v["lct_l"]) - (v["dlc"] - v["dlc_l"]) - (v["txp"] - v["txp_l"]))
            - v["dp"])
    tata = _safe_div(accr, v["at"])

    out = pd.DataFrame(
        {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi, "DEPI": depi,
         "SGAI": np.nan, "LVGI": lvgi, "TATA": tata},
        index=df.index,
    )
    computed = out.notna()
    if winsorize:
        years = df["fyear"].to_numpy()
        for c in COMPONENTS:
            if c == "SGAI":
                continue
            for y in np.unique(years):
                sel = years == y
                col = out.loc[sel, c]
                if col.notna().sum() < 10:
                    continue
                lo, hi = col.quantile([0.01, 0.99])
                out.loc[sel, c] = col.clip(lo, hi)
    for c in COMPONENTS:
        out[c] = out[c].fillna(NEUTRAL[c])
    out["lag_available"] = has_lag
    out["n_components_computed"] = computed.sum(axis=1)
    for c in COMPONENTS:
        out[f"{c}_computed"] = computed[c].to_numpy()
    out["m_score"] = m_score(**{c: out[c].to_numpy() for c in COMPONENTS})
    return out
