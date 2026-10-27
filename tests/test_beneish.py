import numpy as np
import pandas as pd
import pytest

from frauddetect import beneish


def test_neutral_firm_scores_minus_2_48():
    # All indices = 1 and zero accruals: -4.84 + .920 + .528 + .404 + .892 + .115 - .172 - .327
    m = beneish.m_score(DSRI=1, GMI=1, AQI=1, SGI=1, DEPI=1, SGAI=1, LVGI=1, TATA=0)
    assert m == pytest.approx(-2.48, abs=1e-12)


def test_coefficients_match_beneish_1999():
    base = beneish.m_score(1, 1, 1, 1, 1, 1, 1, 0)
    # Each coefficient recovered by bumping one input by 1.
    expected = {"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
                "DEPI": 0.115, "SGAI": -0.172, "LVGI": -0.327, "TATA": 4.679}
    for name, coef in expected.items():
        args = {c: 1.0 for c in beneish.COMPONENTS} | {"TATA": 0.0}
        args[name] += 1
        assert beneish.m_score(**args) - base == pytest.approx(coef)


def _two_year_firm():
    prev = dict(gvkey=1, fyear=2000, act=50.0, che=10.0, lct=30.0, dlc=5.0, txp=2.0,
                dp=4.0, at=100.0, rect=10.0, sale=100.0, cogs=60.0, ppegt=40.0, dltt=20.0)
    cur = dict(gvkey=1, fyear=2001, act=70.0, che=12.0, lct=36.0, dlc=6.0, txp=3.0,
               dp=5.0, at=120.0, rect=24.0, sale=120.0, cogs=78.0, ppegt=40.0, dltt=24.0)
    return pd.DataFrame([prev, cur])


def test_components_hand_computed():
    out = beneish.components(_two_year_firm(), winsorize=False).iloc[1]
    assert out.DSRI == pytest.approx((24 / 120) / (10 / 100))           # 2.0
    assert out.GMI == pytest.approx(0.40 / ((120 - 78) / 120))           # 0.40 / 0.35
    assert out.AQI == pytest.approx((1 - 110 / 120) / (1 - 90 / 100))
    assert out.SGI == pytest.approx(1.2)
    assert out.DEPI == pytest.approx((4 / 44) / (5 / 45))
    assert out.SGAI == 1.0                                               # not in data
    assert out.LVGI == pytest.approx((60 / 120) / (50 / 100))
    # dCA 20 - dCash 2 - (dCL 6 - dDebt 1 - dTax 1) - Dep 5 = 9
    assert out.TATA == pytest.approx(9 / 120)
    assert out.m_score == pytest.approx(beneish.m_score(
        out.DSRI, out.GMI, out.AQI, out.SGI, out.DEPI, 1.0, out.LVGI, out.TATA))
    assert bool(out.lag_available)


def test_first_year_without_lag_is_neutral():
    out = beneish.components(_two_year_firm(), winsorize=False).iloc[0]
    assert not out.lag_available
    for c in ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI"]:
        assert out[c] == 1.0
    assert out.TATA == 0.0
    assert out.m_score == pytest.approx(-2.48)


def test_lag_never_comes_from_a_future_year():
    df = _two_year_firm()
    df.loc[1, "fyear"] = 2003  # gap: 2003's "prior year" (2002) does not exist
    out = beneish.components(df, winsorize=False)
    assert not out.lag_available.any()


def test_zero_denominators_do_not_crash():
    df = _two_year_firm()
    df.loc[0, ["sale", "rect"]] = 0.0
    out = beneish.components(df, winsorize=False)
    assert np.isfinite(out.m_score).all()
