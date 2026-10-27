import numpy as np
import pandas as pd
import pytest

from frauddetect.metrics import evaluate, ndcg_at_k, top_k
from frauddetect.splits import out_of_time_split, rolling_splits


@pytest.fixture
def panel():
    """Toy panel 1990-2010. Firm 1 commits one fraud spanning 1998-2003 (case 500);
    firm 2 has an unrelated fraud in 1995 (case 600); firm 3 has a fraud in 2003
    under a case id with no training years (case 700)."""
    rows = []
    for g in range(1, 6):
        for y in range(1990, 2011):
            rows.append({"gvkey": g, "fyear": y, "misstate": 0, "p_aaer": np.nan})
    df = pd.DataFrame(rows)
    df.loc[(df.gvkey == 1) & df.fyear.between(1998, 2003), ["misstate", "p_aaer"]] = [1, 500]
    df.loc[(df.gvkey == 2) & (df.fyear == 1995), ["misstate", "p_aaer"]] = [1, 600]
    df.loc[(df.gvkey == 3) & (df.fyear == 2003), ["misstate", "p_aaer"]] = [1, 700]
    return df


def test_train_window_ends_two_years_before_test(panel):
    for sp in rolling_splits(panel, range(2003, 2011)):
        train_years = panel.loc[sp.train_idx, "fyear"]
        test_years = panel.loc[sp.test_idx, "fyear"]
        assert train_years.max() == sp.test_year - 2
        assert train_years.min() == 1991
        assert (test_years == sp.test_year).all()
        assert set(sp.train_idx).isdisjoint(sp.test_idx)


def test_no_year_between_gap_and_test_leaks(panel):
    sp = out_of_time_split(panel, 2005, gap=2)
    assert not panel.loc[sp.train_idx, "fyear"].isin([2004, 2005]).any()


def test_serial_fraud_recoded_in_training(panel):
    sp = out_of_time_split(panel, 2003, gap=2)
    train = panel.loc[sp.train_idx].assign(y=sp.y_train)
    # Case 500 is fraudulent in the 2003 test year -> its 1998-2001 training years become 0.
    assert (train.loc[train.p_aaer == 500, "y"] == 0).all()
    assert sp.n_recoded == 4
    # An unrelated case keeps its label.
    assert train.loc[train.p_aaer == 600, "y"].tolist() == [1]
    # Test labels are never altered.
    assert sp.y_test.sum() == 2


def test_no_test_fraud_case_is_positive_in_training(panel):
    for sp in rolling_splits(panel, range(2003, 2011)):
        test = panel.loc[sp.test_idx]
        cases = set(test.loc[test.misstate == 1, "p_aaer"])
        train = panel.loc[sp.train_idx]
        positives = set(train.loc[sp.y_train == 1, "p_aaer"])
        assert cases.isdisjoint(positives)


def test_serial_fix_only_applies_to_cases_active_in_test_year(panel):
    # In 2005 no fraud is active, so nothing in training should be recoded.
    sp = out_of_time_split(panel, 2005, gap=2)
    assert sp.n_recoded == 0
    assert sp.y_train.sum() == panel.loc[sp.train_idx, "misstate"].sum()


def test_can_disable_serial_fix(panel):
    sp = out_of_time_split(panel, 2003, gap=2, handle_serial_fraud=False)
    assert sp.n_recoded == 0
    assert sp.y_train.sum() == 5  # 1998-2001 of case 500 + 1995 of case 600


def test_input_frame_not_mutated(panel):
    before = panel.copy()
    out_of_time_split(panel, 2003)
    pd.testing.assert_frame_equal(panel, before)


def test_ndcg_and_topk_match_matlab_definition():
    assert top_k(5981, 0.01) == 60          # MATLAB round(59.81)
    assert top_k(250, 0.01) == 3            # 2.5 rounds away from zero, like MATLAB
    y = np.array([0, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.0])
    # k = 3: one hit at rank 2 -> DCG = 1/log2(3); ideal (2 hits) = 1 + 1/log2(3)
    assert ndcg_at_k(y, s, 3) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    m = evaluate(y, s, top=0.5)
    assert m["k"] == 3 and m["hits_topk"] == 1
    assert m["precision_topk"] == pytest.approx(1 / 3)
    assert m["sensitivity_topk"] == pytest.approx(1 / 2)
