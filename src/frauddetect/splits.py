"""Out-of-time train/test splits as in Bao et al. (2020).

For test year t the model trains on fiscal years [first_train_year, t - gap].
The default gap of 2 years reflects that AAERs are published long after the
misstated report, so the most recent years cannot have reliable labels.

Serial fraud (Bao et al. 2020, Section 3.3, as corrected in the 2022 erratum):
a fraud case (one AAER id, ``p_aaer``) often spans several years. If a case
that is fraudulent in the test year also has fraud years in the training
window, those training firm-years are relabelled 0, so the model cannot score
well simply by recognising a firm it has already been told is a fraudster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    test_year: int
    train_idx: np.ndarray       # row labels of df used for training
    test_idx: np.ndarray        # row labels of df used for testing
    y_train: np.ndarray         # training labels after serial-fraud recoding
    y_test: np.ndarray
    n_recoded: int              # training fraud firm-years set to 0


def out_of_time_split(
    df: pd.DataFrame,
    test_year: int,
    gap: int = 2,
    first_train_year: int = 1991,
    handle_serial_fraud: bool = True,
    label: str = "misstate",
    case_id: str = "p_aaer",
) -> Split:
    years = df["fyear"].to_numpy()
    train_mask = (years >= first_train_year) & (years <= test_year - gap)
    test_mask = years == test_year
    train = df.loc[train_mask]
    test = df.loc[test_mask]

    y_train = train[label].to_numpy().astype(int).copy()
    y_test = test[label].to_numpy().astype(int)

    n_recoded = 0
    if handle_serial_fraud:
        test_cases = set(test.loc[test[label] == 1, case_id].dropna().unique())
        recode = train[case_id].isin(test_cases).to_numpy() & (y_train == 1)
        n_recoded = int(recode.sum())
        y_train[recode] = 0

    return Split(test_year, train.index.to_numpy(), test.index.to_numpy(),
                 y_train, y_test, n_recoded)


def rolling_splits(df: pd.DataFrame, test_years, **kw):
    for t in test_years:
        yield out_of_time_split(df, t, **kw)
