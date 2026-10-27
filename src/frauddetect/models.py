"""Model definitions. Every learned transform is fitted on the training window only."""
from __future__ import annotations

import numpy as np
from imblearn.ensemble import RUSBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

N_JOBS = 2  # keeps CPU and memory modest when many builds share the machine


class Winsorize(TransformerMixin, BaseEstimator):
    """Clip each column at training-sample quantiles and fill NaN with the training median."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99, signed_log: bool = False):
        self.lower = lower
        self.upper = upper
        self.signed_log = signed_log

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        if self.signed_log:
            X = np.sign(X) * np.log1p(np.abs(X))
        return X

    def fit(self, X, y=None):
        X = self._prep(X)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.med_ = np.nanmedian(X, axis=0)
        return self

    def transform(self, X):
        X = np.clip(self._prep(X), self.lo_, self.hi_)
        return np.where(np.isnan(X), self.med_, X)


def logit_ratios():
    """Dechow et al. (2011)-style logistic regression on the 14 financial ratios."""
    return make_pipeline(Winsorize(), StandardScaler(),
                         LogisticRegression(C=1e4, max_iter=5000))


def logit_raw():
    """Logistic regression on the 28 raw items (signed-log transformed)."""
    return make_pipeline(Winsorize(signed_log=True), StandardScaler(),
                         LogisticRegression(C=1e4, max_iter=5000))


def rusboost(seed: int = 0, n_estimators: int = 300):
    """RUSBoost as in Bao et al.: 300 trees, learning rate 0.1, min leaf 5,
    each round undersamples non-fraud to a 1:1 ratio with fraud."""
    return RUSBoostClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=5, random_state=seed),
        n_estimators=n_estimators, learning_rate=0.1,
        sampling_strategy=1.0, random_state=seed,
    )


def lightgbm(seed: int = 0, n_estimators: int = 300, num_leaves: int = 15,
             learning_rate: float = 0.05):
    return LGBMClassifier(
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=20.0, random_state=seed,
        n_jobs=N_JOBS, verbose=-1,
    )


class BaggedMLP(BaseEstimator):
    """Small neural net. Each member sees all frauds plus a random 10x sample of
    non-frauds; scores are averaged over members."""

    def __init__(self, n_members: int = 5, neg_ratio: int = 10, seed: int = 0):
        self.n_members = n_members
        self.neg_ratio = neg_ratio
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        self.members_ = []
        for i in range(self.n_members):
            take = rng.choice(neg, size=min(len(neg), self.neg_ratio * len(pos)), replace=False)
            idx = np.concatenate([pos, take])
            net = make_pipeline(
                Winsorize(signed_log=True), StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=300,
                              early_stopping=True, n_iter_no_change=15,
                              random_state=self.seed * 100 + i),
            )
            net.fit(X[idx], y[idx])
            self.members_.append(net)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members_], axis=0)
        return np.column_stack([1 - p, p])


def score(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
