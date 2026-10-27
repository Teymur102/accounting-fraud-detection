"""Evaluation metrics, ported from Bao et al.'s evaluate.m.

AUC, plus ranking metrics over the top N% of firm-years by score:
NDCG@k, sensitivity (recall) and precision, with k = round(n * N).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def _matlab_round(x: float) -> int:
    # MATLAB rounds halves away from zero; Python's round() is banker's rounding.
    return int(math.floor(x + 0.5))


def top_k(n: int, top: float) -> int:
    return _matlab_round(n * top)


def ndcg_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    hits = y[order[:k]]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts).sum())
    ideal = min(k, int(y.sum()))
    idcg = float(discounts[:ideal].sum())
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(y_true, scores, top: float = 0.01) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    k = top_k(n, top)
    order = np.argsort(-s, kind="stable")
    tp = int(y[order[:k]].sum())
    pos = int(y.sum())
    return {
        "n": n,
        "n_fraud": pos,
        "k": k,
        "auc": float(roc_auc_score(y, s)) if 0 < pos < n else float("nan"),
        "ndcg_at_k": ndcg_at_k(y, s, k),
        "hits_topk": tp,
        "precision_topk": tp / k if k else float("nan"),
        "sensitivity_topk": tp / pos if pos else float("nan"),
    }
