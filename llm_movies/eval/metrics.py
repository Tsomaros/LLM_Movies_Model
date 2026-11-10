"""Prediction metrics for the p10/p50/p90 outputs. Revenue is in natural-log dollars, score on 0-10.

Same definitions as the app's ml/eval/metrics.py, except where the task brief differs (score width is a median).
Empty input raises ValueError.
"""

import math

import numpy as np

from llm_movies.model import QUANTILES

LN2 = math.log(2)


def _pinball_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-row pinball loss, averaged over the three quantiles (same convention as model/loss.py)."""
    q = np.asarray(QUANTILES, dtype=float)
    diff = true[:, None] - pred
    return np.maximum(q * diff, (q - 1) * diff).mean(axis=1)


def _check(pred, true, weight=None):
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 3 or true.shape != (pred.shape[0],):
        raise ValueError(f"expected pred (N, 3) and true (N,), got {pred.shape} and {true.shape}")
    if len(true) == 0:
        raise ValueError("cannot compute metrics on empty input")
    if weight is None:
        return pred, true
    weight = np.asarray(weight, dtype=float)
    if weight.shape != true.shape:
        raise ValueError("weight must have shape (N,)")
    if weight.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    return pred, true, weight


def revenue_metrics(pred: np.ndarray, true_log: np.ndarray) -> dict:
    pred, true = _check(pred, true_log)
    q10, q50, q90 = pred.T
    return {
        "mae_log": float(np.mean(np.abs(q50 - true))),
        "within_2x": float(np.mean(np.abs(q50 - true) <= LN2 + 1e-12)),  # tolerance as in the app
        "coverage_80": float(np.mean((q10 <= true) & (true <= q90))),
        "median_width_x": float(np.median(np.exp(q90 - q10))),
        "pinball": float(np.mean(_pinball_rows(pred, true))),
    }


def score_metrics(pred: np.ndarray, true: np.ndarray, weight: np.ndarray) -> dict:
    pred, true, weight = _check(pred, true, weight)
    q10, q50, q90 = pred.T
    return {
        "mae": float(np.average(np.abs(q50 - true), weights=weight)),
        "coverage_80": float(np.mean((q10 <= true) & (true <= q90))),  # unweighted, as in the app
        "median_width": float(np.median(q90 - q10)),
        "pinball": float(np.average(_pinball_rows(pred, true), weights=weight)),
    }
