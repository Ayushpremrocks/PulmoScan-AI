"""Classification metrics for binary screening (pure NumPy, no sklearn needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


@dataclass(frozen=True)
class Confusion:
    tp: int
    fn: int
    tn: int
    fp: int

    @property
    def sensitivity(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return _safe_div(self.tn, self.tn + self.fp)

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def accuracy(self) -> float:
        return _safe_div(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn)


def confusion_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Confusion:
    """Predict pneumonia when prob > threshold (same rule as the app)."""
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(probs) > threshold
    return Confusion(
        tp=int(np.sum(preds & labels)),
        fn=int(np.sum(~preds & labels)),
        tn=int(np.sum(~preds & ~labels)),
        fp=int(np.sum(preds & ~labels)),
    )


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (handles ties)."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(probs)).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """ROC points (false positive rate, true positive rate) for every distinct threshold."""
    probs = np.asarray(probs)
    thresholds = np.concatenate([[1.0], np.unique(probs)[::-1], [-1e-9]])
    rows = []
    for t in thresholds:
        c = confusion_at(labels, probs, t)
        rows.append({"threshold": t, "fpr": 1 - c.specificity, "tpr": c.sensitivity})
    return pd.DataFrame(rows)


def threshold_for_sensitivity(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is at least `target`."""
    labels = np.asarray(labels).astype(bool)
    pos = np.sort(np.asarray(probs)[labels])[::-1]
    if len(pos) == 0:
        return 0.5
    k = max(1, int(np.ceil(target * len(pos))))
    # Predictions use `prob > threshold`, so sit just below the k-th highest positive.
    return float(max(np.nextafter(pos[k - 1], -np.inf), 0.0))
