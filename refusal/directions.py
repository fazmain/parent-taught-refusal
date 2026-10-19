"""Difference-in-means directions, projections, AUROC, cosine. All inputs are numpy float32 arrays of shape (n, d)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def diff_in_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Unnormalised mean(pos) - mean(neg). Scale matters for steering, not for projection AUROC."""
    return pos.mean(0) - neg.mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def project(acts: np.ndarray, d: np.ndarray) -> np.ndarray:
    return acts @ unit(d)


def auroc(pos: np.ndarray, neg: np.ndarray, d: np.ndarray) -> float:
    """AUROC of the scalar projection onto d, positives = pos."""
    s = np.concatenate([project(pos, d), project(neg, d)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, s))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(unit(u) @ unit(v))
