"""P05 native-action packet: Euclidean vs action-weighted compile (source-only), no model calls here.

Frozen protocol: docs/GRAIN_P05_NATIVE_ACTION_PACKET_V1.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def source_scale(de_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population SD on source rows; coordinates with SD < 1e-12 are dropped (mask False)."""
    sd = np.asarray(de_source, float).std(axis=0)
    keep = sd >= 1e-12
    return np.where(keep, sd, 1.0), keep


def fit_euclid(Z: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """argmin sum ||y_i - W^T z_i||^2 + alpha ||W||_F^2 (no intercept, no truncation)."""
    Z, Y = np.asarray(Z, float), np.asarray(Y, float)
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ Y)


def euclid_objective(W, Z, Y, alpha):
    R = np.asarray(Y, float) - np.asarray(Z, float) @ W
    return float((R ** 2).sum() + alpha * (W ** 2).sum())


def metric_scale(J: np.ndarray) -> float:
    """s = D / mean_i ||J_i||^2 so an isotropic metric of that trace equals the identity."""
    J = np.asarray(J, float)
    return float(J.shape[1] / np.mean((J ** 2).sum(1)))


def fit_rank1_dual(Z, U, T, alpha, item=None):
    """Ridge over rank-1 metric terms: minimize sum_k (t_k - z_{i(k)}^T W u_k)^2 + alpha ||W||_F^2.

    Returns dual coefficients c (one per term). item maps each term k to its z row.
    """
    Z, U, T = np.asarray(Z, float), np.asarray(U, float), np.asarray(T, float)
    item = np.arange(len(U)) if item is None else np.asarray(item)
    Zk = Z[item]
    K = (Zk @ Zk.T) * (U @ U.T)
    c = np.linalg.solve(K + alpha * np.eye(len(K)), T)
    return c, K


def apply_rank1_dual(c, Z, U, znew, item=None):
    """Compiled change a(z) = sum_k c_k (z_{i(k)} . z) u_k for one or many z rows."""
    Z, U = np.asarray(Z, float), np.asarray(U, float)
    item = np.arange(len(U)) if item is None else np.asarray(item)
    znew = np.atleast_2d(np.asarray(znew, float))
    return (znew @ Z[item].T * c[None, :]) @ U


def rank1_objective(c, K, T, alpha):
    pred = K @ c
    return float(((np.asarray(T) - pred) ** 2).sum() + alpha * c @ K @ c)


def metric_spectrum(U):
    """Nonzero spectrum of sum_i u_i u_i^T via the n x n Gram matrix."""
    w = np.linalg.eigvalsh(np.asarray(U, float) @ np.asarray(U, float).T)
    w = np.sort(w)[::-1]
    rank = int((w > w.max() * 1e-10).sum()) if w.max() > 0 else 0
    return w, rank


class Accounting:
    """Append-only call ledger with a hard cap enforced before every pass."""

    def __init__(self, path: Path, cap: int):
        self.path, self.cap = Path(path), int(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def events(self):
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def passes_attempted(self):
        return sum(e.get("passes", 0) for e in self.events() if e["status"] == "STARTED")

    def done(self):
        return {e["item"]: e for e in self.events() if e["status"] == "OK"}

    def unfinished(self):
        started = {e["item"] for e in self.events() if e["status"] == "STARTED"}
        ended = {e["item"] for e in self.events() if e["status"] in ("OK", "FAILED")}
        return started - ended

    def start(self, item, step, passes):
        if self.unfinished():
            raise RuntimeError(f"unfinished attempts {sorted(self.unfinished())[:3]}; no automatic retry")
        if any(e["status"] == "FAILED" for e in self.events()):
            raise RuntimeError("a prior call failed; stop and report")
        if self.passes_attempted() + passes > self.cap:
            raise RuntimeError("cap would be exceeded before call")
        self._append({"item": item, "step": step, "passes": passes, "status": "STARTED"})

    def finish(self, item, **payload):
        self._append({"item": item, "status": "OK", **payload})

    def fail(self, item, error):
        self._append({"item": item, "status": "FAILED", "error": error})

    def _append(self, rec):
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
