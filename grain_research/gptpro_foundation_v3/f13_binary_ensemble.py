#!/usr/bin/env python3
"""Vectorised QualifiedHistory (filtered mode) across source fits, with B1/B2/M
next-p uncertainty components (state, personal working posterior, source)."""

from __future__ import annotations

import numpy as np

from .f08_estimators import FEATURES

SIGMA2 = 0.25


class EnsembleHistory:
    """Same update semantics as QualifiedHistory(mode='filtered') for F fits."""

    def __init__(self, fits, initial_pairs):
        self.transition = np.array([f["b1"]["source_p"] for f in fits], float).reshape(-1, 2, 2)
        self.source_coef = np.array([f["b2"]["coef"] for f in fits], float)
        self.m_mean = np.array([f["m"]["mean"] for f in fits], float)
        right = np.array([f["m"]["right"] for f in fits], float)
        self.projector = np.einsum("fki,fkj->fij", right, right)
        self.pair = np.array(initial_pairs, float).reshape(-1, 2, 2).copy()
        n = len(fits)
        self.counts = np.zeros((n, 4, 2))
        self.gram = np.zeros((n, 4, 4))
        self.rhs = np.zeros((n, 4))
        self.t = 0

    def observe(self, pole, qualification):
        """Filtered update; a missing time keeps its position (q=0)."""
        likelihood = qualification * np.array([1 - pole, pole]) + (1 - qualification) / 2
        if self.t < 2:
            self.pair *= likelihood[None, :, None] if self.t == 0 else likelihood[None, None, :]
            self.pair /= self.pair.sum(axis=(1, 2))[:, None, None]
        else:
            joint = self.pair[:, :, :, None] * np.stack([1 - self.transition, self.transition], -1) * likelihood
            joint /= joint.sum(axis=(1, 2, 3))[:, None, None, None]
            self.pair = joint.sum(axis=1)
            weighted = qualification * joint.reshape(-1, 4, 2)
            self.counts += weighted
            self.gram += np.einsum("si,fs,sj->fij", FEATURES, weighted.sum(2), FEATURES)
            self.rhs += np.einsum("si,fs->fi", FEATURES, weighted[:, :, 1])
        self.t += 1

    def per_state(self):
        """Per-state predictions (F,3,4), weights (F,4) and personal variances (F,3,4)."""
        eye = np.eye(4)[None]
        a = self.gram + eye
        coef = np.linalg.solve(a, (self.rhs + self.source_coef)[..., None])[..., 0]
        cov = SIGMA2 * np.linalg.inv(a)
        n = self.counts.sum(2)
        raw = (self.counts[:, :, 1] + 1) / (n + 2)
        projected = self.m_mean + np.einsum("fj,fjs->fs", raw - self.m_mean, self.projector)
        raw_var = raw * (1 - raw) / (n + 3)
        m_var = np.einsum("fjs,fj->fs", self.projector**2, raw_var)
        b2 = coef @ FEATURES.T
        b2_var = np.einsum("si,fij,sj->fs", FEATURES, cov, FEATURES)
        values = np.stack([self.transition.reshape(-1, 4), b2, projected], axis=1)
        personal = np.stack([np.zeros_like(b2), b2_var, m_var], axis=1)
        return values, self.pair.reshape(-1, 4), personal


def components(values, weights, personal, canonical=0):
    """Canonical point prediction and variance components for each candidate."""
    raw_means = np.einsum("fcs,fs->fc", values, weights)
    means = np.clip(raw_means, 1e-6, 1 - 1e-6)
    v = values[canonical]
    w = weights[canonical]
    point = means[canonical]
    state = np.einsum("s,cs->c", w, (v - raw_means[canonical][:, None]) ** 2)
    pers = np.einsum("s,cs->c", w, personal[canonical])
    boot = np.delete(means, canonical, axis=0)
    source = boot.var(axis=0) if len(boot) > 1 else np.zeros(3)
    event = np.einsum("s,cs->c", w, np.clip(v, 0, 1) * (1 - np.clip(v, 0, 1)))
    return {"point": point, "state": state, "personal": pers, "source": source, "event": event}
