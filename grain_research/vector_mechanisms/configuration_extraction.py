#!/usr/bin/env python3
"""Fixed input programs for query-configuration observations, without LLM decoding."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

ARMS = ("e5_mean", "e5_query", "e5_distribution", "token_lookup")


def normalized_rows(x):
    """Normalize finite nonempty row vectors without amplifying zero rows."""
    x = np.asarray(x, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Finite nonempty row matrix required")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def query_pool(values, anchor, temperature, absolute=False):
    """Use fixed cosine/pole weighting; this is not a learned attention head."""
    values = np.asarray(values, float)
    anchor = np.asarray(anchor, float)
    unit = normalized_rows(values)
    if anchor.shape != (values.shape[1],) or not np.isfinite(anchor).all():
        raise ValueError("Matching finite query vector required")
    score = unit @ anchor
    if absolute:
        score = np.abs(score)
    weight = np.exp(temperature * (score - score.max()))
    return (weight / weight.sum()) @ values


def token_features(token_ids, table, question_ids):
    """Read frozen token embeddings only; no contextual network is evaluated."""
    ids, qids = np.asarray(token_ids, int), np.asarray(question_ids, int)
    if ids.ndim != 1 or qids.ndim != 1 or min(len(ids), len(qids)) < 1:
        raise ValueError("Nonempty token sequences required")
    if min(ids.min(), qids.min()) < 0 or max(ids.max(), qids.max()) >= len(table):
        raise ValueError("Token outside embedding vocabulary")
    vectors = np.asarray(table[ids], float)
    anchor = np.asarray(table[qids], float).mean(0)
    anchor /= max(np.linalg.norm(anchor), 1e-12)
    return np.stack((vectors.mean(0), query_pool(vectors, anchor, 8.0)))


class InputProgram:
    """Source-only fixed geometry and a two-output ridge readout."""

    def __init__(self, arm, poles, seed=20260910, width=256, penalty=0.01):
        """Store frozen choices rather than choosing from held target performance."""
        if arm not in ARMS or width < 1 or penalty <= 0:
            raise ValueError("Valid fixed program choices required")
        self.arm, self.penalty, self.width = arm, penalty, width
        poles = normalized_rows(poles)
        if len(poles) != 2:
            raise ValueError("One ordered query-pole pair required")
        self.anchor = poles[1] - poles[0]
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(size=(poles.shape[1], width))
        self.phase = rng.uniform(0, 2 * np.pi, size=width)
        self.seed = seed

    def raw_blocks(self, bags, tokens):
        """Preserve specified aggregation differences under the same source map."""
        if self.arm == "token_lookup":
            tokens = np.asarray(tokens, float)
            if (
                tokens.ndim != 3
                or tokens.shape[1] != 2
                or not np.isfinite(tokens).all()
            ):
                raise ValueError("Finite two-pool token features required")
            return [tokens[:, 0], tokens[:, 1]]
        means = np.asarray([np.asarray(b).mean(0) for b in bags])
        if self.arm == "e5_mean":
            return [means]
        if self.arm == "e5_query":
            weighted = np.asarray(
                [query_pool(b, self.anchor, 20, absolute=True) for b in bags]
            )
            return [means, weighted]
        rff = np.asarray(
            [
                (
                    np.sqrt(2 / self.width)
                    * np.cos(np.asarray(b) @ self.omega / self.bandwidth + self.phase)
                ).mean(0)
                for b in bags
            ]
        )
        return [means, rff]

    def fit(self, bags, tokens, target):
        """Fit scales, kernel bandwidth and output coefficients on training only."""
        target = np.asarray(target, float)
        if (
            target.ndim != 2
            or target.shape != (len(bags), 2)
            or not np.isfinite(target).all()
        ):
            raise ValueError("Two finite targets per training view required")
        if self.arm == "e5_distribution":
            flat = np.concatenate(bags)
            rng = np.random.default_rng(self.seed)
            a = rng.integers(len(flat), size=1024)
            b = (a + rng.integers(1, len(flat), size=1024)) % len(flat)
            distance = np.linalg.norm(flat[a] - flat[b], axis=1)
            positive = distance[distance > 1e-12]
            self.bandwidth = float(np.median(positive)) if len(positive) else 1.0
        raw = self.raw_blocks(bags, tokens)
        self.centers = [x.mean(0) for x in raw]
        self.scales = [
            max(float(np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1)))), 1e-12)
            for x, c in zip(raw, self.centers, strict=True)
        ]
        self.head = Ridge(alpha=len(target) * self.penalty)
        self.head.fit(self.transform(bags, tokens), target)
        return self

    def transform(self, bags, tokens):
        """Transform one or many new views without fitting a scoring-batch norm."""
        raw = self.raw_blocks(bags, tokens)
        return np.column_stack(
            [
                (x - c) / s
                for x, c, s in zip(raw, self.centers, self.scales, strict=True)
            ]
        ) / np.sqrt(len(raw))

    def predict(self, bags, tokens):
        """Return configuration and directly estimated configuration remainder."""
        return self.head.predict(self.transform(bags, tokens))

    def linear_instance_map(self):
        """Expose the exact comment-level functional behind the mean-only program."""
        if self.arm != "e5_mean":
            raise ValueError("Only the unweighted linear mean has this decomposition")
        weights = self.head.coef_ / self.scales[0]
        offset = self.head.intercept_ - weights @ self.centers[0]
        return weights, offset
