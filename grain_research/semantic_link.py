#!/usr/bin/env python3
"""Source-fitted links between frozen GRAIN units and semantic affinities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def author_weights(authors):
    """Give each represented author equal total weight, with mean row weight one."""
    _, inverse, counts = np.unique(authors, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def validate_reference_exposure(exposure, author_ids):
    """Allow multiple budget arms per author, but never missing/overlapping authors."""
    selected = exposure[exposure.author_id.isin(author_ids)]
    if set(selected.author_id) != set(author_ids):
        raise ValueError("Missing current-author reference membership")
    for column in ("old_reference_member", "old_e1_member"):
        if (
            selected[column].isna().any()
            or not selected[column].isin([False, True]).all()
        ):
            raise ValueError("Invalid reference membership flags")
        if selected[column].any():
            raise ValueError("Existing scorer/current author overlap")


def author_mean(values, authors):
    """Aggregate equal-weight observations, preserving sorted author identities."""
    ids, ix, counts = np.unique(authors, return_inverse=True, return_counts=True)
    values = np.asarray(values, float)
    out = np.zeros((len(ids), values.shape[1]))
    np.add.at(out, ix, values)
    return ids, out / counts[:, None]


def affinity_targets(h, q, items):
    """Return all facet-wise keyed mean/max cosine contrasts, never trait scores."""
    h, q = np.asarray(h, float), np.asarray(q, float)
    for a in (h, q):
        if a.ndim != 2 or not np.isfinite(a).all():
            raise ValueError("Finite matrix required")
        if np.any(np.linalg.norm(a, axis=1) < 1e-12):
            raise ValueError("Zero vector has no cosine")
    if h.shape[1] != q.shape[1] or len(items) != len(q):
        raise ValueError("Item/vector shape mismatch")
    h = h / np.linalg.norm(h, axis=1)[:, None]
    q = q / np.linalg.norm(q, axis=1)[:, None]
    score = h @ q.T
    columns, outputs = [], []
    for kind in ("mean", "max"):
        for facet in sorted({i["facet"] for i in items}):
            plus = [
                j for j, i in enumerate(items) if i["facet"] == facet and i["key"] == 1
            ]
            minus = [
                j for j, i in enumerate(items) if i["facet"] == facet and i["key"] == -1
            ]
            if not plus or not minus:
                raise ValueError("Every facet needs both keyed item sets")
            aggregate = np.mean if kind == "mean" else np.max
            outputs.append(
                aggregate(score[:, plus], axis=1) - aggregate(score[:, minus], axis=1)
            )
            columns.append(f"{kind}:{facet}")
    return np.column_stack(outputs), columns


def validate_addresses(data):
    """Reject stale offsets, duplicate candidate spans and unauthorized authors."""
    if len({b["author_id"] for b in data["bundles"]}) != len(data["bundles"]):
        raise ValueError("Duplicate author")
    comments = []
    for b in data["bundles"]:
        if b["source_group"] != "big5_only":
            raise ValueError("Unauthorized source group")
        comments.append([c for view in b["views"] for c in view])
    seen = set()
    for s in data["sentences"]:
        c = comments[s["author_index"]][s["comment_index"]]
        key = (s["author_index"], s["comment_index"], s["start"], s["end"])
        if key in seen or c["comment_id"] != s["comment_id"]:
            raise ValueError("Duplicate/stale sentence address")
        if not 0 <= s["start"] < s["end"] <= len(c["body"]):
            raise ValueError("Invalid sentence offsets")
        if c["body"][s["start"] : s["end"]] != s["text"]:
            raise ValueError("Sentence span differs from source")
        seen.add(key)
    return comments


def opportunity_numeric(sentences, comments):
    """Extract declared format/length proxies without interpreting personality."""
    rows = []
    for s in sentences:
        text = s["text"]
        parent = comments[s["author_index"]][s["comment_index"]]["body"]
        n = max(len(text), 1)
        words = re.findall(r"\b\w+\b", text)
        rows.append(
            [
                np.log1p(n),
                np.log1p(len(words)),
                np.log1p(len(parent)),
                sum(c.isdigit() for c in text) / n,
                sum(c in ".,;:!?" for c in text) / n,
                sum(c in "\"'" for c in text) / n,
                len(re.findall(r"\b(?:I|me|my|mine|myself)\b", text, re.IGNORECASE))
                / max(1, len(words)),
                n / max(1, len(parent)),
            ]
        )
    return np.asarray(rows, float)


@dataclass
class SemanticLink:
    """A source-fitted reader; transform/predict never accepts semantic targets."""

    mode: str
    alpha: float
    min_subreddit_authors: int = 10

    def fit(self, g, z, subreddits, authors, target):
        """Fit all coordinate/scaler/head parameters on supplied source rows only."""
        if self.mode not in {
            "opportunity",
            "grain_linear",
            "joint_linear",
            "joint_quadratic",
        }:
            raise ValueError("Unknown link mode")
        if len({len(a) for a in (g, z, subreddits, authors, target)}) != 1:
            raise ValueError("Row count mismatch")
        w = author_weights(authors)
        self.categories_ = sorted(
            name
            for name in set(subreddits)
            if len(set(np.asarray(authors)[np.asarray(subreddits) == name]))
            >= self.min_subreddit_authors
        )
        self.g_scaler_ = StandardScaler().fit(g, sample_weight=w)
        self.poly_ = PolynomialFeatures(2, include_bias=False).fit(g)
        raw = self._raw(g, z, subreddits)
        self.x_scaler_ = StandardScaler().fit(raw, sample_weight=w)
        self.y_scaler_ = StandardScaler().fit(target, sample_weight=w)
        self.head_ = Ridge(alpha=self.alpha, solver="cholesky").fit(
            self.x_scaler_.transform(raw),
            self.y_scaler_.transform(target),
            sample_weight=w,
        )
        return self

    def _raw(self, g, z, subreddits):
        """Build fixed coordinate functions and source-supported context columns."""
        parts = []
        if self.mode != "opportunity":
            h = self.g_scaler_.transform(g)
            if self.mode == "joint_quadratic":
                h = self.poly_.transform(h)
            parts.append(h)
        if self.mode != "grain_linear":
            categories = (
                np.asarray(subreddits)[:, None]
                == np.asarray(self.categories_, dtype=str)[None, :]
            )
            parts.extend([np.asarray(z), categories.astype(float)])
        return np.column_stack(parts)

    def predict(self, g, z, subreddits):
        """Predict affinities from text coordinates/proxies, not teacher vectors."""
        prediction = self.head_.predict(
            self.x_scaler_.transform(self._raw(g, z, subreddits))
        )
        return self.y_scaler_.inverse_transform(
            np.asarray(prediction).reshape(len(g), self.y_scaler_.n_features_in_)
        )


def weighted_metrics(y, p, authors, source_scale):
    """Compute author-balanced sentence agreement and normalized squared error."""
    w = author_weights(authors)
    mean_y, mean_p = np.average(y, axis=0, weights=w), np.average(p, axis=0, weights=w)
    yc, pc = y - mean_y, p - mean_p
    cov = np.average(yc * pc, axis=0, weights=w)
    denom = np.sqrt(
        np.average(yc**2, axis=0, weights=w) * np.average(pc**2, axis=0, weights=w)
    )
    r = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 1e-14)
    mse = np.average((y - p) ** 2, axis=0, weights=w)
    return r, mse, mse / np.asarray(source_scale) ** 2


def paired_author_mse(y, p, reference, authors, scale, draws=500, seed=913):
    """Bootstrap paired error improvement over authors, preserving their sentences."""
    difference = ((y - reference) ** 2 - (y - p) ** 2) / np.asarray(scale) ** 2
    _, author_errors = author_mean(difference, authors)
    values = author_errors.mean(1)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {
        "normalized_mse_improvement": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "authors": len(values),
        "draws": draws,
    }
