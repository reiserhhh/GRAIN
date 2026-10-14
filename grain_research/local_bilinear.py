#!/usr/bin/env python3
"""Local paired/marginal text pooling and source-fitted measurement readers."""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

ARMS = (
    "word_char",
    "semantic",
    "grain",
    "affinity",
    "additive",
    "outer",
    "paired",
    "shuffle924",
    "shuffle925",
    "shuffle926",
)


def chunk_spans(text, budget=128):
    """Partition all original characters at whitespace-token boundaries."""
    if budget < 1:
        raise ValueError("Positive chunk budget required")
    tokens = list(re.finditer(r"\S+", text))
    ends = [tokens[i].start() for i in range(budget, len(tokens), budget)]
    cuts = [0, *ends, len(text)]
    return [(a, b) for a, b in pairwise(cuts) if b > a]


def keyed_directions(vectors, items):
    """Build fixed IPIP facet directions; contrasts do not imply valid evidence."""
    q = np.asarray(vectors, float)
    norms = np.linalg.norm(q, axis=1)
    if len(q) != len(items) or np.any(norms < 1e-12):
        raise ValueError("Invalid item vectors")
    q = q / norms[:, None]
    names, directions = sorted({i["facet"] for i in items}), []
    for facet in names:
        plus = [
            j
            for j, item in enumerate(items)
            if item["facet"] == facet and item["key"] == 1
        ]
        minus = [
            j
            for j, item in enumerate(items)
            if item["facet"] == facet and item["key"] == -1
        ]
        if not plus or not minus:
            raise ValueError("Both item directions required")
        directions.append(q[plus].mean(0) - q[minus].mean(0))
    return np.column_stack(directions), names


def author_blocks(g, h, semantic, authors):
    """Preserve unit marginals while contrasting co-location and random pairing."""
    g, h, semantic = [np.asarray(x, float) for x in (g, h, semantic)]
    authors = np.asarray(authors, int)
    if len({len(x) for x in (g, h, semantic, authors)}) != 1:
        raise ValueError("Unit row mismatch")
    if any(x.ndim != 2 or not np.isfinite(x).all() for x in (g, h, semantic)):
        raise ValueError("Finite matrices required")
    ids = np.unique(authors)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Complete ordered author index required")
    records = {
        k: []
        for k in (
            "semantic",
            "grain",
            "affinity",
            "outer",
            "paired",
            "shuffle924",
            "shuffle925",
            "shuffle926",
        )
    }
    covariance_energy, counts = [], []
    for u in ids:
        ix = np.flatnonzero(authors == u)
        a, b = g[ix], semantic[ix]
        outer = np.outer(a.mean(0), b.mean(0))
        paired = a.T @ b / len(ix)
        records["semantic"].append(h[ix].mean(0))
        records["grain"].append(np.r_[a.mean(0), a.std(0, ddof=0)])
        records["affinity"].append(b.mean(0))
        records["outer"].append(outer.ravel())
        records["paired"].append(paired.ravel())
        covariance_energy.append(float(np.linalg.norm(paired - outer)))
        counts.append(len(ix))
        for seed in (924, 925, 926):
            permutation = np.random.default_rng(
                np.random.SeedSequence([seed, int(u)])
            ).permutation(len(ix))
            records[f"shuffle{seed}"].append((a.T @ b[permutation] / len(ix)).ravel())
    return (
        {k: np.asarray(v) for k, v in records.items()},
        np.asarray(counts),
        np.asarray(covariance_energy),
    )


class ConstructionFeatures:
    """Fit corpus-dependent transforms on one supplied source partition only."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("Unknown arm")
        self.arm = arm

    def fit(self, blocks, documents, indices):
        """Fit TF-IDF/PCA/scaling without evaluation rows or labels."""
        self.fit_indices_ = np.asarray(indices, int).copy()
        if self.arm == "word_char":
            self.text_ = FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            ngram_range=(1, 3),
                            min_df=2,
                            max_features=10000,
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char",
                            ngram_range=(2, 5),
                            min_df=2,
                            max_features=10000,
                            sublinear_tf=True,
                        ),
                    ),
                ]
            ).fit(np.asarray(documents)[indices])
        else:
            if self.arm not in ("grain", "affinity"):
                x = blocks["semantic"][indices]
                self.pca_ = PCA(
                    n_components=min(64, len(x) - 1, x.shape[1]),
                    svd_solver="randomized",
                    random_state=924,
                ).fit(x)
            self.scaler_ = StandardScaler().fit(self._raw(blocks, indices))
        return self

    def _raw(self, blocks, indices):
        """Construct source-frozen additive and same-width interaction controls."""
        if self.arm in ("grain", "affinity"):
            return blocks[self.arm][indices]
        h = self.pca_.transform(blocks["semantic"][indices])
        if self.arm == "semantic":
            return h
        parts = [h, blocks["grain"][indices], blocks["affinity"][indices]]
        if self.arm != "additive":
            parts.append(blocks[self.arm][indices])
        return np.column_stack(parts)

    def transform(self, blocks, documents, indices):
        """Transform individual or batched authors using fixed source coordinates."""
        if self.arm == "word_char":
            return self.text_.transform(np.asarray(documents)[indices])
        return self.scaler_.transform(self._raw(blocks, indices))
