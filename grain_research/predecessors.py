#!/usr/bin/env python3
"""Inspectable predecessor adapters; no claim of reproducing original datasets.

MEM: binary vocabulary, standardized PCA and varimax (Chung/Pennebaker).
Semantic scores: source-fitted PCA/Ridge (Kjell et al. text-package).
Item evidence: sentence-to-item maximum cosine and keyed counts (SIMPA, ch. 9).
"""

from __future__ import annotations

import html
import re

import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

TRAITS = (
    "Extraversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
)
LETTERS = "ENACO"
FACETS = tuple(f"{letter}{i}" for letter in LETTERS for i in range(1, 7))


def parse_ipip_neo(page: str) -> list[dict]:
    """Parse the official public-domain 30-facet, 300-item key; fail on drift."""

    def plain(value):
        return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())

    # Match facet headings only, not the later overlapping domain inventories.
    headings = list(re.finditer(r"([NEOAC][1-6]):\s*([A-Z -]+)", page))
    if len(headings) != 30:
        raise ValueError("Expected exactly 30 official facet headings")
    items = []
    for i, heading in enumerate(headings):
        end = (
            headings[i + 1].start()
            if i + 1 < len(headings)
            else page.find("5 NEO Domains", heading.end())
        )
        if end < 0:
            raise ValueError("IPIP domain-section boundary not found")
        section = page[heading.end() : end]
        sign, local = None, []
        for row in re.findall(
            r"<tr\b[^>]*>(.*?)</tr>", section, re.DOTALL | re.IGNORECASE
        ):
            cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)
            if len(cells) != 2:
                continue
            key, text = map(plain, cells)
            if "keyed" in key:
                sign = 1 if "+" in key else -1
            if not text:
                continue
            if sign is None:
                raise ValueError("Item appeared before its polarity key")
            facet = heading.group(1)
            local.append(
                {
                    "id": f"IPIP_{facet}_{len(local) + 1:02d}",
                    "facet": facet,
                    "trait": TRAITS[LETTERS.index(facet[0])],
                    "key": sign,
                    "original": text,
                    "text": "I " + text[0].lower() + text[1:],
                }
            )
        if len(local) != 10 or {r["key"] for r in local} != {-1, 1}:
            raise ValueError(f"Invalid IPIP key for {heading.group(1)}")
        items.extend(local)
    if len({r["id"] for r in items}) != 300:
        raise ValueError("Duplicate or missing item IDs")
    return items


def first_person_sentences(text: str) -> list[tuple[int, int, str]]:
    """Regex sentence approximation with exact source offsets, retaining literal I.

    Not spaCy's Sentencizer. Abbreviations and quotations can remain ambiguous;
    an I match is candidate availability, not verified autobiographical ownership.
    """
    output, start = [], 0
    for stop in list(re.finditer(r"(?<=[.!?])\s+|\n+", text)) + [None]:
        end = stop.start() if stop is not None else len(text)
        candidate = text[start:end]
        left = start + len(candidate) - len(candidate.lstrip())
        right = end - len(candidate) + len(candidate.rstrip())
        if re.search(r"\bI\b", candidate) and right > left:
            output.append((left, right, text[left:right]))
        start = stop.end() if stop is not None else len(text)
    return output


def varimax(loadings, *, tolerance=1e-7, max_iter=300):
    """Return orthogonal Kaiser-normalized varimax rotation and diagnostics."""
    loadings = np.asarray(loadings, float)
    norms = np.linalg.norm(loadings, axis=1, keepdims=True)
    x = loadings / np.where(norms > 1e-12, norms, 1)
    rotation, previous = np.eye(x.shape[1]), 0.0
    for iteration in range(max_iter):
        projected = x @ rotation
        u, s, vh = svd(
            x.T @ (projected**3 - projected @ np.diag(np.mean(projected**2, axis=0))),
            full_matrices=False,
        )
        rotation, objective = u @ vh, float(s.sum())
        if iteration and abs(objective - previous) <= tolerance * max(1, abs(previous)):
            return rotation, {"iterations": iteration + 1, "converged": True}
        previous = objective
    return rotation, {"iterations": max_iter, "converged": False}


class MeaningExtraction(BaseEstimator, TransformerMixin):
    """Fold-fitted MEM adaptation with out-of-sample component scores.

    English stopwords replace the original custom content-word inventory; no
    lemmatization. Rank is a declared engineering cap, not a personality ontology.
    """

    def __init__(self, n_components=12, max_features=300, min_df=0.05, seed=731):
        self.n_components, self.max_features, self.min_df, self.seed = (
            n_components,
            max_features,
            min_df,
            seed,
        )

    def fit(self, x, y=None):
        """Learn vocabulary, standardized PCA and rotation on source authors only."""
        self.vectorizer_ = CountVectorizer(
            binary=True,
            stop_words="english",
            min_df=self.min_df,
            max_df=0.95,
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        binary = self.vectorizer_.fit_transform(x).toarray().astype(float)
        self.scaler_ = StandardScaler().fit(binary)
        # Sample-standardized input makes PCA loadings correlation loadings.
        self.scaler_.scale_ *= np.sqrt(len(binary) / (len(binary) - 1))
        k = min(self.n_components, binary.shape[1], len(binary) - 1)
        self.pca_ = PCA(n_components=k, svd_solver="full", random_state=self.seed).fit(
            self.scaler_.transform(binary)
        )
        variance = self.pca_.explained_variance_
        self.score_scale_ = np.sqrt(np.maximum(variance, 1e-12))
        self.rotation_, self.rotation_info_ = varimax(
            self.pca_.components_.T * self.score_scale_
        )
        self.loadings_ = (self.pca_.components_.T * self.score_scale_) @ self.rotation_
        return self

    def transform(self, x):
        """Score against the frozen source vocabulary and reference, never the batch."""
        binary = self.vectorizer_.transform(x).toarray().astype(float)
        return (
            self.pca_.transform(self.scaler_.transform(binary)) / self.score_scale_
        ) @ self.rotation_


def match_items(vectors, item_vectors, items, *, threshold=0.6):
    """Match at most one item per candidate and retain both polarity counts."""
    x, q = np.asarray(vectors, float), np.asarray(item_vectors, float)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1] or len(q) != len(items):
        raise ValueError("Candidate/item dimensions disagree")
    if not np.isfinite(x).all() or not np.isfinite(q).all() or not -1 <= threshold <= 1:
        raise ValueError("Nonfinite vectors or invalid threshold")
    if np.any(np.linalg.norm(x, axis=1) < 1e-12) or np.any(
        np.linalg.norm(q, axis=1) < 1e-12
    ):
        raise ValueError("Zero vectors cannot define cosine")
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    similarity = x @ q.T
    best = similarity.argmax(1)
    score = similarity[np.arange(len(x)), best]
    accepted = score >= threshold
    counts = np.zeros((len(FACETS) + len(TRAITS), 2))
    for j in best[accepted]:
        item = items[int(j)]
        if item["key"] not in (-1, 1):
            raise ValueError("Invalid item polarity")
        key = 0 if item["key"] == 1 else 1
        counts[FACETS.index(item["facet"]), key] += 1
        counts[len(FACETS) + TRAITS.index(item["trait"]), key] += 1
    return counts, best, score, accepted


def keyed_proportions(counts):
    """Return positive proportions; lack of evidence remains NaN, not trait zero."""
    c = np.asarray(counts, float)
    if c.shape[-1] != 2 or np.any(c < 0) or not np.isfinite(c).all():
        raise ValueError("Finite nonnegative positive/negative counts required")
    total = c.sum(-1)
    return np.divide(
        c[..., 0], total, out=np.full(total.shape, np.nan), where=total > 0
    )


def reference_percentiles(train, values):
    """Midrank empirical CDF with fixed reference, retaining unavailable scores."""
    train, values = np.asarray(train), np.asarray(values)
    if train.ndim != 2 or values.ndim != 2 or train.shape[1] != values.shape[1]:
        raise ValueError("Reference dimensions disagree")
    out = np.full(values.shape, np.nan)
    for j in range(train.shape[1]):
        ref = np.sort(train[np.isfinite(train[:, j]), j])
        if len(ref):
            good = np.isfinite(values[:, j])
            v = values[good, j]
            out[good, j] = (
                np.searchsorted(ref, v, side="left")
                + np.searchsorted(ref, v, side="right")
            ) / (2 * len(ref))
    return out


def mean_pearson(y, pred):
    """Mean across all traits, refusing to silently omit failed/constant channels."""
    if np.asarray(y).shape != np.asarray(pred).shape:
        raise ValueError("Target/prediction shapes differ")
    values = []
    for a, b in zip(np.asarray(y).T, np.asarray(pred).T, strict=True):
        if (
            not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or np.std(a) < 1e-12
            or np.std(b) < 1e-12
        ):
            return float("nan")
        values.append(pearsonr(a, b).statistic)
    return float(np.mean(values))
