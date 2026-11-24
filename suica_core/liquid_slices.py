#!/usr/bin/env python3
"""Frozen versions of GRAIN's collective cuts; never fit a cut on application texts."""

from itertools import pairwise

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import suica as native


def sentences(txt):
    return [
        " ".join(native.tokenize(s))
        for s in native.sentence_split(txt)
        if native.tokenize(s)
    ]


def gaps(x):
    d = np.linalg.norm(x[:-1], axis=1) * np.linalg.norm(x[1:], axis=1)
    return 1 - np.divide(
        (x[:-1] * x[1:]).sum(1), d, out=np.zeros_like(d), where=d > 1e-12
    )


def fit_policy(items, kind, rind_dir=None):
    if kind == "rind":
        if not rind_dir:
            raise ValueError("rind needs a calibration ledger directory")
        terms = sorted(native._rind_terms(rind_dir))
        if not terms:
            raise ValueError("empty rind ledger")
        return {"kind": kind, "terms": terms}
    if kind != "semantic":
        return {"kind": "fixed_or_turn"}
    groups = [sentences(t) for _, rows in items for _, _, t in rows]
    texts = [s for g in groups for s in g]
    if len(texts) < 20 or not any(len(g) > 1 for g in groups):
        raise ValueError(
            "semantic calibration needs 20 sentences and within-event transitions"
        )
    tf = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=12000,
        min_df=2,
        sublinear_tf=True,
    )
    a = tf.fit_transform(texts)
    sv = TruncatedSVD(n_components=min(32, min(a.shape) - 1), random_state=42)
    raw = sv.fit_transform(a)
    sc = StandardScaler().fit(raw)
    x = sc.transform(raw)
    offsets = np.cumsum([0] + [len(g) for g in groups])
    distances = np.concatenate(
        [gaps(x[l:r]) for l, r in pairwise(offsets) if r - l > 1]
    )
    return {
        "kind": "semantic",
        "vocab": {k: int(v) for k, v in tf.vocabulary_.items()},
        "idf": tf.idf_.tolist(),
        "svd": sv.components_.tolist(),
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "threshold": float(np.quantile(distances, 0.85)),
    }


def cut(txt, width, policy):
    kind = policy.get("kind")

    def fixed(text, n):
        return [
            r["slice_text"]
            for r in native.fixed_token_slices_for_text(
                text, slice_tokens=n, stride=n, min_slice_tokens=1, max_slices=0
            )
        ]

    if kind == "rind":
        chunks = []
        for piece in fixed(txt, 96):
            if len(native.tokenize(piece)) >= 72 and any(
                s in piece.lower() for s in policy["terms"]
            ):
                chunks.extend(fixed(piece, 48))
            else:
                chunks.append(piece)
        return chunks
    if kind == "semantic":
        parts = sentences(txt)
        if len(parts) < 2:
            return parts
        tf = TfidfVectorizer(
            vocabulary=policy["vocab"],
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tf.idf_ = np.array(policy["idf"])
        x = (tf.transform(parts) @ np.array(policy["svd"]).T - policy["mean"]) / policy[
            "scale"
        ]
        chunks = []
        tokens = []
        for sentence, d in zip(parts, [0, *gaps(x)], strict=True):
            words = native.tokenize(sentence)
            if len(tokens) >= 24 and (
                d >= policy["threshold"] or len(tokens) + len(words) > 160
            ):
                chunks.append(" ".join(tokens))
                tokens = []
            tokens.extend(words)
        if tokens:
            chunks.append(" ".join(tokens))
        return chunks
    return fixed(txt, width) if width else [txt]
