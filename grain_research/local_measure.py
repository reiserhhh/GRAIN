#!/usr/bin/env python3
"""Portable source-referenced lexical structural measures, not personality diagnoses."""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


def addressed_tokens(text):
    """Match lowercased tokens while mapping Unicode case expansion to raw offsets."""
    lowered = text.lower()
    positions = [i for i, char in enumerate(text) for _ in char.lower()]
    if len(positions) != len(lowered):
        raise ValueError("Unsupported case-fold address change")
    for m in re.finditer(TOKEN_PATTERN, lowered):
        yield m.group(), positions[m.start()], positions[m.end() - 1] + 1


def word_rates(documents, vocabulary):
    """Count frozen terms over all alphabetic >=2-character tokens in each document."""
    index = {term: i for i, term in enumerate(vocabulary)}
    if len(index) != len(vocabulary):
        raise ValueError("Duplicate vocabulary")
    counts = np.zeros((len(documents), len(index)))
    totals = np.zeros(len(documents), int)
    for i, text in enumerate(documents):
        for match in re.finditer(TOKEN_PATTERN, text.lower()):
            totals[i] += 1
            j = index.get(match.group())
            if j is not None:
                counts[i, j] += 1
    return counts / np.maximum(totals[:, None], 1), totals, counts


def export_reference(feature, source_documents, *, domain, source_ids):
    """Translate an existing source-only model to a finite, inspectable JSON bundle."""
    if len(set(source_ids)) != len(source_ids) or len(source_ids) != len(
        source_documents
    ):
        raise ValueError("One source document per unique author required")
    vocabulary = feature.vectorizer_.get_feature_names_out().tolist()
    rates, tokens, _ = word_rates(source_documents, vocabulary)
    if np.any(tokens == 0):
        raise ValueError("Empty source observation")
    np.testing.assert_allclose(rates, feature._rates(source_documents), atol=1e-15)
    if feature.method == "common_factor":
        weights = feature.model_.weights_
        offset = np.zeros(feature.n_factors)
    elif feature.method == "pca":
        weights = (
            feature.model_.components_.T / feature.score_scale_ @ feature.rotation_
        )
        offset = (
            -feature.model_.mean_ @ feature.model_.components_.T / feature.score_scale_
        ) @ feature.rotation_
    else:
        raise ValueError("Unsupported native reference method")
    raw = ((rates - feature.mean_) / feature.scale_) @ weights + offset
    np.testing.assert_allclose(raw, feature.transform(source_documents), atol=1e-10)
    center, scale = raw.mean(0), raw.std(0, ddof=1)
    if np.any(scale < 1e-12):
        raise ValueError("Degenerate score reference")
    beta = weights / feature.scale_[:, None] / scale
    intercept = ((-feature.mean_ / feature.scale_) @ weights + offset - center) / scale
    reference_z = (raw - center) / scale
    return {
        "schema": "grain_local_lexical_reference_v1",
        "domain": domain,
        "method": feature.method,
        "token_pattern": TOKEN_PATTERN,
        "source_ids": list(source_ids),
        "n_source": len(source_ids),
        "factor_ids": [
            f"{domain}_{feature.method}_F{j + 1:02d}" for j in range(feature.n_factors)
        ],
        "vocabulary": vocabulary,
        "rate_mean": feature.mean_.tolist(),
        "rate_scale": feature.scale_.tolist(),
        "raw_weights": weights.tolist(),
        "raw_offset": offset.tolist(),
        "score_center": center.tolist(),
        "score_scale": scale.tolist(),
        "beta": beta.tolist(),
        "intercept": intercept.tolist(),
        "reference_sorted": np.sort(reference_z, axis=0).tolist(),
        "interpretation": "Local author-panel word-rate profile; unnamed structural reference, not a trait score",
    }


def score_documents(bundle, documents):
    """Compute fixed-reference z and empirical midrank without current-batch fitting."""
    if (
        bundle["schema"] != "grain_local_lexical_reference_v1"
        or bundle["token_pattern"] != TOKEN_PATTERN
    ):
        raise ValueError("Unknown scoring schema")
    rates, totals, counts = word_rates(documents, bundle["vocabulary"])
    beta, intercept = (
        np.asarray(bundle["beta"], float),
        np.asarray(bundle["intercept"], float),
    )
    z = rates @ beta + intercept
    ref = np.asarray(bundle["reference_sorted"], float)
    if ref.shape != (bundle["n_source"], len(intercept)) or np.any(
        np.diff(ref, axis=0) < 0
    ):
        raise ValueError("Invalid source reference distribution")
    if not np.isfinite(z).all() or not np.isfinite(ref).all():
        raise ValueError("Invalid scoring parameters")
    ranks = np.column_stack(
        [
            (
                np.searchsorted(ref[:, j], z[:, j], side="left")
                + np.searchsorted(ref[:, j], z[:, j], side="right")
            )
            / (2 * len(ref))
            for j in range(len(intercept))
        ]
    )
    z[totals == 0] = np.nan
    ranks[totals == 0] = np.nan
    return {
        "z": z,
        "percentile": ranks,
        "tokens": totals,
        "dictionary_hits": counts.sum(1).astype(int),
        "rates": rates,
    }


def explain_document(bundle, text):
    """Exact additive score explanation with original text addresses for every hit."""
    scored = score_documents(bundle, [text])
    total = int(scored["tokens"][0])
    if total == 0:
        return {"status": "NO_ALPHABETIC_EVIDENCE", "hits": [], "score": None}
    by_term = {term: i for i, term in enumerate(bundle["vocabulary"])}
    beta = np.asarray(bundle["beta"])
    hits = []
    for term, start, end in addressed_tokens(text):
        i = by_term.get(term)
        if i is not None:
            hits.append(
                {
                    "term": term,
                    "start": start,
                    "end": end,
                    "contribution": (beta[i] / total).tolist(),
                }
            )
    summed = (
        np.sum([h["contribution"] for h in hits], axis=0)
        if hits
        else np.zeros(beta.shape[1])
    )
    reconstructed = np.asarray(bundle["intercept"]) + summed
    np.testing.assert_allclose(reconstructed, scored["z"][0], atol=1e-10)
    return {
        "status": "SCORED",
        "hits": hits,
        "intercept": bundle["intercept"],
        "score": scored["z"][0].tolist(),
        "max_error": float(np.max(np.abs(reconstructed - scored["z"][0]))),
    }


def material_interval(bundle, units, *, seed=923, draws=200):
    """Resample observed blocks conditional on fixed scoring; not latent trait CI."""
    if len(units) < 2:
        return None
    _, tokens, counts = word_rates(units, bundle["vocabulary"])
    ix = np.random.default_rng(seed).integers(0, len(units), size=(draws, len(units)))
    denominator = tokens[ix].sum(1)
    rates = counts[ix].sum(1) / np.maximum(denominator[:, None], 1)
    samples = rates @ np.asarray(bundle["beta"]) + np.asarray(bundle["intercept"])
    samples = samples[denominator > 0]
    if len(samples) < 2:
        return None
    return {
        "lower": np.quantile(samples, 0.025, axis=0).tolist(),
        "upper": np.quantile(samples, 0.975, axis=0).tolist(),
        "draws": len(samples),
        "blocks": len(units),
        "scope": "conditional_frozen_score_observed_block_resampling",
    }


def essay_blocks(text):
    """Keep all characters in consecutive pairs of approximate sentence spans."""
    ends = [m.end() for m in re.finditer(r"[.!?]+(?:\s+|$)|\n+", text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    cut = [0] + ends[1::2]
    if cut[-1] != len(text):
        cut.append(len(text))
    blocks = [text[a:b] for a, b in pairwise(cut)]
    if "".join(blocks) != text:
        raise ValueError("Block partition lost characters")
    return blocks
