#!/usr/bin/env python3
"""Matched-alphabet query specifications and crossed functional readers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .direct_traits import SYSTEM, question, render

QUERIES = (
    "neuroticism",
    "confirmation_seeking",
    "feeling_numeric",
    "quantitative_content",
)
QUESTIONS = {
    "neuroticism": question("neuroticism"),
    "confirmation_seeking": (
        "Estimate the writer's tendency to seek other people's reassurance, confirmation or approval "
        "when making everyday judgments. Use 0 for very low, 9 for very high, and the digits in "
        "between for intermediate tendencies. Return only one digit from 0 to 9."
    ),
    "feeling_numeric": (
        "Estimate the writer's preference for Feeling (personal values and interpersonal "
        "considerations) rather than Thinking (impersonal logic and consistency) when making "
        "judgments. Use 0 for a very strong Thinking preference, 9 for a very strong Feeling "
        "preference, and the digits in between for intermediate preferences. "
        "Return only one digit from 0 to 9."
    ),
    "quantitative_content": (
        "Estimate how much the writer's excerpts discuss numerical quantities, measurements, "
        "prices or calculations. Use 0 for very little, 9 for very much, and the digits in "
        "between for intermediate amounts. Return only one digit from 0 to 9."
    ),
}


def matched_prompt(texts, query, tokenizer, max_tokens=3072):
    """Change only the final question, preserving the native N template."""
    if query not in QUESTIONS:
        raise ValueError("Unknown frozen functional query")
    user, spans = render(texts, "neuroticism")
    prefix, separator, suffix = user.rpartition("\nQuestion: ")
    expected = question("neuroticism") + "\nAnswer:"
    if not separator or suffix != expected:
        raise ValueError("Frozen native question boundary changed")
    user = prefix + separator + QUESTIONS[query] + "\nAnswer:"
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if not ids or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("Valid compact native token sequence required")
    if len(ids) + 16 > max_tokens:
        raise ValueError("Matched query exceeds frozen context budget; never truncate")
    return list(ids), user, spans


def cross_readers(reference_gradients, recipient_displacements):
    """Contract each reference-query covector with every recipient-query vector.

    Input reference shape is (authors, queries, coordinates); recipient shape is
    (authors, queries, coordinates). Output scores are (recipient, query, reader).
    No recipient gradient, endpoint response or questionnaire is consumed.
    """
    g, d = (
        np.asarray(reference_gradients, float),
        np.asarray(recipient_displacements, float),
    )
    if (
        g.ndim != 3
        or d.ndim != 3
        or min(g.shape) == 0
        or min(d.shape) == 0
        or g.shape[1:] != d.shape[1:]
        or not np.isfinite(g).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError(
            "Finite author/query arrays in the same coordinate basis required"
        )
    references = g.mean(axis=0)
    return references, np.einsum("rd,uqd->uqr", references, d)


def cosine_matrix(rows):
    """Return row-direction cosines, leaving a zero direction explicitly undefined."""
    rows = np.asarray(rows, float)
    norms = np.linalg.norm(rows, axis=1)
    denominator = norms[:, None] * norms[None, :]
    return np.divide(
        rows @ rows.T,
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def effect_quality(effect, prediction):
    """Keep zero-effect conditions visible with undefined normalized metrics."""
    y, p = np.asarray(effect, float), np.asarray(prediction, float)
    if y.ndim != 1 or y.shape != p.shape or len(y) < 2 or not np.isfinite([y, p]).all():
        raise ValueError("Paired finite effects and predictions required")
    mse, energy = float(np.mean((p - y) ** 2)), float(np.mean(y**2))
    return {
        "r": float(np.corrcoef(y, p)[0, 1])
        if np.ptp(y) > 0 and np.ptp(p) > 0
        else None,
        "mse": mse,
        "mse_over_effect_energy": mse / energy if energy > 0 else None,
        "mean_error": float(np.mean(p - y)),
        "sign_agreement": float(np.mean(np.sign(y) == np.sign(p))),
    }
