#!/usr/bin/env python3
"""Controlled outside-appraisal contrasts on complete construct response matrices."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_construct_routing import documents as parent_docs

EVALUATIONS = ("none", "positive", "negative")
CONDITIONS = ((0, 0), (1, 0), (0, 1))


def documents(parent_spec, spec):
    """Keep behaviour and identity fixed, changing only a later outside appraisal."""
    sentences = spec["evaluation_sentences"]
    if set(sentences) != set(EVALUATIONS) or sentences["none"] != "":
        raise ValueError("Require positive/negative/absent appraisal")
    rows = []
    for doc in parent_docs(parent_spec):
        central = (doc["other"], doc["imagined"]) == (0, 0)
        control = (
            doc["role"] == "application"
            and doc["wording"] == 0
            and (doc["other"], doc["imagined"]) in CONDITIONS[1:]
        )
        if not (central or control):
            continue
        marker = ". End of account."
        if not doc["body"].endswith(marker):
            raise ValueError("Exact parent account boundary required")
        base = doc["body"][: -len(" End of account.")]
        for evaluation in EVALUATIONS:
            sentence = sentences[evaluation]
            body = (
                doc["body"]
                if not sentence
                else base + " " + sentence + " End of account."
            )
            start, stop = doc["property_interval"]
            if body[start:stop] != doc["property"]:
                raise ValueError("Appraisal changed the recorded behaviour")
            rows.append(
                {
                    **doc,
                    "hm46_doc_id": doc["doc_id"],
                    "doc_id": len(rows),
                    "evaluation": evaluation,
                    "body": body,
                    "evaluation_interval": [
                        len(base) + 1,
                        len(base) + 1 + len(sentence),
                    ]
                    if sentence
                    else None,
                }
            )
    if len(rows) != 216 or len({r["body"] for r in rows}) != 216:
        raise ValueError("Full216 distinct declared accounts required")
    return rows


def decompose(scores):
    """Exactly decompose a two-behaviour by three-appraisal response table.

    The columns are high/low behaviour and absent/positive/negative appraisal.
    This is an orthogonal finite-design expansion, not a stochastic factor model.
    """
    values = np.asarray(scores, float)
    if values.shape[:2] != (2, 3) or not np.isfinite(values).all():
        raise ValueError("Complete finite2x3 table required")
    behaviour = np.array([[1, 1], [1, -1]], float) / np.sqrt(2)
    appraisal = np.array([[1, 1, 1], [0, 1, -1], [-2, 1, 1]], float)
    appraisal /= np.linalg.norm(appraisal, axis=1, keepdims=True)
    coefficients = np.einsum("ip,jq,pq...->ij...", behaviour, appraisal, values)
    replay = np.einsum("ip,jq,ij...->pq...", behaviour, appraisal, coefficients)
    return coefficients, {
        "reconstruction_max_error": float(np.max(abs(values - replay))),
        "parseval_error": float(
            abs(np.sum(values * values) - np.sum(coefficients * coefficients))
        ),
        "behaviour_contrast": values[0] - values[1],
        "evaluation_contrast_by_behaviour": values[:, 1] - values[:, 2],
        "interaction": (values[0, 1] - values[1, 1]) - (values[0, 2] - values[1, 2]),
        "appraisal_presence_by_behaviour": (values[:, 1] + values[:, 2]) / 2
        - values[:, 0],
    }


def matrix_comparison(reference, current):
    """Describe strength and interaction against a fixed no-evaluation matrix."""
    a, b = np.asarray(reference, float), np.asarray(current, float)
    if (
        a.shape != (6, 6)
        or b.shape != a.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Complete six-by-six signed responses required")
    a2, b2 = float(np.sum(a * a)), float(np.sum(b * b))
    projection = float(np.sum(a * b) / a2) if a2 else None
    residual = (
        float(np.sum((b - projection * a) ** 2)) if projection is not None else None
    )
    return {
        "reference_squared_norm": a2,
        "current_squared_norm": b2,
        "projection": projection,
        "residual_squared_share": residual / b2
        if b2 and residual is not None
        else None,
        "difference_rms": float(np.sqrt(np.mean((b - a) ** 2))),
        "cosine": float(np.sum(a * b) / np.sqrt(a2 * b2)) if a2 and b2 else None,
    }


def array_change(reference, current):
    """Report actual carrier differences without declaring a numerical invariance gate."""
    a, b = np.asarray(reference, float), np.asarray(current, float)
    if (
        a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Aligned finite carrier arrays required")
    delta = b - a
    norm = float(np.linalg.norm(a))
    return {
        "maximum_absolute_change": float(np.max(abs(delta))),
        "delta_norm": float(np.linalg.norm(delta)),
        "relative_delta_norm": float(np.linalg.norm(delta) / norm) if norm else None,
    }
