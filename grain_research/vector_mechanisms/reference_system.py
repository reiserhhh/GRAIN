#!/usr/bin/env python3
"""Frozen, query-conditioned value/response API with explicit reference coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.vector_mechanisms.level_response import affine_parts


def file_digest(path):
    """Hash a file without putting its private contents into a report."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FrozenReferenceSystem:
    """Read fixed-model P/Q states; do not fit, clip, renorm, or infer diagnoses."""

    def __init__(self, arrays, tokens, observer_id, reference_id):
        """Bind every query to its own token coordinates and observer fingerprint."""
        self.arrays = {k: np.asarray(v, dtype=float).copy() for k, v in arrays.items()}
        self.tokens = {q: tuple(v) for q, v in tokens.items()}
        self.observer_id, self.reference_id = observer_id, reference_id
        if not tokens or not observer_id or not reference_id:
            raise ValueError(
                "Declared observer, reference and query coordinates required"
            )
        for query, ids in self.tokens.items():
            values = {
                k: self.arrays[f"{query}__{k}"] for k in ("Z", "h", "G", "g", "b")
            }
            affine_parts(
                values["Z"],
                values["h"],
                values["Z"],
                values["h"],
                values["G"],
                values["g"],
            )
            if (
                len(ids) != len(values["Z"])
                or values["b"].ndim != 0
                or not np.isfinite(values["b"])
            ):
                raise ValueError(
                    "Fixed token identities and finite scalar anchor required"
                )
        for array in self.arrays.values():
            array.setflags(write=False)

    @classmethod
    def load(cls, directory):
        """Load a hash-bound M8 reference; no private text or human labels are needed."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        reference = directory / "reference.npz"
        if file_digest(reference) != manifest["reference_sha256"]:
            raise ValueError("Reference artifact changed")
        with np.load(reference, allow_pickle=False) as arrays:
            return cls(
                dict(arrays),
                manifest["question_tokens"],
                file_digest(directory / "observer.json"),
                manifest["reference_sha256"],
            )

    def score(self, query, z, h, token_ids, observer_id):
        """Score one material in native model units, preserving both P and Q parts."""
        if observer_id != self.observer_id:
            raise ValueError("Observer mismatch; re-encode with the frozen observer")
        if query not in self.tokens or tuple(token_ids) != self.tokens[query]:
            raise ValueError("Query token mismatch; coordinates are query-specific")
        get = lambda k: self.arrays[f"{query}__{k}"]
        parts = affine_parts(z, h, get("Z"), get("h"), get("G"), get("g"))
        b = float(get("b"))
        return {
            "query": query,
            "P": parts["P"],
            "Q": parts["Q"],
            "centered": parts["PQ"],
            "score": b + parts["PQ"],
            "reference_score": b,
            "reference_id": self.reference_id,
            "observer_id": self.observer_id,
        }

    def response(self, query, z, h, edited_z, edited_h, token_ids, observer_id):
        """Use the same value function at both endpoints, with no response refitting."""
        full = self.score(query, z, h, token_ids, observer_id)
        edited = self.score(query, edited_z, edited_h, token_ids, observer_id)
        return {
            "full": full,
            "edited": edited,
            "natural": edited["centered"] - full["centered"],
            "anchor_candidate": -full["centered"],
        }


def reference_coordinates(values, center, scale):
    """Map declared level/response coordinates using fixed, not same-batch, norms."""
    values, center, scale = (
        np.asarray(v, dtype=float) for v in (values, center, scale)
    )
    if (
        values.ndim not in (1, 2)
        or center.ndim != 1
        or scale.shape != center.shape
        or values.shape[-1] != len(center)
    ):
        raise ValueError("Matched one- or two-dimensional profiles required")
    if not all(np.isfinite(v).all() for v in (values, center, scale)) or np.any(
        scale <= 0
    ):
        raise ValueError("Finite fixed reference with positive scales required")
    return (values - center) / scale


def query_function_gram(values):
    """Return functional covariance over matched materials, not mismatched token axes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Two or more matched materials by query required")
    centered = values - values.mean(axis=0)
    return centered.T @ centered / len(values)
