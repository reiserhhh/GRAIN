#!/usr/bin/env python3
"""Portable compact query readout; frozen neural prefix remains an input contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .functional_profiles import finite_matrix


class FrozenFunctionalBundle:
    """Read matched query states without generation, refitting or batch renorm."""

    @classmethod
    def load(cls, directory):
        """Load finite coefficient/reference artifacts bound to exact file digests."""
        directory = Path(directory)
        manifest = json.loads((directory / "bundle.json").read_text())
        if manifest["protocol"] != "GRAIN_FUNCTIONAL_BUNDLE_M26_V1":
            raise ValueError("Unknown functional bundle protocol")
        for name, expected in manifest["files"].items():
            if Path(name).name != name:
                raise ValueError("Bundle entries must be local basenames")
            with (directory / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Bundle artifact changed: " + name)
        obj = cls()
        obj.manifest = manifest
        with np.load(directory / "compact_functions.npz", allow_pickle=False) as z:
            obj.arrays = {k: np.asarray(v, float).copy() for k, v in z.items()}
        obj.reference = json.loads((directory / "author_reference.json").read_text())
        obj.projections = json.loads(
            (directory / "matched_query_projection.json").read_text()
        )
        for a in obj.arrays.values():
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite bundle coefficients")
            a.setflags(write=False)
        return obj

    def score(self, query, p, q, token_ids, observer_configuration):
        """Return unclipped full/mean/configuration/final values in model units."""
        if observer_configuration != self.manifest["observer_configuration"]:
            raise ValueError("Observer configuration mismatch")
        if (
            query not in self.manifest["question_tokens"]
            or list(token_ids) != self.manifest["question_tokens"][query]
        ):
            raise ValueError("Query token coordinates mismatch")
        get = lambda k: self.arrays[f"{query}__{k}"]
        p, q = finite_matrix(p), np.asarray(q, float)
        if (
            p.shape != get("p_ref").shape
            or q.shape != get("q_ref").shape
            or not np.isfinite(q).all()
        ):
            raise ValueError("State dimensions or values mismatch")
        delta = p - get("p_ref")
        mean = delta.mean(0)
        if f"{query}__configuration" in self.arrays:
            configuration = float(np.sum(get("configuration") * (delta - mean)))
        else:
            configuration = float(
                np.sum((get("left").T @ (delta - mean)) * get("right"))
            )
        values = {
            "mean": float(get("mean_coef") @ mean),
            "configuration": configuration,
            "final": float(get("q_coef") @ (q - get("q_ref"))),
        }
        values["score"] = float(get("anchor")) + sum(values.values())
        return values

    def coordinates(self, raw_profiles):
        """Map nine profile values to the frozen twelve-author reference."""
        x = finite_matrix(raw_profiles)
        ref = self.reference
        center, scale, active = (
            np.asarray(ref[k]) for k in ("center", "scale", "active")
        )
        if x.shape[1] != len(center) or np.any(scale <= 0):
            raise ValueError("Invalid profile coordinates/reference")
        return (x - center) / scale * active

    def profile(self, states, token_ids, observer_configuration):
        """Score one author-view full/E1/E2 panel in the declared coordinate order."""
        queries = self.manifest["queries"]
        if set(states) != set(queries) or set(token_ids) != set(queries):
            raise ValueError("All frozen profile queries required")
        parts = {}
        for query in queries:
            if set(states[query]) != {"full", "e1", "e2"}:
                raise ValueError("Matched full/E1/E2 states required")
            parts[query] = {
                c: self.score(query, *pair, token_ids[query], observer_configuration)
                for c, pair in states[query].items()
            }
        raw = [parts[q]["full"]["score"] for q in queries]
        raw += [
            parts[q][e]["configuration"] - parts[q]["full"]["configuration"]
            for e in ("e1", "e2")
            for q in queries
        ]
        return {
            "coordinate_names": self.reference["coordinates"],
            "raw": raw,
            "positions": self.coordinates([raw])[0].tolist(),
            "parts": parts,
        }

    def shared_private(self, response, view, edit, signal="configuration"):
        """Apply one reference-only projection with an explicit material condition."""
        m = self.projections[signal]
        key = f"{view}/{edit}"
        if key not in m["centers"]:
            raise ValueError("Material condition has no reference calibration")
        x = np.asarray(response, float)
        if x.shape != (len(m["queries"]),) or not np.isfinite(x).all():
            raise ValueError("Matched finite query responses required")
        z = (
            (x - np.asarray(m["centers"][key]))
            / np.asarray(m["scale"])
            * np.asarray(m["active"])
        )
        shared = z @ np.asarray(m["coef"])
        return {
            "queries": m["queries"],
            "standardized": z.tolist(),
            "shared": shared.tolist(),
            "private": (z - shared).tolist(),
        }
