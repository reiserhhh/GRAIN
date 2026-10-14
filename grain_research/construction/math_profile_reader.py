#!/usr/bin/env python3
"""Source-calibrated conditional target contrasts with explicit common reference."""

import hashlib
from pathlib import Path

import numpy as np

from grain_research.construction import math_vector_lift as V

VERSION = "HM182_conditional_profile_reader_v1"
TARGETS = V.TARGETS
COMPONENTS = ("marginal", "matched", "reference_shift", "profile_composition")


def code_hash():
    """Bind this finite reader implementation without target outcomes at scoring."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def profile_codes(y, q):
    """Encode the other four binary outcomes in their retained target order."""
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, (0, 1)).all():
        raise ValueError("Five inherited binary source outcomes required")
    if type(q) is not int or q not in range(5):
        raise ValueError("Unknown source target")
    return np.delete(y, q, axis=1) @ (2 ** np.arange(4))


def construct(features, labels):
    """Retain marginal,matched and composition fields,without regressions or grids."""
    x, y = np.asarray(features, float), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Finite author-balanced feature observations required")
    profile_codes(y, 0)
    fields, edges, counts, omega, weights, means = [], [], [], [], [], []
    for q in range(5):
        b = profile_codes(y, q).astype(int)
        n = np.array([np.bincount(b[y[:, q] == g], minlength=16) for g in (0, 1)])
        if np.min(n) == 0:
            raise ValueError(
                "Unobserved target/profile side;no smoothing or truncation"
            )
        m = np.array(
            [[x[(y[:, q] == g) & (b == c)].mean(0) for c in range(16)] for g in (0, 1)]
        )
        p = n / n.sum(1, keepdims=True)
        w = n.prod(0) / n.sum(0)
        w = w / w.sum()
        d = m[1] - m[0]
        marginal = p[1] @ m[1] - p[0] @ m[0]
        matched = w @ d
        reference = ((p[1] + p[0]) / 2 - w) @ d
        composition = (p[1] - p[0]) @ ((m[1] + m[0]) / 2)
        author_w = np.where(y[:, q] == 1, w[b] / n[1, b], -w[b] / n[0, b])
        fields.append(np.stack((marginal, matched, reference, composition)))
        edges.append(d)
        counts.append(n)
        omega.append(w)
        weights.append(author_w)
        means.append(m)
    return {
        "fields": np.array(fields),
        "edges": np.array(edges),
        "counts": np.array(counts),
        "omega": np.array(omega),
        "weights": np.array(weights),
        "means": np.array(means),
    }


def validate_bundle(b):
    """Reject source IDs,labels,extra fields and altered reader contracts."""
    if set(b) != {
        "version",
        "code_hash",
        "targets",
        "components",
        "fields",
        "edges",
        "omega",
        "interpretation",
    }:
        raise ValueError("Unexpected cold reader fields")
    if (
        b["version"] != VERSION
        or b["code_hash"] != code_hash()
        or tuple(b["targets"]) != TARGETS
        or tuple(b["components"]) != COMPONENTS
    ):
        raise ValueError("Reader identity changed")
    if (
        b["interpretation"]
        != "source_descriptive_profile_contrasts_not_probability_or_causal_trait"
    ):
        raise ValueError("Reader meaning changed")
    for k, shape in (
        ("fields", (2, 5, 4, 4096)),
        ("edges", (2, 5, 16, 4096)),
        ("omega", (5, 16)),
    ):
        x = np.asarray(b[k], float)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError("Invalid cold coefficients: " + k)
    w = np.asarray(b["omega"])
    if np.min(w) <= 0 or not np.allclose(w.sum(1), 1, atol=2e-12, rtol=0):
        raise ValueError("Invalid positive common reference")


def cold_record(b, row):
    """Read addressed vectors only;other target labels are never application inputs."""
    if set(row) != {"frame_key", "view", "events"} or row["view"] not in (0, 1):
        raise ValueError("Unexpected observation fields")
    if not isinstance(row["frame_key"], str):
        raise TypeError("Opaque frame key required")
    ids = []
    for e in row["events"]:
        if set(e) != {"event_id", "slot", "coordinates"} or type(e["slot"]) is not int:
            raise ValueError("Unexpected event fields")
        if not isinstance(e["event_id"], str):
            raise TypeError("Opaque event identity required")
        ids.append(e["event_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evidence event")
    x = np.asarray([e["coordinates"] for e in row["events"]], float).reshape(
        len(ids), 15
    )
    phi, n, counts = V.subset_feature(
        x, np.array([e["slot"] for e in row["events"]], int)
    )
    fields = np.einsum("vqck,k->vqc", np.asarray(b["fields"]), phi)
    edges = np.einsum("vqbk,k->vqb", np.asarray(b["edges"]), phi)
    return {
        "frame_key": row["frame_key"],
        "view": row["view"],
        "triples": n,
        "role_counts": counts,
        "retained_mass": n / 64,
        "retained_fields": fields.tolist(),
        "retained_edges": edges.tolist(),
        "conditional_fields": (fields * 64 / n).tolist() if n else None,
        "conditional_edges": (edges * 64 / n).tolist() if n else None,
    }
