#!/usr/bin/env python3
"""Author-balanced target anchors on a fixed finite observation family."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import math_observed_family as F

VERSION = "HM171_trait_observation_anchor_v1"
TRAITS = (
    "Extraversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
)


def code_hash():
    """Bind the numeric cold rule to its exact implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def characters():
    """Return the standard eight-cell Boolean character matrix, including1."""
    cells = np.arange(8)
    return np.column_stack(
        [
            np.prod(
                [np.where(cells & (1 << b), 1, -1) for b in range(3) if m & (1 << b)],
                axis=0,
            )
            if m
            else np.ones(8, int)
            for m in range(8)
        ]
    )


def validate_mass(p):
    """Check complete probability distributions, not arbitrary feature vectors."""
    a = np.asarray(p, float)
    if (
        a.ndim != 2
        or a.shape[1] != 8
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or not np.allclose(a.sum(1), 1)
    ):
        raise ValueError("Complete finite8 author mass required")
    return a


def compile_law(author_mass, binary_labels, coordinate_cutoffs, label_cutoffs):
    """Calibrate a five-target source law with equal total mass per author."""
    p, y = validate_mass(author_mass), np.asarray(binary_labels, float)
    if y.shape != (len(p), 5) or not np.isin(y, (0, 1)).all() or not len(p):
        raise ValueError("Source author labels must align, be binary and nonempty")
    ref, prior = p.mean(0), y.mean(0)
    if np.any((prior == 0) | (prior == 1)):
        raise ValueError("Both source label groups required")
    joint = p.T @ y / len(p)
    probability = np.divide(
        joint, ref[:, None], out=np.full((8, 5), np.nan), where=ref[:, None] > 0
    )
    rho1, rho0 = joint / prior, (ref[:, None] - joint) / (1 - prior)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "traits": list(TRAITS),
        "source_authors": len(p),
        "coordinate_cutoffs": np.asarray(coordinate_cutoffs, float).tolist(),
        "label_cutoffs": np.asarray(label_cutoffs, float).tolist(),
        "reference_mass": ref.tolist(),
        "prior": prior.tolist(),
        "joint_positive_mass": joint.tolist(),
        "probabilities": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in probability
        ],
        "rho1": rho1.tolist(),
        "rho0": rho0.tolist(),
        "regime": "author_balanced_self_selected_cached_publications",
    }


def verify_law(b):
    """Verify source units and the Bayes density-contrast identity."""
    required = {
        "version",
        "code_hash",
        "traits",
        "source_authors",
        "coordinate_cutoffs",
        "label_cutoffs",
        "reference_mass",
        "prior",
        "joint_positive_mass",
        "probabilities",
        "rho1",
        "rho0",
        "regime",
    }
    if (
        set(b) != required
        or b["version"] != VERSION
        or b["code_hash"] != code_hash()
        or b["traits"] != list(TRAITS)
        or b["source_authors"] < 1
        or b["regime"] != "author_balanced_self_selected_cached_publications"
    ):
        raise ValueError("Unknown source target rule or extra latent field")
    r = validate_mass([b["reference_mass"]])[0]
    pi, j, psi, rho1, rho0 = [
        np.asarray(b[k], float)
        for k in ("prior", "joint_positive_mass", "probabilities", "rho1", "rho0")
    ]
    if (
        pi.shape != (5,)
        or any(v.shape != (8, 5) for v in (j, psi, rho1, rho0))
        or not all(np.isfinite(v).all() for v in (pi, j, rho1, rho0))
        or np.any((pi <= 0) | (pi >= 1))
    ):
        raise ValueError("Invalid source law dimensions or source groups")
    support = r > 0
    if (
        np.any(j < -1e-12)
        or np.any(j > r[:, None] + 1e-12)
        or not np.allclose(j.sum(0), pi)
        or not np.array_equal(np.isfinite(psi), np.repeat(support[:, None], 5, axis=1))
        or np.any((psi < 0) | (psi > 1))
    ):
        raise ValueError("Probability/reference/support disagreement")
    if (
        not np.allclose(psi[support] * r[support, None], j[support])
        or not np.allclose(rho1, j / pi)
        or not np.allclose(rho0, (r[:, None] - j) / (1 - pi))
    ):
        raise ValueError("Bayes group probability mismatch")
    F.states(np.zeros((0, 15)), b["coordinate_cutoffs"])
    cut = np.asarray(b["label_cutoffs"], float)
    if cut.shape != (5,) or not np.isfinite(cut).all() or np.any((cut < 0) | (cut > 1)):
        raise ValueError("Frozen external label cutoffs required")


def cold_author(bundle, record):
    """Read an outcome-free author window; identifier is never a feature."""
    verify_law(bundle)
    if set(record) != {"author_key", "events"}:
        raise ValueError("Cold input excludes labels, future outcomes and truth")
    events = record["events"]
    if any(set(e) != {"event_id", "condition", "coordinates"} for e in events):
        raise ValueError("Only existing current event fields allowed")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(v, str) for v in ids) or len(ids) != len(set(ids)):
        raise ValueError("Distinct event identities required")
    result = {
        "author_key": record["author_key"],
        "events": len(events),
        "traits": list(TRAITS),
    }
    if not events:
        return result | {"status": "NO_OBSERVATIONS", "relative": None}
    c = np.asarray([e["condition"] for e in events])
    if not np.isin(c, (0, 1)).all():
        raise ValueError("Unknown recorded condition")
    s = F.states(
        np.asarray([e["coordinates"] for e in events]), bundle["coordinate_cutoffs"]
    )
    p = np.bincount(4 * c.astype(int) + s, minlength=8) / len(events)
    r, pi, psi = [
        np.asarray(bundle[k], float)
        for k in ("reference_mass", "prior", "probabilities")
    ]
    if np.any((p > 0) & (r == 0)):
        return result | {"status": "UNOBSERVED_SOURCE_CELLS", "relative": None}
    psi = np.nan_to_num(psi, nan=0.0)
    w = psi - pi
    mean = p @ psi
    rc, pc = r.reshape(2, 4).sum(1), p.reshape(2, 4).sum(1)
    mu = np.divide(
        (r[:, None] * psi).reshape(2, 4, 5).sum(1),
        rc[:, None],
        out=np.zeros((2, 5)),
        where=rc[:, None] > 0,
    )
    choice = (pc - rc) @ mu
    within = (p[:, None] * (psi - np.repeat(mu, 4, axis=0))).sum(0)
    chi = characters()
    coef = chi.T @ w / 8
    contributions = ((p - r) @ chi)[:, None] * coef
    return result | {
        "status": "OBSERVED_TRAIT_FUNCTION",
        "mean": mean.tolist(),
        "relative": (mean - pi).tolist(),
        "choice": choice.tolist(),
        "within": within.tolist(),
        "mass": p.tolist(),
        "walsh_contributions": contributions.tolist(),
        "reference": pi.tolist(),
        "person_posterior": False,
    }
