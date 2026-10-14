#!/usr/bin/env python3
"""Source empirical laws for same-author evidence composition, without fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

VERSION = "HM174_joint_evidence_law_v1"
MODELS = ("event_mean", "target_iid", "shared_author")
VARIANTS = (
    "full",
    "remove_selected",
    "remove_control",
    "selected_only",
    "control_only",
)


def code_hash():
    """Bind a portable SOURCE law to its scientific implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_source(p, y):
    """Require normalized eight-cell laws and aligned five binary targets."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if (
        p.ndim != 2
        or p.shape[1] != 8
        or len(p) < 2
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, rtol=0, atol=2e-12)
        or y.shape != (len(p), 5)
        or not np.isin(y, (0, 1)).all()
        or np.any((y.mean(0) == 0) | (y.mean(0) == 1))
    ):
        raise ValueError("Normalized8-cell SOURCE laws and both binary groups required")
    return p, y


def build_bundle(p, y):
    """Keep the empirical author catalogue, not a fitted class count or teacher."""
    p, y = validate_source(p, y)
    ref, pi = p.mean(0), y.mean(0)
    if np.any(ref == 0):
        raise ValueError("Missing one-event SOURCE support;no smoothing rescue")
    joint = p.T @ y / len(p)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "source_mass": p.tolist(),
        "source_binary_targets": y.tolist(),
        "prior": pi.tolist(),
        "event_probability": (joint / ref[:, None]).tolist(),
        "rho1": (joint / pi).tolist(),
        "rho0": ((ref[:, None] - joint) / (1 - pi)).tolist(),
        "observation_law": "empirical_source_author_resampling;conditional_iid_given_index",
        "smoothing": 0,
    }


def validate_bundle(b):
    """Reject altered source laws or undeclared extra fitting fields."""
    if b.get("version") != VERSION or b.get("code_hash") != code_hash():
        raise ValueError("Unknown scientific implementation")
    p, y = validate_source(b["source_mass"], b["source_binary_targets"])
    rebuilt = build_bundle(p, y)
    if set(b) != set(rebuilt) or any(
        not np.array_equal(np.asarray(b[k]), np.asarray(rebuilt[k])) for k in rebuilt
    ):
        raise ValueError("Changed source law or extra fitting field")
    return p, y


def counts(cells):
    """Count finite observation categories, never author identifiers."""
    a = np.asarray(cells)
    if a.size == 0:
        return np.zeros(8, int)
    if (
        a.ndim != 1
        or not np.issubdtype(a.dtype, np.integer)
        or np.any((a < 0) | (a > 7))
    ):
        raise ValueError("In-range integer event cells required")
    return np.bincount(a, minlength=8)


def log_likelihoods(p, n):
    """Keep true zero support; inactive zero cells never create0*log0."""
    n = np.asarray(n)
    active = n > 0
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, float)[:, active]) @ n[active]


def evaluate(b, cells):
    """Compute three fixed rules; binary log odds only for actual posteriors."""
    p, y = np.asarray(b["source_mass"]), np.asarray(b["source_binary_targets"])
    n = counts(cells)
    pi, psi = np.asarray(b["prior"]), np.asarray(b["event_probability"])
    mean = n @ psi / n.sum() if n.sum() else pi.copy()
    author_ll = log_likelihoods(p, n)
    joint_shared = np.array(
        [
            [logsumexp(author_ll[y[:, q] == g]) - np.log(len(p)) for g in (0, 1)]
            for q in range(5)
        ]
    )
    joint_iid = np.column_stack(
        [
            np.log(1 - pi) + log_likelihoods(np.asarray(b["rho0"]).T, n),
            np.log(pi) + log_likelihoods(np.asarray(b["rho1"]).T, n),
        ]
    )
    joints = np.stack([joint_iid, joint_shared])
    norm = logsumexp(joints, axis=-1)
    with np.errstate(invalid="ignore"):
        posterior = np.exp(joints[..., 1] - norm)
        odds = joints[..., 1] - joints[..., 0]
    return {
        "probabilities": np.vstack([mean, posterior]),
        "binary_log_odds": odds,
        "posterior_supported": np.isfinite(norm),
        "finite_log_odds": np.isfinite(odds),
        "log_evidence": norm,
        "events": int(n.sum()),
    }


def group_indices(selected, control):
    """Keep original identities and order in the fixed five subset conditions."""
    s, c = np.asarray(selected), np.asarray(control)
    if (
        s.shape != (4,)
        or c.shape != (4,)
        or not np.issubdtype(s.dtype, np.integer)
        or not np.issubdtype(c.dtype, np.integer)
        or len(set(s.tolist()) | set(c.tolist())) != 8
        or np.any((np.r_[s, c] < 0) | (np.r_[s, c] >= 12))
    ):
        raise ValueError("Require disjoint4-event identities in12-comment view")
    i = np.arange(12)
    return (
        i,
        i[~np.isin(i, s)],
        i[~np.isin(i, c)],
        np.sort(s),
        np.sort(c),
    )


def evidence_groups(cells, selected, control):
    """Return the five SAME identity-preserving D2 evidence families."""
    a = np.asarray(cells)
    counts(a)
    if a.shape != (12,):
        raise ValueError("Require12-comment view")
    return tuple(a[i] for i in group_indices(selected, control))


def composition(values, coefficient=1.0):
    """Compare S-vs-C with and without the SAME remaining context R."""
    x = np.asarray(values, float)
    if x.shape[0] != 5:
        raise ValueError("Five fixed natural conditions required")
    return x[2] - x[1] - coefficient * (x[3] - x[4])


def categorical_contrasts(logits):
    """All9 digit-vs0 log odds; cancel full-vocabulary/softmax gauge exactly."""
    z = np.asarray(logits, float)
    if z.shape[-1] != 10 or not np.isfinite(z).all():
        raise ValueError("Ten finite native candidate logits required")
    return z[..., 1:] - z[..., :1]


def cold_record(bundle, record):
    """Score outcome-free identified events with no native reader or labels."""
    if set(record) != {"author_key", "events"}:
        raise ValueError("Cold input excludes labels and native outcomes")
    events = record["events"]
    if any(set(e) != {"event_id", "cell"} for e in events):
        raise ValueError("Only identified observed cells allowed")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct event identities required")
    a = evaluate(bundle, [e["cell"] for e in events])
    clean = lambda v: np.where(np.isfinite(v), v, None).tolist()
    return {
        "author_key": record["author_key"],
        "events": len(events),
        "status": "SCORED" if events else "NO_OBSERVATIONS",
        "probabilities": clean(a["probabilities"]),
        "binary_log_odds": clean(a["binary_log_odds"]),
        "posterior_supported": a["posterior_supported"].tolist(),
        "finite_log_odds": a["finite_log_odds"].tolist(),
    }
