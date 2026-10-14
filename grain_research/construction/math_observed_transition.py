#!/usr/bin/env python3
"""Target evidence in identified observed continuations, not accuracy search."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_observed_family as F

VERSION = "HM195_observed_transition_target_v1"
REGIME = "self_selected_next_cached_record"
TARGETS = ("E", "N", "A", "C", "O")
BITS = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)


def frequencies(current, outcome):
    """Count each author's actual window transitions, preserving both endpoints."""
    a, o = np.asarray(current), np.asarray(outcome)
    if (
        a.ndim != 3
        or a.shape != o.shape
        or a.shape[1] != 2
        or a.shape[2] == 0
        or not np.isin(a, np.arange(8)).all()
        or not np.isin(o, np.arange(8)).all()
    ):
        raise ValueError("Aligned author/two-window/current-outcome addresses required")
    p = np.zeros((*a.shape[:2], 8, 8))
    for u in range(len(a)):
        for w in range(2):
            p[u, w] = (
                np.bincount(
                    8 * a[u, w].astype(int) + o[u, w].astype(int), minlength=64
                ).reshape(8, 8)
                / a.shape[2]
            )
    return p


def compile_laws(probabilities, codes, folds, prior):
    """Use common profile priors; every observed source author has equal weight."""
    p, g, b, pi = (
        np.asarray(probabilities, float),
        np.asarray(codes),
        np.asarray(folds),
        np.asarray(prior, float),
    )
    if (
        p.shape != (len(g), 2, 8, 8)
        or b.shape != g.shape
        or g.ndim != 1
        or not np.isin(g, np.arange(32)).all()
        or not np.isin(b, (0, 1)).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum((2, 3)), 1, atol=1e-12, rtol=0)
        or pi.shape != (32,)
        or not np.isfinite(pi).all()
        or np.any(pi <= 0)
        or not np.isclose(pi.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Complete fixed-profile SOURCE probability laws required")
    j = np.empty((2, 2, 32, 8, 8))
    for bank in range(2):
        for profile in range(32):
            ix = (b == bank) & (g == profile)
            if not ix.any():
                raise ValueError("Missing source profile; no smoothing or rescue")
            j[bank, :, profile] = pi[profile] * p[ix].mean(0)
    return j, screened_law(j)


def screened_law(joint):
    """Match profile-current and current-outcome margins with Y independent of O|A."""
    j = np.asarray(joint, float)
    if (
        j.shape[-3:] != (32, 8, 8)
        or not np.isfinite(j).all()
        or np.any(j < 0)
        or not np.allclose(j.sum((-3, -2, -1)), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Normalized nonnegative full-profile transition law required")
    ga, ao = j.sum(-1), j.sum(-3)
    a = ao.sum(-1)
    return np.divide(
        ga[..., :, :, None] * ao[..., None, :, :],
        a[..., None, :, None],
        out=np.zeros_like(j),
        where=a[..., None, :, None] > 0,
    )


def target_fields(joint):
    """Read before/after target means, keeping unsupported transitions unavailable."""
    j = np.asarray(joint, float)
    screened_law(j)
    ao, ga = j.sum(-3), j.sum(-1)
    a = ga.sum(-2)
    positive = np.einsum("...gao,gq->...qao", j, BITS)
    singleton = np.einsum("...ga,gq->...qa", ga, BITS)
    after = np.divide(
        positive,
        ao[..., None, :, :],
        out=np.full_like(positive, np.nan),
        where=ao[..., None, :, :] > 0,
    )
    before = np.divide(
        singleton,
        a[..., None, :],
        out=np.full_like(singleton, np.nan),
        where=a[..., None, :] > 0,
    )
    return {
        "before": before,
        "after": after,
        "increment": after - before[..., :, :, None],
        "supported": ao > 0,
    }


def agreement(fields, support, reference):
    """Compare author banks across disjoint windows, not predictive accuracy."""
    h, s, rho = (
        np.asarray(fields, float),
        np.asarray(support, bool),
        np.asarray(reference, float),
    )
    if h.shape != (2, 2, 5, 8, 8) or s.shape != (2, 2, 8, 8) or rho.shape != (8, 8):
        raise ValueError("Four fixed field families and unlabelled reference required")
    if not np.isfinite(rho).all() or np.any(rho < 0) or not np.isclose(rho.sum(), 1):
        raise ValueError("Normalized reference required; no coverage renormalization")
    common = s.all((0, 1)) & (rho > 0)
    if not np.isfinite(h[:, :, :, common]).all():
        raise ValueError("Nonfinite supported action")
    x = h[:, :, :, common]
    r = rho[common]
    cross = 0.5 * np.sum(r * (x[0, 0] * x[1, 1] + x[0, 1] * x[1, 0]), axis=-1)
    energy = np.einsum("i,bwqi->bwq", r, x * x)
    return {
        "by_target": cross,
        "restricted_mean": float(cross.mean()),
        "reference_mass": float(r.sum()),
        "missing_mass": float(rho[~common].sum()),
        "full_primary_available": not np.any((rho > 0) & ~common),
        "energy": energy,
        "common": common,
    }


def compile_bundle(joint, thresholds, representation_id):
    """Export a single-transition observer, with no source author catalogue."""
    j = np.asarray(joint, float)
    screened_law(j)
    if (
        j.shape != (32, 8, 8)
        or not isinstance(representation_id, str)
        or not representation_id
    ):
        raise ValueError("Aggregate law and representation identity required")
    F.states(np.zeros((0, 15)), thresholds)
    return {
        "version": VERSION,
        "regime": REGIME,
        "targets": list(TARGETS),
        "thresholds": list(thresholds),
        "representation_id": representation_id,
        "profile_current_outcome_law": j.tolist(),
        "reading": "one_transition_source_posterior_NOT_personality_truth_or_full_history",
    }


def query(bundle, record):
    """Read two identified successive numeric records; no hidden labels accepted."""
    if (
        set(bundle)
        != {
            "version",
            "regime",
            "targets",
            "thresholds",
            "representation_id",
            "profile_current_outcome_law",
            "reading",
        }
        or bundle["version"] != VERSION
        or bundle["regime"] != REGIME
        or bundle["targets"] != list(TARGETS)
        or bundle["reading"]
        != "one_transition_source_posterior_NOT_personality_truth_or_full_history"
        or set(record) != {"representation_id", "regime", "current", "next"}
        or record["regime"] != REGIME
        or record["representation_id"] != bundle["representation_id"]
    ):
        raise ValueError(
            "Frozen representation/regime and identified observations required"
        )
    required = {
        "author_key",
        "event_id",
        "timestamp",
        "venue_code",
        "condition",
        "coordinates",
    }
    e = [record["current"], record["next"]]
    for v in e:
        if (
            set(v) != required
            or not isinstance(v["author_key"], str)
            or not v["author_key"]
            or not isinstance(v["event_id"], str)
            or not v["event_id"]
            or type(v["venue_code"]) is not int
            or type(v["condition"]) is not int
            or v["condition"] not in (0, 1)
            or type(v["timestamp"]) not in (int, float)
            or not np.isfinite(v["timestamp"])
        ):
            raise ValueError("Only provenance and numeric observation fields allowed")
    if (
        e[0]["author_key"] != e[1]["author_key"]
        or e[0]["event_id"] == e[1]["event_id"]
        or e[0]["timestamp"] >= e[1]["timestamp"]
    ):
        raise ValueError(
            "Distinct strictly ordered records from the SAME author required"
        )
    st = F.states(np.asarray([v["coordinates"] for v in e]), bundle["thresholds"])
    a = 4 * e[0]["condition"] + int(st[0])
    o = 4 * int(e[0]["venue_code"] == e[1]["venue_code"]) + int(st[1])
    j = np.asarray(bundle["profile_current_outcome_law"], float)
    screened_law(j)
    if j.shape != (32, 8, 8):
        raise ValueError("Changed aggregate observation law")
    mass = float(j[:, a, o].sum())
    result = {
        "status": "SUPPORTED" if mass > 0 else "UNOBSERVED_TRANSITION",
        "current_cell": a,
        "outcome_cell": o,
        "event_ids": [v["event_id"] for v in e],
        "targets": list(TARGETS),
    }
    if mass == 0:
        return result | {"before": None, "after": None, "increment": None}
    before = j[:, a].sum(-1) / j[:, a].sum()
    after = j[:, a, o] / mass
    return result | {
        "before": (before @ BITS).tolist(),
        "after": (after @ BITS).tolist(),
        "increment": ((after - before) @ BITS).tolist(),
        "profile_before": before.tolist(),
        "profile_after": after.tolist(),
        "observable_probability": mass / float(j[:, a].sum()),
    }
