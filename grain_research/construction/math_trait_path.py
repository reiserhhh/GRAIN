#!/usr/bin/env python3
"""Source-oriented edge observers on qualified numerical event ledgers.

The complete-graph decomposition is standard Hodge algebra. A fixed-end
permutation reference retains each event bag; it does not simulate new texts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import math_observed_family as O
from .math_trait_anchor import TRAITS

VERSION = "HM172_trait_path_observation_v1"
STATES = 8


def code_hash():
    """Bind the cold measurement to its scientific implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def path(value):
    """Validate the declared eight-state numerical event sequence."""
    p = np.asarray(value)
    if (
        p.ndim != 1
        or len(p) < 4
        or not np.isfinite(p).all()
        or not np.equal(p, np.floor(p)).all()
        or np.any((p < 0) | (p >= STATES))
    ):
        raise ValueError("At least four valid eight-state events required")
    return p.astype(int)


def edge_mass(value):
    """Count actual open adjacency without closing endpoints or adding edges."""
    p = path(value)
    counts = np.zeros((STATES, STATES), int)
    np.add.at(counts, (p[:-1], p[1:]), 1)
    return counts / (len(p) - 1)


def fixed_end_mean(value):
    """Exact edge expectation under identity permutations of internal events."""
    p = path(value)
    m = np.bincount(p[1:-1], minlength=STATES).astype(float)
    total = len(p) - 2
    counts = np.outer(m, m) - np.diag(m)
    counts[p[0], :] += m
    counts[:, p[-1]] += m
    return counts / (total * (len(p) - 1))


def parts(value):
    """Split signed fields into traffic, endpoint gradient and cyclic current.

    Axes0/1 are the complete unweighted graph; extra axes are observers.
    This does not infer a continuous-time physical drift or full simplicial
    curl/harmonic separation.
    """
    f = np.asarray(value, float)
    if f.shape[:2] != (STATES, STATES) or not np.isfinite(f).all():
        raise ValueError("Finite aligned eight-state matrix required")
    symmetric = (f + f.swapaxes(0, 1)) / 2
    skew = (f - f.swapaxes(0, 1)) / 2
    divergence = skew.sum(1)
    gradient = (divergence[:, None] - divergence[None, :]) / STATES
    return symmetric, gradient, skew - gradient


def contract(mass, field):
    """Pair an empirical signed edge measure with every fixed target observer."""
    return np.einsum("ij,ijq->q", mass, field)


def compile_law(masses, binary, cuts, node_law, reader):
    """Calibrate one five-target law with exactly equal mass per source author."""
    j, y = np.asarray(masses, float), np.asarray(binary, float)
    if (
        j.ndim != 3
        or j.shape[1:] != (STATES, STATES)
        or not len(j)
        or y.shape != (len(j), 5)
        or not np.isfinite(j).all()
        or np.any(j < 0)
        or not np.allclose(j.sum((1, 2)), 1)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Aligned probability ledgers and source labels required")
    ref, prior = j.mean(0), y.mean(0)
    if np.any(ref <= 0) or np.any((prior <= 0) | (prior >= 1)):
        raise ValueError("Unsupported source edge or label group;no smoothing")
    joint = np.einsum("uij,uq->ijq", j, y) / len(j)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "traits": list(TRAITS),
        "reader": reader,
        "source_authors": len(j),
        "coordinate_cutoffs": np.asarray(cuts, float).tolist(),
        "reference": ref.tolist(),
        "prior": prior.tolist(),
        "field": (joint / ref[:, :, None]).tolist(),
        "node_field": node_law["probabilities"],
        "node_prior": node_law["prior"],
        "regime": "self_selected_cached_publication_order_not_causal_response",
    }


def verify_law(bundle):
    """Reject unknown units, unsupported edges and altered implementations."""
    keys = {
        "version",
        "code_hash",
        "traits",
        "reader",
        "source_authors",
        "coordinate_cutoffs",
        "reference",
        "prior",
        "field",
        "node_field",
        "node_prior",
        "regime",
    }
    if (
        set(bundle) != keys
        or bundle["version"] != VERSION
        or bundle["code_hash"] != code_hash()
        or bundle["traits"] != list(TRAITS)
        or bundle["source_authors"] <= 0
        or bundle["regime"]
        != "self_selected_cached_publication_order_not_causal_response"
    ):
        raise ValueError("Unknown or altered source observer")
    r, f, pi, node, npi = [
        np.asarray(bundle[k], float)
        for k in ("reference", "field", "prior", "node_field", "node_prior")
    ]
    if (
        r.shape != (8, 8)
        or f.shape != (8, 8, 5)
        or pi.shape != (5,)
        or node.shape != (8, 5)
        or npi.shape != (5,)
        or not all(np.isfinite(v).all() for v in (r, f, pi, node, npi))
        or np.any(r <= 0)
        or not np.isclose(r.sum(), 1)
        or np.any((f < -1e-12) | (f > 1 + 1e-12))
        or not np.allclose(contract(r, f), pi)
    ):
        raise ValueError("Inconsistent source probability law")
    O.states(np.zeros((0, 15)), bundle["coordinate_cutoffs"])


def accounts(joint, bag, bundle):
    """Retain both the bag/end law and its signed chronological remainder."""
    f, r = np.asarray(bundle["field"]), np.asarray(bundle["reference"])
    j, b = np.asarray(joint, float), np.asarray(bag, float)
    delta = j - b
    fs, fg, fc = parts(f)
    ms, mg, mc = parts(j - r)
    bs, bg, bc = parts(b - r)
    ds, dg, dc = parts(delta)
    return {
        "relative": contract(j - r, f),
        "bag_expected": contract(b - r, f),
        "reorder": contract(delta, f),
        "traffic": contract(ms, fs),
        "endpoint": contract(mg, fg),
        "cycle": contract(mc, fc),
        "bag_traffic": contract(bs, fs),
        "bag_endpoint": contract(bg, fg),
        "bag_cycle": contract(bc, fc),
        "reorder_traffic": contract(ds, fs),
        "reorder_endpoint": contract(dg, fg),
        "reorder_cycle": contract(dc, fc),
        "old_relative": j.sum(1) @ np.asarray(bundle["node_field"])
        - bundle["node_prior"],
    }


def cold_author(bundle, record):
    """Score the same events without application labels or model calls."""
    verify_law(bundle)
    if set(record) != {"author_key", "events"}:
        raise ValueError("Cold input permits only author handle and event ledger")
    events = record["events"]
    if len(events) != 18 or any(
        set(e) != {"event_id", "condition", "coordinates"} for e in events
    ):
        raise ValueError("Exactly18 qualified numerical events required")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct opaque event identities required")
    c = np.asarray([e["condition"] for e in events])
    if not np.isin(c, (0, 1)).all():
        raise ValueError("Recorded binary condition required")
    s = O.states(
        np.asarray([e["coordinates"] for e in events]), bundle["coordinate_cutoffs"]
    )
    p = 4 * c.astype(int) + s
    j, b = edge_mass(p), fixed_end_mean(p)
    a = accounts(j, b, bundle)
    return {
        "author_key": record["author_key"],
        "status": "OBSERVED_DIRECTED_TARGET_FUNCTION",
        "events": 18,
        "edges": 17,
        "reader": bundle["reader"],
        "traits": list(TRAITS),
        "labels_received": False,
        "person_posterior": False,
        "accounts": {k: v.tolist() for k, v in a.items()},
    }
