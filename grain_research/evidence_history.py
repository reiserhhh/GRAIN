#!/usr/bin/env python3
"""Versioned evidence dependencies and source-learned conditional updates.

Revision changes a declared claim, not the occurrence of an event. Gaussian
conditioning supplies a linear working predictor for the non-Gaussian text bank.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import nnls


def active_records(history):
    """Replay explicit dependencies within one author; never infer event identity.

    Aliases and summaries annotate provenance, not new likelihood terms. An
    explicit revision replaces one claim; other reports of that event remain.
    """
    seen, active, events, annotations = {}, {}, {}, []
    for original in history:
        r = deepcopy(original)
        identity, kind = r["record_id"], r["kind"]
        if identity in seen:
            raise ValueError("Duplicate record ID; use an explicit alias")
        if kind in {"new", "repeat", "revise"}:
            key = r["event_id"]
            context = (float(r["condition"]), r["block_id"])
            if not np.isfinite(context[0]) or context[0] not in (-1, 1):
                raise ValueError("Declared endpoint condition required")
            if key in events and events[key] != context:
                raise ValueError("Same event has conflicting context")
            if kind == "new" and key in events:
                raise ValueError("Existing event is not an independent new event")
            if kind == "repeat" and key not in events:
                raise ValueError("Repeat has no observed event")
            if kind == "revise":
                old = active.get(r["parent_id"])
                if old is None or old["event_id"] != key:
                    raise ValueError("Revision requires an active same-event parent")
                del active[r["parent_id"]]
            events[key] = context
            active[identity] = r
        elif kind in {"alias", "retract"}:
            parent = r["parent_id"]
            if parent not in active:
                raise ValueError("Reference must name an active claim")
            if kind == "retract":
                del active[parent]
            else:
                annotations.append(r)
        elif kind == "summary":
            if not r["parents"] or not set(r["parents"]) <= set(active):
                raise ValueError("Summary must cite active claims")
            annotations.append(r)
        else:
            raise ValueError("Unknown evidence operation")
        seen[identity] = r
    stale = [
        r["record_id"]
        for r in annotations
        if not set(r.get("parents", [r.get("parent_id")])) <= set(active)
    ]
    return {
        "records": list(active.values()),
        "history": list(seen.values()),
        "stale_derived": stale,
        "observed_events": len(events),
        "active_events": len({r["event_id"] for r in active.values()}),
    }


def restore_summary(summary):
    """Restore the numeric dependency appendix, not a scalar-only summary."""
    if summary.get("format") != "versioned_numeric_evidence_v1":
        raise ValueError("Current score is insufficient for targeted history updates")
    return active_records(summary["history"])


def kernels(records):
    """Five variance-component bases from declared condition/record relations."""
    z = np.array([r["condition"] for r in records], float)
    if not len(z) or not np.isfinite(z).all():
        raise ValueError("Nonempty finite observations required")
    x = np.c_[np.ones(len(z)), z]
    block = np.array([r["block_id"] for r in records])
    event = np.array([r["event_id"] for r in records])
    record = np.array([r.get("origin_id", r["record_id"]) for r in records])
    return x, np.array(
        [
            np.ones((len(z), len(z))),
            np.outer(z, z),
            block[:, None] == block[None, :],
            event[:, None] == event[None, :],
            record[:, None] == record[None, :],
        ],
        float,
    )


def fit_source(values, records, *, floor=1e-9):
    """Learn working moments from source authors, never their latent parameters."""
    y = np.asarray(values, float)
    x, bases = kernels(records)
    if y.ndim != 2 or y.shape[1] != len(x) or len(y) < 8 or not np.isfinite(y).all():
        raise ValueError("Eight source authors with common observed schedule required")
    design = bases.reshape(5, -1).T
    if np.linalg.matrix_rank(design) != 5:
        raise ValueError("Source design cannot separate the five covariance components")
    mean = np.linalg.lstsq(x, y.mean(axis=0), rcond=None)[0]
    empirical = np.cov(y, rowvar=False)
    raw, residual = nnls(design, empirical.ravel())
    variance = np.maximum(raw, floor)
    return {
        "mean": mean,
        "components": variance,
        "raw_components": raw,
        "floor_count": int((raw < floor).sum()),
        "source_authors": len(y),
        "source_records": len(x),
        "covariance_fit_residual": float(residual),
        "empirical_covariance": empirical,
        "working_model": "independent_level_response_plus_block_event_report",
    }


def observation_moments(model, records, mode="full"):
    """Shared blocks/events/report aliases determine working noise, not text count."""
    x, bases = kernels(records)
    c = np.asarray(model["components"], float)
    v = np.diag(c[:2])
    if mode == "full":
        r = np.einsum("k,kij->ij", c[2:], bases[2:])
    elif mode == "independent":
        r = np.eye(len(x)) * c[2:].sum()
    else:
        raise ValueError("Unknown dependence model")
    return x, v, r


def score_arrays(model, values, x, v, r):
    """Return linear conditional moments, material weights and prior contribution."""
    y = np.atleast_2d(np.asarray(values, float))
    if y.shape[1] != len(x) or not np.isfinite(y).all():
        raise ValueError("Aligned finite observation values required")
    k = x @ v @ x.T + r
    inv = np.linalg.pinv(k, hermitian=True, rcond=1e-12)
    unsupported = y - y @ (k @ inv).T
    if np.max(np.abs(unsupported)) > 1e-7:
        raise ValueError("Conflicting observations in an exact duplicate direction")
    w = v @ x.T @ inv
    offset = np.asarray(model["mean"]) - w @ x @ np.asarray(model["mean"])
    covariance = v - w @ x @ v
    return {
        "mean": y @ w.T + offset,
        "covariance": (covariance + covariance.T) / 2,
        "weights": w,
        "offset": offset,
        "observation_covariance": k,
    }


def score_records(model, values, records, mode="full"):
    """Score one observed record set, or an explicit event-only averaging control."""
    x, v, r = observation_moments(
        model, records, "full" if mode == "event_only" else mode
    )
    y = np.atleast_2d(values)
    if mode == "event_only":
        ids = np.array([s["event_id"] for s in records])
        unique = list(dict.fromkeys(ids))
        transform = np.array([(ids == u) / (ids == u).sum() for u in unique])
        x, y = transform @ x, y @ transform.T
        r = np.diag(np.diag(transform @ r @ transform.T))
        result = score_arrays(model, y, x, v, r)
        result["weights"] = result["weights"] @ transform
        return result
    return score_arrays(model, y, x, v, r)


def append_innovation(model, old_values, old_records, new_values, new_record):
    """Update through a correlated Gaussian innovation; exact duplicates add zero.

    Formula is exact for the working joint moments. It does not declare a binary
    text observation model Gaussian or turn report revision into append-only data.
    """
    old_values = np.atleast_2d(np.asarray(old_values, float))
    new_values = np.asarray(new_values, float).reshape(-1)
    old = score_records(model, old_values, old_records)
    x, v, r = observation_moments(model, old_records + [new_record])
    k = x @ v @ x.T + r
    inv = np.linalg.pinv(k[:-1, :-1], hermitian=True, rcond=1e-12)
    regress = inv @ k[:-1, -1]
    innovation_variance = float(k[-1, -1] - k[-1, :-1] @ regress)
    expected = x[-1] @ model["mean"] + (old_values - x[:-1] @ model["mean"]) @ regress
    innovation = new_values - expected
    cross = v @ x[-1] - v @ x[:-1].T @ regress
    if innovation_variance < -1e-8:
        raise ValueError("Negative innovation variance")
    if innovation_variance < 1e-10:
        if np.max(np.abs(innovation)) > 1e-7 or np.max(np.abs(cross)) > 1e-7:
            raise ValueError("Conflicting exact duplicate")
        gain = np.zeros(2)
    else:
        gain = cross / innovation_variance
    return {
        "mean": old["mean"] + innovation[:, None] * gain,
        "covariance": old["covariance"] - np.outer(gain, cross),
        "innovation": innovation,
        "innovation_variance": max(0.0, innovation_variance),
        "gain": gain,
    }
