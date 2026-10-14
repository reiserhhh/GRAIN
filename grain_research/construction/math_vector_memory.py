#!/usr/bin/env python3
"""Actual-vector memory, readouts, and profile-preserving memory interventions."""

import hashlib
import json

import numpy as np
from scipy.special import logsumexp

TARGETS = ("E", "N", "A", "C", "O")
BITS = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
STATE_KEYS = {
    "schema",
    "bank_key",
    "frame_key",
    "next_block",
    "event_ledger",
    "log_weights",
    "compression",
}


def compile_bank(prototypes, codes, variance, source_ids):
    """Freeze three source channels and their full-profile prior without fitting."""
    z, g = np.asarray(prototypes, float), np.asarray(codes)
    ids = np.asarray(source_ids)
    if (
        z.ndim != 3
        or z.shape[1] != 3
        or not z.shape[2]
        or not np.isfinite(z).all()
        or g.shape != (len(z),)
        or not np.isin(g, np.arange(32)).all()
        or ids.shape != (len(z),)
        or ids.dtype.kind != "U"
        or len(set(ids)) != len(ids)
        or any(not uid for uid in ids)
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError("Finite three-block source bank and unique IDs required")
    counts = np.bincount(g.astype(int), minlength=32)
    if counts.min() == 0:
        raise ValueError("Unsupported source profiles cannot be neutral scores")
    h = hashlib.sha256()
    for a in (z.astype("<f8"), g.astype("<i8"), np.array(variance, "<f8")):
        h.update(a.tobytes())
    h.update(json.dumps(ids.tolist(), ensure_ascii=True).encode())
    return {
        "prototypes": z,
        "codes": g.astype(int),
        "counts": counts,
        "variance": np.array(variance),
        "source_ids": ids,
        "bank_key": np.array(h.hexdigest()),
    }


def normalize(log_weights):
    """Normalize finite log weights without clipping a source component."""
    w = np.asarray(log_weights, float)
    if w.ndim not in (1, 2) or not np.isfinite(w).all():
        raise ValueError("Finite component log weights required")
    return w - logsumexp(w, axis=-1, keepdims=True)


def prior(bank):
    """Use the declared uniform-profile, uniform-within-profile reference."""
    return -np.log(32.0) - np.log(bank["counts"][bank["codes"]])


def likelihood(bank, points, block):
    """Return Gaussian log kernels; query-only normalizers are omitted explicitly."""
    x = np.asarray(points, float)
    z = bank["prototypes"]
    if (
        type(block) is not int
        or block not in (0, 1, 2)
        or x.ndim != 2
        or x.shape[1] != z.shape[-1]
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite registered block means required")
    s2 = float(bank["variance"])
    return x @ z[:, block].T / s2 - np.sum(z[:, block] ** 2, 1) / (2 * s2)


def readout(bank, log_weights):
    """Read all32 profiles/five marginals without changing the stored memory."""
    w = normalize(log_weights)
    if w.shape[-1] != len(bank["codes"]):
        raise ValueError("Wrong bank size")
    logs = np.stack(
        [logsumexp(w[..., bank["codes"] == g], axis=-1) for g in range(32)],
        axis=-1,
    )
    target_logs = np.stack(
        [
            np.stack(
                [logsumexp(logs[..., BITS[:, q] == b], axis=-1) for b in (0, 1)],
                axis=-1,
            )
            for q in range(5)
        ],
        axis=-2,
    )
    return {
        "log_profile": logs,
        "profile": np.exp(logs),
        "target_logs": target_logs,
        "targets": np.exp(target_logs[..., 1]),
        "log_odds": target_logs[..., 1] - target_logs[..., 0],
    }


def reset_weights(bank, log_weights):
    """Preserve every current profile, reset only invisible within-profile weights."""
    p = readout(bank, log_weights)["log_profile"]
    return p[..., bank["codes"]] - np.log(bank["counts"][bank["codes"]])


def compare_next(bank, prefix, next_log_kernel):
    """Compare the same new observation after fine versus score-preserving reset."""
    a, k = normalize(prefix), np.asarray(next_log_kernel, float)
    if a.ndim != 2 or a.shape != k.shape or not np.isfinite(k).all():
        raise ValueError("Matched prefix and next-observation kernels required")
    b = reset_weights(bank, a)
    fa, fb = normalize(a + k), normalize(b + k)
    before, before_reset = readout(bank, a), readout(bank, b)
    after, after_reset = readout(bank, fa), readout(bank, fb)
    r = np.exp(a) - np.exp(b)
    kl = np.exp(k - np.max(k, axis=1, keepdims=True))
    denominator = np.sum(np.exp(a) * kl, axis=1)
    delta = after["targets"] - after_reset["targets"]
    identity = (
        np.sum(
            r[:, :, None]
            * kl[:, :, None]
            * (BITS[bank["codes"]][None] - after_reset["targets"][:, None]),
            axis=1,
        )
        / denominator[:, None]
    )
    return {
        "prefix": a,
        "reset_prefix": b,
        "fine_next": fa,
        "reset_next": fb,
        "before_profile": before["profile"],
        "before_reset_profile": before_reset["profile"],
        "fine_profile": after["profile"],
        "reset_profile": after_reset["profile"],
        "fine_target_logs": after["target_logs"],
        "reset_target_logs": after_reset["target_logs"],
        "delta_targets": delta,
        "delta_log_odds": after["log_odds"] - after_reset["log_odds"],
        "residual_identity": identity,
        "prefix_memory_TV": 0.5 * np.sum(np.abs(r), axis=1),
        "observation_log_evidence_ratio": logsumexp(a + k, axis=1)
        - logsumexp(b + k, axis=1),
    }


def experiment(bank, points):
    """Induce the prefix from actual first two blocks; never select a witness sign."""
    x = np.asarray(points, float)
    if x.ndim != 3 or x.shape[1:] != bank["prototypes"].shape[1:]:
        raise ValueError("Three full blocks required")
    kernels = [likelihood(bank, x[:, t], t) for t in range(3)]
    prefix = normalize(prior(bank)[None] + kernels[0] + kernels[1])
    return compare_next(bank, prefix, kernels[2])


def initialize(bank, frame_key):
    """Create a serializable bank/frame-bound state, not a predicted author ID."""
    if not isinstance(frame_key, str) or not frame_key:
        raise ValueError("Nonempty frame key required")
    return {
        "schema": "grain_vector_memory_v1",
        "bank_key": str(bank["bank_key"]),
        "frame_key": frame_key,
        "next_block": 0,
        "event_ledger": [],
        "log_weights": prior(bank).tolist(),
        "compression": "none",
    }


def validate_state(bank, state):
    """Reject a changed reference, malformed checkpoint, or false acquisition step."""
    if (
        not isinstance(state, dict)
        or set(state) != STATE_KEYS
        or state["schema"] != "grain_vector_memory_v1"
        or state["bank_key"] != str(bank["bank_key"])
        or not isinstance(state["frame_key"], str)
        or not state["frame_key"]
        or type(state["next_block"]) is not int
        or state["next_block"] not in (0, 1, 2, 3)
        or state["compression"] not in ("none", "within_profile_reset")
        or not isinstance(state["event_ledger"], list)
        or len(state["event_ledger"]) != 8 * state["next_block"]
    ):
        raise ValueError("Memory schema, bank or acquisition step changed")
    ids, addresses = set(), set()
    for e in state["event_ledger"]:
        if (
            not isinstance(e, dict)
            or set(e) != {"event_id", "view", "slot", "condition", "vector_sha256"}
            or not isinstance(e["event_id"], str)
            or not e["event_id"]
            or e["event_id"] in ids
            or type(e["view"]) is not int
            or e["view"] not in (0, 1)
            or type(e["slot"]) is not int
            or not 0 <= e["slot"] < 4 * state["next_block"]
            or (e["view"], e["slot"]) in addresses
            or type(e["condition"]) is not int
            or e["condition"] not in (0, 1)
            or not isinstance(e["vector_sha256"], str)
            or len(e["vector_sha256"]) != 64
        ):
            raise ValueError("Changed or repeated evidence ledger")
        ids.add(e["event_id"])
        addresses.add((e["view"], e["slot"]))
    w = np.asarray(state["log_weights"], float)
    if (
        w.shape != (len(bank["codes"]),)
        or not np.isfinite(w).all()
        or abs(logsumexp(w)) > 1e-10
    ):
        raise ValueError("Wrong or unnormalized component memory")
    return w


def ingest(bank, state, record):
    """Ingest exactly one addressed8-comment block without labels or model answers."""
    w = validate_state(bank, state)
    t = state["next_block"]
    if (
        t >= 3
        or not isinstance(record, dict)
        or set(record) != {"frame_key", "events"}
        or record["frame_key"] != state["frame_key"]
        or not isinstance(record["events"], list)
        or len(record["events"]) != 8
    ):
        raise ValueError("Same frame and next complete block required")
    seen = {e["event_id"] for e in state["event_ledger"]}
    addresses, vectors, ledger = set(), [], []
    for e in record["events"]:
        if not isinstance(e, dict) or set(e) != {
            "event_id",
            "vector",
            "condition",
            "view",
            "slot",
        }:
            raise ValueError("Only addressed vectors and observed conditions permitted")
        uid, v = e["event_id"], np.asarray(e["vector"], float)
        if (
            not isinstance(uid, str)
            or not uid
            or uid in seen
            or v.shape != (bank["prototypes"].shape[-1],)
            or not np.isfinite(v).all()
            or type(e["condition"]) is not int
            or e["condition"] not in (0, 1)
            or type(e["view"]) is not int
            or e["view"] not in (0, 1)
            or type(e["slot"]) is not int
            or not 4 * t <= e["slot"] < 4 * (t + 1)
            or (e["view"], e["slot"]) in addresses
        ):
            raise ValueError("Repeated, invalid or wrong-block event")
        seen.add(uid)
        addresses.add((e["view"], e["slot"]))
        vectors.append(v)
        ledger.append(
            {
                **{k: e[k] for k in ("event_id", "view", "slot", "condition")},
                "vector_sha256": hashlib.sha256(v.astype("<f8").tobytes()).hexdigest(),
            }
        )
    next_w = normalize(w + likelihood(bank, np.mean(vectors, 0)[None], t)[0])
    return {
        **state,
        "next_block": t + 1,
        "event_ledger": [*state["event_ledger"], *ledger],
        "log_weights": next_w.tolist(),
    }


def reset_memory(bank, state):
    """Apply an explicit software reset, retaining frame and entire event history."""
    w = validate_state(bank, state)
    return {
        **state,
        "log_weights": reset_weights(bank, w).tolist(),
        "compression": "within_profile_reset",
    }


def describe(bank, state):
    """Render present readouts separately from their recursively sufficient memory."""
    r = readout(bank, validate_state(bank, state))
    return {
        "frame_key": state["frame_key"],
        "next_block": state["next_block"],
        "event_count": len(state["event_ledger"]),
        "profiles": r["profile"].tolist(),
        "target_readings": dict(zip(TARGETS, r["targets"].tolist(), strict=True)),
        "compression": state["compression"],
    }
