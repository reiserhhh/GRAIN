#!/usr/bin/env python3
"""Actual-reader transport of categorical text observations, without neural fitting."""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from . import f16_dynamic_text as dynamic
from . import f16_t1_text as static

ARMS = (
    "old_matched",
    "changed_uncalibrated",
    "changed_pilot_plugin",
    "changed_pilot_mixture",
    "changed_known_pool",
    "oracle_labels",
)


def material_bank():
    """Build a role-disjoint controlled bank without any author/latent parameter."""
    roles = (("source", 20), ("pilot", 25), ("application", 50))
    frames = {
        "source": ("For the written minutes", "In my own account", "At the review"),
        "pilot": (
            "To clarify the record",
            "In this short explanation",
            "For the follow-up",
        ),
        "application": (
            "Here is what happened",
            "To explain the exchange",
            "In the final account",
        ),
    }
    endings = (
        "The meeting concerned a proposed timetable.",
        "The discussion concerned a proposed budget.",
        "The proposal concerned a shared workspace.",
        "The meeting concerned a proposed schedule.",
        "The proposal concerned a community project.",
    )
    moments = ("Monday", "Tuesday", "Wednesday", "Thursday")
    rows = []
    for role, repetitions in roles:
        for variant in range(repetitions):
            for b in (0, 1):
                for q in (0, 1):
                    action, opposite = (
                        ("rejected", "supported")[b],
                        ("supported", "rejected")[b],
                    )
                    status = (("quoted", "hypothetical"), ("actual", "negated"))[q][
                        variant % 2
                    ]
                    bodies = {
                        "actual": f"I actually {action} the plan.",
                        "quoted": f"My colleague said, 'I actually {action} the plan.' That is my colleague's decision, not mine.",
                        "hypothetical": f"If I had to decide, I might have {action} the plan. This is hypothetical; I have not made that decision.",
                        "negated": f"It is wrong to say I {opposite} the plan. What I actually did was this: I {action} the plan.",
                    }
                    text = (
                        f"{frames[role][variant % 3]} on {moments[(variant // 15) % 4]}, "
                        f"{bodies[status]} {endings[(variant // 3) % 5]}"
                    )
                    rows.append(
                        {
                            "material_id": f"transport_{role}_{variant:02}_{b}_{q}",
                            "role": role,
                            "pilot_partition": ("fit" if variant < 15 else "validation")
                            if role == "pilot"
                            else None,
                            "pole": b,
                            "qualification": q,
                            "subject_status": status,
                            "variant": variant,
                            "text": text,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
    if len({r["text_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate designed material")
    return rows


def calibrated_channel(rows, qualification_key, *, required_role, partition=None):
    """Count observed output categories in specified known pilot/source strata."""
    counts = np.zeros((2, 2, 4), dtype=int)
    ids = set()
    for r in rows:
        if r["role"] != required_role or (
            partition and r["pilot_partition"] != partition
        ):
            raise ValueError("calibration role/partition mismatch")
        if r["material_id"] in ids:
            raise ValueError("duplicate calibration material")
        ids.add(r["material_id"])
        b, q = r["pole"], r["qualification"]
        if b not in (0, 1) or q not in (0, 1):
            raise ValueError("invalid designed strata")
        counts[b, q, static.category(r["pole_readout"], r[qualification_key])] += 1
    absent = np.argwhere(counts.sum(-1) == 0).tolist()
    return {
        "status": "MISSING_ANCHOR" if absent else "SUPPORTED",
        "required_strata": absent,
        "counts": counts.tolist(),
        "material_ids": sorted(ids),
    }


def channels(calibration, seed, count, mode="mixture"):
    """Return frozen error laws, refusing missing anchors before adding a prior."""
    if calibration["status"] != "SUPPORTED":
        raise ValueError("MISSING_ANCHOR: cannot calibrate unseen strata")
    c = np.asarray(calibration["counts"], float)
    if mode == "mixture":
        return static.channel_draws(c, seed, count)
    if mode not in ("plugin", "known_pool"):
        raise ValueError("unknown channel estimator")
    numerator = c + (0.5 if mode == "plugin" else 0)
    e = numerator / numerator.sum(-1, keepdims=True)
    return np.repeat(e[None], count, axis=0)


def observation_cache(rows, reader):
    """Export only readable text/scores; never carry designed labels into scoring."""
    if reader not in ("old", "new"):
        raise ValueError("unknown reader")
    key = "grain_qualification" if reader == "old" else "nli_qualification"
    return {
        r["material_id"]: {
            "text": r["text"],
            "text_sha256": r["text_sha256"],
            "pole_readout": float(r["pole_readout"]),
            "grain_qualification": float(r[key]),
            "nli_qualification": float(r[key]),
        }
        for r in rows
    }


def bundle_for(base, cache, channel):
    """Replace the observation channel only; preserve source units and models."""
    result = copy.deepcopy(base)
    result["materials"] = cache
    result["channel_draws"] = {
        k: np.asarray(channel).tolist() for k in ("hybrid_joint", "nli_joint")
    }
    return result


def score_panel(inputs, bundle, dynamic_target):
    """Run the actual frozen target estimator, returning both dynamic targets."""
    if dynamic_target:
        return dynamic.score_panel(inputs, bundle)
    counts = np.array([static.counts_from_input(x, bundle)[0] for x in inputs])
    return static.score_counts(counts, bundle)


def channel_diagnostics(calibration):
    """Describe finite output rank and eligible content contrast, not personality."""
    if calibration["status"] != "SUPPORTED":
        return {
            "status": calibration["status"],
            "required_strata": calibration["required_strata"],
        }
    c = np.asarray(calibration["counts"], float)
    e = c / c.sum(-1, keepdims=True)
    return {
        "empirical_rank": int(np.linalg.matrix_rank(e.reshape(4, 4))),
        "singular_values": np.linalg.svd(e.reshape(4, 4), compute_uv=False).tolist(),
        "eligible_pole_total_variation": float(np.abs(e[1, 1] - e[0, 1]).sum() / 2),
        "counts": c.astype(int).tolist(),
    }
