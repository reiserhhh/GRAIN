#!/usr/bin/env python3
"""Verify weighted exterior contraction without application target observations."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_current_transport import fundamental_currents
from scripts.run_grain_current_transport_hm98 import check, settings, stage_seal
from scripts.run_grain_observer_parity_hm96 import seal, sha, table, write


def run():
    """Keep both the finite contraction bounds and all actually pushed currents."""
    spec, out = settings()
    check(out)
    stage_seal(out, 4)
    prediction = np.load(out / "04/predictions.npz", allow_pickle=False)
    dest = out / "07_weighted_companion"
    dest.mkdir()
    pairs = np.array(list(itertools.combinations(range(17), 2)))
    d, _, _, _ = fundamental_currents(18)
    p = np.full(18, 1 / 18)
    rp = np.sqrt(p)
    rows, errors, arrays = [], [], {}
    for context in range(2):
        for arm in (*spec["banks"], "uniform_null", "known_correspondence"):
            t = (
                prediction[f"map_{context}_{arm}"]
                if arm in spec["banks"]
                else np.full((18, 18), 1 / 18)
                if arm == "uniform_null"
                else np.eye(18)
            )
            r = t.T @ p
            rr = np.sqrt(r)
            a = rp[:, None] * t / rr[None, :]
            c = a - np.outer(rp, rr)
            errors.extend(
                [
                    float(np.max(np.abs(a @ rr - rp))),
                    float(np.max(np.abs(a.T @ rp - rr))),
                ]
            )
            ev0, vectors0 = np.linalg.eigh(np.eye(18) - np.outer(rp, rp))
            ev1, vectors1 = np.linalg.eigh(np.eye(18) - np.outer(rr, rr))
            u, v = vectors0[:, ev0 > 0.5], vectors1[:, ev1 > 0.5]
            local = u.T @ c @ v
            singular = np.linalg.svd(local, compute_uv=False)
            compound = (
                local[pairs[:, 0, None], pairs[None, :, 0]]
                * local[pairs[:, 1, None], pairs[None, :, 1]]
                - local[pairs[:, 0, None], pairs[None, :, 1]]
                * local[pairs[:, 1, None], pairs[None, :, 0]]
            )
            expected = np.sort(singular[pairs[:, 0]] * singular[pairs[:, 1]])[::-1]
            actual = np.linalg.svd(compound, compute_uv=False)
            errors.append(float(np.max(np.abs(actual - expected))))
            gains = []
            for current in d:
                whitened = current / rp[:, None] / rp[None, :]
                pushed = (t.T @ current @ t) / rr[:, None] / rr[None, :]
                errors.append(float(np.max(np.abs(pushed - c.T @ whitened @ c))))
                gain = float(np.linalg.norm(pushed) / np.linalg.norm(whitened))
                gains.append(gain)
                if gain > expected[0] + spec["arithmetic_atol"]:
                    raise ValueError("Current exceeds declared exterior bound")
            if singular[0] > 1 + spec["arithmetic_atol"]:
                raise ValueError("Weighted Markov contraction failed")
            arrays[f"node_singular_{context}_{arm}"] = singular
            arrays[f"exterior_singular_{context}_{arm}"] = actual
            arrays[f"current_gains_{context}_{arm}"] = np.array(gains)
            rows.append(
                {
                    "context": context,
                    "arm": arm,
                    "weighted_node_gain": float(singular[0]),
                    "weighted_current_gain_bound": float(expected[0]),
                    "actual_current_gain_min": min(gains),
                    "actual_current_gain_max": max(gains),
                    "source_reference_min": float(r.min()),
                    "source_reference_max": float(r.max()),
                }
            )
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Weighted companion mismatch")
    np.savez_compressed(dest / "spectra.npz", **arrays)
    table(dest / "weighted_gains.csv", rows)
    summary = {
        "status": "VERIFIED_WEIGHTED_EXTERIOR_COMPANION",
        "max_error": max(errors),
        "native_application_target_reads": 0,
        "explicit_compound_checks": 8 * 136,
        "actual_current_checks": 8 * 136,
        "fitting": False,
        "original_scope_complete": False,
        "companion_code": {
            str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
            "docs/GRAIN_CURRENT_TRANSPORT_WEIGHTED_COMPANION_HM98.md": sha(
                ROOT / "docs/GRAIN_CURRENT_TRANSPORT_WEIGHTED_COMPANION_HM98.md"
            ),
        },
    }
    write(dest / "summary.json", summary)
    write(out / "07_seal.json", seal(out, dest))
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
