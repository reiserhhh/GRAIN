#!/usr/bin/env python3
"""Finite nonlinear separation counterpoint; NOT a transported predictor.

An inadequate area summary does not imply that distinct factor-coordinate
nodes are intrinsically inseparable by every nonlinear observation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_cycle_capacity_hm97 import check, settings
from scripts.run_grain_observer_parity_hm96 import load, seal, sha, table, verify, write


def cardinal(points, queries):
    """Evaluate a finite cardinal polynomial constructed only from the nodes."""
    x, y = np.asarray(points, float), np.asarray(queries, float)
    distance = np.sum((x[:, None] - x[None, :]) ** 2, axis=-1)
    if np.min(distance[np.triu_indices(len(x), 1)]) <= 1e-20:
        raise ValueError("Aliased node support cannot use this cardinal construction")
    output = []
    for i in range(len(x)):
        other = np.arange(len(x)) != i
        numerators = np.sum((y[:, None] - x[None, other]) ** 2, axis=-1)
        output.append(np.prod(numerators / distance[i, other], axis=1))
    return np.stack(output, axis=-1)


def run():
    """Check node separation and exact finite target reconstruction, no fitting."""
    spec, out = settings()
    check(out)
    verify(out, load(out / "03_seal.json"))
    if load(out / "verification.json")["status"] != "VERIFIED_FINITE_HM97_NOT_GLOBAL":
        raise ValueError("Primary mechanism verification required")
    dest = out / "06_cardinal_counterpoint"
    dest.mkdir()
    data = np.load(out / "03/measured_capacity.npz")
    records, errors = [], []
    n = spec["states"]
    for context in range(4):
        x = data[f"points_{context}_0"]
        distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        separation = float(np.min(distance[np.triu_indices(n, 1)]))
        phi = cardinal(x, x)
        identity_error = float(np.max(np.abs(phi - np.eye(n))))
        target = data["target_kernel"][context]
        reconstructed = np.einsum("ia,abq,jb->ijq", phi, target, phi)
        reconstruction_error = float(np.max(np.abs(reconstructed - target)))
        errors.extend((identity_error, reconstruction_error))
        records.append(
            {
                "context": context,
                "min_factor_node_distance": separation,
                "cardinal_identity_error": identity_error,
                "finite_target_table_reconstruction_error": reconstruction_error,
                "polynomial_degree": 2 * (n - 1),
                "node_channels": n,
                "interpretation": "finite node-address lookup;not independent prediction",
            }
        )
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Finite cardinal identity failed")
    table(dest / "finite_separation.csv", records)
    summary = {
        "status": "VERIFIED_FINITE_CARDINAL_COUNTERPOINT",
        "rows": records,
        "no_predictor_fit": True,
        "uses_measured_target_table_as_lookup": True,
        "no_transport_or_psychological_specificity_claim": True,
        "max_error": max(errors),
        "companion_code": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
    }
    write(dest / "summary.json", summary)
    write(out / "06_seal.json", seal(out, dest))
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
