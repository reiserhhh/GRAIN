#!/usr/bin/env python3
"""Result-after arithmetic companion for HM66's zero-collision branch."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_grain_math_natural_summary as N
from scripts import run_grain_math_observation_fibers as F


def run(out):
    """Check all richer-input observations, not vacuous zero-witness counters."""
    F.sealed(out)
    for name, expected in N.read(out / "output_seal.json").items():
        if N.sha(out / name) != expected:
            raise ValueError("Frozen primary output changed: " + name)
    if N.read(out / "checkpoint.json")["status"] != "COMPLETED_HM66":
        raise ValueError("Terminal HM66 required")
    destination = out / "04_rich_input_arithmetic"
    destination.mkdir(exist_ok=False)
    N.write(
        destination / "manifest.json",
        {
            "code_sha256": N.sha(Path(__file__).resolve()),
            "primary_output_seal_sha256": N.sha(out / "output_seal.json"),
            "result_after_arithmetic_not_new_replication": True,
            "all_rows_einsum_plus_five_evenly_spaced_scalar_checks_per_query": True,
            "new_calls_fits": [0, 0],
        },
    )
    cohort, _ = F.cases()
    largest, scalar, envelope_violation, values, scalar_values = 0.0, 0.0, 0.0, 0, 0
    with np.load(out / "private_numeric_readouts.npz", allow_pickle=False) as saved:
        for group, c in cohort.items():
            for fold, w in enumerate(c["maps"]):
                key = f"{group}_{fold}"
                expected = saved[key + "__y"]
                alternate = np.einsum("ij,jk->ik", c["h"], w, optimize=False)
                differences = abs(alternate - expected)
                bound = F.M.roundoff_envelope(c["h"], w)
                largest = max(largest, float(differences.max()))
                envelope_violation = max(
                    envelope_violation, float(np.maximum(differences - bound, 0).max())
                )
                values += expected.size
                for i in np.linspace(0, len(c["h"]) - 1, 5, dtype=int):
                    for q in range(w.shape[1]):
                        actual = math.fsum(
                            c["h"][i, j] * w[j, q] for j in range(w.shape[0])
                        )
                        scalar = max(scalar, abs(actual - expected[i, q]))
                        scalar_values += 1
    if envelope_violation or max(largest, scalar) > 1e-9:
        raise ArithmeticError("Richer input control differs beyond arithmetic envelope")
    result = {
        "status": "VERIFIED_RESULT_AFTER_HM66_RICH_INPUT_ARITHMETIC",
        "actual_einsum_observation_entries": values,
        "actual_scalar_fsum_observations": scalar_values,
        "maximum_einsum_error": largest,
        "maximum_scalar_error": scalar,
        "maximum_envelope_violation": envelope_violation,
        "primary_no_fiber_witness_dot_counters_are_not_global_validation": True,
        "algebraic_h_to_psi_not_new_personality_model": True,
        "same_agent_not_independent_replication": True,
    }
    N.write(destination / "verification.json", result)
    print(result, flush=True)


def main():
    """Run only the arithmetic companion; never reacquire primary outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=F.OUT)
    with threadpool_limits(limits=1):
        run(p.parse_args().out)


if __name__ == "__main__":
    main()
