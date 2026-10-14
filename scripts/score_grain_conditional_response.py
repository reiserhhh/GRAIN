#!/usr/bin/env python3
"""Cold model-labelled conditional completion from the frozen reference package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.conditional_reference import completion_envelope, score_reference
from grain_research.conditional_response import (
    complete_functional,
    condition_kernel,
    observed_cells,
)
from scripts.prepare_grain_predecessor_reference import write


def restore_reference(value):
    """Restore numeric JSON arrays without importing a serialized model object."""
    keys = (
        "mean",
        "weights",
        "common5_weights",
        "source_indices",
        "source_authors_per_cell",
        "descriptor",
        "descriptor_mean",
        "descriptor_scale",
        "descriptor_raw",
        "noise",
    )
    return {k: np.asarray(v) if k in keys else v for k, v in value.items()}


def score_packet(
    conditions, vectors, ref, direction, targets, alpha, beta, target="full"
):
    """Keep finite observed scores separate from Gaussian imputation and its support."""
    h = np.asarray(vectors, float)
    if h.ndim != 2 or h.shape[1] != direction.shape[0] or not np.isfinite(h).all():
        raise ValueError("Finite same-encoder original-unit vectors required")
    norm = np.linalg.norm(h, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("Zero evidence vector")
    if target not in ("full", "common5"):
        raise ValueError("Unknown fixed reference target")
    values = h / norm[:, None] @ direction
    empirical = score_reference(conditions, values, ref, target)
    cells = observed_cells(conditions, values, ref)
    kernel = condition_kernel(ref, alpha, beta)
    result = complete_functional(
        ref,
        kernel,
        cells["indices"],
        cells["means"],
        cells["counts"],
        ref["weights" if target == "full" else "common5_weights"],
    )
    lo, hi, _, _ = completion_envelope(
        result["observed"], result["missing_mass"], direction, targets
    )
    prediction = result["model_completed"] @ targets
    return {
        "protocol": "GRAIN_CONDITIONAL_RESPONSE_V1",
        "target": target,
        "alpha": alpha,
        "beta": beta,
        "epistemic_status": "model_completed_functional_not_extra_observation_or_questionnaire_score",
        "model_sd_scope": "missing_latent_functional_given_frozen_moments_and_working_covariance",
        "actual_observed_score": empirical["actual"].tolist(),
        "unknown_material_mass": empirical["unknown_material_mass"],
        "known_cells": len(cells["indices"]),
        "unknown_cells": cells["unknown_cells"],
        "envelope_violations": int(
            np.sum((prediction < lo - 1e-12) | (prediction > hi + 1e-12))
        ),
        **{
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()
        },
    }


def main():
    """Score one private input packet; encoder/prefix/unit compatibility is required."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--arm", choices=["source_prior", "intercept", "grain_gp"], required=True
    )
    p.add_argument("--directions", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", choices=["full", "common5"], default="full")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an earlier individual score")
    ref = restore_reference(
        json.loads((args.model / "held/private_reference.json").read_text())
    )
    params = (
        {"alpha": 0, "beta": 0}
        if args.arm == "source_prior"
        else json.loads((args.model / "selected.json").read_text())[args.arm]
    )
    with (
        np.load(args.directions, allow_pickle=False) as z,
        np.load(args.input, allow_pickle=False) as data,
        threadpool_limits(limits=2),
    ):
        result = score_packet(
            data["conditions"],
            data["vectors"],
            ref,
            z["direction"],
            z["targets"],
            params["alpha"],
            params["beta"],
            args.target,
        )
    write(args.output, result)
    print(
        json.dumps(
            {
                "status": "SCORED",
                "coverage": result["coverage"],
                "empirical_complete": result["empirical_point"] is not None,
                "arm": args.arm,
            }
        )
    )


if __name__ == "__main__":
    main()
