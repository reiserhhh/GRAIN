#!/usr/bin/env python3
"""Post-outcome target-structure description, not calibration or factor promotion."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_comparison_geometry_hm226 as G
from scripts import run_grain_comparison_observable_contract_hm228 as C
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H


def geometry(x):
    """Describe every target using raw and equal-column-scale Gram accounts."""
    centered = x - x.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms == 0):
        raise ValueError("Undefined target structure must not be filled with zero")
    unit = centered / norms
    s, su = (
        np.linalg.svd(centered, compute_uv=False),
        np.linalg.svd(unit, compute_uv=False),
    )
    cosine = unit.T @ unit
    # An independent scalar correlation account is a check, not a fitted map.
    scalar = np.asarray(
        [
            [
                np.dot(centered[:, i], centered[:, j]) / (norms[i] * norms[j])
                for j in range(5)
            ]
            for i in range(5)
        ]
    )
    eigen = np.linalg.eigvalsh(cosine)[::-1] / 5
    return {
        "target_cosine": cosine.tolist(),
        "raw_singular_squared_ratio": (s * s / (s @ s)).tolist(),
        "equal_column_singular_squared_ratio": (su * su / (su @ su)).tolist(),
        "scalar_Gram_max": float(np.max(abs(scalar - cosine))),
        "eigen_singular_ratio_max": float(np.max(abs(eigen - su * su / (su @ su)))),
        "not_personality_variance_fraction": True,
    }


def describe(out):
    """Retain full five-target structure and all signs after primary analysis."""
    R.check(out, G.STAGES + ("ordinal_seal.json", "ordinal_verification_seal.json"))
    z = H.selected(out / "analysis.npz", ("objects", "labels", "odd", "even"))
    names = (
        "comparison_position",
        "standalone_level",
        "source_full_position",
        "source_departure_position",
    )
    objects = {name: geometry(z["objects"][:, i, :]) for i, name in enumerate(names)}
    objects["human_anchor"] = geometry(z["labels"])
    objects["comparison_odd_edges"] = geometry(z["odd"])
    checks = [
        v[k]
        for v in objects.values()
        for k in ("scalar_Gram_max", "eigen_singular_ratio_max")
    ]
    payload = {
        "post_outcome_exploratory_description": True,
        "targets": ["E", "N", "A", "C", "O"],
        "objects": objects,
        "comparison_even_mean": z["even"].mean(0).tolist(),
        "max_alternate_geometry_error": max(checks),
        "new_model_API_private_train_calls": 0,
        "no_target_relabeling_or_shared_component_deletion": True,
        "normalization_is_descriptive_not_cold_rescoring": True,
        "same_agent_not_blind": True,
    }
    R.write(out / "target_structure_description.json", payload)
    R.write(
        out / "target_structure_provenance.json",
        {
            "script": R.digest(Path(__file__)),
            "primary_code_and_results_unchanged": True,
        },
    )
    R.seal(
        out,
        "target_structure_seal.json",
        ("target_structure_description.json", "target_structure_provenance.json"),
    )
    if max(checks) > 1e-10:
        raise ValueError("Alternate structure accounts disagree;retain failure")
    print(
        "HM228 post-outcome all-target structure saved;no model/calibration", flush=True
    )


def main():
    """Write a separately identified descriptive companion once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=C.DEFAULT)
    args = parser.parse_args()
    describe(args.out)


if __name__ == "__main__":
    main()
