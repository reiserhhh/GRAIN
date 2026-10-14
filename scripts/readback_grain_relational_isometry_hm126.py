#!/usr/bin/env python3
"""Alternate scalar/Gram/expectation verification of the fixed geometry packet."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_relational_dual_frame_hm125 import budget
from scripts.run_grain_relational_isometry_hm126 import check


def root_psd(matrix):
    """Compute the symmetric positive root for an independent residual formula."""
    values, vectors = np.linalg.eigh(matrix)
    if min(values) < -1e-10:
        raise ValueError("Unexpected indefinite Gram matrix")
    return (vectors * np.sqrt(np.maximum(0, values))) @ vectors.T


def audit(out):
    """Verify all numerical identities, without refitting any held-out transport."""
    if H.load(out / "status.json")["state"] != "COLLECTED":
        raise RuntimeError("Only a complete terminal collection can be audited")
    with budget(out, "audit") as state:
        spec = check(out)["spec"]
        H.verify_hashes(out, H.load(out / "output_seal.json"))
        errors, entries = {}, 0
        with np.load(out / "00/profiles.npz", allow_pickle=False) as z:
            pairs = z["pairs"]
        for case in H.load(out / "00/cases.json"):
            key = case["key"]
            with np.load(out / f"00/{key}_bundle.npz", allow_pickle=False) as z:
                bundle = dict(z)
            with np.load(out / f"00/{key}_held.npz", allow_pickle=False) as z:
                held = z["vectors"]
            with np.load(out / f"01/{key}.npz", allow_pickle=False) as z:
                data = dict(z)
            error = 0.0
            for arm in spec["arms"]:
                if arm == "no_rotation":
                    coefficient = bundle["dual"].T
                else:
                    p = "matched" if arm == "matched_anchors" else "wrong"
                    left, right = bundle[p + "_left"], bundle[p + "_right"]
                    coefficient = left @ (right.T @ bundle["dual"].T)
                    error = max(
                        error,
                        float(np.max(abs(left.T @ left - np.eye(left.shape[1])))),
                        float(np.max(abs(right.T @ right - np.eye(right.shape[1])))),
                    )
                scores = (held - bundle["context_center"]) @ coefficient
                error = max(
                    error, float(np.max(abs(scores - data[arm + "_coordinates"])))
                )
                author = np.array(
                    [
                        sum(p[i] * scores[i] for i in range(len(scores)))
                        for p in data["laws"]
                    ]
                )
                contrast = (
                    len(case["cells"]) * (author[pairs[:, 0]] - author[pairs[:, 1]]) / 2
                )
                error = max(
                    error,
                    float(np.max(abs(author - data[arm + "_author_coordinates"]))),
                    float(np.max(abs(contrast - data[arm + "_contrast"]))),
                )
                entries += scores.size + author.size + contrast.size
            for prefix in ("calibration", "held"):
                gx, gy = data[prefix + "_source_gram"], data[prefix + "_target_gram"]
                root = root_psd(gx)
                eigenvalues = np.linalg.eigvalsh(root @ gy @ root)
                minimum = (
                    np.trace(gx)
                    + np.trace(gy)
                    - 2 * np.sqrt(np.maximum(0, eigenvalues)).sum()
                )
                summary = H.load(out / "summary.json")["cases"][key]
                field = "calibration" if prefix == "calibration" else "held_geometry"
                error = max(
                    error,
                    abs(float(minimum) - summary[field]["best_rotation_residual_sq"]),
                )
            if key == "common_condition_minus__fresh_instance":
                with np.load(out / "cold_scores.npz", allow_pickle=False) as z:
                    for name in z.files:
                        error = max(
                            error,
                            float(
                                np.max(abs(z[name] - data["matched_anchors_" + name]))
                            ),
                        )
            errors[key] = error
            if error > spec["numeric_atol"]:
                raise ValueError("Alternate arithmetic mismatch")
        H.write(
            out / "verification.json",
            {
                "status": "VERIFIED_FINITE_HM126_NOT_GLOBAL",
                "max_errors": errors,
                "numeric_score_entries": entries,
                "same_agent_second_formula_not_independent_reviewer": True,
                "no_heldout_inverse_or_target_fit": True,
            },
        )
        H.write(
            out / "audit_seal.json",
            {n: H.sha(out / n) for n in ("verification.json", "cold_scores.npz")},
        )
        state["state"] = "VERIFIED"
    print("HM126 alternate Gram and score audit PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/relational_isometry_r1",
    )
    args = parser.parse_args()
    audit(args.out.resolve())
