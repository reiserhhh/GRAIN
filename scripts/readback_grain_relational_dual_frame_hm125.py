#!/usr/bin/env python3
"""Alternative least-squares/direct-sum audit, never a transport rescue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_relational_dual_frame_hm125 import VIEWS, budget, check


def audit(out):
    """Reconstruct all scores and controls with independent arithmetic formulas."""
    if H.load(out / "status.json")["state"] != "COLLECTED":
        raise RuntimeError("Audit only a terminal fully collected packet")
    with budget(out, "audit") as state:
        spec = check(out)["spec"]
        H.verify_hashes(out, H.load(out / "output_seal.json"))
        with np.load(out / "00/frame.npz", allow_pickle=False) as z:
            frame = dict(z)
        with np.load(out / "00/profiles.npz", allow_pickle=False) as z:
            pairs, theta = z["pairs"], z["theta"]
        d = frame["columns"]
        errors, entries = {}, 0
        for name in VIEWS:
            with np.load(out / f"01/{name}.npz", allow_pickle=False) as z:
                data = dict(z)
            x = data["vectors"] - frame["center"]
            independent = np.linalg.lstsq(d, x.T, rcond=None)[0].T
            score_error = float(np.max(abs(independent - data["coordinates"])))
            reconstruction = data["coordinates"] @ d.T + data["complement"]
            orthogonal = float(np.max(abs(data["complement"] @ d)))
            sums = np.array(
                [
                    sum(p[i] * independent[i] for i in range(len(x)))
                    for p in data["laws"]
                ]
            )
            author_error = float(np.max(abs(sums - data["author_coordinates"])))
            factor_cols = []
            rows = H.load(out / "01/rows.json")[name]
            indices = data["visible_indices"]
            for j in indices:
                f, c = divmod(int(j), 2)
                plus = [
                    i
                    for i, r in enumerate(rows)
                    if r["family"] == f
                    and (r["condition"] + 1) // 2 == c
                    and r["polarity"] == r["owner"]
                ]
                minus = [
                    i
                    for i, r in enumerate(rows)
                    if r["family"] == f
                    and (r["condition"] + 1) // 2 == c
                    and r["polarity"] != r["owner"]
                ]
                factor_cols.append(
                    (data["vectors"][plus].mean(0) - data["vectors"][minus].mean(0)) / 2
                )
            dv = np.array(factor_cols).T
            transport = np.linalg.lstsq(d, dv, rcond=None)[0]
            formula = (
                theta[pairs[:, 0]].reshape(len(pairs), 12)[:, indices] @ transport.T
            )
            contrast = len(indices) * (sums[pairs[:, 0]] - sums[pairs[:, 1]]) / 2
            marginal_error = 0.0
            for field in ("wording", "family", "condition", "polarity", "owner"):
                for value in {r[field] for r in rows}:
                    ids = [i for i, r in enumerate(rows) if r[field] == value]
                    mass = data["laws"][:, ids].sum(1)
                    marginal_error = max(
                        marginal_error, float(np.max(abs(mass - mass[0])))
                    )
            missing = 0.0
            if name == "fresh_instance":
                absent = np.arange(1, 12, 2)
                missing_ids = np.concatenate((pairs[absent].ravel(), [0]))
                missing = float(np.max(abs(sums[missing_ids] - sums[0])))
                if not np.array_equal(
                    data["laws"][missing_ids],
                    np.repeat(data["laws"][:1], len(missing_ids), axis=0),
                ):
                    raise ValueError(
                        "Unobserved condition fabricated distinct evidence laws"
                    )
            cold_error = 0.0
            if name == "fresh_instance":
                with np.load(out / "cold_scores.npz", allow_pickle=False) as z:
                    cold_error = float(np.max(abs(z["coordinates"] - independent)))
            errors[name] = {
                "score_lstsq_max_error": score_error,
                "vector_reconstruction_max_error": float(
                    np.max(abs(reconstruction - x))
                ),
                "orthogonal_complement_max_error": orthogonal,
                "author_direct_sum_max_error": author_error,
                "transport_lstsq_max_error": float(
                    np.max(abs(transport - data["transport"]))
                ),
                "action_formula_max_error": float(np.max(abs(contrast - formula))),
                "all_margin_max_error": marginal_error,
                "unobserved_profile_max_error": missing,
                "cold_cli_max_error": cold_error,
            }
            entries += (
                independent.size + reconstruction.size + sums.size + transport.size
            )
            if max(errors[name].values()) > spec["numeric_atol"]:
                raise ValueError("Independent numeric account failed")
        dual_error = float(np.max(abs(frame["dual"] @ d - np.eye(12))))
        if dual_error > spec["numeric_atol"]:
            raise ValueError("Source dual failed")
        H.write(
            out / "verification.json",
            {
                "status": "VERIFIED_FINITE_HM125_NOT_GLOBAL",
                "errors": errors,
                "dual_identity_max_error": dual_error,
                "numeric_entries_read_back": entries,
                "audit_is_second_formula_same_agent_not_blind_replication": True,
                "no_application_inverse_or_recalibration": True,
            },
        )
        H.write(
            out / "audit_seal.json",
            {n: H.sha(out / n) for n in ("verification.json", "cold_scores.npz")},
        )
        state["state"] = "VERIFIED"
    print("HM125 alternative audit PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/relational_dual_frame_r1",
    )
    args = parser.parse_args()
    audit(args.out.resolve())
