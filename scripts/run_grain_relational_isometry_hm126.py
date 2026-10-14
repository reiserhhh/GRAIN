#!/usr/bin/env python3
"""One finite relation-geometry/anchor-transport test, with held-out wording."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relational_dual_frame as F
from grain_research.construction import math_relational_isometry as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_relational_dual_frame_hm125 import (
    VIEWS,
    budget,
    save,
)
from scripts.run_grain_relational_dual_frame_hm125 import check as previous_check

SPEC = ROOT / "configs/grain_relational_isometry_hm126.json"
CODE = (
    "configs/grain_relational_isometry_hm126.json",
    "docs/GRAIN_RELATIONAL_ISOMETRY_PLAN_HM126.md",
    "grain_research/construction/math_relational_isometry.py",
    "grain_research/construction/math_relational_dual_frame.py",
    "scripts/run_grain_relational_isometry_hm126.py",
    "scripts/readback_grain_relational_isometry_hm126.py",
    "scripts/score_grain_relational_isometry_cold.py",
    "scripts/run_grain_relational_dual_frame_hm125.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "tests/test_grain_relational_isometry.py",
)


def check(out):
    """Authenticate the fixed numeric inputs and construction before each phase."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def select(x, rows, wording, conditions):
    """Select declared calibration or held-out evidence, not a winning subset."""
    ids = [
        i
        for i, r in enumerate(rows)
        if r["wording"] == wording and (r["condition"] + 1) // 2 in conditions
    ]
    return x[ids], [rows[i] for i in ids]


def prepare(out):
    """Fit transport on wording0 only, then seal all held-out numeric material."""
    if out.exists():
        raise FileExistsError("Preserve previous and failed attempts")
    (out / "00").mkdir(parents=True)
    H.write(out / "status.json", {"state": "PREPARING", "cpu_seconds": 0.0})
    with budget(out, "prepare") as state:
        spec = H.load(SPEC)
        if spec["max_cpu_seconds"] != 600:
            raise ValueError("Inherited runner budget must remain600seconds")
        src = ROOT / spec["input"]
        previous_check(src)
        H.verify_hashes(src, H.load(src / "output_seal.json"))
        H.verify_hashes(src, H.load(src / "audit_seal.json"))
        all_rows = H.load(src / "01/rows.json")
        with np.load(src / "01/source.npz", allow_pickle=False) as z:
            original = z["vectors"]
        with np.load(src / "00/profiles.npz", allow_pickle=False) as z:
            save(out / "00/profiles.npz", theta=z["theta"], pairs=z["pairs"])
        cases, inputs = (
            [],
            [
                src / "00/profiles.npz",
                src / "01/rows.json",
                src / "00/manifest.json",
                src / "output_seal.json",
                src / "audit_seal.json",
            ],
        )
        for stratum, conditions in spec["strata"].items():
            cells = list(itertools.product(range(6), conditions))
            x, calibration_rows = select(original, all_rows["source"], 0, conditions)
            xh, source_held_rows = select(original, all_rows["source"], 1, conditions)
            columns = M.action_columns(x, calibration_rows, cells)
            frame = F.fit_frame(x, columns)
            for view in VIEWS:
                if view == "fresh_instance" and len(conditions) == 2:
                    continue
                with np.load(src / f"01/{view}.npz", allow_pickle=False) as z:
                    values = z["vectors"]
                y, calibration = select(values, all_rows[view], 0, conditions)
                yh, held_rows = select(values, all_rows[view], 1, conditions)
                target_columns = M.action_columns(y, calibration, cells)
                matched = M.supported_polar(columns, target_columns)
                permutation = [cells.index(((f + 1) % 6, c)) for f, c in cells]
                wrong = M.supported_polar(columns, target_columns[:, permutation])
                key = stratum + "__" + view
                save(
                    out / f"00/{key}_bundle.npz",
                    source_columns=columns,
                    source_center=frame["center"],
                    context_center=y.mean(0),
                    dual=frame["dual"],
                    source_singular=frame["singular"],
                    target_columns=target_columns,
                    matched_left=matched["left"],
                    matched_right=matched["right"],
                    matched_cross_singular=matched["cross_singular"],
                    wrong_left=wrong["left"],
                    wrong_right=wrong["right"],
                )
                save(out / f"00/{key}_held.npz", vectors=yh, source_vectors=xh)
                cases.append(
                    {
                        "key": key,
                        "view": view,
                        "stratum": stratum,
                        "cells": cells,
                        "calibration_rows": calibration,
                        "held_rows": held_rows,
                        "source_held_rows": source_held_rows,
                    }
                )
                inputs.append(src / f"01/{view}.npz")
        H.write(out / "00/cases.json", cases)
        H.write(
            out / "00/manifest.json",
            {
                "spec": spec,
                "code": {n: H.sha(ROOT / n) for n in CODE},
                "source": {str(p.relative_to(ROOT)): H.sha(p) for p in set(inputs)},
                "inputs": H.seals(out, out / "00"),
                "context_calibration_is_paid_known_public_design": True,
                "held_wording_never_fit": True,
                "native_answers_read": False,
                "new_model_calls": 0,
                "original_scope_complete": False,
            },
        )
        state["state"] = "PREPARED"
    print("HM126 seven public-anchor bundles sealed;wording1 held out", flush=True)


def account_without_arrays(account):
    """Keep scalar diagnostics in JSON and complete signed Gram matrices in NPZ."""
    return {k: v for k, v in account.items() if not isinstance(v, np.ndarray)}


def execute(out):
    """Evaluate fixed geometric laws and author contrasts without any refit."""
    if H.load(out / "status.json")["state"] != "PREPARED":
        raise RuntimeError("Do not re-execute terminal or failed runs")
    with budget(out, "execute") as state:
        spec = check(out)["spec"]
        (out / "01").mkdir()
        with np.load(out / "00/profiles.npz", allow_pickle=False) as z:
            theta, pairs = z["theta"], z["pairs"]
        summaries = {}
        for case in H.load(out / "00/cases.json"):
            key = case["key"]
            cells = [tuple(c) for c in case["cells"]]
            with np.load(out / f"00/{key}_bundle.npz", allow_pickle=False) as z:
                bundle = dict(z)
            with np.load(out / f"00/{key}_held.npz", allow_pickle=False) as z:
                yh, xh = z["vectors"], z["source_vectors"]
            source_held = M.action_columns(xh, case["source_held_rows"], cells)
            target_held = M.action_columns(yh, case["held_rows"], cells)
            calibration_account = M.gram_account(
                bundle["source_columns"], bundle["target_columns"]
            )
            held_account = M.gram_account(source_held, target_held)
            gx, gy = held_account["source_gram"], held_account["target_gram"]
            permuted = []
            for permutation in itertools.permutations(range(6)):
                indices = [cells.index((permutation[f], c)) for f, c in cells]
                gp = gy[np.ix_(indices, indices)]
                permuted.append(
                    float(np.sum(gx * gp) / (np.linalg.norm(gx) * np.linalg.norm(gp)))
                )
            laws = F.make_profile_laws(case["held_rows"], theta, cells)
            source_laws = F.make_profile_laws(case["source_held_rows"], theta, cells)
            source_scores = (xh - bundle["source_center"]) @ bundle["dual"].T
            source_author = source_laws @ source_scores
            source_contrast = (
                len(cells)
                * (source_author[pairs[:, 0]] - source_author[pairs[:, 1]])
                / 2
            )
            reference_matrix = bundle["dual"] @ source_held
            arrays = {
                "held_source_gram": gx,
                "held_target_gram": gy,
                "calibration_source_gram": calibration_account["source_gram"],
                "calibration_target_gram": calibration_account["target_gram"],
                "source_author": source_author,
                "source_contrast": source_contrast,
                "source_reference_matrix": reference_matrix,
                "permutation_alignments": np.array(permuted),
                "laws": laws,
            }
            arm_accounts = {}
            for arm in spec["arms"]:
                scored = M.score_vectors(yh, bundle, arm)
                author = F.compose_profiles(laws, scored)
                contrast = (
                    len(cells)
                    * (
                        author["coordinates"][pairs[:, 0]]
                        - author["coordinates"][pairs[:, 1]]
                    )
                    / 2
                )
                matrix = M.action_columns(
                    scored["coordinates"], case["held_rows"], cells
                )
                ids = [2 * f + c for f, c in cells]
                predicted = (
                    theta[pairs[:, 0]].reshape(len(pairs), 12)[:, ids] @ matrix.T
                )
                if np.max(abs(contrast - predicted)) > spec["numeric_atol"]:
                    raise ValueError("Finite author expectation identity failed")
                for name, value in scored.items():
                    arrays[arm + "_" + name] = value
                    arrays[arm + "_author_" + name] = author[name]
                arrays[arm + "_contrast"] = contrast
                arrays[arm + "_response_matrix"] = matrix
                arm_accounts[arm] = {
                    "matrix_discrepancy_over_source_held_norm": float(
                        np.linalg.norm(matrix - reference_matrix)
                        / np.linalg.norm(reference_matrix)
                    ),
                    "author_contrast_discrepancy_over_source_norm": float(
                        np.linalg.norm(contrast - source_contrast)
                        / np.linalg.norm(source_contrast)
                    ),
                    "author_identity_max_error": float(
                        np.max(abs(contrast - predicted))
                    ),
                    "mean_complement_energy": float(
                        np.mean(np.sum(scored["complement"] ** 2, axis=1))
                    ),
                }
            save(out / f"01/{key}.npz", **arrays)
            summaries[key] = {
                "calibration": account_without_arrays(calibration_account),
                "held_geometry": account_without_arrays(held_account),
                "held_geometry_correct_family_rank": 1
                + int(
                    np.sum(
                        np.array(permuted)
                        > held_account["normalized_gram_alignment"] + 1e-12
                    )
                ),
                "permutation_count": len(permuted),
                "held_records": len(yh),
                "calibration_records": len(yh),
                "coordinates": len(cells),
                "profiles": len(laws),
                "origin_coordinate_norm": float(
                    np.linalg.norm(
                        (bundle["context_center"] - bundle["source_center"])
                        @ bundle["dual"].T
                    )
                ),
                "unobserved_condition_plus": len(cells) == 6,
                "arms": arm_accounts,
            }
        H.write(
            out / "summary.json",
            {
                "status": "FINITE_RELATION_GEOMETRY_AND_ANCHOR_TRANSPORT_ACCOUNTED",
                "cases": summaries,
                "new_model_calls": 0,
                "geometry_only_does_not_identify_polarity": True,
                "held_best_rotation_is_diagnostic_oracle_not_deployed": True,
                "anchor_calibration_is_not_free_source_only_generalization": True,
                "scorer_keeps_origin_and_complement_channels": True,
                "original_scope_complete": False,
            },
        )
        H.write(
            out / "output_seal.json",
            H.seals(out, out / "01") | {"summary.json": H.sha(out / "summary.json")},
        )
        state["state"] = "COLLECTED"
    print(
        "HM126 seven fixed cases accounted;no target answers or model calls", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "execute"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/relational_isometry_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute}[args.phase](args.out.resolve())
