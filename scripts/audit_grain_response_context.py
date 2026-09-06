#!/usr/bin/env python3
"""Recompute response-context results from sealed scalar metadata only."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Verify case/assignment identities and recompute contrasts, not model states."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh response audit directory required")
    summary = verify_outputs(a.run_dir)
    manifest = json.loads((a.run_dir / "manifest.json").read_text())
    if (
        summary["status"] != "COMPLETE"
        or summary["parent_identity"] != manifest["identity"]
        or manifest["identity"]
        != material_hash({k: v for k, v in manifest.items() if k != "identity"})
    ):
        raise ValueError("Incomplete or altered observer run")
    material = json.loads((a.run_dir / "private_material_manifest.json").read_text())
    if material_hash(material["cases"]) != material["identity"]:
        raise ValueError("Changed case/assignment manifest")
    cases = pd.DataFrame(material["cases"]).set_index("participant_code")
    frame = pd.read_csv(a.run_dir / "private_response_metrics.csv").set_index(
        "participant_code"
    )
    geometry = pd.read_csv(a.run_dir / "private_geometry_metrics.csv")
    coverage = pd.read_csv(a.run_dir / "coverage.csv")
    if (
        cases.index.has_duplicates
        or frame.index.has_duplicates
        or set(cases.index) != set(frame.index)
    ):
        raise ValueError("Unpaired or duplicate authors")
    frame = frame.loc[cases.index]
    if (
        len(frame) != summary["participants"]
        or summary["forwards"] != 6 + 6 * len(frame)
        or not frame.response_tokens.equals(cases.response_tokens)
        or not np.isfinite(frame.to_numpy()).all()
    ):
        raise ValueError("Invalid observation counts or scalar values")
    if not frame.self_patch_max_error.eq(0).all():
        raise ValueError("Self-patch failed")
    if not set(coverage.loc[coverage.eligible, "participant_code"]) == set(cases.index):
        raise ValueError("Eligibility differs from analyzed cases")
    positions = {code: i for i, code in enumerate(cases.index)}
    lengths = cases.context_tokens.to_numpy()
    cost = abs(np.log(lengths[:, None]) - np.log(lengths[None, :]))
    cost += np.random.default_rng(731).uniform(0, 1e-10, size=cost.shape)
    np.fill_diagonal(cost, np.inf)
    expected_matched = linear_sum_assignment(cost)[1]
    rng = np.random.default_rng(1731)
    for _ in range(10000):
        expected_random = rng.permutation(len(frame))
        if not (expected_random == np.arange(len(frame))).any():
            break
    else:
        raise ValueError("No random derangement")
    matching = {}
    for kind, expected in (("matched", expected_matched), ("random", expected_random)):
        donor = np.array([positions[c] for c in cases[kind + "_donor_code"]])
        if not np.array_equal(donor, expected):
            raise ValueError("Donor assignment does not replay")
        matching[kind] = {
            "mean_abs_log_length_gap": float(
                abs(np.log(lengths) - np.log(lengths[donor])).mean()
            ),
            "identical_context_hashes": int(
                np.sum(
                    cases.context_sha256.to_numpy()
                    == cases.context_sha256.to_numpy()[donor]
                )
            ),
        }
    derived = {
        "actual_minus_matched": frame.actual_mean_logp - frame.matched_mean_logp,
        "actual_minus_random": frame.actual_mean_logp - frame.random_mean_logp,
        "actual_minus_none": frame.actual_mean_logp - frame.none_mean_logp,
        "patch_minus_matched_tail": frame.patch_tail_logp - frame.matched_tail_logp,
        "actual_minus_matched_tail": frame.actual_tail_logp - frame.matched_tail_logp,
    }
    rows = []
    for name, x in derived.items():
        if not np.allclose(x, frame[name], atol=1e-12, rtol=0):
            raise ValueError("Incorrect paired arithmetic")
        idx = np.random.default_rng(731).integers(len(x), size=(1000, len(x)))
        lower, upper = np.quantile(x.to_numpy()[idx].mean(axis=1), [0.025, 0.975])
        values = {
            "mean": float(x.mean()),
            "lower": float(lower),
            "upper": float(upper),
            "positive_count": int((x > 0).sum()),
            "participants": len(x),
        }
        if any(
            not np.isclose(v, summary["effects"][name][k], atol=1e-12, rtol=0)
            for k, v in values.items()
        ):
            raise ValueError("Paired interval failed replay")
        rows.append({"contrast": name, **values})
    keys = ["participant_code", "site", "comparison"]
    if (
        geometry.duplicated(keys).any()
        or len(geometry) != len(frame) * 12
        or set(geometry.participant_code) != set(frame.index)
    ):
        raise ValueError("Geometry coverage mismatch")
    if (
        not np.isfinite(geometry[["cosine", "relative_l2"]]).all().all()
        or not geometry.cosine.between(-1, 1).all()
        or not (geometry.relative_l2 >= 0).all()
    ):
        raise ValueError("Invalid geometry scalars")
    if not geometry.loc[geometry.site == "input_embedding", "relative_l2"].eq(0).all():
        raise ValueError("Input embedding changed")
    aggregate = (
        geometry.groupby(["site", "comparison"])
        .agg(
            mean_cosine=("cosine", "mean"),
            mean_relative_l2=("relative_l2", "mean"),
            n=("participant_code", "nunique"),
        )
        .reset_index()
    )
    aggregate.to_csv(out / "geometry_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "paired_effects.csv", index=False)
    arm_means = {
        name: float(frame[name + "_mean_logp"].mean())
        for name in ("actual", "matched", "random", "none")
    }
    result = {
        "status": "COMPLETE_SCALAR_AND_ASSIGNMENT_AUDIT",
        "parent_identity": manifest["identity"],
        "participants": len(frame),
        "available_free_chat": len(coverage),
        "unique_response_hashes": cases.response_sha256.nunique(),
        "response_tokens_total": int(cases.response_tokens.sum()),
        "matching": matching,
        "arm_mean_logp": arm_means,
        "effects": rows,
        "raw_text_reopened": False,
        "new_model_forwards": 0,
        "labels_read": False,
        "scope": "case/assignment/scalar replay, not independent model replay or personality validity",
    }
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in [
            Path(__file__),
            ROOT / "scripts/audit_grain_meps_material.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ]:
            z.write(path, path.relative_to(ROOT))
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
