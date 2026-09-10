#!/usr/bin/env python3
"""NEXT21-23: one support calibration, no-fit logit geometry, exact dependency accounts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.action_geometry import (
    axis_account,
    fit_support_correction,
    output_gradient,
    read_support,
    soft_support,
)
from grain_research.vector_mechanisms.comment_carrier import (
    partition,
    reference_state,
    score,
)
from scripts.run_grain_joint_carrier_readout import (
    OLD,
    STATES,
    check,
    donors_and_design,
)
from scripts.run_grain_joint_carrier_readout import (
    verify as verify_parent,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write

PARENT = BASE / "logic_joint_carrier_readout_v1"
CELLS = BASE / "configuration_query_factorial_v1/private_packets"
PLAN = ROOT / "docs/GRAIN_LOGIC_ACTION_GEOMETRY_PLAN_V1.md"


def prepare(out):
    """Bind old states, matrices, endpoints and this finite program before fitting."""
    parent = verify_parent(PARENT)
    source = read(OLD / "manifest.json")
    inputs = {}
    for name in (
        "manifest.json",
        "joint_freeze.json",
        "carrier_freeze.json",
        "carrier_nodes_freeze.json",
    ):
        p = PARENT / name
        inputs[str(p.relative_to(ROOT))] = sha(p)
    for name in (
        "joint_freeze.json",
        "carrier_freeze.json",
        "carrier_nodes_freeze.json",
    ):
        for path, digest in read(PARENT / name)["files"].items():
            check(PARENT / path, digest)
            inputs[str((PARENT / path).relative_to(ROOT))] = digest
    for r in read(OLD / "private_design.json"):
        p = CELLS / f"author_{r['index']:03d}.json"
        check(p, source["sources"][str(p.relative_to(ROOT))])
        inputs[str(p.relative_to(ROOT))] = sha(p)
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/action_geometry.py",
        ROOT / "scripts/analyze_grain_action_geometry.py",
    ]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_ACTION_GEOMETRY_V1",
            "inputs": inputs,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "source_outcome_sha256": parent["source_outcome_sha256"],
            "human_labels_for_fitting": False,
            "new_neural_calls": 0,
            "fits": 1,
            "penalty": 0.01,
            "reference_count": 4,
            "evaluation_count": 20,
        },
    )
    print({"status": "PREPARED", "fits": 1, "new_neural_calls": 0})


def verify(out):
    """Keep upstream and local source/code lineage immutable without opening Y."""
    verify_parent(PARENT)
    m = read(out / "manifest.json")
    for group in ("inputs", "code_hashes"):
        for name, digest in m[group].items():
            check(ROOT / name, digest)
    return m


def carrier(out):
    """Calibrate on four exact model allocations and score only unchanged source states."""
    donors, design = donors_and_design()
    displacement, matrices = [], []
    for row in design:
        v = row["views"][0]
        groups, _, _, rest = partition(v["groups"], v["p_mask"])
        with np.load(STATES / f"author_{row['index']:03d}.npz") as z:
            h = z["view_0"].copy()
        ref, interpolation = reference_state(h, groups, v["p_mask"], donors)
        if interpolation or len(rest) != 136:
            raise ValueError("Exact old retained-state role grid required")
        displacement.append((h[rest].astype(float) - ref[rest].astype(float)).ravel())
        if row["role"] == "reference":
            parts = []
            for body_case in (0, 1):
                with np.load(
                    PARENT
                    / "private_nodes"
                    / f"ref_{row['index']:03d}_m{body_case}_integral.npz"
                ) as z:
                    parts.append(z["matrix"].ravel())
            matrices.append(np.mean(parts, axis=0))
    d, matrices = np.array(displacement), np.array(matrices)
    keep = np.array([r["role"] == "reference" for r in design])
    prior = matrices.mean(0)
    old = pd.read_csv(
        PARENT / "private_carrier_predictions.csv", dtype={"author_id": str}
    )
    base = (
        old[old.arm == "frozen_full"]
        .set_index("author_id")
        .loc[[str(r["author_id"]) for r in design]]
    )
    y = base.actual_retained.to_numpy()
    if not np.allclose(d @ prior, base.predicted_retained, rtol=0, atol=1e-12):
        raise ValueError("Reference prior no longer reproduces old full reader")
    model = fit_support_correction(d[keep], y[keep], prior)
    predictions = read_support(d, model)
    support = soft_support(d, model)
    np.savez_compressed(out / "private_support_model.npz", **model)
    np.savez_compressed(out / "private_displacements.npz", displacements=d)
    fresh = base.reset_index().copy()
    fresh["arm"] = "support_calibrated"
    fresh["predicted_retained"] = predictions
    fresh["oracle_baseline_body_reconstruction"] = (
        fresh.actual_baseline + fresh.actual_body_allocated + predictions
    )
    fresh["soft_reference_support"] = support
    retained = old[old.arm.isin(["frozen_full", "zero", "reference_mean"])].copy()
    pd.concat([retained, fresh], ignore_index=True).to_csv(
        out / "private_carrier_predictions.csv", index=False
    )
    own = np.sum(matrices * d[keep], axis=1)
    common = d[keep] @ prior
    deviation = np.sum((matrices - prior) * d[keep], axis=1)
    numerical = y[keep] - own
    if np.max(abs(y[keep] - common - deviation - numerical)) > 1e-12:
        raise ValueError("Reference action contraction identity failed")
    refs = base.reset_index()[keep][["author_id", "role", "actual_retained"]].copy()
    refs["own_integrated"] = own
    refs["common_action"] = common
    refs["deviation_action"] = deviation
    refs["integration_error"] = numerical
    refs.to_csv(out / "private_reference_actions.csv", index=False)
    np.savez_compressed(
        out / "private_reference_contractions.npz",
        matrix_on_displacement=matrices @ d[keep].T,
    )
    summary = {
        "n_reference": int(keep.sum()),
        "n_evaluation": int((~keep).sum()),
        "dimensions": d.shape[1],
        "singular_values": model["singular_values"].tolist(),
        "effective_df": model["effective_df"],
        "evaluation_soft_support_mean": float(support[~keep].mean()),
        "evaluation_soft_support_min": float(support[~keep].min()),
        "evaluation_soft_support_max": float(support[~keep].max()),
        "reference_mean_displacement_norm": float(np.linalg.norm(d[keep].mean(0))),
        "reference_rms_displacement_norm": float(
            np.sqrt(np.mean(np.sum(d[keep] ** 2, axis=1)))
        ),
        "reference_actual_mean_action": float(y[keep].mean()),
        "reference_prior_mean_action": float(common.mean()),
        "reference_calibrated_mean_action": float(predictions[keep].mean()),
        "reference_integration_max_error": float(abs(numerical).max()),
        "reference_common_rms": float(np.sqrt(np.mean(common**2))),
        "reference_deviation_rms": float(np.sqrt(np.mean(deviation**2))),
        "reference_own_rms": float(np.sqrt(np.mean(own**2))),
        "reference_action_common_energy": float(np.mean(common**2)),
        "reference_action_deviation_energy": float(np.mean(deviation**2)),
        "reference_action_cross_term": float(2 * np.mean(common * deviation)),
        "reference_action_own_energy": float(np.mean(own**2)),
        "prior_replay_max": float(
            abs(common - base.predicted_retained.to_numpy()[keep]).max()
        ),
        "stationarity_norm": float(
            np.linalg.norm(
                d[keep].T @ (predictions[keep] - y[keep])
                + model["regularization"] * model["correction"]
            )
        ),
    }
    write(out / "support_summary.json", summary)


def joint_geometry(out):
    """Define one training-mean direction per pair/fold and account for each old arm."""
    records = read(PARENT / "private_joint_predictions.json")
    unique = [r for r in records if r["arm"] == "zero"]
    directions, output = [], []
    for pair in ("adjacent", "distant"):
        for fold in range(4):
            train = [r for r in unique if r["pair"] == pair and r["fold"] != fold]
            if len(train) != 18:
                raise ValueError("Exactly18 training authors per pair/fold required")
            mean = np.mean([r["actual_mixed"] for r in train], axis=0)
            v = mean / np.linalg.norm(mean)
            directions.append(
                {
                    "pair": pair,
                    "fold": fold,
                    "direction": v.tolist(),
                    "mean": mean.tolist(),
                }
            )
            for r in [r for r in records if r["pair"] == pair and r["fold"] == fold]:
                y, pred = np.array(r["actual_mixed"]), np.array(r["predicted_mixed"])
                if r["arm"] == "pair_train_mean" and not np.allclose(
                    mean, pred, rtol=0, atol=1e-12
                ):
                    raise ValueError("Old fold-local pair mean changed")
                a = axis_account(y[None], pred[None], v)
                z, values = r["z_add"], r["values"]
                grad = output_gradient(z, values)
                actual_linear, predicted_linear = float(grad @ y), float(grad @ pred)
                actual_internal = r["actual_score"] - score(z, values)
                predicted_internal = r["oracle_singletons_reconstruction"] - score(
                    z, values
                )
                output.append(
                    {k: r[k] for k in ("author_id", "query", "pair", "fold", "arm")}
                    | {
                        **{k: float(x[0]) for k, x in a.items()},
                        "actual_norm": float(np.linalg.norm(y)),
                        "predicted_norm": float(np.linalg.norm(pred)),
                        "actual_internal": actual_internal,
                        "predicted_internal": predicted_internal,
                        "actual_linear_output": actual_linear,
                        "predicted_linear_output": predicted_linear,
                        "actual_output_remainder": actual_internal - actual_linear,
                        "predicted_output_remainder": predicted_internal
                        - predicted_linear,
                    }
                )
    pd.DataFrame(output).to_csv(out / "private_joint_geometry.csv", index=False)
    write(out / "private_joint_directions.json", directions)


def dependency_accounts(out):
    """Replay nested B and native-F identities while preserving their different endpoints."""
    rows = []
    for r in read(OLD / "private_design.json"):
        packet = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        cells = read(CELLS / f"author_{r['index']:03d}.json")
        if (
            str(packet["author_id"]) != str(cells["author_id"])
            or cells["query"] != r["query"]
        ):
            raise ValueError("Nested account identity mismatch")
        v = cells["views"][0]
        f00, f10, f01, f11 = [v[k] for k in ("f00", "f10", "f01", "f11")]
        p = packet["scores"]
        replay = max(
            abs(p["zero"]["conditional_score"] - f00),
            abs(p["full"]["conditional_score"] - f10),
        )
        a, q = f10 - f00, f01 - f00
        interaction = f11 - f10 - f01 + f00
        row = {
            "author_id": r["author_id"],
            "query": r["query"],
            "B": f00,
            "configuration_at_x0": a,
            "query_at_collapsed": q,
            "configuration_query_interaction": interaction,
            "native_F": f11,
            "full_F_closure_error": abs(f11 - f00 - a - q - interaction),
            "old_endpoint_replay": replay,
        }
        if r["query"] == "TF_cont":
            b00, b10, b01 = [p[k]["conditional_score"] for k in ("b00", "b10", "b01")]
            phi_m, phi_r = (b10 - b00 + f00 - b01) / 2, (b01 - b00 + f00 - b10) / 2
            row.update(
                layout_reference=b00,
                body_allocation=phi_m,
                retained_allocation=phi_r,
                B_closure_error=abs(f00 - b00 - phi_m - phi_r),
            )
        rows.append(row)
    data = pd.DataFrame(rows)
    if (
        len(data) != 48
        or data.full_F_closure_error.max() > 1e-12
        or data.old_endpoint_replay.max() > 2e-6
        or data.B_closure_error.max() > 1e-12
    ):
        raise ValueError("Measured full dependency accounting failed")
    data.to_csv(out / "private_dependency_accounts.csv", index=False)
    output = []
    for query, g in data.groupby("query"):
        output.append(
            {
                "query": query,
                "n": len(g),
                "native_closure_max": float(g.full_F_closure_error.max()),
                "old_endpoint_replay_max": float(g.old_endpoint_replay.max()),
                "B_native_difference_rms": float(
                    np.sqrt(np.mean((g.native_F - g.B) ** 2))
                ),
                "B_closure_max": float(g.B_closure_error.max())
                if query == "TF_cont"
                else None,
            }
        )
    write(out / "dependency_summary.json", output)


def construct(out):
    """Complete one fit and all numeric diagnostics before opening questionnaire Y."""
    verify(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Preserve frozen outputs")
    started = time.monotonic()
    carrier(out)
    joint_geometry(out)
    dependency_accounts(out)
    names = (
        "private_support_model.npz",
        "private_displacements.npz",
        "private_carrier_predictions.csv",
        "private_reference_actions.csv",
        "private_reference_contractions.npz",
        "support_summary.json",
        "private_joint_geometry.csv",
        "private_joint_directions.json",
        "private_dependency_accounts.csv",
        "dependency_summary.json",
    )
    write(
        out / "prediction_freeze.json",
        {"files": {n: sha(out / n) for n in names}, "human_outcomes_read": False},
    )
    write(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "seconds": time.monotonic() - started,
            "fits": 1,
            "new_neural_calls": 0,
            "new_participants": 0,
        },
    )
    print(read(out / "run_status.json"))


def main():
    """Run one named finite phase under ignored research outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "construct"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_action_geometry_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Use ignored results")
    {"prepare": prepare, "construct": construct}[args.phase](out)


if __name__ == "__main__":
    main()
