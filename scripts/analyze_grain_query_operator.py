#!/usr/bin/env python3
"""Audit P1D and test an explicitly conditional, endpoint-oracle-free gap readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash


def fit_gap(x, y, train):
    """Train one RMS normalization and fixed Ridge without inspecting held targets."""
    scale = float(np.sqrt(np.mean(np.sum(x[train] ** 2, axis=1))))
    if scale <= 1e-12:
        raise ValueError("Degenerate training differences")
    model = Ridge(alpha=1.0, fit_intercept=False).fit(x[train] / scale, y[train])
    return model.predict(x / scale), model.coef_, scale


def analyze(parent: Path, conduit: Path, model_dir: Path):
    """Compute audits and all predeclared state-gap models using cached synthetic arrays."""
    m = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] > 800:
        raise ValueError("Complete P1D required")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as z:
        for name, sha in m["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Source checksum")
    for name, sha in json.loads((parent / "output_hashes.json").read_text()).items():
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Output checksum")
    scores = pd.read_csv(parent / "scores.csv", dtype={"mask": str})
    recorded = pd.read_csv(parent / "operator_scores.csv")
    native = pd.read_csv(conduit / "baselines.csv").set_index("case_id")
    if len(scores) != 576 or scores.duplicated(["case_id", "mask"]).any():
        raise ValueError("P1D grid")
    for _, sub in scores.groupby("case_id"):
        if set(sub["mask"]) != {f"{i:04b}" for i in range(16)}:
            raise ValueError("Missing mask")
    arithmetic = float(
        np.max(abs(scores.margin - scores.case_id.map(native.margin) - scores["shift"]))
    )
    donor_error = float(
        np.max(
            abs(
                scores.donor_gap
                - (
                    scores.donor_case.map(native.margin)
                    - scores.case_id.map(native.margin)
                )
            )
        )
    )
    fit_groups = sorted(scores.group_id.unique())[:3]
    coefficient_error = 0.0
    copy_controls = []
    for variant, sub in scores.groupby("variant"):
        full = sub[sub["mask"] == "1111"].set_index("case_id")["shift"]
        predicted = (
            sub.case_id.map(full).to_numpy() * (sub["mask"] != "0000").to_numpy()
        )
        select = (~sub.group_id.isin(fit_groups)).to_numpy()
        copy_controls.append(
            {
                "variant": variant,
                "split": "heldout",
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (predicted[select] - sub["shift"].to_numpy()[select]) ** 2
                        )
                    )
                ),
                "full_query_oracle_used": True,
            }
        )
        if variant == "venue":
            continue
        mask = np.array([list(map(int, s)) for s in sub["mask"]], dtype=float)
        train = sub.group_id.isin(fit_groups).to_numpy()
        target = sub["shift"].to_numpy() / sub.donor_gap.to_numpy()
        for degree in (1, 2):
            x = (
                mask
                if degree == 1
                else np.column_stack(
                    [mask]
                    + [mask[:, i] * mask[:, j] for i, j in combinations(range(4), 2)]
                )
            )
            beta = (
                LinearRegression(fit_intercept=False).fit(x[train], target[train]).coef_
            )
            previous = json.loads(
                recorded[
                    (recorded.variant == variant)
                    & (recorded.degree == degree)
                    & (recorded.split == "fit")
                ].coefficients.iloc[0]
            )
            coefficient_error = max(
                coefficient_error, float(np.max(abs(beta - previous)))
            )
    if max(arithmetic, donor_error, coefficient_error) > 1e-8:
        raise ValueError("Saved operator replay failed")
    layouts = json.loads((conduit / "token_layouts.json").read_text())
    bins = json.loads((parent / "bins.json").read_text())
    for case, parts in bins.items():
        if [j for part in parts for j in part] != layouts[case]["partitions"]["Q"]:
            raise ValueError("Bin coverage")
    states = {
        case: np.load(conduit / "activations" / (case + ".npz"), allow_pickle=False)[
            "block_16"
        ].astype(float)
        for case in scores.case_id.unique()
    }
    summaries, predictions, models = [], [], {}
    for variant in ("event_target", "trait_code1"):
        cases = (
            scores[scores.variant == variant]
            .drop_duplicates("case_id")
            .sort_values("case_id")
        )
        train = cases.group_id.isin(fit_groups).to_numpy()
        y = cases.donor_gap.to_numpy()
        shape = np.array(
            json.loads(
                recorded[
                    (recorded.variant == variant)
                    & (recorded.degree == 1)
                    & (recorded.split == "fit")
                ].coefficients.iloc[0]
            )
        )
        for view in ("E", "Q", "D"):
            x = np.stack(
                [
                    (
                        states[r.donor_case][layouts[r.case_id]["partitions"][view]]
                        - states[r.case_id][layouts[r.case_id]["partitions"][view]]
                    ).mean(axis=0)
                    for r in cases.itertuples()
                ]
            )
            pred, beta, scale = fit_gap(x, y, train)
            models[f"{variant}__{view}"] = beta
            models[f"{variant}__{view}__scale"] = np.array([scale])
            lookup = dict(zip(cases.case_id, pred))
            all_rows = scores[scores.variant == variant].copy()
            mask = np.array([list(map(int, s)) for s in all_rows["mask"]], dtype=float)
            predicted = (mask @ shape) * all_rows.case_id.map(lookup).to_numpy()
            held = ~all_rows.group_id.isin(fit_groups).to_numpy()
            observed = all_rows["shift"].to_numpy()
            denominator = float(np.sum(observed[held] ** 2))
            gap_denom = float(np.sum(y[~train] ** 2))
            summaries.append(
                {
                    "variant": variant,
                    "view": view,
                    "held_scenarios": 3,
                    "held_pairs": 3,
                    "gap_rmse": float(
                        np.sqrt(np.mean((pred[~train] - y[~train]) ** 2))
                    ),
                    "gap_relative_error_energy": float(
                        np.sum((pred[~train] - y[~train]) ** 2) / gap_denom
                    ),
                    "effect_rmse": float(
                        np.sqrt(np.mean((predicted[held] - observed[held]) ** 2))
                    ),
                    "effect_relative_error_energy": float(
                        np.sum((predicted[held] - observed[held]) ** 2) / denominator
                    ),
                    "held_endpoint_gap_used_as_input": False,
                    "donor_activations_used": True,
                }
            )
            for i, r in enumerate(all_rows.to_dict("records")):
                predictions.append(
                    {
                        **r,
                        "view": view,
                        "split": "heldout" if held[i] else "fit",
                        "predicted_shift": float(predicted[i]),
                        "predicted_gap": lookup[r["case_id"]],
                    }
                )
    # Text is inspected after vector partitioning; the only text here is authored synthetic material.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    decoded = []
    for case, parts in bins.items():
        if not case.startswith("e_photos") or "__-1__" not in case:
            continue
        for j, part in enumerate(parts):
            decoded.append(
                {
                    "case_id": case,
                    "bin": j + 1,
                    "start": part[0],
                    "end": part[-1],
                    "text": tokenizer.decode([layouts[case]["ids"][i] for i in part]),
                }
            )
    receipt = {
        "status": "VERIFIED",
        "parent_identity": m["identity"],
        "score_arithmetic_max": arithmetic,
        "donor_gap_max": donor_error,
        "sklearn_operator_coefficient_max": coefficient_error,
        "new_neural_calls": 0,
        "fit_groups": fit_groups,
        "held_groups": sorted(set(scores.group_id) - set(fit_groups)),
        "training_alpha": 1.0,
        "interpretation": "conditional vector contrast prediction, not autonomous personality prediction",
    }
    return (
        receipt,
        {
            "endpoint_free_operator_scores": pd.DataFrame(summaries),
            "endpoint_free_predictions": pd.DataFrame(predictions),
            "full_query_oracle_control": pd.DataFrame(copy_controls),
            "decoded_bins": pd.DataFrame(decoded),
        },
        models,
    )


def main():
    """Write fresh analysis, source and fitted small models without new neural work."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--conduit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    a = p.parse_args()
    result, tables, models = analyze(a.parent_dir, a.conduit_dir, a.model_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(a.output_dir / (name + ".csv"), index=False)
    np.savez_compressed(a.output_dir / "linear_gap_models.npz", **models)
    (a.output_dir / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["outputs"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in a.output_dir.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    result["identity"] = material_hash(result)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
