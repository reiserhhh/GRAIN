#!/usr/bin/env python3
"""Independent algebra replay of HM10A frozen vectors, units and joint laws."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, sha, write_json

SOURCE = ROOT / "results/grain_math_response_v1/binding_r1"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_law_r1"
DEST = ROOT / "results/grain_math_response_v1/joint_law_verification_r1"
TARGETS = ("coordination_action", "verification_action")


def read_csv(path):
    """Read frozen numeric tables without importing production scoring code."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def expanded_condition(model, action, obj):
    """Expand the centered tensor kernel into augmented polynomial coefficients."""
    aa = np.column_stack([np.ones(len(model["a"])), model["a"]])
    oo = np.column_stack([np.ones(len(model["b"])), model["b"]])
    dual = model["dual"]
    corrected = dual - dual.sum() / len(dual)
    tensor = aa.T @ (corrected[:, None] * oo)
    intercept = (
        model["bias"] - model["column_mean"] @ dual + model["grand_mean"] * dual.sum()
    )
    return (
        np.einsum(
            "ij,jk,ik->i",
            np.column_stack([np.ones(len(action)), action]),
            tensor,
            np.column_stack([np.ones(len(obj)), obj]),
        )
        + intercept
    )


def subset_sum(values, mask):
    """Direct inclusion-exclusion, distinct from the production butterfly."""
    sub = mask
    result = np.zeros_like(values[0], dtype=float)
    while True:
        sign = -1 if (mask.bit_count() - sub.bit_count()) % 2 else 1
        result += sign * values[sub]
        if sub == 0:
            return result
        sub = (sub - 1) & mask


def verify(packet, output):
    """Replay every observed/declared readout and every source-only completion."""
    hashes_checked = 0
    for root in (SOURCE, packet):
        hashes = json.loads((root / "04/artifact_hashes.json").read_text())
        for name, digest in hashes.items():
            if sha(root / name) != digest:
                raise ValueError("hash mismatch: " + str(root / name))
            hashes_checked += 1
    bank = read_bank(packet / "00/documents.jsonl")
    indexed = json.loads((packet / "01/observation_indices.json").read_text())
    docs = read_csv(packet / "03/document_readouts.csv")
    conditions = read_csv(packet / "03/condition_scores.csv")
    condition_lookup = {
        (
            r["reader"],
            r["material_id"],
            r["arm"],
            int(r["report"]),
            int(r["trigger"]),
        ): r
        for r in conditions
    }
    doc_lookup = {(r["reader"], r["material_id"]): r for r in docs}
    errors = defaultdict(float)
    counts = defaultdict(int)

    def check(name, a, b):
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        errors[name] = max(errors[name], diff)
        counts[name] += int(np.asarray(a).size)
        if diff > 1e-9:
            raise ValueError(f"{name} numeric mismatch: {diff}")

    for reader in MODELS:
        old = np.load(SOURCE / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(packet / f"01B/embeddings/{reader}.npy").astype(float)

        def vector(ref, old=old, new=new):
            return old[ref[1]] if ref[0] == "source" else new[ref[1]]

        models = [load_model(SOURCE / f"02/{reader}_{t}.npz") for t in TARGETS]
        for material, entry in zip(bank, indexed, strict=True):
            row = doc_lookup[reader, material["material_id"]]
            h = vector(entry["document_ref"])
            for qi, name in enumerate(("C", "V")):
                model = models[qi]["whole_factors"]
                result = (
                    h @ model["w"][:, 0] + (model["bias"] - model["mu"] @ model["w"])[0]
                )
                check("whole_readouts", result, float(row["whole_" + name]))
            for arm in ("observed", "declared"):
                reports = entry[arm]
                eligible = bool(reports) and all(
                    r["status"] == "ok" and r["experiencer"] == "self" for r in reports
                )
                cap, joint, hard = [], [], []
                for ri, report in enumerate(reports):
                    tt = report["triggers"]
                    if any(t["object_ref"] is None for t in tt):
                        eligible = False
                        continue
                    aa = np.array([vector(t["action_ref"]) for t in tt])
                    oo = np.array([vector(t["object_ref"]) for t in tt])
                    k = np.column_stack(
                        [expanded_condition(m["auto_tensor"], aa, oo) for m in models]
                    )
                    for ti in range(len(tt)):
                        stored = condition_lookup[
                            reader, material["material_id"], arm, ri, ti
                        ]
                        check(
                            "condition_readouts",
                            k[ti],
                            [float(stored["C"]), float(stored["V"])],
                        )
                    g = np.clip((k + 1) / 2, 0, 1)
                    c = g.max(0) if len(g) else np.zeros(2)
                    cap.append(c)
                    joint.append(min(c))
                    hard.append(min((k > 0).astype(float).max(0)) if len(k) else 0)
                if eligible:
                    check(
                        "report_units", len(reports), float(row[arm + "_report_mass"])
                    )
                    check(
                        "document_marginals",
                        np.mean(cap, axis=0),
                        [float(row[arm + "_C"]), float(row[arm + "_V"])],
                    )
                    check("document_joint", np.mean(joint), float(row[arm + "_joint"]))
                    check(
                        "document_joint", np.mean(hard), float(row[arm + "_hard_joint"])
                    )
                elif any(
                    row[arm + "_" + k] != ""
                    for k in ("C", "V", "joint", "hard_joint", "report_mass")
                ):
                    raise ValueError("unresolved observation imputed")
        cubes = defaultdict(dict)
        for material, entry in zip(bank, indexed, strict=True):
            cubes[material["cube_id"]][material["mask"]] = (material, entry)
        vm = np.load(packet / f"02/{reader}_vector_mobius.npy")
        for ci, (_, cells) in enumerate(cubes.items()):
            hs = np.array([vector(cells[m][1]["document_ref"]) for m in range(8)])
            for mask in range(8):
                actual = subset_sum(hs, mask)
                check("vector_inclusion_exclusion", actual, vm[ci, mask])
                if mask:
                    for qi, name in enumerate(("C", "V")):
                        field = np.array(
                            [
                                float(
                                    doc_lookup[reader, cells[m][0]["material_id"]][
                                        "whole_" + name
                                    ]
                                )
                                for m in range(8)
                            ]
                        )
                        readout_difference = subset_sum(field, mask)
                        check(
                            "interaction_dual_action",
                            actual @ models[qi]["whole_factors"]["w"][:, 0],
                            readout_difference,
                        )

    cube_rows = defaultdict(dict)
    for row in docs:
        cube_rows[row["reader"], row["cube_id"]][int(row["mask"])] = row
    coefficient_rows = read_csv(packet / "03/mobius_coefficients.csv")
    for row in coefficient_rows:
        cells = cube_rows[row["reader"], row["cube_id"]]
        f = np.array([float(cells[m][row["metric"]]) for m in range(8)])
        check(
            "scalar_inclusion_exclusion",
            subset_sum(f, int(row["mask"])),
            float(row["coefficient"]),
        )
    tables = read_csv(packet / "03/four_corner_tables.csv")
    table_lookup = {
        (r["reader"], r["cube_id"], r["metric"], int(r["incidence"])): r for r in tables
    }
    source_group = defaultdict(list)
    for r in tables:
        y = [float(r[f"y{i}"]) for i in range(4)]
        check(
            "four_corner_response", y[3] - y[1] - y[2] + y[0], float(r["interaction"])
        )
        if r["role"] == "source":
            source_group[r["reader"], r["metric"]].append(r)
    predictions = read_csv(packet / "03/cold_joint_predictions.csv")
    for row in predictions:
        if row["role"] not in {"validation", "application"}:
            raise ValueError("source rows mislabelled cold")
        src = source_group[row["reader"], row["metric"]]
        if row["arm"] == "additive":
            expected = 0.0
        elif row["arm"] == "source_global":
            expected = float(np.mean([float(r["interaction"]) for r in src]))
        else:
            frame, incidence = int(row["frame"]), int(row["incidence"])
            if row["arm"] == "source_shifted_key_control":
                frame, incidence = (frame + 1) % 4, 1 - incidence
            selected = [
                r
                for r in src
                if int(r["frame"]) == frame
                and int(r["incidence"]) == incidence
                and r["actor"] == row["actor"]
            ]
            expected = (
                float(np.mean([float(r["interaction"]) for r in selected]))
                if selected
                else None
            )
        if expected is None:
            if (
                row["source_correction"]
                or row["predicted_joint"]
                or row["squared_error"]
            ):
                raise ValueError("missing source support imputed")
            continue
        check("source_only_corrections", expected, float(row["source_correction"]))
        table = table_lookup[
            row["reader"], row["cube_id"], row["metric"], int(row["incidence"])
        ]
        y = [float(table[f"y{i}"]) for i in range(4)]
        prediction = y[1] + y[2] - y[0] + expected
        check("cold_predictions", prediction, float(row["predicted_joint"]))
        check(
            "cold_squared_errors", (prediction - y[3]) ** 2, float(row["squared_error"])
        )
    summaries = read_csv(packet / "03/completion_summary.csv")
    for row in summaries:
        chosen = [
            r
            for r in predictions
            if all(row[k] == r[k] for k in ("reader", "metric", "role", "arm"))
            and r["squared_error"] != ""
        ]
        if not chosen:
            continue
        check(
            "summary_rmse",
            np.sqrt(np.mean([float(r["squared_error"]) for r in chosen])),
            float(row["rmse"]),
        )
        check(
            "summary_improvement",
            np.mean(
                [
                    float(r["additive_squared_error"]) - float(r["squared_error"])
                    for r in chosen
                ]
            ),
            float(row["mse_improvement"]),
        )
    # Reconstruct product-character energy directly from complete cell tables.
    spec = json.loads((packet / "00/protocol_before_run.json").read_text())["spec"]
    energy_rows = read_csv(packet / "03/fourier_energies.csv")
    energy_by = defaultdict(list)
    for r in energy_rows:
        energy_by[r["reader"], r["cube_id"], r["metric"], r["reference"]].append(r)
    for (reader, cube, metric, reference), rr in energy_by.items():
        cells = cube_rows[reader, cube]
        f = np.array([float(cells[m][metric]) for m in range(8)])
        p = np.array(
            spec["factor_probabilities"]
            if reference == "uniform_product"
            else spec["alternate_reference_only"]
        )
        bits = (np.arange(8)[:, None] >> np.arange(3)) & 1
        weights = np.prod(np.where(bits, p, 1 - p), axis=1)
        total = 0.0
        for r in rr:
            mask = int(r["mask"])
            phi = np.ones(8)
            for j in range(3):
                if mask & (1 << j):
                    phi *= (bits[:, j] - p[j]) / np.sqrt(p[j] * (1 - p[j]))
            coefficient = sum(weights[m] * f[m] * phi[m] for m in range(8))
            check("fourier_energy", coefficient**2, float(r["energy"]))
            if mask:
                total += coefficient**2
        check(
            "fourier_variance",
            total,
            sum(weights[m] * (f[m] - weights @ f) ** 2 for m in range(8)),
        )
    witness = json.loads((packet / "00/support_witness.json").read_text())
    x = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]])
    check("support_null", x @ np.array(witness["null_vector"]), np.zeros(3))
    result = {
        "packet": "HM-10A",
        "status": "PASS",
        "hashes_checked": hashes_checked,
        "checks": dict(counts),
        "max_errors": dict(errors),
        "max_formula_error": max(errors.values()),
        "new_encoder_parser_api_calls": 0,
        "same_agent_alternate_algebra": True,
        "independent_human_review": False,
    }
    write_json(output / "verification.json", result)
    write_json(
        output / "artifact_hashes.json",
        {
            str(f.relative_to(output)): sha(f)
            for f in sorted(output.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Audit a frozen new packet without invoking any observer."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, default=DEST)
    args = p.parse_args()
    if (args.output / "verification.json").exists():
        raise ValueError("verification frozen")
    verify(args.packet, args.output)


if __name__ == "__main__":
    main()
