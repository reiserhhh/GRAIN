#!/usr/bin/env python3
"""Alternate HM27 pivoted-QR constraints, physical actions and covariance replay."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, orth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_source_field import OUT, check

DEFAULT = ROOT / "results/grain_math_response_v1/source_field_verification_r1"


def map_matrix(count):
    """Form the monomial weights independently with explicit token positions."""
    positions = [0.0] if count == 1 else [2 * i / (count - 1) - 1 for i in range(count)]
    return np.array([[p**k / count for p in positions] for k in range(4)])


def direct_extension(g, frame):
    """Solve each underdetermined extension with pivoted QR, not the primary SVD."""
    a = (frame @ map_matrix(len(g))).T
    solution, _, rank, _ = lstsq(
        a, g, cond=100 * np.finfo(float).eps * max(a.shape), lapack_driver="gelsy"
    )
    return solution, rank


def direct_common(gradients, frame):
    """Solve stacked equal-family constraints and independently form the row space."""
    count = len(gradients)
    a = np.vstack(
        [(frame @ map_matrix(len(g))).T / np.sqrt(len(g) * count) for g in gradients]
    )
    b = np.vstack([g / np.sqrt(len(g) * count) for g in gradients])
    cond = 100 * np.finfo(float).eps * max(a.shape)
    solution, _, rank, _ = lstsq(a, b, cond=cond, lapack_driver="gelsy")
    basis = orth(a.T, rcond=cond)
    return (
        solution,
        basis @ basis.T,
        rank,
        float(np.sqrt(np.sum((a @ solution - b) ** 2) / b.shape[1])),
    )


def verify(packet, out):
    """Check source anchors/all cold actions/frame probes/metrics and signed accounts."""
    m = check(packet)
    if out.exists():
        raise FileExistsError("preserve prior verification")
    out.mkdir(parents=True)
    (out / "verify.py").write_bytes(Path(__file__).read_bytes())
    for name, sha in read_json(packet / "02/artifact_hashes.json").items():
        if digest(packet / name) != sha:
            raise ValueError("HM27 output changed " + name)
    spec, design = m["spec"], read_json(packet / "00/design.json")
    parent, frame = ROOT / spec["parent"], np.diag(spec["frame_diagonal"])
    weights = np.array([1, 4, 2, 4, 2, 4, 2, 4, 1], float) / 24
    maxima = {
        key: 0.0
        for key in (
            "source_gradient",
            "anchor",
            "source_geometry",
            "action",
            "frame",
            "metrics",
            "covariance",
        )
    }

    def compare(kind, actual, expected, tolerance=2e-10):
        error = float(np.max(abs(np.asarray(actual) - np.asarray(expected)), initial=0))
        maxima[kind] = max(maxima[kind], error)
        if error > tolerance:
            raise ValueError(f"{kind}: {error} > {tolerance}")

    models, geometry = {}, {}
    for query, record in read_json(packet / "01/source_manifest.json").items():
        addresses = [r for r in design if r["query"] == query and r["role"] == "source"]
        if [(r["family"], r["hm24_unit"]) for r in addresses] != [
            (r["family"], r["unit"]) for r in record["addresses"]
        ]:
            raise ValueError("source membership changed")
        with np.load(packet / record["file"]) as p:
            saved = {name: p[name].copy() for name in p.files}
        starts, paths, source_actions = [], [], []
        for address in addresses:
            gs = []
            for node in range(9):
                with np.load(
                    node_path(parent, address["hm24_unit"], node).with_suffix(".npz")
                ) as p:
                    gs.append(p["g_native_" + address["span"]].copy())
            path = sum(w * g for w, g in zip(weights, gs, strict=True))
            starts.append(gs[0])
            paths.append(path)
            compare(
                "source_gradient", gs[0], saved[f"family_{address['family']}_start"]
            )
            compare("source_gradient", path, saved[f"family_{address['family']}_path"])
            r0, r1 = (
                read_json(node_path(parent, address["hm24_unit"], n)) for n in (0, 8)
            )
            source_actions.append(
                r1["native"]["conditional_score"] - r0["native"]["conditional_score"]
            )
        model = {"constant_action": float(np.mean(source_actions))}
        compare("anchor", model["constant_action"], saved["constant_action"])
        for label, gradients in (("start", starts), ("path", paths)):
            individual = [direct_extension(g, np.eye(4))[0] for g in gradients]
            average = np.mean(individual, axis=0)
            joint, support, rank, residual = direct_common(gradients, np.eye(4))
            mean_frame = np.mean(
                [direct_extension(g, frame)[0] for g in gradients], axis=0
            )
            joint_frame, _, _, _ = direct_common(gradients, frame)
            model.update(
                {
                    label + "_mean": average,
                    label + "_joint": joint,
                    label + "_support": support,
                    "fresh_frame_" + label + "_mean": mean_frame,
                    "fresh_frame_" + label + "_joint": joint_frame,
                }
            )
            for key in (
                label + "_mean",
                label + "_joint",
                label + "_support",
                "fresh_frame_" + label + "_mean",
                "fresh_frame_" + label + "_joint",
            ):
                compare("anchor", model[key], saved[key])
            geometry[query, label] = {
                "common_rank": rank,
                "weighted_joint_gradient_residual_rms": residual,
                "source_individual_extension_max_error": max(
                    np.max(abs(map_matrix(len(g)).T @ c - g))
                    for c, g in zip(individual, gradients, strict=True)
                ),
                "individual_extension_dispersion_rms": float(
                    np.sqrt(np.mean([np.sum((c - average) ** 2) for c in individual]))
                ),
                "mean_joint_covector_distance": float(np.linalg.norm(average - joint)),
            }
        models[query] = model
    for r in csv.DictReader((packet / "01/source_geometry.csv").open()):
        for key, value in geometry[r["query"], r["anchor"]].items():
            compare("source_geometry", value, float(r[key]))
    compiled = {
        (int(r["unit"]), r["method"]): float(r["predicted_action"])
        for r in csv.DictReader(
            (ROOT / spec["compiled_parent"] / "02/actions.csv").open()
        )
        if r["representation"] == "full24_Q_layout"
    }
    action_table = list(csv.DictReader((packet / "02/actions.csv").open()))
    lookup = {r["hm24_unit"]: r for r in design}
    predictions, displacements = {}, {}
    for r in action_table:
        address = lookup[int(r["unit"])]
        if (r["query"], r["role"], int(r["family"])) != (
            address["query"],
            address["role"],
            address["family"],
        ):
            raise ValueError("action address changed")
        with np.load(
            node_path(parent, address["hm24_unit"], 0).with_suffix(".npz")
        ) as p:
            delta = p["path_delta"].copy()
        x = map_matrix(len(delta)) @ delta
        displacements[address["hm24_unit"]] = x
        method, model = r["method"], models[address["query"]]
        if method in ("start_mean", "path_mean", "start_joint", "path_joint"):
            value = float(np.einsum("ij,ij->", model[method], x))
            support = model[method.split("_")[0] + "_support"]
            compare(
                "action",
                np.linalg.norm(x - support @ x),
                float(r["outside_joint_source_support_norm"]),
            )
        elif method == "zero":
            value = 0.0
        elif method == "constant_action":
            value = model[method]
        else:
            value = compiled[address["hm24_unit"], method.removeprefix("compiled_")]
        predictions[int(r["unit"]), method] = value
        r0, r1 = (read_json(node_path(parent, address["hm24_unit"], n)) for n in (0, 8))
        native = r1["native"]["conditional_score"] - r0["native"]["conditional_score"]
        compare("action", native, float(r["native_action"]), 0)
        compare("action", value, float(r["prediction"]))
        compare("action", value - native, float(r["error"]))
    frame_rows = list(csv.DictReader((packet / "02/frame_comparisons.csv").open()))
    for r in frame_rows:
        model, x, method = (
            models[r["query"]],
            displacements[int(r["unit"])],
            r["method"],
        )
        original = predictions[int(r["unit"]), method]
        proper = float(
            np.einsum("ij,ij->", np.linalg.solve(frame.T, model[method]), frame @ x)
        )
        fresh = float(np.einsum("ij,ij->", model["fresh_frame_" + method], frame @ x))
        for key, value in {
            "original": original,
            "proper_transport": proper,
            "fresh_minimum_norm": fresh,
            "proper_difference": proper - original,
            "fresh_difference": fresh - original,
        }.items():
            compare("frame", value, float(r[key]))
    metrics = list(csv.DictReader((packet / "02/metrics.csv").open()))
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    for r in metrics:
        chosen = [
            a
            for a in action_table
            if all(a[k] == r[k] for k in ("query", "role", "method"))
        ]
        native = np.array([float(a["native_action"]) for a in chosen])
        prediction = np.array(
            [predictions[int(a["unit"]), a["method"]] for a in chosen]
        )
        values = {
            "families": len(chosen),
            "native_rms": rms(native),
            "prediction_rms": rms(prediction),
            "rmse": rms(native - prediction),
            "sign_agreements": sum(native * prediction > 0),
        }
        if np.ptp(prediction) > 0 and np.ptp(native) > 0:
            values["r"] = np.corrcoef(native, prediction)[0, 1]
        elif r["r"]:
            raise ValueError("constant prediction must have undefined correlation")
        for key, value in values.items():
            compare("metrics", value, float(r[key]))
    covariances = list(csv.DictReader((packet / "02/covariance_accounts.csv").open()))
    cov = lambda a, b: float(np.cov(a, b, ddof=0)[0, 1])
    for row in covariances:
        pairs = [
            {
                int(r["family"]): r
                for r in action_table
                if (r["query"], r["role"], r["method"])
                == (q, row["role"], row["method"])
            }
            for q in (row["query_left"], row["query_right"])
        ]
        families = sorted(pairs[0])
        if set(families) != set(pairs[1]):
            raise ValueError("unpaired covariance")
        y, z = (
            np.array([float(p[f]["native_action"]) for f in families]) for p in pairs
        )
        a, b = (
            np.array([predictions[int(p[f]["unit"]), row["method"]] for f in families])
            for p in pairs
        )
        values = {
            "native": cov(y, z),
            "PP": cov(a, b),
            "PR": cov(a, z - b),
            "RP": cov(y - a, b),
            "RR": cov(y - a, z - b),
        }
        for key, value in values.items():
            compare("covariance", value, float(row[key]), 2e-12)
        compare(
            "covariance",
            values["native"],
            sum(values[k] for k in ("PP", "PR", "RP", "RR")),
            2e-12,
        )
    summary = {
        "status": "VERIFIED_FINITE_HM27_NOT_GLOBAL",
        "counts": {
            "query_models": 3,
            "actions": len(action_table),
            "metrics": len(metrics),
            "frames": len(frame_rows),
            "covariances": len(covariances),
        },
        "max_errors": maxima,
        "new_model_calls": 0,
        "alternate_same_agent_not_human_review": True,
        "results_or_tolerances_changed": False,
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_hashes.json",
        {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(summary, flush=True)


def main():
    """Verify one new output without editing primary scientific artifacts."""
    p = argparse.ArgumentParser()
    p.add_argument("--packet", type=Path, default=OUT)
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        verify(a.packet, a.out)
    except BaseException as exc:
        if a.out.exists():
            write_json(a.out / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
