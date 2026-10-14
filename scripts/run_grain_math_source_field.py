#!/usr/bin/env python3
"""HM27 source-common field controls and fixed coordinate-frame identification."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_source_field as S
from grain_research.construction.math_decoder_paths import simpson_weights
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_hermite_field import check as parent_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_source_field_v1.json"
PLAN = ROOT / "docs/GRAIN_SOURCE_FIELD_TRANSPORT_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/source_field_transport_r1"
CODE = (
    "grain_research/construction/math_source_field.py",
    "scripts/run_grain_math_source_field.py",
    "scripts/verify_grain_math_source_field.py",
    "tests/test_grain_math_source_field.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Validate frozen inputs/code/design before a resumed finite stage."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen input changed " + name)
    if digest(out / "00/design.json") != m["design_sha256"]:
        raise ValueError("frozen design changed")
    return m


def prepare(out):
    """Preserve all48 previous paths and freeze controls before source assembly."""
    if out.exists():
        raise FileExistsError("preserve previous packet")
    spec = read_json(SPEC)
    compiled = ROOT / spec["compiled_parent"]
    m = parent_check(compiled)
    if (
        read_json(ROOT / spec["compiled_verification"] / "summary.json")["status"]
        != "VERIFIED_FINITE_HM26_NOT_GLOBAL"
    ):
        raise ValueError("HM26 verification required")
    design = read_json(compiled / "00/design.json")
    if len(design) != spec["units"] or {r["query"] for r in design} != set(
        spec["queries"]
    ):
        raise ValueError("path/query inventory changed")
    for q in spec["queries"]:
        roles = {
            role: {r["family"] for r in design if r["query"] == q and r["role"] == role}
            for role in ("source", "validation", "application")
        }
        if (
            tuple(map(len, roles.values())) != (8, 4, 4)
            or len(set.union(*roles.values())) != 16
        ):
            raise ValueError("source/held family partition changed")
    files = dict(m["files"])
    for packet in (ROOT / spec["parent"], compiled):
        for name, sha in read_json(packet / "02/artifact_hashes.json").items():
            if digest(packet / name) != sha:
                raise ValueError("parent output changed " + name)
            files[str((packet / name).relative_to(ROOT))] = sha
    files.update({name: digest(ROOT / name) for name in CODE})
    (out / "00/code").mkdir(parents=True)
    (out / "01/source").mkdir(parents=True)
    write_json(out / "00/design.json", design)
    for name in CODE:
        dst = out / "00/code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {"files": files, "spec": spec, "design_sha256": digest(out / "00/design.json")},
    )
    write_json(out / "run_status.json", {"status": "PREPARED", "new_model_calls": 0})
    print(
        {
            "status": "PREPARED",
            "units": len(design),
            "candidates": len(spec["methods"]),
        },
        flush=True,
    )


def source(out):
    """Freeze all query anchors using source gradients/values, never held outputs."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError("source stage already ran; do not overwrite")
    spec, design = m["spec"], read_json(out / "00/design.json")
    parent, frame = ROOT / spec["parent"], np.diag(spec["frame_diagonal"])
    weights = simpson_weights(np.linspace(0, 1, 9))
    inventory, geometry = {}, []
    for query in spec["queries"]:
        records = [r for r in design if r["query"] == query and r["role"] == "source"]
        starts, paths, native_actions, family_data = [], [], [], []
        arrays = {"frame": frame, "weights": weights}
        for address in records:
            gradients = []
            for node in range(9):
                with np.load(
                    node_path(parent, address["hm24_unit"], node).with_suffix(".npz")
                ) as p:
                    gradients.append(p["g_native_" + address["span"]].copy())
            start = gradients[0]
            path = np.tensordot(weights, np.array(gradients), axes=(0, 0))
            starts.append(start)
            paths.append(path)
            r0, r1 = (
                read_json(node_path(parent, address["hm24_unit"], n)) for n in (0, 8)
            )
            if r0["address"] != address or r1["address"] != address:
                raise ValueError("source native address mismatch")
            native_actions.append(
                r1["native"]["conditional_score"] - r0["native"]["conditional_score"]
            )
            family_data.append(
                {
                    "family": address["family"],
                    "unit": address["hm24_unit"],
                    "tokens": len(start),
                }
            )
            arrays[f"family_{address['family']}_start"] = start
            arrays[f"family_{address['family']}_path"] = path
        for label, gradients in (("start", starts), ("path", paths)):
            extensions = [S.extension(g) for g in gradients]
            original = np.mean([r["covector"] for r in extensions], axis=0)
            fresh = np.mean(
                [S.extension(g, frame)["covector"] for g in gradients], axis=0
            )
            joint, joint_frame = (
                S.common_covector(gradients),
                S.common_covector(gradients, frame),
            )
            arrays[label + "_mean"] = original
            arrays["fresh_frame_" + label + "_mean"] = fresh
            arrays[label + "_joint"] = joint["covector"]
            arrays["fresh_frame_" + label + "_joint"] = joint_frame["covector"]
            arrays[label + "_support"] = joint["support"]
            arrays[label + "_singular"] = joint["singular"]
            geometry.append(
                {
                    "query": query,
                    "anchor": label,
                    "source_families": len(records),
                    "token_counts": [len(g) for g in gradients],
                    "common_rank": joint["rank"],
                    "weighted_joint_gradient_residual_rms": joint[
                        "weighted_residual_rms"
                    ],
                    "source_individual_extension_max_error": max(
                        r["residual_max"] for r in extensions
                    ),
                    "individual_extension_dispersion_rms": float(
                        np.sqrt(
                            np.mean(
                                [
                                    np.sum((r["covector"] - original) ** 2)
                                    for r in extensions
                                ]
                            )
                        )
                    ),
                    "mean_joint_covector_distance": float(
                        np.linalg.norm(original - joint["covector"])
                    ),
                }
            )
        arrays["constant_action"] = np.array(np.mean(native_actions))
        path = out / "01/source" / (query + ".npz")
        np.savez_compressed(path, **arrays)
        inventory[query] = {
            "addresses": family_data,
            "file": str(path.relative_to(out)),
            "sha256": digest(path),
            "native_source_mean_action": float(np.mean(native_actions)),
        }
    write_csv(out / "01/source_geometry.csv", geometry)
    write_json(out / "01/source_manifest.json", inventory)
    write_json(
        out / "run_status.json",
        {
            "status": "SOURCE_ANCHORS_FROZEN",
            "query_anchors": 3,
            "common_covector_solves": 6,
            "source_moments": 6,
            "frame_diagnostic_solves": 6,
            "new_model_calls": 0,
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def rms(v):
    """Compute physical magnitude without treating tiny ratios as discoveries."""
    return float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))


def evaluate(out):
    """Evaluate every finite candidate/frame on all frozen family addresses."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "SOURCE_ANCHORS_FROZEN":
        raise ValueError("source anchors must be frozen")
    spec, design = m["spec"], read_json(out / "00/design.json")
    parent = ROOT / spec["parent"]
    rows, frames = [], []
    compiled = list(
        csv.DictReader((ROOT / spec["compiled_parent"] / "02/actions.csv").open())
    )
    compiled = {
        (int(r["unit"]), r["method"]): float(r["predicted_action"])
        for r in compiled
        if r["representation"] == "full24_Q_layout"
    }
    for query, metadata in read_json(out / "01/source_manifest.json").items():
        if digest(out / metadata["file"]) != metadata["sha256"]:
            raise ValueError("source anchor changed")
        with np.load(out / metadata["file"]) as p:
            model = {key: p[key].copy() for key in p.files}
        for address in [r for r in design if r["query"] == query]:
            with np.load(
                node_path(parent, address["hm24_unit"], 0).with_suffix(".npz")
            ) as p:
                delta = p["path_delta"].copy()
            x = S.moment_map(len(delta)) @ delta
            r0, r1 = (
                read_json(node_path(parent, address["hm24_unit"], n)) for n in (0, 8)
            )
            if r0["address"] != address or r1["address"] != address:
                raise ValueError("native endpoint mismatch")
            actual = (
                r1["native"]["conditional_score"] - r0["native"]["conditional_score"]
            )
            common = {
                "unit": address["hm24_unit"],
                "family": address["family"],
                "query": query,
                "role": address["role"],
                "tokens": len(delta),
            }
            for method in spec["methods"]:
                support_error = None
                if method in ("start_mean", "path_mean", "start_joint", "path_joint"):
                    prediction = S.contract(model[method], x)
                    support = model[method.split("_")[0] + "_support"]
                    support_error = float(np.linalg.norm(x - support @ x))
                    frame = model["frame"]
                    proper = S.contract(
                        np.linalg.solve(frame.T, model[method]), frame @ x
                    )
                    fresh = S.contract(model["fresh_frame_" + method], frame @ x)
                    frames.append(
                        {
                            **common,
                            "method": method,
                            "original": prediction,
                            "proper_transport": proper,
                            "fresh_minimum_norm": fresh,
                            "proper_difference": proper - prediction,
                            "fresh_difference": fresh - prediction,
                            "outside_joint_source_support_norm": support_error,
                        }
                    )
                elif method == "constant_action":
                    prediction = float(model["constant_action"])
                elif method == "zero":
                    prediction = 0.0
                else:
                    prediction = compiled[
                        address["hm24_unit"], method.removeprefix("compiled_")
                    ]
                rows.append(
                    {
                        **common,
                        "method": method,
                        "native_action": actual,
                        "prediction": prediction,
                        "error": prediction - actual,
                        "outside_joint_source_support_norm": support_error,
                    }
                )
    stats, covariances = [], []
    for query, role, method in itertools.product(
        spec["queries"], ("source", "validation", "application"), spec["methods"]
    ):
        chosen = [
            r
            for r in rows
            if (r["query"], r["role"], r["method"]) == (query, role, method)
        ]
        a, b = (
            np.array([r["native_action"] for r in chosen]),
            np.array([r["prediction"] for r in chosen]),
        )
        stats.append(
            {
                "query": query,
                "role": role,
                "method": method,
                "families": len(chosen),
                "native_rms": rms(a),
                "prediction_rms": rms(b),
                "rmse": rms(a - b),
                "r": float(np.corrcoef(a, b)[0, 1])
                if np.ptp(a) > 0 and np.ptp(b) > 0
                else None,
                "sign_agreements": int(sum(a * b > 0)),
            }
        )
    for role, method in itertools.product(
        ("source", "validation", "application"), spec["methods"]
    ):
        for q1, q2 in itertools.combinations(spec["queries"], 2):
            chosen = [
                {
                    r["family"]: r
                    for r in rows
                    if (r["query"], r["role"], r["method"]) == (query, role, method)
                }
                for query in (q1, q2)
            ]
            families = sorted(chosen[0])
            if set(chosen[1]) != set(families):
                raise ValueError("unpaired query families")
            y, z = ([c[f]["native_action"] for f in families] for c in chosen)
            a, b = ([c[f]["prediction"] for f in families] for c in chosen)
            covariances.append(
                {
                    "query_left": q1,
                    "query_right": q2,
                    "role": role,
                    "method": method,
                    "families": len(families),
                    **S.paired_covariance(y, z, a, b),
                }
            )
    (out / "02").mkdir()
    write_csv(out / "02/actions.csv", rows)
    write_csv(out / "02/metrics.csv", stats)
    write_csv(out / "02/frame_comparisons.csv", frames)
    write_csv(out / "02/covariance_accounts.csv", covariances)
    write_json(
        out / "02/summary.json",
        {
            "status": "EVALUATED_FINITE_HM27_NOT_GLOBAL",
            "actions": len(rows),
            "metrics": len(stats),
            "frame_comparisons": len(frames),
            "covariance_accounts": len(covariances),
            "new_model_calls": 0,
            "held_gradients_used_in_anchors": False,
            "proper_frame_max_error": max(abs(r["proper_difference"]) for r in frames),
            "fresh_frame_max_change": max(abs(r["fresh_difference"]) for r in frames),
            "outside_joint_source_support_max_norm": max(
                r["outside_joint_source_support_norm"] for r in frames
            ),
            "scope": "fixed controlled query actions, not natural-author psychological accuracy",
        },
    )
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(read_json(out / "02/summary.json"), flush=True)
    print([r for r in stats if r["role"] == "application"], flush=True)


def main():
    """Execute only one declared stage; preserve all previous scientific results."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "source", "evaluate", "check"))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    collector_guard(ROOT)
    if args.stage == "check":
        check(args.out)
        print("HM27_FROZEN_FILES_VALID", flush=True)
    else:
        {"prepare": prepare, "source": source, "evaluate": evaluate}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
