#!/usr/bin/env python3
"""HM28 exact source-position certificates and cold identified-action replay."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_position_quotient as Q
from grain_research.construction import math_source_field as S
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_source_field import check as parent_check
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_position_quotient_v1.json"
PLAN = ROOT / "docs/GRAIN_POSITION_IDEAL_PLAN_V1.md"
OUT = ROOT / "results/grain_math_response_v1/position_quotient_r1"
CODE = (
    "grain_research/construction/math_position_quotient.py",
    "scripts/run_grain_math_position_quotient.py",
    "scripts/verify_grain_math_position_quotient.py",
    "tests/test_grain_math_position_quotient.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Read back every frozen input, source law and design before continuation."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen input changed " + name)
    if digest(out / "00/design.json") != manifest["design_sha256"]:
        raise ValueError("frozen design changed")
    return manifest


def prepare(out):
    """Freeze the follow-up without adding or selecting any held family."""
    if out.exists():
        raise FileExistsError("preserve previous packet")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    manifest = parent_check(parent)
    if (
        read_json(ROOT / spec["parent_verification"] / "summary.json")["status"]
        != "VERIFIED_FINITE_HM27_NOT_GLOBAL"
    ):
        raise ValueError("HM27 verification required")
    design = read_json(parent / "00/design.json")
    if len(design) != spec["units"]:
        raise ValueError("original48 paths required")
    files = dict(manifest["files"])
    for name, sha in read_json(parent / "02/artifact_hashes.json").items():
        if digest(parent / name) != sha:
            raise ValueError("parent output changed " + name)
        files[str((parent / name).relative_to(ROOT))] = sha
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
        {
            "files": files,
            "spec": spec,
            "design_sha256": digest(out / "00/design.json"),
        },
    )
    write_json(out / "run_status.json", {"status": "PREPARED", "new_model_calls": 0})
    print("HM28_PREPARED_48_ORIGINAL_PATHS", flush=True)


def source(out):
    """Learn six position laws solely from the saved source gradient families."""
    spec = check(out)["spec"]
    if read_json(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError("source already ran")
    parent = ROOT / spec["parent"]
    records, models = [], {}
    for query, entry in read_json(parent / "01/source_manifest.json").items():
        with np.load(parent / entry["file"]) as data:
            arrays = {name: data[name].copy() for name in data.files}
        for anchor in spec["anchors"]:
            gs = [arrays[f"family_{a['family']}_{anchor}"] for a in entry["addresses"]]
            counts = [len(g) for g in gs]
            cert, field = Q.certificate(counts), Q.fit_node_field(gs)
            if list(map(str, cert["positions"])) != ["-1", "0", "1"]:
                raise ValueError("this packet requires the declared source quotient")
            path = out / "01/source" / f"{query}_{anchor}.npz"
            np.savez_compressed(
                path, coefficients=field["coefficients"], values=field["values"]
            )
            key = query + "/" + anchor
            records.append(
                {
                    "query": query,
                    "anchor": anchor,
                    "counts": counts,
                    "positions": list(map(str, cert["positions"])),
                    "vanishing_polynomial": list(map(str, cert["polynomial"])),
                    "reduction": [[str(x) for x in row] for row in cert["reduction"]],
                    "rank": cert["rank"],
                    "nullity": cert["nullity"],
                    "node_weights": list(map(str, field["weights"])),
                    "ambient_coefficient_difference": float(
                        np.linalg.norm(
                            field["coefficients"] - arrays[anchor + "_joint"]
                        )
                    ),
                }
            )
            models[key] = {"file": str(path.relative_to(out)), "sha256": digest(path)}
    write_json(out / "01/source_certificates.json", records)
    write_json(out / "01/source_manifest.json", models)
    write_json(
        out / "run_status.json",
        {
            "status": "SOURCE_FROZEN",
            "source_position_laws": len(models),
            "new_model_calls": 0,
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def evaluate(out):
    """Compare quotient and joint actions; explicitly identify absent support."""
    spec = check(out)["spec"]
    if read_json(out / "run_status.json")["status"] != "SOURCE_FROZEN":
        raise ValueError("source must precede held evaluation")
    parent = ROOT / spec["parent"]
    grand = ROOT / read_json(parent / "00/manifest.json")["spec"]["parent"]
    design = read_json(out / "00/design.json")
    old = {
        (int(r["unit"]), r["method"]): r
        for r in csv.DictReader((parent / "02/actions.csv").open())
    }
    rows, witness = [], []
    for key, entry in read_json(out / "01/source_manifest.json").items():
        query, anchor = key.split("/")
        if digest(out / entry["file"]) != entry["sha256"]:
            raise ValueError("source model changed")
        with np.load(out / entry["file"]) as data:
            model = {name: data[name].copy() for name in data.files}
        field = {
            "positions": Q.certificate([1, 2])["positions"],
            "values": model["values"],
        }
        n = np.array([0.0, -1.0, 0.0, 1.0])
        for address in [a for a in design if a["query"] == query]:
            with np.load(
                node_path(grand, address["hm24_unit"], 0).with_suffix(".npz")
            ) as data:
                delta = data["path_delta"].copy()
            moments = S.moment_map(len(delta)) @ delta
            action = Q.node_action(field, delta)
            polynomial = S.contract(model["coefficients"], moments)
            joint = old[address["hm24_unit"], anchor + "_joint"]
            actual = float(joint["native_action"])
            rows.append(
                {
                    "unit": address["hm24_unit"],
                    "query": query,
                    "anchor": anchor,
                    "family": address["family"],
                    "role": address["role"],
                    "tokens": len(delta),
                    "node_action": action,
                    "polynomial_action": polynomial,
                    "joint_action": float(joint["prediction"]),
                    "native_action": actual,
                    "error": action - actual,
                    "node_polynomial_difference": action - polynomial,
                    "node_joint_difference": action - float(joint["prediction"]),
                    "moment_alias_norm": float(np.linalg.norm(n @ moments)),
                    "source_null_unit_shift_action": float((n @ moments)[0]),
                }
            )
        p = Q.certificate([1, 2])["polynomial"]
        for count in spec["witness_counts"]:
            w = Q.exact_witness(count, p)
            witness.append(
                {
                    "query": query,
                    "anchor": anchor,
                    "tokens": count,
                    "positions": ";".join(map(str, w["positions"])),
                    "delta_h": ";".join(map(str, w["values"])),
                    "null_action": str(w["single_action"]),
                    "paired_difference": str(w["paired_action_difference"]),
                    "augmented_cubic_rank": Q.certificate([1, 2, count])["rank"],
                    "scope": "exact toy support witness; no decoder called",
                }
            )
    stats = []
    for query in spec["queries"]:
        for anchor in spec["anchors"]:
            for role in ("source", "validation", "application"):
                rs = [
                    r
                    for r in rows
                    if (r["query"], r["anchor"], r["role"]) == (query, anchor, role)
                ]
                stats.append(
                    {
                        "query": query,
                        "anchor": anchor,
                        "role": role,
                        "families": len(rs),
                        "rmse": float(np.sqrt(np.mean([r["error"] ** 2 for r in rs]))),
                        "joint_max_difference": max(
                            abs(r["node_joint_difference"]) for r in rs
                        ),
                    }
                )
    (out / "02").mkdir()
    write_csv(out / "02/actions.csv", rows)
    write_csv(out / "02/metrics.csv", stats)
    write_csv(out / "02/support_witnesses.csv", witness)
    write_json(
        out / "02/summary.json",
        {
            "status": "EVALUATED_FINITE_HM28_NOT_GLOBAL",
            "actions": len(rows),
            "metrics": len(stats),
            "exact_support_witnesses": len(witness),
            "position_laws": 6,
            "source_rank": 3,
            "source_vanishing_polynomial": "t^3-t",
            "new_position_acquisition_count": 4,
            "node_joint_max_error": max(abs(r["node_joint_difference"]) for r in rows),
            "node_polynomial_max_error": max(
                abs(r["node_polynomial_difference"]) for r in rows
            ),
            "actual_moment_alias_max_norm": max(r["moment_alias_norm"] for r in rows),
            "new_model_calls": 0,
            "held_gradients_used_in_fit": False,
            "scope": spec["scope"],
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


def main():
    """Run one finite stage with explicit resume state and immutable inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "source", "evaluate", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    collector_guard(ROOT)
    if args.stage == "check":
        check(args.out)
        print("HM28_FROZEN_FILES_VALID", flush=True)
    else:
        {"prepare": prepare, "source": source, "evaluate": evaluate}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
