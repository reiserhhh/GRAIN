#!/usr/bin/env python3
"""Alternate dense-formula HM-09A audit and finite-reference intersection witness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, sha, write_csv, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/response_geometry_r1"
DEST = ROOT / "results/grain_math_response_v1/response_geometry_verification_r1"
TARGETS = ("coordination_action", "verification_action")


def dense(model):
    """Build the full tensor matrix independently from centered kernel columns."""
    a, o, d = model["a"], model["b"], model["dual"]
    e = d - np.mean(d)
    matrix = a.T @ (e[:, None] * o)
    constant = (
        e.sum()
        - model["column_mean"] @ d
        + model["grand_mean"] * d.sum()
        + model["bias"]
    )
    return constant, a.T @ e, o.T @ e, matrix


def rank(x):
    """Record standard numerical rank including the exact resolution used."""
    s = np.linalg.svd(x, compute_uv=False)
    tol = np.finfo(float).eps * max(x.shape) * s[0] if len(s) else 0
    return int(np.sum(s > tol)), float(tol)


def verify(out, dest):
    """Rebuild readouts, drifts, source units and dimensional-forced overlaps."""
    if (dest / "verification.json").exists():
        raise ValueError("completed verification frozen")
    manifest = json.loads((out / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(out / name) != digest:
            raise ValueError("changed primary: " + name)
    protocol = json.loads((out / "00/protocol_before_run.json").read_text())
    source, cached = (
        ROOT / protocol["spec"][k] for k in ("source_packet", "cached_packet")
    )
    for folder in (source, cached):
        if (
            sha(folder / "04/artifact_hashes.json")
            != protocol["dependencies"][str(folder.relative_to(ROOT))]
        ):
            raise ValueError("dependency manifest changed")
        for name, digest in json.loads(
            (folder / "04/artifact_hashes.json").read_text()
        ).items():
            if sha(folder / name) != digest:
                raise ValueError("changed dependency: " + name)
    bank, idx = (
        read_bank(out / "00/documents.jsonl"),
        json.loads((out / "01/observation_indices.json").read_text()),
    )
    cb, ci = (
        read_bank(cached / "00/documents.jsonl"),
        json.loads((cached / "01/observation_indices.json").read_text()),
    )
    cold_events = {
        r["material_id"]: d["observed"][0]
        for r, d in zip(bank, idx, strict=True)
        if len(d["observed"]) == 1 and d["observed"][0]["status"] == "ok"
    }
    cache_events = {
        (r["material_id"], e["node_id"]): e
        for r, d in zip(cb, ci, strict=True)
        for e in d["observed"]
    }
    fields = list(csv.DictReader((out / "01A/source_response_fields.csv").open()))
    scores = list(csv.DictReader((out / "02/cold_readouts.csv").open()))
    responses = [
        (cached, list(csv.DictReader((out / "01A/cached_responses.csv").open()))),
        (
            out,
            list(csv.DictReader((out / "02/cold_response_decomposition.csv").open())),
        ),
    ]
    errors, dimensions, counts = {}, [], {}
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new_by_folder = {
            folder: np.load(folder / f"01B/embeddings/{reader}.npy").astype(float)
            for folder in (out, cached)
        }
        ops = [
            dense(load_model(source / f"02/{reader}_{t}.npz")["auto_tensor"])
            for t in TARGETS
        ]

        support = np.load(out / f"01A/{reader}_geometry.npz")["support"]
        dense_cache = {}

        def vector(ref, folder, old=old, new_by_folder=new_by_folder):
            return old[ref[1]] if ref[0] == "source" else new_by_folder[folder][ref[1]]

        def score(a, o, op, dense_cache=dense_cache):
            c, ca, co, matrix = op
            key = (id(matrix), a.tobytes())
            if key not in dense_cache:
                dense_cache[key] = (float(c + a @ ca), co + a @ matrix)
            intercept, slope = dense_cache[key]
            return float(intercept + slope @ o)

        lookup = {
            r["text"]: i
            for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
        }
        sr = [r for r in fields if r["reader"] == reader]
        reconstructed = []
        for r in sr:
            a = old[lookup[f"Action: {r['action']}."]]
            before, after = [
                old[lookup[f"Object: {r[k]}."]] for k in ("before", "after")
            ]
            shifts = [score(a, after, op) - score(a, before, op) for op in ops]
            reconstructed.append(shifts)
            errors["source_fields"] = max(
                errors.get("source_fields", 0),
                *(
                    abs(s - float(r[k]))
                    for s, k in zip(
                        shifts,
                        ("coordination_response", "verification_response"),
                        strict=True,
                    )
                ),
            )
        reconstructed = np.array(reconstructed)
        geometry = json.loads((out / "01A/source_geometry.json").read_text())[reader]
        for x, y, name in (
            (0, 1, "coordination_to_verification"),
            (1, 0, "verification_to_coordination"),
        ):
            beta = float(
                reconstructed[:, x]
                @ reconstructed[:, y]
                / (reconstructed[:, x] @ reconstructed[:, x])
            )
            errors["source_conversion"] = max(
                errors.get("source_conversion", 0), abs(beta - geometry[name]["beta"])
            )
        for r in (r for r in scores if r["reader"] == reader):
            e = cold_events[r["material_id"]]
            a = vector(e["action_ref"], out)
            o = vector(e["object_ref" if r["view"] == "strict" else "legal_ref"], out)
            ss = [score(a, o, op) for op in ops]
            got = ss[TARGETS.index(r["target"])]
            errors["cold_scores"] = max(
                errors.get("cold_scores", 0),
                abs(got - float(r["k"])),
                abs(np.mean(ss) - float(r["shared_k"])),
            )
            if (got > 0) != (r["target_gate"] == "True"):
                raise ValueError("target gate changed")
        for folder, rows in responses:
            for r in (r for r in rows if r["reader"] == reader):
                if folder == cached:
                    e = cache_events[r["material_id"], r["node_id"]]
                    a, before, after = (
                        vector(e[k], folder)
                        for k in (
                            "action_ref",
                            "object_ref",
                            "legal_ref" if r["view"] == "legal_joint" else "naive_ref",
                        )
                    )
                else:
                    e0, e1 = cold_events[r["material_id"]], cold_events[r["after_id"]]
                    field = (
                        "object_ref" if r["view"].startswith("strict") else "legal_ref"
                    )
                    a, before, after = (
                        vector(e0["action_ref"], folder),
                        vector(e0[field], folder),
                        vector(e1[field], folder),
                    )
                delta = after - before
                sd = support.T @ (support @ delta)
                for t, op, k in zip(
                    TARGETS,
                    ops,
                    ("coordination_response", "verification_response"),
                    strict=True,
                ):
                    direct = score(a, after, op) - score(a, before, op)
                    j = dense_cache[id(op[3]), a.tobytes()][1]
                    errors["responses"] = max(
                        errors.get("responses", 0),
                        abs(direct - float(r[k])),
                        abs(j @ sd - float(r[t + "_supported"])),
                        abs(j @ (delta - sd) - float(r[t + "_unobserved"])),
                    )
                x, y = (
                    float(r["coordination_response"]),
                    float(r["verification_response"]),
                )
                errors["shared_contrast"] = max(
                    errors.get("shared_contrast", 0),
                    abs(float(r["shared_response"]) - float(r["target_contrast"]) - x),
                    abs(float(r["shared_response"]) + float(r["target_contrast"]) - y),
                )
        arrays = np.load(out / f"01A/{reader}_geometry.npz")
        f0, f1 = arrays["reference_coordination"], arrays["reference_verification"]
        r0, _ = rank(f0)
        r1, _ = rank(f1)
        joint, _ = rank(np.vstack((f0, f1)))
        ambient, _ = rank(arrays["source_deltas"])
        intersection = r0 + r1 - joint
        lower = max(0, r0 + r1 - ambient)
        cos = geometry["empirical_principal_cosines"]
        dimensions.append(
            {
                "reader": reader,
                "rank_coordination": r0,
                "rank_verification": r1,
                "ambient_reference_rank": ambient,
                "joint_rank": joint,
                "intersection_rank": intersection,
                "dimension_forced_intersection_lower_bound": lower,
                "principal_cosines_near_one": sum(abs(c - 1) < 1e-12 for c in cos),
                "interpretation": "dimension-forced overlap is not common psychological meaning",
            }
        )
        counts[reader] = {
            "source_fields": len(sr),
            "cold_scores": sum(r["reader"] == reader for r in scores),
            "cached_responses": sum(r["reader"] == reader for r in responses[0][1]),
            "cold_responses": sum(r["reader"] == reader for r in responses[1][1]),
        }
    if any(v > 1e-9 for v in errors.values()):
        raise ValueError("alternate formulas disagree: " + repr(errors))
    result = {
        "status": "verified",
        "primary_hashes": len(manifest),
        "counts": counts,
        "max_errors": errors,
        "dimension_witness": dimensions,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
        "new_encoder_calls": 0,
        "source_refits": 0,
    }
    write_csv(dest / "dimension_intersection_witness.csv", dimensions)
    write_json(dest / "verification.json", result)
    write_json(
        dest / "artifact_hashes.json",
        {
            str(p.relative_to(dest)): sha(p)
            for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(result), flush=True)


def main():
    """Run a bounded independent-formula audit, never overwrite the primary."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    verify(args.out, args.dest)


if __name__ == "__main__":
    main()
