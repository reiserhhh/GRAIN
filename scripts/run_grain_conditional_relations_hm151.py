#!/usr/bin/env python3
"""Test response-relation constraints on HM150 without fitting a predictor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_relations as M
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_semantic_anchor_hm147 as H
from scripts.run_grain_joint_persona_style_enumeration_hm150 import ScientificOutputPath

SOURCE = ROOT / "results/grain_math_response_v1/joint_persona_style_r2"
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_relation_circuits_r1"
CODE = (
    "grain_research/construction/math_conditional_relations.py",
    "scripts/run_grain_conditional_relations_hm151.py",
    "tests/test_grain_conditional_relations.py",
    "docs/GRAIN_CONDITIONAL_RELATION_CIRCUITS_PLAN_HM151.md",
)
SCOPES = ("discovery", "development", "evaluation", "all")


def prepare(out):
    """Authenticate all observations and freeze aligned response matrices."""
    source = ScientificOutputPath(SOURCE)
    manifest = J.check(source)
    rows, _ = J.generated(source)
    if len(rows) != 702 or len({r["text_id"] for r in rows}) != 702:
        raise ValueError("All702 unique actual observations required")
    for seal in ("observation_seal.json", "output_seal.json"):
        for name, checksum in H.load(source / seal).items():
            if H.digest(source / name) != checksum:
                raise ValueError("Upstream sealed observation changed: " + name)
    raw = H.archive(source / "numeric_observations.npz", manifest["representations"])
    raw = {k: np.asarray(v, float) for k, v in raw.items()}
    program = H.archive(
        H.DEFAULT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )
    z = (raw["grain4_slices"] - program["origin"]) / program["scale"][None, :, None]
    views = {
        "qwen_input_mean": raw["qwen_input_mean"],
        "qwen_block16_mean": raw["qwen_block16_mean"],
        "grain4_coordinates": z.reshape(len(z), -1),
        "grain4_tensor": H.M.tensor_features(z),
    }
    tasks = list(dict.fromkeys(r["situation"] for r in rows))
    roles = {r["situation"]: r["role"] for r in rows}
    keys = [(a, f, t) for a in J.M.TARGETS for f in J.M.FAMILIES for t in tasks]
    addresses = [
        {"axis": a, "family": f, "situation": t, "role": roles[t]} for a, f, t in keys
    ]
    lookup = {
        tuple(r[k] for k in ("axis", "family", "situation", "style", "seed", "pole")): i
        for i, r in enumerate(rows)
    }
    arrays = {}
    for name, v in views.items():
        contrasts = []
        for style in J.M.STYLES:
            seeds = []
            for seed in J.M.SEEDS:
                seeds.append(
                    [
                        v[lookup[a, f, t, style, seed, 1]]
                        - v[lookup[a, f, t, style, seed, -1]]
                        for a, f, t in keys
                    ]
                )
            contrasts.append(seeds)
        arrays[name] = np.asarray(contrasts, float)
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Unobserved contrast must not be imputed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    H.write_npz(out / "inputs.npz", **arrays)
    H.write_json(out / "addresses.json", addresses)
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM151",
            "parent": str(SOURCE),
            "code": {p: H.digest(ROOT / p) for p in CODE},
            "inputs": {p: H.digest(out / p) for p in ("inputs.npz", "addresses.json")},
            "upstream": {
                str(p.relative_to(ROOT)): H.digest(p)
                for p in (
                    source / "numeric_observations.npz",
                    source / "manifest.json",
                    source / "observation_seal.json",
                    source / "output_seal.json",
                    H.DEFAULT / "program.npz",
                )
            },
            "views": list(arrays),
            "styles": list(J.M.STYLES),
            "seeds": list(J.M.SEEDS),
            "scopes": list(SCOPES),
            "cpu_cap_seconds": 120,
            "scope": "Finite response constraints,not population mechanism/personality/accuracy;0newforwards",
        },
    )
    H.write_json(out / "status.json", {"status": "PREPARED", "actual_articles": 702})


def check(out):
    """Verify fixed inputs and code before any scientific calculation."""
    m = H.load(out / "manifest.json")
    for root, section in ((ROOT, "code"), (ROOT, "upstream"), (out, "inputs")):
        for name, checksum in m[section].items():
            if H.digest(root / name) != checksum:
                raise ValueError("Frozen input or code changed: " + name)
    return m


def execute(out):
    """Calculate all declared views/scopes, preserving exact-fit vacuity."""
    start = time.process_time()
    m = check(out)
    if H.load(out / "status.json")["status"] != "PREPARED":
        raise ValueError("Refuse completed or failed experiment replay")
    raw = H.archive(out / "inputs.npz", m["views"])
    addresses = H.load(out / "addresses.json")
    rows, repeats, numeric = [], [], {}
    max_qr_error = 0.0
    for name, full in raw.items():
        for scope in SCOPES:
            selected = [
                i
                for i, r in enumerate(addresses)
                if scope == "all" or r["role"] == scope
            ]
            x = full[0][:, selected]
            for si, style in enumerate(m["styles"][1:], 1):
                y = full[si][:, selected]
                for unit, source, target in (
                    ("three_seed_mean", x.mean(axis=0), y.mean(axis=0)),
                    ("seed11_relation", x[0], y[0]),
                ):
                    if time.process_time() - start > m["cpu_cap_seconds"]:
                        raise TimeoutError("Finite CPU cap reached;no renewal")
                    summary, q, c, singular = M.relation_certificate(source, target)
                    basis, _, _ = qr(source, mode="economic", pivoting=True)
                    r = summary["source_rank"]
                    alt_q = (
                        np.zeros_like(q)
                        if r == len(source)
                        else np.eye(len(source)) - basis[:, :r] @ basis[:, :r].T
                    )
                    err = float(np.max(np.abs(q - alt_q)))
                    max_qr_error = max(max_qr_error, err)
                    if err > 1e-8:
                        raise ArithmeticError(
                            "SVD and pivoted QR relation projectors disagree"
                        )
                    key = f"{name}__{scope}__{style}__{unit}"
                    rows.append(
                        {
                            "representation": name,
                            "scope": scope,
                            "style": style,
                            "unit": unit,
                            **summary,
                            "alternate_projector_error": err,
                        }
                    )
                    numeric[key + "__q"] = q
                    numeric[key + "__singular_values"] = singular
                    if c is not None:
                        numeric[key + "__witness"] = c
                        if unit == "seed11_relation":
                            repeats.append(
                                {
                                    "representation": name,
                                    "scope": scope,
                                    "style": style,
                                    **M.replicate_relation(c, x[1:], y[1:]),
                                }
                            )
    H.write_csv(out / "relation_certificates.csv", rows)
    if repeats:
        H.write_csv(out / "held_replicate_relations.csv", repeats)
    H.write_npz(out / "certificates.npz", **numeric)
    summary = {
        "finite_certificates": len(rows),
        "held_replicate_relations": len(repeats),
        "max_svd_qr_error": max_qr_error,
        "cpu_seconds": time.process_time() - start,
        "new_model_calls": 0,
        "accuracy_optimization": False,
        "by_view": {
            name: {
                "three_seed_mean_certificates": sum(
                    r["representation"] == name and r["unit"] == "three_seed_mean"
                    for r in rows
                ),
                "saturated_mean_certificates": sum(
                    r["representation"] == name
                    and r["unit"] == "three_seed_mean"
                    and r["arbitrary_target_transport_possible"]
                    for r in rows
                ),
            }
            for name in raw
        },
        "scope": m["scope"],
        "original_scope_complete": False,
    }
    H.write_json(out / "summary.json", summary)
    names = ["relation_certificates.csv", "certificates.npz", "summary.json"]
    if repeats:
        names.append("held_replicate_relations.csv")
    H.write_json(out / "output_seal.json", {p: H.digest(out / p) for p in names})
    # Status is a mutable receipt, never an immutable scientific result.
    with (out / "status.json").open("w") as stream:
        json.dump(
            {"status": "COMPLETE_FINITE_NOT_GLOBAL", **summary},
            stream,
            indent=2,
            allow_nan=False,
        )
    print(summary, flush=True)


def main():
    """Run exactly one prepare or finite arithmetic phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        (prepare if args.stage == "prepare" else execute)(args.output)
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", flush=True)
        receipt = args.output / "status.json"
        if receipt.is_file() and H.load(receipt)["status"] == "PREPARED":
            with receipt.open("w") as stream:
                json.dump(
                    {"status": "FAILED_PRESERVED", "error": repr(error)},
                    stream,
                    indent=2,
                    allow_nan=False,
                )
        raise


if __name__ == "__main__":
    main()
