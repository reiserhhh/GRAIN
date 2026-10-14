#!/usr/bin/env python3
"""Construct and inspect source-anchored GRAIN functions without native answers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_semantic_anchor as M

SOURCE = ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded"
DEFAULT = ROOT / "results/grain_math_response_v1/semantic_anchor_r1"
TARGETS = ("extraversion", "neuroticism")
ARMS = ("full", "single", "source_relinked")
CODE = (
    "grain_research/construction/math_semantic_anchor.py",
    "scripts/run_grain_semantic_anchor_hm147.py",
    "tests/test_grain_semantic_anchor.py",
    "docs/GRAIN_SEMANTIC_ANCHOR_PLAN_HM147.md",
)


def load(path):
    """Read a JSON artifact without running any model or source code."""
    with path.open() as stream:
        return json.load(stream)


def digest(path):
    """Hash a complete local artifact for immutable numeric provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    """Create a new artifact; never silently overwrite an existing experiment."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_npz(path, **arrays):
    """Create a pickle-free numeric artifact exclusively."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_csv(path, rows):
    """Create a complete finite table with no outcome-dependent row filtering."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive(path, keys):
    """Reject arrays that inject labels or metadata into the cold scoring path."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(keys):
            raise ValueError(f"Unexpected fields in {path.name}")
        return {key: data[key].copy() for key in keys}


def check(out):
    """Authenticate prepared sources and frozen new analysis inputs/code."""
    manifest = load(out / "manifest.json")
    for root, section in ((ROOT, "code"), (SOURCE, "source"), (out, "inputs")):
        for name, value in manifest[section].items():
            if digest(root / name) != value:
                raise ValueError("Frozen ancestry changed: " + name)
    return manifest


def prepare(out):
    """Prepare only cached vectors and design metadata; no text/model answers."""
    manifest = load(SOURCE / "manifest.json")
    role_by_task = {
        task: role for role, tasks in manifest["split_roles"].items() for task in tasks
    }
    if role_by_task != {
        "diary": "discovery",
        "email": "discovery",
        "opinion": "discovery",
        "advice": "development",
        "review": "evaluation",
        "neighbor": "evaluation",
    }:
        raise ValueError("Original source/application split changed")
    raw = load(SOURCE / "readout_grain.json")["grain"]
    materials = load(SOURCE / "materials.json")
    if (
        len(materials) != 270
        or len({r["text_id"] for r in materials}) != 270
        or set(raw) != {r["text_id"] for r in materials}
    ):
        raise ValueError("Missing or duplicate historical source observations")
    rows, values, errors = [], [], []
    for item in materials:
        row = {
            key: item[key]
            for key in (
                "text_id",
                "condition",
                "axis",
                "pole",
                "family",
                "situation",
                "seed",
            )
        }
        row["axis"], row["pole"] = row["axis"] or "", int(row["pole"] or 0)
        row["role"] = role_by_task[row["situation"]]
        value = np.asarray(raw[row["text_id"]]["slices"], float)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("Invalid cached GRAIN slice observation")
        errors.append(float(abs(value.mean(axis=0) - raw[row["text_id"]]["g"]).max()))
        rows.append(row)
        values.append(value)
    if max(errors) > 1e-8:
        raise ValueError("Historical pooled coordinates are not the slice means")
    source_ids = [i for i, row in enumerate(rows) if row["role"] == "discovery"]
    cold_ids = [i for i, row in enumerate(rows) if row["role"] != "discovery"]
    if len(source_ids) != 135 or len(cold_ids) != 135:
        raise ValueError("Prepared role counts changed")
    out.mkdir(parents=True, exist_ok=False)
    source_rows, cold_rows = ([rows[i] for i in ids] for ids in (source_ids, cold_ids))
    write_json(out / "source_metadata.json", source_rows)
    write_json(out / "application_metadata.json", cold_rows)
    write_npz(out / "source_input.npz", views=np.asarray(values)[source_ids])
    write_npz(out / "cold_input.npz", views=np.asarray(values)[cold_ids])
    inputs = (
        "source_metadata.json",
        "application_metadata.json",
        "source_input.npz",
        "cold_input.npz",
    )
    write_json(
        out / "manifest.json",
        {
            "packet": "HM147",
            "source_rows": 135,
            "application_rows": 135,
            "historical_g_mean_error": max(errors),
            "targets": TARGETS,
            "arms": ARMS,
            "code": {name: digest(ROOT / name) for name in CODE},
            "source": {
                name: digest(SOURCE / name)
                for name in ("manifest.json", "materials.json", "readout_grain.json")
            },
            "inputs": {name: digest(out / name) for name in inputs},
            "scope": "Historical synthetic prompt assignment anchors, not human traits or a blind new corpus",
            "forbidden_inputs": "native/DeepSeek answers, generation text, questionnaires, private data, application labels during construction/scoring",
        },
    )
    print(json.dumps({"prepared": str(out), "source": 135, "application": 135}))


def source_objects(out):
    """Build source-only primitives from the preserved experimental assignments."""
    rows = load(out / "source_metadata.json")
    x = archive(out / "source_input.npz", ("views",))["views"]
    origin, scale = M.source_units(x)
    z = M.standardize(x, origin, scale)
    groups = ["|".join((r["situation"], r["family"], r["condition"])) for r in rows]
    relinked = M.relink_source(z, groups, [r["seed"] for r in rows])
    weights = M.anchor_weights(
        [r["axis"] for r in rows], [r["pole"] for r in rows], TARGETS
    )
    return rows, x, z, relinked, weights, origin, scale


def fit(out):
    """Seal the complete source witness before inspecting new cold effects."""
    check(out)
    start = time.monotonic()
    rows, _, z, relinked, weights, origin, scale = source_objects(out)
    coefficients, norms = M.compile_witness(z, weights, relinked)
    sectors = M.feature_sectors()
    single = np.asarray([mask.bit_count() <= 1 for mask in sectors])
    single_error = float(
        abs(coefficients[0, :, single] - coefficients[2, :, single]).max()
    )
    if single_error > 1e-12:
        raise ValueError("Conditional marginal preservation failed")
    write_npz(
        out / "program.npz",
        origin=origin,
        scale=scale,
        coefficients=coefficients,
        norms=norms,
    )
    energies = [
        {
            "target": name,
            "sector": mask,
            "order": mask.bit_count(),
            "normalized_energy": float(
                np.sum(coefficients[0, k, sectors == mask] ** 2)
            ),
        }
        for k, name in enumerate(TARGETS)
        for mask in range(16)
    ]
    write_csv(out / "source_sector_energy.csv", energies)
    summary = {
        "source_prompt_class_counts": {
            name: {
                str(p): sum(r["axis"] == name and r["pole"] == p for r in rows)
                for p in (-1, 1)
            }
            for name in TARGETS
        },
        "witness_norms": norms.tolist(),
        "relinked_single_error": single_error,
        "source_target_witness_cosine": float(coefficients[0, 0] @ coefficients[0, 1]),
        "higher_order_source_energy": {
            name: sum(
                r["normalized_energy"]
                for r in energies
                if r["target"] == name and r["order"] >= 2
            )
            for name in TARGETS
        },
        "source_seconds": time.monotonic() - start,
        "meaning": "Generator assignment anchored witness; empirical norm includes sampling variation and is not human trait validity",
    }
    write_json(out / "source_summary.json", summary)
    write_json(
        out / "source_seal.json",
        {
            name: digest(out / name)
            for name in (
                "program.npz",
                "source_sector_energy.csv",
                "source_summary.json",
            )
        },
    )
    print(json.dumps(summary))


def score(program, input_path, output_path):
    """Run a standalone label-free numeric process on the fixed source program."""
    p = archive(program, ("origin", "scale", "coefficients", "norms"))
    x = archive(input_path, ("views",))["views"]
    scores, sectors = M.score_views(x, p["origin"], p["scale"], p["coefficients"])
    write_npz(output_path, scores=scores, sectors=sectors)
    print(
        json.dumps(
            {
                "rows": len(x),
                "score_shape": scores.shape,
                "labels_or_text_received": False,
            }
        )
    )


def analyze(out):
    """Join labels only after the source program and cold scores are sealed."""
    check(out)
    for name, value in load(out / "source_seal.json").items():
        if digest(out / name) != value:
            raise ValueError("Source program changed before analysis")
    rows = load(out / "application_metadata.json")
    result = archive(out / "cold_scores.npz", ("scores", "sectors"))
    scores, sectors = result["scores"], result["sectors"]
    if scores.shape != (135, 3, 2) or sectors.shape != (135, 3, 2, 16):
        raise ValueError("Incomplete cold score support")
    lookup = {
        (r["axis"], r["pole"], r["situation"], r["family"], r["seed"]): i
        for i, r in enumerate(rows)
    }
    pairs = []
    for i, row in enumerate(rows):
        if row["axis"] not in TARGETS or row["pole"] != 1:
            continue
        j = lookup[row["axis"], -1, row["situation"], row["family"], row["seed"]]
        for a, arm in enumerate(ARMS):
            for k, target in enumerate(TARGETS):
                parts = sectors[i, a, k] - sectors[j, a, k]
                pairs.append(
                    {
                        "role": row["role"],
                        "task": row["situation"],
                        "family": row["family"],
                        "seed": row["seed"],
                        "manipulated_axis": row["axis"],
                        "target": target,
                        "arm": arm,
                        "high_id": row["text_id"],
                        "low_id": rows[j]["text_id"],
                        "contrast": float(scores[i, a, k] - scores[j, a, k]),
                        **{f"sector_{m:02d}": float(parts[m]) for m in range(16)},
                    }
                )
    groups = {}
    for row in pairs:
        key = tuple(
            row[key]
            for key in ("role", "task", "family", "manipulated_axis", "target", "arm")
        )
        groups.setdefault(key, []).append(row)
    cells = []
    for key, items in sorted(groups.items()):
        cells.append(
            dict(
                zip(
                    ("role", "task", "family", "manipulated_axis", "target", "arm"), key
                )
            )
            | {
                "paired_seeds": len(items),
                "mean_contrast": float(np.mean([r["contrast"] for r in items])),
                "positive_pairs": sum(r["contrast"] > 0 for r in items),
                "single_contribution": float(
                    np.mean(
                        [
                            sum(
                                r[f"sector_{m:02d}"]
                                for m in range(16)
                                if m.bit_count() == 1
                            )
                            for r in items
                        ]
                    )
                ),
                "joint_contribution": float(
                    np.mean(
                        [
                            sum(
                                r[f"sector_{m:02d}"]
                                for m in range(16)
                                if m.bit_count() >= 2
                            )
                            for r in items
                        ]
                    )
                ),
            }
        )
    controls = []
    by_control = {
        (r["condition"], r["situation"], r["seed"]): i
        for i, r in enumerate(rows)
        if not r["axis"]
    }
    for (condition, task, seed), i in by_control.items():
        if not condition.startswith("nonpersona"):
            continue
        j = by_control["neutral", task, seed]
        for a, arm in enumerate(ARMS):
            for k, target in enumerate(TARGETS):
                controls.append(
                    {
                        "condition": condition,
                        "task": task,
                        "seed": seed,
                        "target": target,
                        "arm": arm,
                        "minus_neutral": float(scores[i, a, k] - scores[j, a, k]),
                    }
                )
    totals = []
    for role in ("development", "evaluation", "all_application"):
        for manipulated in TARGETS:
            for target in TARGETS:
                for arm in ARMS:
                    sel = [
                        r
                        for r in pairs
                        if (role == "all_application" or r["role"] == role)
                        and r["manipulated_axis"] == manipulated
                        and r["target"] == target
                        and r["arm"] == arm
                    ]
                    totals.append(
                        {
                            "role": role,
                            "manipulated_axis": manipulated,
                            "target": target,
                            "arm": arm,
                            "pairs": len(sel),
                            "mean_contrast": float(
                                np.mean([r["contrast"] for r in sel])
                            ),
                            "positive_pairs": sum(r["contrast"] > 0 for r in sel),
                        }
                    )
    write_csv(out / "paired_sector_effects.csv", pairs)
    write_csv(out / "effects_by_task_family.csv", cells)
    write_csv(out / "control_effects.csv", controls)
    write_csv(out / "effects_by_role.csv", totals)
    write_json(
        out / "summary.json",
        {
            "status": "EXECUTED_SOURCE_ANCHORED_JOINT_VALUATION_NOT_HUMAN_VALIDITY",
            "source": load(out / "source_summary.json"),
            "cold_rows": 135,
            "paired_effect_rows": len(pairs),
            "cells": len(cells),
            "control_effect_rows": len(controls),
            "effects_by_role": totals,
            "new_neural_API_training_private_calls": 0,
            "scientific_scope": "Historical generated texts; prompt allocation meaning, fixed source units and no native answer fit. No human personality accuracy or irreducible synergy claim.",
        },
    )
    names = (
        "cold_scores.npz",
        "paired_sector_effects.csv",
        "effects_by_task_family.csv",
        "control_effects.csv",
        "effects_by_role.csv",
        "summary.json",
    )
    write_json(out / "output_seal.json", {name: digest(out / name) for name in names})
    print(
        json.dumps(
            {
                "paired_effect_rows": len(pairs),
                "cells": len(cells),
                "control_effect_rows": len(controls),
            }
        )
    )


def verify(out):
    """Independently recompute scores via kernels and authenticate all ancestors."""
    manifest = check(out)
    for seal in ("source_seal.json", "output_seal.json"):
        for name, value in load(out / seal).items():
            if digest(out / name) != value:
                raise ValueError("Sealed scientific artifact changed: " + name)
    rows, x, z, r, weights, origin, scale = source_objects(out)
    cold = archive(out / "cold_input.npz", ("views",))["views"]
    program = archive(out / "program.npz", ("origin", "scale", "coefficients", "norms"))
    actual = archive(out / "cold_scores.npz", ("scores", "sectors"))
    y = M.standardize(cold, origin, scale)
    norms = np.sqrt(np.diag(weights.T @ M.product_kernel(z, z) @ weights))
    expected = np.empty((len(cold), 3, 2, 16))
    for mask in range(16):
        for a, source in enumerate((z, z, r)):
            kernel = np.ones((len(cold), len(z)))
            for v in range(4):
                if mask & (1 << v):
                    kernel *= y[:, v] @ source[:, v].T
            expected[:, a, :, mask] = kernel @ weights / norms
            if a == 1 and mask.bit_count() > 1:
                expected[:, a, :, mask] = 0
    checks = {
        "program_origin": float(abs(program["origin"] - origin).max()),
        "program_scale": float(abs(program["scale"] - scale).max()),
        "norm_kernel_vs_tensor": float(abs(program["norms"] - norms).max()),
        "all_sector_kernel_vs_tensor": float(abs(actual["sectors"] - expected).max()),
        "cold_total_kernel_vs_tensor": float(
            abs(actual["scores"] - expected.sum(axis=-1)).max()
        ),
    }
    feature_error = 0.0
    for group in sorted(
        {(row["situation"], row["family"], row["condition"]) for row in rows}
    ):
        ids = [
            i
            for i, row in enumerate(rows)
            if (row["situation"], row["family"], row["condition"]) == group
        ]
        for v in range(4):
            if sorted(map(tuple, z[ids, v])) != sorted(map(tuple, r[ids, v])):
                raise ValueError("Source conditional marginal multiset changed")
            feature_error = max(
                feature_error,
                float(abs(z[ids, v].sum(axis=0) - r[ids, v].sum(axis=0)).max()),
            )
    checks["conditional_marginal_sum"] = feature_error
    rng = np.random.default_rng(147)
    rotations = np.stack([np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(4)])
    xr, yr = (np.einsum("nvd,vde->nve", t, rotations) for t in (x, cold))
    om, sm = M.source_units(xr)
    zr = M.standardize(xr, om, sm)
    groups = [
        "|".join((row["situation"], row["family"], row["condition"])) for row in rows
    ]
    rr = M.relink_source(zr, groups, [row["seed"] for row in rows])
    wc, _ = M.compile_witness(zr, weights, rr)
    rotated_scores, rotated_parts = M.score_views(yr, om, sm, wc)
    checks["coordinate_gauge_total"] = float(
        abs(rotated_scores - actual["scores"]).max()
    )
    checks["coordinate_gauge_sectors"] = float(
        abs(rotated_parts - actual["sectors"]).max()
    )
    if max(checks.values()) > 1e-10:
        raise ValueError("Alternate numerical check failed: " + str(checks))
    source_ids = {row["text_id"] for row in rows}
    cold_rows = load(out / "application_metadata.json")
    if source_ids & {row["text_id"] for row in cold_rows}:
        raise ValueError("Source/application overlap")
    print(
        json.dumps(
            {
                "verified": True,
                "checks": checks,
                "replayed_sector_scores": int(actual["sectors"].size),
                "cold_input_fields": ["views"],
                "ancestry_packet": manifest["packet"],
                "scope": "Same-agent alternate arithmetic, not independent blind replication",
            }
        )
    )


def main():
    """Expose distinct construction, cold scoring and metadata-analysis stages."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action", choices=("prepare", "fit", "score", "analyze", "verify")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        if not all((args.program, args.input, args.output)):
            parser.error("Cold scoring requires explicit program, input and output")
        score(args.program, args.input, args.output)
    else:
        {"prepare": prepare, "fit": fit, "analyze": analyze, "verify": verify}[
            args.action
        ](args.out)


if __name__ == "__main__":
    main()
