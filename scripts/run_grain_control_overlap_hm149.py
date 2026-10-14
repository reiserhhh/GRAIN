#!/usr/bin/env python3
"""Test target/control response overlap without changing frozen target scores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_control_overlap as M
from scripts import run_grain_semantic_anchor_hm147 as H

PARENT = H.DEFAULT
DEFAULT = ROOT / "results/grain_math_response_v1/control_overlap_r1"
CODE = (
    "grain_research/construction/math_control_overlap.py",
    "scripts/run_grain_control_overlap_hm149.py",
    "tests/test_grain_control_overlap.py",
    "docs/GRAIN_CONTROL_OVERLAP_PLAN_HM149.md",
)
PARENT_FILES = (
    "manifest.json",
    "source_input.npz",
    "source_metadata.json",
    "cold_input.npz",
    "application_metadata.json",
    "program.npz",
    "source_seal.json",
)
PROGRAM_KEYS = ("origin", "scale", "parts", "basis", "singular_values", "dual")


def check(out):
    """Verify historical ancestry and every newly frozen code/input artifact."""
    H.check(PARENT)
    for name, value in H.load(PARENT / "source_seal.json").items():
        if H.digest(PARENT / name) != value:
            raise ValueError("The frozen parent source changed: " + name)
    manifest = H.load(out / "manifest.json")
    for root, section in ((ROOT, "code"), (PARENT, "parent"), (out, "inputs")):
        for name, value in manifest[section].items():
            if H.digest(root / name) != value:
                raise ValueError("Frozen ancestry changed: " + name)
    return manifest


def parent_program():
    """Load fixed parent units/covectors without fitting an answer or target."""
    return H.archive(
        PARENT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )


def prepare(out):
    """Prepare control design and vectors, excluding text and native answers."""
    H.check(PARENT)
    rows = H.load(PARENT / "source_metadata.json")
    weights, controls = M.control_design(rows)
    source = H.archive(PARENT / "source_input.npz", ("views",))["views"]
    cold = H.archive(PARENT / "cold_input.npz", ("views",))["views"]
    if (
        weights.shape != (18, 135)
        or source.shape != (135, 4, 4)
        or cold.shape != source.shape
    ):
        raise ValueError("The original finite material design changed")
    program = parent_program()
    out.mkdir(parents=True, exist_ok=False)
    H.write_npz(out / "source_input.npz", views=source, contrast_weights=weights)
    H.write_npz(out / "cold_input.npz", views=cold)
    H.write_json(out / "source_control_metadata.json", controls)
    inputs = ("source_input.npz", "cold_input.npz", "source_control_metadata.json")
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM149",
            "source_rows": len(source),
            "cold_rows": len(cold),
            "source_control_contrasts": len(controls),
            "rank_relative_tolerance": 1e-10,
            "code": {name: H.digest(ROOT / name) for name in CODE},
            "parent": {name: H.digest(PARENT / name) for name in PARENT_FILES},
            "inputs": {name: H.digest(out / name) for name in inputs},
            "parent_unit_norms": program["norms"].tolist(),
            "scope": "Historical synthetic assignment contrasts, not new human data or causal nuisance",
            "original_scope_complete": False,
            "forbidden": "text/API/native answers/private data/new neural training/label-based model selection",
        },
    )
    print(json.dumps({"prepared": str(out), "control_contrasts": len(controls)}))


def source_objects(out):
    """Construct the finite response span from source controls only."""
    source = H.archive(out / "source_input.npz", ("views", "contrast_weights"))
    parent = parent_program()
    z = H.M.standardize(source["views"], parent["origin"], parent["scale"])
    b = source["contrast_weights"] @ H.M.tensor_features(z)
    projection = M.response_projection(b, parent["coefficients"], 1e-10)
    return parent, source, z, b, projection


def fit(out):
    """Freeze an exact bookkeeping split; do not change target definitions."""
    check(out)
    start = time.monotonic()
    parent, _, _, b, projection = source_objects(out)
    H.write_npz(
        out / "program.npz",
        origin=parent["origin"],
        scale=parent["scale"],
        **projection,
    )
    metadata = H.load(out / "source_control_metadata.json")
    support = []
    for task in sorted({r["situation"] for r in metadata}):
        excluded = np.asarray([r["situation"] == task for r in metadata])
        loo = M.response_projection(b[~excluded], parent["coefficients"], 1e-10)
        covered = M.coverage(b[excluded], loo["basis"])
        ids = np.flatnonzero(excluded)
        for i, fraction in zip(ids, covered, strict=True):
            support.append(
                {
                    **metadata[i],
                    "basis_source_tasks": "+".join(
                        sorted(
                            {r["situation"] for r in metadata if r["situation"] != task}
                        )
                    ),
                    "basis_rank": len(loo["basis"]),
                    "covered_squared_norm_fraction": float(fraction),
                }
            )
    H.write_csv(out / "source_held_task_control_coverage.csv", support)
    geometry = []
    grams = M.target_couplings(projection["parts"])
    for arm, arm_name in enumerate(H.ARMS):
        for k, target in enumerate(H.TARGETS):
            energy = np.sum(projection["parts"][arm, k] ** 2, axis=1)
            geometry.append(
                {
                    "arm": arm_name,
                    "target": target,
                    "full_covector_squared_norm": float(energy.sum()),
                    "control_overlap_squared_norm": float(energy[0]),
                    "complement_squared_norm": float(energy[1]),
                    "overlap_fraction": float(energy[0] / energy.sum()),
                }
            )
    H.write_csv(out / "source_target_geometry.csv", geometry)
    H.write_json(
        out / "source_summary.json",
        {
            "basis_rank": len(projection["basis"]),
            "singular_values": projection["singular_values"].tolist(),
            "target_grams": {key: value.tolist() for key, value in grams.items()},
            "source_overlap": geometry,
            "source_held_task_coverage_mean": float(
                np.mean([r["covered_squared_norm_fraction"] for r in support])
            ),
            "seconds": time.monotonic() - start,
            "self_orthogonality_is_validation": False,
            "no_components_removed": True,
        },
    )
    names = (
        "program.npz",
        "source_summary.json",
        "source_target_geometry.csv",
        "source_held_task_control_coverage.csv",
    )
    H.write_json(
        out / "program_seal.json", {name: H.digest(out / name) for name in names}
    )
    print(
        json.dumps(
            {"frozen": str(out / "program.npz"), "basis_rank": len(projection["basis"])}
        )
    )


def score(out):
    """Cold scoring sees only sealed numeric program and numeric observations."""
    check(out)
    for name, value in H.load(out / "program_seal.json").items():
        if H.digest(out / name) != value:
            raise ValueError("The frozen control-response split changed")
    program = H.archive(out / "program.npz", PROGRAM_KEYS)
    x = H.archive(out / "cold_input.npz", ("views",))["views"]
    z = H.M.standardize(x, program["origin"], program["scale"])
    parts = M.score_features(H.M.tensor_features(z), program["parts"])
    H.write_npz(out / "scores.npz", parts=parts)
    H.write_json(out / "score_seal.json", {"scores.npz": H.digest(out / "scores.npz")})
    print(
        json.dumps(
            {
                "cold_rows": len(x),
                "shape": list(parts.shape),
                "no_metadata_in_scorer": True,
            }
        )
    )


def paired_effects(rows, values):
    """Summarize all cold high/low pairs after numerical scoring is frozen."""
    groups = {}
    for i, row in enumerate(rows):
        if row["axis"] in H.TARGETS:
            key = (
                row["role"],
                row["situation"],
                row["family"],
                row["axis"],
                row["seed"],
            )
            pole = int(row["pole"])
            if pole in groups.setdefault(key, {}):
                raise ValueError("Duplicate cold assignment pole")
            groups[key][pole] = i
    effects = []
    for key, ids in sorted(groups.items()):
        if set(ids) != {-1, 1}:
            raise ValueError("A complete observed cold assignment pair is required")
        delta = values[ids[1]] - values[ids[-1]]
        role, task, family, axis, seed = key
        for arm, arm_name in enumerate(H.ARMS):
            for k, target in enumerate(H.TARGETS):
                effects.append(
                    {
                        "role": role,
                        "situation": task,
                        "family": family,
                        "manipulated_axis": axis,
                        "seed": seed,
                        "arm": arm_name,
                        "target": target,
                        "full_effect": float(delta[arm, k].sum()),
                        "control_overlap_effect": float(delta[arm, k, 0]),
                        "complement_effect": float(delta[arm, k, 1]),
                    }
                )
    return effects


def aggregate(effects, keys):
    """Retain every role/task/axis/target cell and every signed component."""
    groups = {}
    for row in effects:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "n_observed_pairs": len(group),
            **{
                name: float(np.mean([row[name] for row in group]))
                for name in (
                    "full_effect",
                    "control_overlap_effect",
                    "complement_effect",
                )
            },
        }
        for key, group in sorted(groups.items())
    ]


def analysis_objects(out):
    """Join historical design labels only after all cold scalar values exist."""
    rows = H.load(PARENT / "application_metadata.json")
    values = H.archive(out / "scores.npz", ("parts",))["parts"]
    effects = paired_effects(rows, values)
    weights, metadata = M.control_design(rows)
    control_values = np.einsum("in,nakp->iakp", weights, values)
    controls = []
    for i, item in enumerate(metadata):
        for arm, arm_name in enumerate(H.ARMS):
            for k, target in enumerate(H.TARGETS):
                controls.append(
                    {
                        **item,
                        "arm": arm_name,
                        "target": target,
                        "full_effect": float(control_values[i, arm, k].sum()),
                        "control_overlap_effect": float(control_values[i, arm, k, 0]),
                        "complement_effect": float(control_values[i, arm, k, 1]),
                    }
                )
    cells = aggregate(
        effects, ("role", "situation", "family", "manipulated_axis", "arm", "target")
    )
    roles = aggregate(effects, ("role", "manipulated_axis", "arm", "target"))
    all_roles = aggregate(effects, ("manipulated_axis", "arm", "target"))
    return effects, controls, cells, roles, all_roles


def analyze(out):
    """Report descriptive effects, never select a new predictor or polarity."""
    check(out)
    effects, controls, cells, roles, all_roles = analysis_objects(out)
    for name, rows in (
        ("effects.csv", effects),
        ("control_effects.csv", controls),
        ("effects_by_cell.csv", cells),
        ("effects_by_role.csv", roles),
        ("effects_all_application.csv", all_roles),
    ):
        H.write_csv(out / name, rows)
    primary = [
        r for r in roles if r["arm"] == "full" and r["target"] == r["manipulated_axis"]
    ]
    H.write_json(
        out / "summary.json",
        {
            "packet": "HM149",
            "source": H.load(out / "source_summary.json"),
            "primary_diagonal_roles": primary,
            "primary_diagonal_all": [
                r
                for r in all_roles
                if r["arm"] == "full" and r["target"] == r["manipulated_axis"]
            ],
            "actual_observed_cold_persona_pairs": len(effects) // 6,
            "actual_observed_cold_control_pairs": len(controls) // 6,
            "paired_effect_rows": len(effects),
            "control_effect_rows": len(controls),
            "cell_rows": len(cells),
            "new_model_API_training_private_calls": 0,
            "noise_or_personality_information_fraction": False,
            "cold_corpus_is_new_blind_data": False,
            "original_scope_complete": False,
        },
    )
    names = (
        "effects.csv",
        "control_effects.csv",
        "effects_by_cell.csv",
        "effects_by_role.csv",
        "effects_all_application.csv",
        "summary.json",
    )
    H.write_json(
        out / "output_seal.json", {name: H.digest(out / name) for name in names}
    )
    print(json.dumps({"analyzed": str(out), "primary_diagonal_roles": primary}))


def verify(out):
    """Independently replay scores with source kernels and all finite tables."""
    check(out)
    for seal in ("program_seal.json", "score_seal.json", "output_seal.json"):
        for name, value in H.load(out / seal).items():
            if H.digest(out / name) != value:
                raise ValueError("An output was changed: " + name)
    parent, source, z, b, fitted = source_objects(out)
    program = H.archive(out / "program.npz", PROGRAM_KEYS)
    cold = H.archive(out / "cold_input.npz", ("views",))["views"]
    cold_z = H.M.standardize(cold, parent["origin"], parent["scale"])
    values = H.archive(out / "scores.npz", ("parts",))["parts"]
    kernel_contrasts = source["contrast_weights"] @ H.M.product_kernel(z, cold_z)
    overlap_kernel = np.einsum("aki,in->nak", program["dual"], kernel_contrasts)
    full = H.M.score_views(
        cold, parent["origin"], parent["scale"], parent["coefficients"]
    )[0]
    grams = M.target_couplings(program["parts"])
    errors = {
        "source_program_replay": float(abs(program["parts"] - fitted["parts"]).max()),
        "full_score_unchanged": float(abs(values.sum(axis=-1) - full).max()),
        "independent_kernel_overlap": float(
            abs(overlap_kernel - values[:, :, :, 0]).max()
        ),
        "source_complement_control_action": float(
            abs(np.einsum("if,akf->iak", b, program["parts"][:, :, 1])).max()
        ),
        "covector_sum": float(
            abs(program["parts"].sum(axis=2) - parent["coefficients"]).max()
        ),
        "orthogonal_parts": float(abs(grams["cross"]).max()),
        "target_gram_sum": float(
            abs(grams["full"] - grams["overlap"] - grams["complement"]).max()
        ),
        "dual_coefficient_replay": float(
            abs(
                np.einsum("aki,if->akf", program["dual"], b) - program["parts"][:, :, 0]
            ).max()
        ),
    }
    for name, rows in zip(
        (
            "effects.csv",
            "control_effects.csv",
            "effects_by_cell.csv",
            "effects_by_role.csv",
            "effects_all_application.csv",
        ),
        analysis_objects(out),
        strict=True,
    ):
        saved = list(H.csv.DictReader((out / name).open()))
        if len(saved) != len(rows):
            raise ValueError("Missing effect rows")
        for old, new in zip(saved, rows, strict=True):
            for key, value in new.items():
                if isinstance(value, float):
                    if abs(float(old[key]) - value) > 1e-12:
                        raise ValueError("An effect cell did not replay")
                elif old[key] != str(value):
                    raise ValueError("An effect address changed")
    if max(errors.values()) > 1e-9:
        raise ValueError("Arithmetic verification failed: " + str(errors))
    print(
        json.dumps(
            {
                "status": "VERIFIED_FINITE_CONTROL_OVERLAP_NOT_PSYCHOLOGICAL_CLOSURE",
                "errors": errors,
                "original_scope_complete": False,
            }
        )
    )


def main():
    """Run one bounded stage; existing output paths never silently overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "fit", "score", "analyze", "verify")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.out.resolve())


if __name__ == "__main__":
    main()
