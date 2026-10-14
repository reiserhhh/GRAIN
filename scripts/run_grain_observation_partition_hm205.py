#!/usr/bin/env python3
"""Execute one provenance-bound observation-partition construction, without grids."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observation_partition as M
from grain_research.construction import math_vector_likelihood as V
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_observation_partition_hm205.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_OBSERVATION_PARTITION_PLAN_HM205.md",
    "grain_research/construction/math_observation_partition.py",
    "scripts/run_grain_observation_partition_hm205.py",
    "tests/test_grain_observation_partition.py",
    "tests/test_grain_vector_likelihood.py",
    "grain_research/construction/math_vector_likelihood.py",
    "scripts/run_grain_observed_family_hm170.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/observation_partition_r1"
LAWS = ("independent", "shared_occasion")
READERS = ("joint", "diagonal")
ARMS = ("full", "remove_selected", "remove_control", "selected_only", "control_only")


def load(path):
    """Read sealed numerical arrays only, without pickle or natural text."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def parameters(s2, law):
    """Use two declared worlds with exactly the same 12-event mean variance."""
    return (12 * s2, 0.0) if law == "independent" else (6 * s2, s2 / 2)


def frames(z):
    """Bind each subset to its existing root comments, not artificial group IDs."""
    for u, rep, view, arm in np.ndindex(24, 2, 2, 5):
        ss, cc, whole = z["selected"][u, view], z["control"][u, view], np.arange(12)
        slots = (whole, whole[~np.isin(whole, ss)], whole[~np.isin(whole, cc)], ss, cc)[
            arm
        ]
        x = z["panel_vectors"][u, rep, view, slots]
        yield (
            {
                "author_index": u,
                "representation": rep,
                "view": view,
                "arm": ARMS[arm],
                "slots": slots.tolist(),
            },
            x,
        )


def prepare(out):
    """Freeze old source reference and same-unit observations before new scoring."""
    cfg = R.read(ROOT / CONFIG)
    parent, event = ROOT / cfg["parent"], ROOT / cfg["event_parent"]
    stages = (
        "compiled_seal.json",
        "scored_seal.json",
        "analysis_seal.json",
        "verification_seal.json",
    )
    R.check(parent, stages)
    R.check(event, stages)
    old, e = R.prepared(parent), R.prepared(event)
    kernel = load(parent / "source_kernel.npz")
    if not np.array_equal(
        old["source_indices"], e["source_indices"]
    ) or not np.array_equal(old["panel_indices"], e["panel_indices"]):
        raise ValueError("Old source/application event linkage changed")
    if not np.array_equal(kernel["labels"], old["labels"]) or e[
        "panel_vectors"
    ].shape != (24, 2, 2, 12, 1024):
        raise ValueError("Declared source/profile/vector support changed")
    if set(old["source_indices"]) & set(old["panel_indices"]):
        raise ValueError("Source/application overlap")
    inputs = [
        parent / "manifest.json",
        parent / "source_bundle.json",
        parent / "source_kernel.npz",
        parent / "analysis.npz",
        event / "observations.npz",
    ]
    for root in (parent, event):
        inputs.extend(root / name for name in ("prepared_seal.json",) + stages)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        panel_vectors=e["panel_vectors"],
        selected=e["selected"],
        control=e["control"],
        prototypes=kernel["prototypes"],
        labels=kernel["labels"],
        splits=old["splits"],
    )
    R.write(
        out / "source_bundle.json",
        {
            "source_variance": R.read(parent / "source_bundle.json")["variance"],
            "targets": list(V.TARGETS),
            "reference": kernel["labels"].mean(0).tolist(),
            "law_status": "declared_Gaussian_worlds;not_identified_natural_noise",
            "new_model_calls": 0,
            "application_labels_read": 0,
        },
    )
    R.write(
        out / "frame_descriptors.json", [meta for meta, _ in frames(R.prepared(out))]
    )
    R.write(
        out / "data_schema.json",
        {
            "source_authors": 386,
            "application_authors": 24,
            "frames": 480,
            "representations": ["cue_mask", "raw"],
            "root_events_per_subset": [12, 8, 8, 4, 4],
            "dimensions": 1024,
            "application_labels_read": 0,
            "cohort": "historical;N_enriched_selection_and_representation_overlap_retained",
            "grouping": "numeric_linear_averages;not_reencoded_raw_text_slices",
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "source_bundle.json",
            "frame_descriptors.json",
            "data_schema.json",
        ),
    )


def compute(out):
    """Run all groupings/worlds once, retaining every target and signed reading."""
    R.check(out)
    z, b = R.prepared(out), R.read(out / "source_bundle.json")
    family = list(frames(z))
    s2 = b["source_variance"]
    centers, precision, geometry_rows = [], [], []
    lookup = {}
    for n in (4, 8, 12):
        for li, law in enumerate(LAWS):
            sigma, tau = parameters(s2, law)
            for gi, (name, a) in enumerate(M.programs(n).items()):
                for ri, reader in enumerate(READERS):
                    g = M.geometry(a, sigma, tau, ri == 1)
                    lookup[n, li, gi, ri] = g
                    geometry_rows.append(
                        {
                            "root_events": n,
                            "law": law,
                            "program": name,
                            "reader": reader,
                            "group_rows": len(a),
                            "rank": g["rank"],
                            "information": g["alpha"],
                            "root_information": g["root_information"],
                            "mean_recovery_defect": g["mean_recovery_defect"],
                        }
                    )
    for meta, x in family:
        for li, gi, ri in np.ndindex(2, 7, 2):
            info = lookup[len(x), li, gi, ri]
            centers.append(info["root_weights"] @ x)
            precision.append(info["alpha"])
    centers, precision = np.array(centers), np.array(precision)
    post, profile = [], []
    for bank in range(2):
        result = M.posterior(z["prototypes"][bank], z["labels"], centers, precision)
        post.append(result["posterior"].reshape(480, 2, 7, 2, 5))
        profile.append(result["profile"].reshape(480, 2, 7, 2, 32))
    post, profile = np.stack(post, axis=4), np.stack(profile, axis=4)
    rows, metrics = [], []
    ref = np.asarray(b["reference"])
    for f, li, gi, ri, bank in np.ndindex(480, 2, 7, 2, 2):
        row = {
            "frame": f,
            **family[f][0],
            "law": LAWS[li],
            "program": M.PROGRAMS[gi],
            "reader": READERS[ri],
            "source_window": bank,
        }
        row.pop("slots")
        row.update(
            {q: float(post[f, li, gi, ri, bank, qi]) for qi, q in enumerate(V.TARGETS)}
        )
        rows.append(row)
    for li, gi, ri in np.ndindex(2, 7, 2):
        actual, base = post[:, li, gi, ri], post[:, li, 0, 0]
        delta = actual - base
        for qi, q in enumerate(V.TARGETS):
            aa, bb = actual[..., qi] - ref[qi], base[..., qi] - ref[qi]
            metrics.append(
                {
                    "law": LAWS[li],
                    "program": M.PROGRAMS[gi],
                    "reader": READERS[ri],
                    "target": q,
                    "same_events_delta_RMS": float(
                        np.sqrt(np.mean(delta[..., qi] ** 2))
                    ),
                    "same_events_max_abs_delta": float(np.max(np.abs(delta[..., qi]))),
                    "relative_sign_reversals": int(
                        np.sum((aa * bb < 0) & (np.abs(bb) > 1e-10))
                    ),
                    "cells": 960,
                    "lossless_program": gi != 6,
                }
            )
    law_rows = []
    for n in (4, 8, 12):
        mask = np.array([len(x) == n for _, x in family])
        d = post[mask, 1, 0, 0] - post[mask, 0, 0, 0]
        law_rows.append(
            {
                "root_events": n,
                "frames": int(mask.sum()),
                "all_target_RMS": float(np.sqrt(np.mean(d**2))),
                **{
                    q: float(np.sqrt(np.mean(d[..., qi] ** 2)))
                    for qi, q in enumerate(V.TARGETS)
                },
            }
        )
    invariant = np.max(np.abs(post[:, :, :6, 0] - post[:, :, 0, 0][:, :, None]))
    R.archive(
        out / "analysis.npz",
        posterior=post,
        profile=profile,
        precisions=precision.reshape(480, 2, 7, 2),
    )
    R.table(out / "readings.csv", rows)
    R.table(out / "partition_effects.csv", metrics)
    R.table(out / "program_geometry.csv", geometry_rows)
    R.table(out / "law_contrasts.csv", law_rows)
    R.write(
        out / "summary.json",
        {
            "packet": "HM205",
            "frames": 480,
            "source_authors": 386,
            "application_authors": 24,
            "grouped_readings": len(rows),
            "target_values": len(rows) * 5,
            "lossless_joint_max_difference": float(invariant),
            "law_contrasts": law_rows,
            "new_model_calls": 0,
            "application_labels_read": 0,
            "parameter_search": False,
            "scientific_status": "positive_provenance_bound_constructor;not_human_accuracy_or_new_Gaussian_theorem",
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        (
            "analysis.npz",
            "readings.csv",
            "partition_effects.csv",
            "program_geometry.csv",
            "law_contrasts.csv",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def verify(out):
    """Audit using full quadratic likelihoods and a separate whitened SVD route."""
    R.check(out, ("computed_seal.json",))
    z, saved = R.prepared(out), load(out / "analysis.npz")
    b = R.read(out / "source_bundle.json")
    s2, family = b["source_variance"], list(frames(z))
    literal_error, info_error = 0.0, 0.0
    checks = 0
    for li, bank, gi in np.ndindex(2, 2, 7):
        index = (31 * gi + 19 * bank + 5 * li) % 480
        _, x = family[index]
        a = M.programs(len(x))[M.PROGRAMS[gi]]
        sigma, tau = parameters(s2, LAWS[li])
        expected = M.literal(z["prototypes"][bank], z["labels"], a @ x, a, sigma, tau)
        literal_error = max(
            literal_error,
            float(
                np.max(np.abs(expected - saved["posterior"][index, li, gi, 0, bank]))
            ),
        )
        checks += 1
    for n in (4, 8, 12):
        for li, law in enumerate(LAWS):
            sigma, tau = parameters(s2, law)
            chol = np.linalg.cholesky(sigma * np.eye(n) + tau * np.ones((n, n)))
            direction = np.linalg.solve(chol, np.ones(n))
            for a in M.programs(n).values():
                _, values, vt = np.linalg.svd(a @ chol, full_matrices=False)
                keep = values > 1e-12 * values.max()
                alpha = float(np.sum((vt[keep] @ direction) ** 2))
                info_error = max(
                    info_error, abs(alpha - M.geometry(a, sigma, tau)["alpha"])
                )
    profile = saved["profile"]
    bits = (np.arange(32)[:, None] // (2 ** np.arange(5))) % 2
    marginal_error = float(np.max(np.abs(profile @ bits - saved["posterior"])))
    old = load(
        ROOT / R.read(out / "manifest.json")["config"]["parent"] / "analysis.npz"
    )["posterior"].reshape(480, 2, 5)
    full = np.array([m["arm"] == "full" for m, _ in family])
    old_error = float(
        np.max(np.abs(saved["posterior"][full, :, 0, 0] - old[full, None]))
    )
    invariant = R.read(out / "summary.json")["lossless_joint_max_difference"]
    result = {
        "PASS": max(literal_error, info_error, marginal_error, old_error, invariant)
        < 1e-9,
        "literal_quadratic_checks": checks,
        "literal_quadratic_error": literal_error,
        "whitened_SVD_information_error": info_error,
        "joint_profile_marginal_error": marginal_error,
        "HM204_full_frame_replay_error": old_error,
        "lossless_joint_max_difference": invariant,
        "same_agent_not_independent": True,
        "new_model_calls": 0,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result))
    if not result["PASS"]:
        raise RuntimeError("Registered construction audit failed; preserve result")


def main():
    """Run one exclusive stage without auto-resume or scientific retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    start = time.perf_counter()
    globals()[args.stage](args.out)
    R.write(
        args.out / (args.stage + "_timing.json"),
        {"elapsed_seconds": time.perf_counter() - start},
    )


if __name__ == "__main__":
    main()
