#!/usr/bin/env python3
"""Test local gradient predictions, task-sensitive directions and historical access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.memory import render_records
from grain_research.vector_mechanisms.natural_patch import (
    current_difference_basis,
    match_quantized_row_norms,
    matched_random_rows,
    patch_message,
    project_rows,
)
from grain_research.vector_mechanisms.sensitivity import (
    local_prediction,
    unit_matrix_direction,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Use target-question derivatives on both target and collateral tasks."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--gradient-dir", type=Path, required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    parent = read(a.parent_dir / "manifest.json")
    gm = read(a.gradient_dir / "manifest.json")
    for directory in (a.parent_dir, a.gradient_dir):
        if read(directory / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Incomplete parent")
    if gm["parent_identity"] != parent["identity"]:
        raise ValueError("Gradient parent differs")
    cases = read(a.materials / "cases.json")
    if material_hash(cases) != parent["material_sha256"]:
        raise ValueError("Source material changed")
    by_case = {r["case_id"]: r for r in cases}
    gradient_jobs = {
        (r["case_id"], r["task"]): (i, r) for i, r in enumerate(gm["jobs"])
    }
    original = pd.read_csv(a.parent_dir / "interventions.csv")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    manifest = {
        "protocol": "GRAIN_VM04_SENSITIVITY_INTERVENTIONS_V1",
        "parent_identity": parent["identity"],
        "gradient_identity": gm["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "sites": parent["sites"],
        "ranks": [2, 4],
        "alphas": [0.25, 0.5, 1.0],
        "max_calls": 4096,
        "max_seconds": 1800,
        "basis_source": "disposition gradient only; applied unchanged to collateral task",
        "matching_tolerance": 0.01,
        "region_epsilons": [0.002, 0.01],
        "region_norm_reference": "same current-state Frobenius norm for history and current",
        "labels": "none",
        "split": "existing six-author discovery pilot",
        "jobs": parent["jobs"],
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    torch.manual_seed(731)
    torch.set_num_threads(4)
    obs = parent["observer"]
    observer = DecoderObserver.load(
        obs["model"],
        revision=obs["revision"],
        device="cuda",
        dtype="bfloat16",
        max_tokens=obs["max_tokens"],
        pad_to_length=obs["pad_to_length"],
        local_files_only=True,
    )
    candidates = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if any(len(x) != 1 for x in candidates):
        raise ValueError("Candidate tokenization changed")
    yes, no = candidates[0][0], candidates[1][0]
    quantize = lambda x: (
        torch.as_tensor(x).to(torch.bfloat16).float().numpy().astype(float)
    )
    rows, controls, geometry = [], [], []
    q_cache = {}
    start = time.monotonic()
    status = "RUNNING"
    write_json(out / "run_status.json", {"status": status})
    try:
        with threadpool_limits(limits=4):
            for ji, job in enumerate(parent["jobs"]):
                if (
                    time.monotonic() - start > manifest["max_seconds"]
                    or observer.calls + 100 > 4096
                ):
                    status = "PARTIAL_BUDGET"
                    break
                case = by_case[job["case_id"]]
                user, _ = patch_message(
                    render_records(case["history"]),
                    case["current"]["body"],
                    job["task"],
                )
                ids = observer.prompt_ids(SYSTEM, user)
                if material_hash(ids) != job["token_hashes"][0]:
                    raise ValueError("Prompt identity mismatch")
                gi, grad_job = gradient_jobs[job["case_id"], job["task"]]
                ti, target_job = gradient_jobs[job["case_id"], "disposition"]
                subset = original[
                    (original.case_id == job["case_id"])
                    & (original.rule == job["rule"])
                    & (original.task == job["task"])
                ]
                base = float(subset.baseline_margin.iloc[0])
                natural = float(subset.natural_donor_margin.iloc[0])
                _, initial = observer.capture(ids)
                if float(initial[yes] - initial[no]) != base:
                    raise RuntimeError("Original baseline not exactly reproduced")
                meta = {
                    k: job[k]
                    for k in ("case_id", "author_id", "donor_id", "rule", "task")
                }
                own = np.load(a.parent_dir / "private_cache" / f"j{ji:02d}_a0.npz")
                donor = np.load(a.parent_dir / "private_cache" / f"j{ji:02d}_a1.npz")

                def run_arm(
                    site,
                    positions,
                    h,
                    gradient,
                    value,
                    *,
                    family,
                    method,
                    rank=0,
                    alpha=1.0,
                    match_error=None,
                    scale=None,
                    ids=ids,
                    base=base,
                    natural=natural,
                    meta=meta,
                ):
                    actual = quantize(value)
                    _, logits = observer.capture(
                        ids, patch=Patch(site, positions, value)
                    )
                    score = float(logits[yes] - logits[no])
                    predicted = local_prediction(gradient, h, actual)
                    delta = score - base
                    rows.append(
                        meta
                        | {
                            "site": site,
                            "family": family,
                            "method": method,
                            "rank": rank,
                            "alpha": alpha,
                            "baseline_margin": base,
                            "natural_donor_margin": natural,
                            "patched_margin": score,
                            "delta": delta,
                            "oriented_delta": delta * np.sign(natural - base),
                            "predicted_delta": predicted,
                            "linear_error": delta - predicted,
                            "effect_norm": float(np.linalg.norm(actual - h)),
                            "relative_effect_norm": float(
                                np.linalg.norm(actual - h)
                                / max(np.linalg.norm(h), 1e-12)
                            ),
                            "norm_match_max_relative_error": match_error,
                            "max_scale": scale,
                            "decision_flip": int((score > 0) != (base > 0)),
                        }
                    )
                    return score

                for site in parent["sites"]:
                    full = np.load(
                        a.gradient_dir / "private_cache" / f"j{gi:02d}_{site}.npz"
                    )
                    target = np.load(
                        a.gradient_dir / "private_cache" / f"j{ti:02d}_{site}.npz"
                    )
                    positions = grad_job["regions"]["current"]
                    h, g = (
                        full["state"][positions].astype(float),
                        full["gradient"][positions].astype(float),
                    )
                    h_other = donor[site][:-1].astype(float)
                    if not np.array_equal(h, own[site][:-1]):
                        raise RuntimeError("Current state no longer aligned")
                    delta = h_other - h
                    qg, _ = current_difference_basis(delta, 4)
                    key = (job["case_id"], site)
                    if key not in q_cache:
                        tg = target["gradient"][
                            target_job["regions"]["current"]
                        ].astype(float)
                        qt, energy = current_difference_basis(tg, 4)
                        q_cache[key] = qt, energy
                    qt, energy = q_cache[key]
                    geometry.append(
                        meta
                        | {
                            "site": site,
                            "target_gradient_rank4_energy": energy,
                            "rank2_subspace_overlap": float(
                                np.linalg.norm(qg[:, :2].T @ qt[:, :2]) ** 2 / 2
                            ),
                            "rank4_subspace_overlap": float(
                                np.linalg.norm(qg.T @ qt) ** 2 / 4
                            ),
                        }
                    )
                    full_score = run_arm(
                        site,
                        positions,
                        h,
                        g,
                        h_other,
                        family="current_subspace",
                        method="full",
                    )
                    expected_full = subset[
                        (subset.site == site)
                        & (subset.span == "current_token_span")
                        & (subset.arm == "full")
                    ]
                    if full_score != float(expected_full.patched_margin.iloc[0]):
                        raise RuntimeError("Full current patch replay failed")
                    self_score = run_arm(
                        site,
                        positions,
                        h,
                        g,
                        h,
                        family="current_subspace",
                        method="self",
                    )
                    if self_score != base:
                        raise RuntimeError("Self patch changed margin")
                    for rank in (2, 4):
                        geom, sens = (
                            project_rows(delta, qg[:, :rank]),
                            project_rows(delta, qt[:, :rank]),
                        )
                        for method, effect in (
                            ("geometry", geom),
                            ("sensitivity_natural", sens),
                        ):
                            for alpha in (0.25, 0.5, 1.0):
                                value = h + alpha * effect
                                result = run_arm(
                                    site,
                                    positions,
                                    h,
                                    g,
                                    value,
                                    family="current_subspace",
                                    method=method,
                                    rank=rank,
                                    alpha=alpha,
                                )
                                if method == "geometry":
                                    expected = subset[
                                        (subset.site == site)
                                        & (subset.span == "current_token_span")
                                        & (subset.arm == "candidate")
                                        & (subset["rank"] == rank)
                                        & (subset.alpha == alpha)
                                    ]
                                    if result != float(expected.patched_margin.iloc[0]):
                                        raise RuntimeError(
                                            "Geometry candidate replay failed"
                                        )
                        matched, scales, errors = match_quantized_row_norms(
                            h, sens, h + geom, quantize
                        )
                        run_arm(
                            site,
                            positions,
                            h,
                            g,
                            matched,
                            family="current_subspace",
                            method="sensitivity_matched",
                            rank=rank,
                            match_error=float(errors.max()),
                            scale=float(scales.max()),
                        )
                        for mode, target_value in (
                            ("natural", h + sens),
                            ("matched", matched),
                        ):
                            for rep in range(2):
                                direction = matched_random_rows(
                                    delta,
                                    target_value - h,
                                    rank,
                                    [
                                        job["case_id"],
                                        job["rule"],
                                        site,
                                        rank,
                                        rep,
                                        "sensitivity",
                                    ],
                                )
                                value, scales, errors = match_quantized_row_norms(
                                    h, direction, target_value, quantize
                                )
                                run_arm(
                                    site,
                                    positions,
                                    h,
                                    g,
                                    value,
                                    family="current_subspace",
                                    method=f"random_{mode}_{rep}",
                                    rank=rank,
                                    match_error=float(errors.max()),
                                    scale=float(scales.max()),
                                )

                    # Same regional contrast is evaluated once, not duplicated by donor rule.
                    if job["rule"] == "other_condition":
                        reference_norm = float(np.linalg.norm(h))
                        for region in ("history", "current"):
                            rp = grad_job["regions"][region]
                            tp = target_job["regions"][region]
                            if rp != tp:
                                raise RuntimeError(
                                    "Pre-question regional alignment failed"
                                )
                            hr = full["state"][rp].astype(float)
                            gr = full["gradient"][rp].astype(float)
                            target_gradient = target["gradient"][tp].astype(float)
                            direction = unit_matrix_direction(target_gradient)
                            for epsilon in (0.002, 0.01):
                                for sign in (-1, 1):
                                    value = (
                                        hr + sign * epsilon * reference_norm * direction
                                    )
                                    run_arm(
                                        site,
                                        rp,
                                        hr,
                                        gr,
                                        value,
                                        family=region + "_access",
                                        method="target_gradient",
                                        alpha=sign * epsilon,
                                    )
                            target_value = hr + 0.01 * reference_norm * direction
                            for rep in range(2):
                                seed = int(
                                    material_hash(
                                        [job["case_id"], site, region, rep, "access"]
                                    )[:16],
                                    16,
                                )
                                random = unit_matrix_direction(
                                    np.random.default_rng(seed).normal(size=hr.shape)
                                )
                                random *= 0.01 * reference_norm
                                value, scales, errors = match_quantized_row_norms(
                                    hr.reshape(1, -1),
                                    random.reshape(1, -1),
                                    target_value.reshape(1, -1),
                                    quantize,
                                    steps=24,
                                )
                                run_arm(
                                    site,
                                    rp,
                                    hr,
                                    gr,
                                    value.reshape(hr.shape),
                                    family=region + "_access",
                                    method=f"random_{rep}",
                                    alpha=0.01,
                                    match_error=float(errors.max()),
                                    scale=float(scales.max()),
                                )
                _, restored = observer.capture(ids)
                restore = abs(float(restored[yes] - restored[no]) - base)
                if restore != 0:
                    raise RuntimeError("Restoration failed")
                controls.append(meta | {"restoration_error": restore})
                for name, data in (
                    ("interventions", rows),
                    ("geometry", geometry),
                    ("controls", controls),
                ):
                    pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
                write_json(
                    out / "run_status.json",
                    {"status": "RUNNING", "jobs": ji + 1, "calls": observer.calls},
                )
                print(
                    f"SENSITIVITY_PATCH {ji + 1}/48 calls={observer.calls}", flush=True
                )
            else:
                status = "COMPLETE"
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_CONTROL_OR_RUNTIME",
                "calls": observer.calls,
                "completed_jobs": len(controls),
                "error": type(exc).__name__ + ": " + str(exc),
            },
        )
        raise
    write_json(
        out / "run_status.json",
        {
            "status": status,
            "jobs": len(controls),
            "rows": len(rows),
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "input_tokens": observer.input_tokens,
            "processed_tokens": observer.processed_tokens,
            "restore_max_error": max(
                (r["restoration_error"] for r in controls), default=0
            ),
            "self_max_error": max(
                (abs(r["delta"]) for r in rows if r["method"] == "self"), default=0
            ),
            "replayed_geometry_and_full": True,
            "psychological_labels_used": False,
        },
    )
    print(json.dumps(read(out / "run_status.json")), flush=True)


if __name__ == "__main__":
    main()
