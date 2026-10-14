#!/usr/bin/env python3
"""One conditional evidence-routing experiment on sealed HM209 observations."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_evidence_routing as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_relational_diagnostic_update_hm209 as P

CONFIG = "configs/grain_target_evidence_routing_hm210.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_evidence_routing_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_EVIDENCE_ROUTING_PLAN_HM210.md",
    "grain_research/construction/math_target_evidence_routing.py",
    "scripts/run_grain_target_evidence_routing_hm210.py",
    "tests/test_grain_target_evidence_routing.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_relational_diagnostic_update_hm209.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Bind unchanged source/held/cold evidence and old science before analysis."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    obs = R.prepared(parent)
    with np.load(parent / "analysis.npz", allow_pickle=False) as z:
        data = {
            k: z[k]
            for k in z.files
            if k.endswith(("log_profile", "log_factor", "targets"))
        }
    for b in (0, 1):
        ix = obs["folds"] != b
        data[f"held{b}_codes"] = obs["codes"][ix]
        data[f"held{b}_indices"] = obs["source_indices"][ix]
    data["author_ids"] = obs["author_ids"]
    if (len(data["held0_codes"]), len(data["held1_codes"])) != (183, 203):
        raise ValueError("Changed source folds;do not create a new split")
    if any(data[f"panel{b}_log_profile"].shape != (216, 2, 32) for b in (0, 1)):
        raise ValueError("Changed complete application panel")
    files = (
        "observations.npz",
        "analysis.npz",
        "source_banks.npz",
        "cold_example.json",
        "summary.json",
        "manifest.json",
        "verified_seal.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in files
            },
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **data)
    R.write(out / "cold_example.json", R.read(parent / "cold_example.json"))
    quality = {
        "source_authors": 386,
        "source_folds": [203, 183],
        "application_authors": 24,
        "frames_per_bank": 216,
        "conditional_cells_per_frame": 80,
        "no_application_labels_or_native_answers": True,
        "scale_prior_readings_unchanged": True,
        "original_scope_complete": False,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_example.json", "input_quality.json"),
    )
    print(json.dumps(quality))


def compute(out):
    """Explain all signed target changes without selecting a favorable account."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.perf_counter()
    arrays, held_rows, panel_rows, summaries = {}, [], [], []
    for b in (0, 1):
        for kind in ("held", "panel"):
            prefix = f"{kind}{b}"
            a = M.analyze(z[f"{prefix}_log_profile"], z[f"{prefix}_log_factor"])
            arrays.update({f"{prefix}_{k}": v for k, v in a.items()})
            n = len(a["corners"])
            if kind == "held":
                codes = z[f"{prefix}_codes"]
                score = M.scored_accounts(a["corners"], codes)
                arrays[f"{prefix}_log_score_accounts"] = score
                counts = np.bincount(codes, minlength=32)
                balanced = 1 / (32 * counts[codes])
                summaries.append(
                    {
                        "source_bank": b,
                        "held_authors": n,
                        "mean_log_score_accounts": score.mean(0).tolist(),
                        "profile_balanced_log_score_accounts": np.einsum(
                            "n,nqa->qa", balanced, score
                        ).tolist(),
                        "probability_account_RMS": np.sqrt(
                            np.mean(a["probability_accounts"] ** 2, 0)
                        ).tolist(),
                        "conditional_log_odds_RMS": np.sqrt(
                            np.mean(a["conditional_log_odds_update"] ** 2, axis=(0, 2))
                        ).tolist(),
                        "positive_conditional_edges_NOT_independent_people": (
                            a["conditional_log_odds_update"] > 0
                        )
                        .sum(axis=(0, 2))
                        .tolist(),
                        "conditional_edges_per_target": n * 16,
                    }
                )
                for u, q in np.ndindex(n, 5):
                    row = {
                        "source_bank": b,
                        "held_source_author": int(z[f"{prefix}_indices"][u]),
                        "target": M.P.TARGETS[q],
                    }
                    row.update(
                        {
                            f"log_score_{name}": float(score[u, q, j])
                            for j, name in enumerate(M.ACCOUNTS)
                        }
                    )
                    row["full_log_score_change"] = float(score[u, q].sum())
                    held_rows.append(row)
            else:
                for u, q in np.ndindex(n, 5):
                    row = {
                        "source_bank": b,
                        "application_author": str(z["author_ids"][u // 9]),
                        "frame": P.FRAME_NAMES[u % 9],
                        "target": M.P.TARGETS[q],
                    }
                    row.update(
                        {
                            f"probability_{name}": float(
                                a["probability_accounts"][u, q, j]
                            )
                            for j, name in enumerate(M.ACCOUNTS)
                        }
                    )
                    row["full_probability_change"] = float(
                        a["probability_accounts"][u, q].sum()
                    )
                    panel_rows.append(row)
        if time.perf_counter() - start > cfg["max_stage_seconds"]:
            R.write(
                out / "compute_failure.json", {"finite_cap_reached": True, "bank": b}
            )
            raise RuntimeError("Finite cap reached;no rerun or cap extension")
    x, y = [
        arrays[f"panel{b}_conditional_log_odds_update"].reshape(24, 9, 5, 16)[:, 0]
        for b in (0, 1)
    ]
    summary = {
        "targets": list(M.P.TARGETS),
        "account_order": list(M.ACCOUNTS),
        "banks": summaries,
        "cold_full_frame_conditional_edge_signed_products": np.mean(
            x * y, axis=(0, 2)
        ).tolist(),
        "cold_full_frame_conditional_edge_same_sign_counts": (x * y > 0)
        .sum(axis=(0, 2))
        .tolist(),
        "cold_full_frame_cells_per_target": 24 * 16,
        "new_capability": "conditional_target_atlas_and_exact_target_update_routing_on_SAME_vectors",
        "maturity": "finite_disintegration_and_loglinear_conditioning_known_not_new_information_geometry_theorem",
        "qualification": "diagnostic_cross_readings_not_causal_mediation_or_independent_personality_paths",
        "original_scope_complete": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.table(out / "held_target_accounts.csv", held_rows)
    R.table(out / "cold_target_accounts.csv", panel_rows)
    R.archive(out / "analysis.npz", **arrays)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "held_target_accounts.csv",
            "cold_target_accounts.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One scalar/table audit,not a precision optimization or new science run."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z, start = R.prepared(out), time.perf_counter()
    with np.load(out / "analysis.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    errors, checks = [], 0
    parent = R.read(ROOT / cfg["parent"] / "summary.json")
    for b in (0, 1):
        for kind in ("held", "panel"):
            prefix = f"{kind}{b}"
            corners, accounts = (
                a[f"{prefix}_corners"],
                a[f"{prefix}_probability_accounts"],
            )
            targets = z[f"{prefix}_targets"]
            errors.extend(
                (
                    np.max(np.abs(corners[..., 0] - targets[:, 1])),
                    np.max(np.abs(corners[..., 3] - targets[:, 0])),
                    np.max(np.abs(accounts.sum(-1) - targets[:, 0] + targets[:, 1])),
                )
            )
            for u, q in np.ndindex(min(8, len(corners)), 5):
                literal = M.literal(z[f"{prefix}_log_profile"][u], q)
                errors.append(np.max(np.abs(literal - corners[u, q])))
                low, high = M.edges(q)
                for c, (lo, hi) in enumerate(zip(low, high, strict=True)):
                    lp = z[f"{prefix}_log_profile"][u]
                    delta = lp[0, hi] - lp[0, lo] - lp[1, hi] + lp[1, lo]
                    errors.append(
                        abs(delta - a[f"{prefix}_conditional_log_odds_update"][u, q, c])
                    )
                    checks += 1
                checks += 4
        score = a[f"held{b}_log_score_accounts"]
        errors.append(
            np.max(
                np.abs(
                    score.sum(-1).mean(0)
                    - parent["banks"][b]["target_log_score_mean_change"]
                )
            )
        )
        codes = z[f"held{b}_codes"]
        w = 1 / (32 * np.bincount(codes, minlength=32)[codes])
        errors.append(
            np.max(
                np.abs(
                    w @ score.sum(-1)
                    - parent["banks"][b]["target_log_score_profile_balanced_change"]
                )
            )
        )
    error = float(max(errors))
    if not np.isfinite(error) or error > cfg["comparison_tolerance"]:
        R.write(out / "verification_failure.json", {"error": error, "no_retry": True})
        raise ValueError("Diagnostic account audit failed;do not tune tolerance")
    receipt = {
        "PASS": True,
        "alternate_max_error": error,
        "literal_corner_and_conditional_odds_values": checks,
        "all_old_probability_and_log_score_updates_retained": True,
        "same_agent_not_independent": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verified_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def cold(out):
    """Read actual vectors in a separate CLI and retain all80 conditional cells."""
    cfg = R.check(out, ("computed_seal.json", "verified_seal.json"))["config"]
    r = M.cold(P.banks(ROOT / cfg["parent"])[0], R.read(out / "cold_example.json"))
    R.write(out / "cold_readback.json", r)
    R.seal(out, "cold_seal.json", ("cold_readback.json",))
    print(
        json.dumps(
            {k: v for k, v in r.items() if k != "conditional_profiles"}
            | {"conditional_cells": len(r["conditional_profiles"])}
        )
    )


def main():
    """Run an explicit finite stage once;existing outputs cannot be replaced."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "cold"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify, "cold": cold}[
        args.stage
    ](args.out)


if __name__ == "__main__":
    main()
