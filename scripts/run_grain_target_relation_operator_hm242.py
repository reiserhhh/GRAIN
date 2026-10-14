#!/usr/bin/env python3
"""Finite complete-vector target-relation construction and literal verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_relation_operator as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_target_relation_operator_hm242.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_relation_operator_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_RELATION_OPERATOR_PLAN_HM242.md",
    "grain_research/construction/math_target_relation_operator.py",
    "scripts/run_grain_target_relation_operator_hm242.py",
    "tests/test_grain_target_relation_operator.py",
    "scripts/run_grain_observed_family_hm170.py",
)
STAGES = ("computed_seal.json", "verification_seal.json")


def selected(path, keys):
    """Read only authorized numeric arrays, not other ancestor NPZ fields."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def prepare(out):
    """Authenticate identical source authors and addressed application evidence."""
    cfg = R.read(ROOT / CONFIG)
    source, panel = ROOT / cfg["source_parent"], ROOT / cfg["panel_parent"]
    R.check(source, ("computed_seal.json", "verification_seal.json"))
    R.check(panel, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    keys = ("means", "codes", "folds", "source_indices")
    z = selected(source / "observations.npz", keys)
    old = selected(
        panel / "observations.npz",
        (
            "source_means",
            "codes",
            "folds",
            "source_indices",
            "panel_points",
            "author_ids",
        ),
    )
    if (
        z["means"].shape != (386, 2, 1024)
        or not np.array_equal(z["means"], old["source_means"])
        or any(not np.array_equal(z[k], old[k]) for k in keys[1:])
        or not np.array_equal(np.bincount(z["folds"]), [203, 183])
        or old["panel_points"].shape != (24, 9, 2, 1024)
        or len(set(z["source_indices"].tolist())) != 386
    ):
        raise ValueError("The frozen source/panel identity or scope changed")
    example = R.read(panel / "cold_example.json")
    point, addresses = M.cold_point(example, 1024)
    if not np.allclose(point, old["panel_points"][0, 0], rtol=0, atol=1e-14):
        raise ValueError(
            "Addressed cold example differs from the same cached event mean"
        )
    out.mkdir(parents=True, exist_ok=False)
    files = [
        p / name
        for p in (source, panel)
        for name in ("manifest.json", "observations.npz", "prepared_seal.json")
    ] + [
        panel / "cold_example.json",
        source / "computed_seal.json",
        panel / "built_seal.json",
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "previous_turn": "PROGRESS_RESEARCH_DECISION:precision_branch_closed;source_meaning_relation_selected",
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **z,
        panel_points=old["panel_points"],
        author_ids=old["author_ids"],
    )
    R.write(out / "cold_example.json", example)
    quality = {
        "source_authors": 386,
        "source_folds": [203, 183],
        "source_profiles_retained": np.bincount(z["codes"], minlength=32).tolist(),
        "application_authors": 24,
        "dependent_frames": 216,
        "cold_example_event_ids": addresses,
        "source_arrays_match_HM209_exactly": True,
        "source_keys_read": list(keys),
        "panel_keys_read": list(old),
        "raw_private_application_labels_model_answers_new_neural_API_training": 0,
        "upstream_embedding_neural": True,
        "new_estimator_reference": "author-balanced binary covariance;not uniform32profile prior",
    }
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "quality.json", "cold_example.json"),
    )
    print(json.dumps(quality), flush=True)


def load_banks(out):
    """Load the frozen complete-vector source witness, not a fitted predictor."""
    with np.load(out / "source_banks.npz", allow_pickle=False) as z:
        return [
            {k[3:]: z[k] for k in z.files if k.startswith(f"b{b}_")} for b in (0, 1)
        ]


def finite(v):
    """Serialize an unavailable operator cosine as missing rather than zero."""
    return float(v) if np.isfinite(v) else None


def compute(out):
    """Compile both sources and all declared signed operator accounts, once."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.perf_counter()
    banks = [
        M.compile_bank(z["means"][z["folds"] == b], z["codes"][z["folds"] == b])
        for b in (0, 1)
    ]
    grams = np.array(
        [[M.operator_gram(left, right) for right in banks] for left in banks]
    )
    cosines = M.cosine_gram(grams[0, 1], grams[0, 0], grams[1, 1])
    scores = np.array([M.score_points(b, z["panel_points"]) for b in banks])
    source_arrays = {f"b{i}_{k}": v for i, b in enumerate(banks) for k, v in b.items()}
    R.archive(out / "source_banks.npz", **source_arrays)
    audits = []
    for b in banks:
        moved = b["means"].copy()
        moved[:, 1] = moved[M.cyclic_relink(b["codes"]), 1]
        relink = M.compile_bank(moved, b["codes"])
        rr, br, bb = (
            M.operator_gram(relink, relink),
            M.operator_gram(b, relink),
            M.operator_gram(b, b),
        )
        delta = rr + bb - br - br.transpose(2, 3, 0, 1)
        audits.append(delta)
    audit_grams = np.asarray(audits)
    R.archive(
        out / "accounts.npz",
        grams=grams,
        cosines=cosines,
        scores=scores,
        audit_difference_grams=audit_grams,
    )
    rows = []
    for m, mode in enumerate(M.COMPONENTS):
        for q, target in enumerate(M.TARGETS):
            for n, other_mode in enumerate(M.COMPONENTS):
                for r, other_target in enumerate(M.TARGETS):
                    rows.append(
                        {
                            "left_component": mode,
                            "left_target": target,
                            "right_component": other_mode,
                            "right_target": other_target,
                            "cross_source_signed_inner_product": float(
                                grams[0, 1, m, q, n, r]
                            ),
                            "cross_source_cosine": finite(cosines[m, q, n, r]),
                        }
                    )
    R.table(out / "operator_gram.csv", rows)
    cold_rows = [
        {
            "bank": b,
            "author_key": str(z["author_ids"][u]),
            "frame": f,
            "component": mode,
            "target": target,
            "score": float(scores[b, u, f, m, q]),
        }
        for b in (0, 1)
        for u in range(24)
        for f in range(9)
        for m, mode in enumerate(M.COMPONENTS)
        for q, target in enumerate(M.TARGETS)
    ]
    R.table(out / "cold_readings.csv", cold_rows)
    primary = [
        {
            "target": t,
            "signed_cross_product": float(grams[0, 1, 1, q, 1, q]),
            "cosine": finite(cosines[1, q, 1, q]),
            "total_cosine": finite(cosines[0, q, 0, q]),
            "between_cosine": finite(cosines[2, q, 2, q]),
        }
        for q, t in enumerate(M.TARGETS)
        if q
    ]
    cold_agreement = []
    for m, mode in enumerate(M.COMPONENTS):
        for q, target in enumerate(M.TARGETS):
            a, b = scores[:, :, 0, m, q]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cold_agreement.append(
                {
                    "component": mode,
                    "target": target,
                    "full_panel_score_cosine": finite(a @ b / denom) if denom else None,
                }
            )
    elapsed = time.perf_counter() - start
    summary = {
        "status": "COMPUTED_NOT_ACCURACY_PROMOTED",
        "primary_within_profile_targets": primary,
        "positive_within_target_count": sum(
            r["signed_cross_product"] > 0 for r in primary
        ),
        "common_within_source_cosine": finite(cosines[1, 0, 1, 0]),
        "cold_full_panel_agreement": cold_agreement,
        "binary_counts_by_bank": [b["binary_counts"].tolist() for b in banks],
        "singleton_profiles_by_bank": [
            np.flatnonzero(b["profile_counts"] == 1).tolist() for b in banks
        ],
        "audit_between_difference_squared_max": float(
            max(abs(audit_grams[b, 2, q, 2, q]) for b in (0, 1) for q in range(6))
        ),
        "audit_within_difference_squared": [
            [float(audit_grams[b, 1, q, 1, q]) for q in range(6)] for b in (0, 1)
        ],
        "dependent_cold_rows": len(cold_rows),
        "operator_accounts": len(rows),
        "seconds": elapsed,
        "new_neural_API_private_training_application_labels": 0,
        "original_scope_complete": False,
        "interpretation": "target-linked paired-vector functional,not causal target law or personality accuracy",
    }
    R.write(
        out / "cold_output.json",
        M.cold_record(banks, R.read(out / "cold_example.json")),
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "source_banks.npz",
            "accounts.npz",
            "operator_gram.csv",
            "cold_readings.csv",
            "cold_output.json",
            "summary.json",
        ),
    )
    if elapsed > cfg["max_cpu_seconds"]:
        raise RuntimeError("Completed finite computation exceeded registered budget")
    print(json.dumps(summary), flush=True)


def ambient_block(x, codes, start, stop):
    """Independently build covariance blocks using literal centered author rows."""
    dims = x.shape[2]
    result = np.zeros((3, 6, stop - start, dims))
    profile_means = {g: x[codes == g].mean(0) for g in np.unique(codes)}
    profile_sums = {}
    for g, mean in profile_means.items():
        centered = x[codes == g] - mean
        profile_sums[g] = centered[:, 0, start:stop].T @ centered[:, 1]
    groups = [(0, np.arange(len(x)), 1)] + [
        (q + 1, np.flatnonzero(((codes >> q) & 1) == a), 2 * a - 1)
        for q in range(5)
        for a in (0, 1)
    ]
    for q, ix, sign in groups:
        mean, weight = x[ix].mean(0), sign / (len(ix) - 1)
        centered = x[ix] - mean
        result[0, q] += weight * centered[:, 0, start:stop].T @ centered[:, 1]
        for g in np.unique(codes[ix]):
            result[1, q] += weight * profile_sums[g]
            diff = profile_means[g] - mean
            number = np.count_nonzero(codes[ix] == g)
            result[2, q] += weight * number * np.outer(diff[0, start:stop], diff[1])
    return result


def verify(out):
    """Verify actual full-dimensional Grams/cold scores by ambient covariance sums."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    banks, z = load_banks(out), R.prepared(out)
    expected = selected(out / "accounts.npz", ("grams", "scores"))
    start = time.perf_counter()
    grams = np.zeros_like(expected["grams"])
    cold = np.zeros_like(expected["scores"])
    centered = [z["panel_points"] - b["reference"] for b in banks]
    errors = {"component_closure": 0.0, "relink_between": 0.0}
    for lo in range(0, 1024, 128):
        hi = min(1024, lo + 128)
        blocks = [ambient_block(b["means"], b["codes"], lo, hi) for b in banks]
        for b, block in enumerate(blocks):
            errors["component_closure"] = max(
                errors["component_closure"],
                float(abs(block[0] - block[1] - block[2]).max()),
            )
            moved = banks[b]["means"].copy()
            moved[:, 1] = moved[M.cyclic_relink(banks[b]["codes"]), 1]
            relinked = ambient_block(moved, banks[b]["codes"], lo, hi)
            errors["relink_between"] = max(
                errors["relink_between"], float(abs(block[2] - relinked[2]).max())
            )
            flat = centered[b].reshape(216, 2, 1024)
            literal = np.einsum(
                "ai,mqij,aj->amq", flat[:, 0, lo:hi], block, flat[:, 1], optimize=True
            )
            cold[b] += literal.reshape(24, 9, 3, 6)
        for b in (0, 1):
            for c in (0, 1):
                grams[b, c] += np.einsum(
                    "mqij,nrij->mqnr", blocks[b], blocks[c], optimize=True
                )
    errors["ambient_all_operator_grams"] = float(abs(grams - expected["grams"]).max())
    errors["literal_all_cold_scores"] = float(abs(cold - expected["scores"]).max())
    replay = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "cold",
                "--output",
                str(out),
            ],
            text=True,
        )
    )
    old = R.read(out / "cold_output.json")
    errors["independent_process_cold"] = float(
        abs(np.asarray(replay["scores"]) - np.asarray(old["scores"])).max()
    )
    if (
        replay["event_ids"] != old["event_ids"]
        or replay["frame_key"] != old["frame_key"]
    ):
        raise ValueError("Cold process changed evidence identity")
    result = {
        "status": "PASS"
        if max(errors.values()) <= cfg["comparison_tolerance"]
        else "FAIL",
        "errors": errors,
        "operator_gram_accounts": int(grams.size),
        "cold_accounts": int(cold.size),
        "full_ambient_dimensions": 1024,
        "seconds": time.perf_counter() - start,
        "audit": "same-agent alternate calculation;not independent blind replication",
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS":
        raise ValueError("Literal full-vector verification failed;retain failure")


def main():
    """Run exactly one immutable stage or a read-only cold/check command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "verify", "check", "cold")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.stage in ("prepare", "compute", "verify"):
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](
            args.output
        )
    elif args.stage == "check":
        R.check(args.output, STAGES)
        print(
            json.dumps(
                {"status": "PASS", "packet": "HM242", "original_scope_complete": False}
            )
        )
    else:
        R.check(args.output, ("computed_seal.json",))
        record = R.read(args.record or args.output / "cold_example.json")
        print(json.dumps(M.cold_record(load_banks(args.output), record)), flush=True)


if __name__ == "__main__":
    main()
