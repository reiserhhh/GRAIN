#!/usr/bin/env python3
"""One finite relation-assignment mechanism, not an accuracy search."""

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
from grain_research.construction import math_cross_window as W
from grain_research.construction import math_joint_meaning as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_joint_meaning_hm177.json"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_relation_meaning_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_JOINT_MEANING_PLAN_HM177.md",
    "grain_research/construction/math_joint_meaning.py",
    "scripts/run_grain_joint_meaning_hm177.py",
    "scripts/score_grain_joint_meaning_cold.py",
    "scripts/verify_grain_joint_meaning_hm177.py",
    "tests/test_grain_joint_meaning.py",
)


def prepare(out):
    """Freeze method and parent identities, then build SOURCE references once."""
    start = time.perf_counter()
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["parent"]
    manifest = R.check(parent, ("scored_seal.json", "analysis_seal.json"))
    W.validate_bundle(R.read(parent / "source_bundle.json"))
    inputs = dict(manifest["inputs"])
    for stage in (
        "manifest.json",
        "prepared_seal.json",
        "scored_seal.json",
        "analysis_seal.json",
    ):
        files = (
            (stage,) if stage == "manifest.json" else (stage, *R.read(parent / stage))
        )
        for name in files:
            p = str((parent / name).relative_to(ROOT))
            inputs[p] = R.digest(ROOT / p)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": manifest["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
            "cohort_reuse": "Historical source and held cohorts;not an untouched lockbox",
        },
    )
    z = R.prepared(parent)
    if (
        len(z["source_mass"]) != spec["source_authors"]
        or len(z["held_cells"]) != spec["application_authors"]
    ):
        raise ValueError("Changed author scope")
    bundle = M.build_bundle(
        z["source_mass"],
        z["source_targets"],
        spec["ipf_tolerance"],
        spec["ipf_max_sweeps"],
    )
    R.write(out / "source_bundle.json", bundle)
    R.archive(out / "observations.npz", **z)
    with (
        (parent / "cold_input.jsonl").open() as source,
        (out / "cold_input.jsonl").open("x") as sink,
    ):
        for line in source:
            sink.write(line)
    receipt = {
        "source_authors": len(z["source_mass"]),
        "profiles": len(bundle["profiles"]),
        "application_authors": len(z["held_cells"]),
        "events_per_window": 17,
        "solver": bundle["solver_receipt"],
        "new_neural_calls": 0,
        "application_labels_read": False,
        "seconds": time.perf_counter() - start,
    }
    if receipt["profiles"] != spec["profiles"]:
        raise ValueError("Full profile scope changed")
    R.write(out / "input_quality.json", receipt)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "source_bundle.json",
            "observations.npz",
            "cold_input.jsonl",
            "input_quality.json",
        ),
    )
    print(json.dumps(receipt))


def score(out):
    """Run the standalone cold CLI with only IDs and numerical observations."""
    R.check(out)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_joint_meaning_cold.py"),
            "--bundle",
            str(out / "source_bundle.json"),
            "--input",
            str(out / "cold_input.jsonl"),
            "--output",
            str(out / "cold_output.jsonl"),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    z = R.prepared(out)
    if [r["author_key"] for r in rows] != [f"held/{u}" for u in z["held"]] or any(
        r["events_per_window"] != [17, 17] for r in rows
    ):
        raise ValueError("Cold observation identity changed")
    R.archive(
        out / "cold_accounts.npz",
        readout=np.asarray([r["pair_target_readout"] for r in rows], float),
        supported=np.asarray([r["supported"] for r in rows], bool),
    )
    receipt = {
        "rows": len(rows),
        "seconds": time.perf_counter() - start,
        "application_labels_read": False,
        "new_model_calls": 0,
    }
    R.write(out / "cold_receipt.json", receipt)
    R.seal(
        out,
        "scored_seal.json",
        ("cold_output.jsonl", "cold_accounts.npz", "cold_receipt.json"),
    )
    print(json.dumps(receipt))


def rms(x):
    """Report magnitude in its actual field units, never prediction accuracy."""
    return float(np.sqrt(np.mean(np.square(x))))


def analyze(out):
    """Retain shared,profile-specific and signed multi-target relation effects."""
    start = time.perf_counter()
    spec = R.check(out, ("scored_seal.json",))["config"]
    b = M.validate_bundle(R.read(out / "source_bundle.json"))
    z = R.prepared(out)
    profiles, tables, laws = (
        np.asarray(b[k]) for k in ("profiles", "profile_tables", "joint_laws")
    )
    psi = np.asarray(b["pair_readout"], float)
    pair_weight = tables[2].sum(0)
    specific, shared, total = psi[2] - psi[1], psi[1] - psi[0], psi[2] - psi[0]
    delta = laws[2] - laws[1]
    den = laws[2].sum(1)
    algebra = np.divide(
        delta[:, 1] - psi[1] * delta.sum(1),
        den,
        out=np.full_like(den, np.nan),
        where=den > 0,
    )
    known = pair_weight > 0
    gram = (specific[:, known] * pair_weight[known]) @ specific[:, known].T
    norms = np.sqrt(np.diag(gram))
    cosine = np.divide(
        gram,
        np.outer(norms, norms),
        out=np.full_like(gram, np.nan),
        where=np.outer(norms, norms) > 0,
    )
    observed, null, tail, solver = M.matched_reference(
        z["source_mass"],
        z["source_targets"],
        spec["permutations"],
        spec["permutation_seed"],
        spec["ipf_tolerance"],
        spec["ipf_max_sweeps"],
    )
    margin_errors = []
    for z0 in range(2):
        for z1 in range(2):
            s0, s1 = slice(4 * z0, 4 * z0 + 4), slice(4 * z1, 4 * z1 + 4)
            margin_errors += [
                float(np.max(np.abs(a - v)))
                for a, v in zip(
                    M.margins(tables[1, :, s0, s1]),
                    M.margins(tables[2, :, s0, s1]),
                    strict=True,
                )
            ]
    R.table(
        out / "profile_law_accounts.csv",
        [
            {
                "model": M.MODELS[k],
                "profile": c,
                "profile_bits_ENACO": "".join(map(str, profiles[c])),
                "early_cell": a,
                "late_cell": v,
                "mass": float(tables[k, c, a, v]),
            }
            for k, c, a, v in np.ndindex(tables.shape)
        ],
    )
    R.table(
        out / "target_law_accounts.csv",
        [
            {
                "model": M.MODELS[k],
                "target": spec["targets"][q],
                "group": g,
                "early_cell": a,
                "late_cell": v,
                "mass": float(laws[k, q, g, a, v]),
            }
            for k, q, g, a, v in np.ndindex(laws.shape)
        ],
    )
    R.table(
        out / "pair_field_accounts.csv",
        [
            {
                "model": M.MODELS[k],
                "target": spec["targets"][q],
                "early_cell": a,
                "late_cell": v,
                "supported": bool(np.isfinite(psi[k, q, a, v])),
                "readout": float(psi[k, q, a, v])
                if np.isfinite(psi[k, q, a, v])
                else None,
            }
            for k, q, a, v in np.ndindex(psi.shape)
        ],
    )
    R.table(
        out / "cross_target_gram.csv",
        [
            {
                "target_1": spec["targets"][q],
                "target_2": spec["targets"][v],
                "gram": float(gram[q, v]),
                "cosine": float(cosine[q, v]) if np.isfinite(cosine[q, v]) else None,
            }
            for q, v in np.ndindex(gram.shape)
        ],
    )
    names = ["assignment_KL", *spec["targets"]]
    R.table(
        out / "matched_reference.csv",
        [
            {
                "statistic": name,
                "observed": float(observed[i]),
                "null_mean": float(null[:, i].mean()),
                "null_min": float(null[:, i].min()),
                "null_max": float(null[:, i].max()),
                "tail_rank": float(tail[i]),
            }
            for i, name in enumerate(names)
        ],
    )
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        cold, supported = a["readout"], a["supported"]
    cold_rows, differences = [], {}
    for q, name in enumerate(spec["targets"]):
        ok = supported[:, :, q].all(1)
        differences[name] = {
            "supported_authors": int(ok.sum()),
            "common_minus_independence_rms": rms(cold[ok, 1, q] - cold[ok, 0, q])
            if ok.any()
            else None,
            "full_minus_common_rms": rms(cold[ok, 2, q] - cold[ok, 1, q])
            if ok.any()
            else None,
            "full_minus_common_mean": float(np.mean(cold[ok, 2, q] - cold[ok, 1, q]))
            if ok.any()
            else None,
        }
    for u, k, q in np.ndindex(cold.shape):
        cold_rows.append(
            {
                "author_key": f"held/{z['held'][u]}",
                "model": M.MODELS[k],
                "target": spec["targets"][q],
                "supported": bool(supported[u, k, q]),
                "readout": float(cold[u, k, q]) if supported[u, k, q] else None,
            }
        )
    R.table(out / "cold_readout_accounts.csv", cold_rows)
    R.archive(
        out / "mechanism_accounts.npz",
        profile_tables=tables,
        joint_laws=laws,
        pair_fields=psi,
        shared=shared,
        specific=specific,
        total=total,
        gram=gram,
        cosine=cosine,
        matched_observed=observed,
        matched_null=null,
        matched_tail=tail,
    )
    summary = {
        "status": "FINITE_RELATION_ASSIGNMENT_COMPLETE_NOT_SCOPE_COMPLETE",
        "source_authors": len(z["source_mass"]),
        "profiles": len(profiles),
        "source_pairs": len(z["source_mass"]) * 17 * 17,
        "models": list(M.MODELS),
        "maximum_preserved_margin_error": max(margin_errors),
        "target_pooled_pair_error": float(np.max(np.abs(delta.sum(1)))),
        "exact_target_delta_account_error": float(
            np.nanmax(np.abs(algebra - specific))
        ),
        "total_account_error": float(np.nanmax(np.abs(total - shared - specific))),
        "matched_statistics": dict(zip(names, observed.tolist(), strict=True)),
        "matched_tail_ranks": dict(zip(names, tail.tolist(), strict=True)),
        "solver": solver,
        "specific_positive_cells": [
            (specific[q, known] > 0).sum().item() for q in range(5)
        ],
        "specific_negative_cells": [
            (specific[q, known] < 0).sum().item() for q in range(5)
        ],
        "cross_target_cosine": cosine.tolist(),
        "cold_readout_differences": differences,
        "cold_supported_values": int(supported.sum()),
        "cold_values": int(supported.size),
        "application_labels_read": False,
        "accuracy_optimization": False,
        "psychological_cause_identified": False,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "profile_law_accounts.csv",
            "target_law_accounts.csv",
            "pair_field_accounts.csv",
            "cross_target_gram.csv",
            "matched_reference.csv",
            "cold_readout_accounts.csv",
            "mechanism_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Execute one explicit stage without an automatic research queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
