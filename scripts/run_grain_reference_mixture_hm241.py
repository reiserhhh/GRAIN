#!/usr/bin/env python3
"""Freeze and audit SOURCE-mixture comparisons on existing, identical evidence."""

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_reference_mixture as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_relational_diagnostic_update_hm209 import FRAME_NAMES

CONFIG = "configs/grain_reference_mixture_hm241.json"
DEFAULT = ROOT / "results/grain_math_response_v1/reference_mixture_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_REFERENCE_MIXTURE_PLAN_HM241.md",
    "docs/GRAIN_REFERENCE_MIXTURE_PREFLIGHT_HM241.md",
    "grain_research/construction/math_reference_mixture.py",
    "scripts/run_grain_reference_mixture_hm241.py",
    "tests/test_grain_reference_mixture.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_relational_diagnostic_update_hm209.py",
)
STAGES = ("prepared_seal.json", "computed_seal.json", "verification_seal.json")


def check(out, stages=STAGES):
    """Check frozen code, authenticated parent input and results without reruns."""
    return R.check(out, stages)


def prepare(out):
    """Whitelist common-unit source prototypes, existing points and old endpoints."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    table = pd.read_csv(
        parent / "cold_readings.csv",
        dtype=str,
        usecols=["source_bank", "application_author", "frame"],
    )
    addr = table[table.source_bank == "0"].drop_duplicates()
    ids = list(addr.application_author.unique())
    expected = [(uid, frame) for uid in ids for frame in FRAME_NAMES]
    if len(ids) != cfg["authors"] or len(expected) != 216:
        raise ValueError("Complete fixed24-author scope changed")
    for b in (0, 1):
        a = table[table.source_bank == str(b)].drop_duplicates()
        if list(zip(a.application_author, a.frame, strict=True)) != expected:
            raise ValueError("Source likelihood addresses changed")
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        points = z["panel_points"]
    if points.shape != (24, 9, 2, 1024):
        raise ValueError("Changed registered points/dimensions")
    with np.load(parent / "source_banks.npz", allow_pickle=False) as z:
        banks = [
            {k: z[f"b{b}_{k}"] for k in ("prototypes", "codes", "variance")}
            for b in (0, 1)
        ]
    if banks[0]["variance"] != banks[1]["variance"]:
        raise ValueError(
            "Different Gaussian units require an explicit normalization contract"
        )
    with np.load(parent / "analysis.npz", allow_pickle=False) as z:
        old = np.stack([z[f"panel{b}_log_profile"] for b in (0, 1)], axis=2)
    if old.shape != (216, 2, 2, 32):
        raise ValueError("Changed original32-profile/law/source endpoints")
    out.mkdir(parents=True, exist_ok=False)
    inputs = (
        "observations.npz",
        "source_banks.npz",
        "analysis.npz",
        "cold_readings.csv",
        "manifest.json",
        "prepared_seal.json",
        "built_seal.json",
        "analyzed_seal.json",
        "verified_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in inputs
            },
            "whitelist": [
                "panel_points",
                "b0/b1_prototypes/codes/variance",
                "panel0/panel1_log_profile",
                "opaque source/frame/author CSV addresses",
            ],
            "source_contract": "HM209/common Gaussian variance+dimension/uniform32 prior",
            "query_only_density_factors": "common to both sources for each query;Gaussian normalizing constant omitted",
            "new_neural_API_private_training_application_labels": 0,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "inputs.npz",
        points=points,
        old_log_profile=old,
        **{f"b{b}_{k}": v for b, bank in enumerate(banks) for k, v in bank.items()},
    )
    R.write(out / "addresses.json", {"authors": ids, "frames": list(FRAME_NAMES)})
    R.seal(out, STAGES[0], ("inputs.npz", "addresses.json"))
    print(
        json.dumps(
            {
                "authors": 24,
                "source_counts": [len(b["codes"]) for b in banks],
                "comparisons": 24840,
                "no_new_model_calls": True,
            }
        )
    )


def load(out):
    """Read only this packet's explicitly copied numeric input."""
    with np.load(out / "inputs.npz", allow_pickle=False) as z:
        points, old = z["points"], z["old_log_profile"]
        banks = [
            {k: z[f"b{b}_{k}"] for k in ("prototypes", "codes", "variance")}
            for b in (0, 1)
        ]
    return points, old, banks


def compute(out):
    """Evaluate all frozen comparisons analytically, without a lambda grid."""
    cfg = check(out, (STAGES[0],))["config"]
    if (out / "likelihoods.npz").exists():
        raise ValueError("Exclusive scientific calculation, no overwrite or repeat")
    points, old, banks = load(out)
    ids = R.read(out / "addresses.json")["authors"]
    start = time.perf_counter()
    logs = np.stack(
        [M.source_log_mass(b, points.reshape(-1, 2, 1024)) for b in banks], axis=2
    )
    replay = float(np.max(np.abs(logs - logsumexp(logs, axis=-1, keepdims=True) - old)))
    if replay > cfg["arithmetic_tolerance"]:
        R.write(
            out / "compute_failure.json",
            {"endpoint_replay_error": replay, "no_rerun": True},
        )
        raise ValueError("Likelihood units do not replay old scientific endpoints")
    logs = logs.reshape(24, 9, 2, 2, 32)
    R.archive(out / "likelihoods.npz", log_joint_mass=logs)
    records, comparisons, partials = [], [], []
    for frame, name in enumerate(FRAME_NAMES):
        for law, label in enumerate(("paired", "product")):
            for u, uid in enumerate(ids):
                moment = M.moments(logs[u, frame, law])
                half = M.mix(logs[u, frame, law], 0.5)
                for b in (0, 1):
                    records.append(
                        {
                            "author": uid,
                            "frame": name,
                            "law": label,
                            "source": b,
                            "log_evidence": float(moment["log_evidence"][b]),
                            "source1_weight_at_half": half["source1_responsibility"],
                            **{
                                t: float(moment["endpoint_means"][b, q])
                                for q, t in enumerate(M.TARGETS)
                            },
                        }
                    )
            for u, v in combinations(range(24), 2):
                result = M.compare(
                    logs[u, frame, law], logs[v, frame, law], tol=cfg["sign_tolerance"]
                )
                for q, (target, cert) in enumerate(zip(M.TARGETS, result, strict=True)):
                    row = {
                        "author_u": ids[u],
                        "author_v": ids[v],
                        "u": u,
                        "v": v,
                        "frame": name,
                        "frame_index": frame,
                        "law": label,
                        "law_index": law,
                        "target": target,
                        "q": q,
                        "primary": frame == 0,
                        **cert,
                    }
                    for key in (
                        "quadratic",
                        "polynomial_extrema_weights",
                        "crossing_weights",
                        "endpoint_deltas",
                        "delta_extrema_weights",
                    ):
                        row[key] = json.dumps(row[key])
                    comparisons.append(row)
                    if cert["status"] in ("u_above_all", "u_below_all"):
                        high, low = (
                            (ids[u], ids[v])
                            if cert["status"] == "u_above_all"
                            else (ids[v], ids[u])
                        )
                        partials.append(
                            {
                                "frame": name,
                                "law": label,
                                "target": target,
                                "higher": high,
                                "lower": low,
                                "scope": "entire fixed two-source mixture family only",
                            }
                        )
    R.table(out / "source_evidence_records.csv", records)
    R.table(out / "comparison_certificates.csv", comparisons)
    df = pd.DataFrame(comparisons)
    summaries = []
    for (frame, law, target), group in df.groupby(
        ["frame", "law", "target"], sort=False
    ):
        consensus = group.endpoint_consensus.astype(bool)
        summaries.append(
            {
                "frame": frame,
                "law": law,
                "target": target,
                "pairs": len(group),
                "primary": frame == "full",
                "endpoint_consensus": int(consensus.sum()),
                "endpoint_consensus_interior_reversals": int(
                    group.endpoint_consensus_interior_reversal.sum()
                ),
                **{
                    status: int((group.status == status).sum())
                    for status in (
                        "u_above_all",
                        "u_below_all",
                        "crossing",
                        "tie_within_tolerance",
                        "boundary_or_unresolved",
                    )
                },
            }
        )
    R.table(out / "reference_order_summary.csv", summaries)
    R.write(
        out / "partial_orders.json",
        {
            "authors": ids,
            "edges": partials,
            "no_total_order_forced": True,
            "sources_fixed_not_all_human_populations": True,
        },
    )
    tu, tv = M.toy()
    toy = {
        "certificate": M.compare(tu, tv)[0],
        "actual_half_deltas": (
            M.mix(tu, 0.5)["means"] - M.mix(tv, 0.5)["means"]
        ).tolist(),
        "naive_endpoint_average_delta": (
            M.moments(tu)["endpoint_means"].mean(0)
            - M.moments(tv)["endpoint_means"].mean(0)
        ).tolist(),
        "source1_weights_at_half": [
            M.mix(x, 0.5)["source1_responsibility"] for x in (tu, tv)
        ],
        "scope": "constructed positive likelihood example,not people or new psychology",
    }
    R.write(out / "toy_counterexample.json", toy)
    example = {
        "evidence_keys": [f"{ids[u]}/full/paired" for u in (0, 1)],
        "source_contract": "HM209/same Gaussian units/uniform32 profile prior/source203183",
        "log_joint_mass": logs[[0, 1], 0, 0].tolist(),
    }
    R.write(out / "cold_input.json", example)
    R.write(out / "cold_expected.json", M.cold(example))
    primary = df[df.primary]
    primary_records = pd.DataFrame(records)
    weights = primary_records[
        (primary_records.frame == "full") & (primary_records.source == 0)
    ]
    summary = {
        "authors": 24,
        "source_counts": [len(b["codes"]) for b in banks],
        "comparisons": len(df),
        "primary_comparisons": len(primary),
        "primary": [s for s in summaries if s["primary"]],
        "primary_endpoint_consensus": int(primary.endpoint_consensus.sum()),
        "primary_consensus_interior_reversals": int(
            primary.endpoint_consensus_interior_reversal.sum()
        ),
        "primary_crossing_comparisons": int((primary.status == "crossing").sum()),
        "all_consensus_interior_reversals": int(
            df.endpoint_consensus_interior_reversal.sum()
        ),
        "all_crossing_comparisons": int((df.status == "crossing").sum()),
        "primary_source1_responsibility_at_half_range": [
            float(weights.source1_weight_at_half.min()),
            float(weights.source1_weight_at_half.max()),
        ],
        "normalized_endpoint_replay_error": replay,
        "elapsed_seconds": time.perf_counter() - start,
        "new_neural_API_private_training_application_labels": 0,
        "new_capability": "cold source-qualified author partial order and analytic reference crossings",
        "scope": "declared source-instrument family;not psychological prediction or universal transport",
        "original_scope_complete": False,
    }
    if summary["elapsed_seconds"] > cfg["max_seconds"]:
        R.write(
            out / "compute_failure.json",
            {"stage": "finite_cap", "summary": summary, "no_rerun": True},
        )
        raise RuntimeError("Finite cap reached")
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        STAGES[1],
        (
            "likelihoods.npz",
            "source_evidence_records.csv",
            "comparison_certificates.csv",
            "reference_order_summary.csv",
            "partial_orders.json",
            "toy_counterexample.json",
            "cold_input.json",
            "cold_expected.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def literal(bank, point):
    """Audit common-unit mass by direct distances and literal profile products."""
    z, g, s2 = bank["prototypes"], bank["codes"], float(bank["variance"])
    k = np.array(
        [
            [np.exp(-np.sum((point[v] - row[v]) ** 2) / (2 * s2)) for row in z]
            for v in (0, 1)
        ]
    )
    result = np.empty((2, 32))
    for code in range(32):
        a, b = k[:, g == code]
        result[0, code] = (
            sum(float(aa * bb) for aa, bb in zip(a, b, strict=True)) / len(a) / 32
        )
        result[1, code] = sum(float(aa * bb) for aa in a for bb in b) / len(a) ** 2 / 32
    return np.log(result)


def verify(out):
    """Independently evaluate actual mixture laws at algebraically selected points."""
    cfg = check(out, STAGES[:2])["config"]
    if (out / "verification.json").exists():
        raise ValueError("Exclusive verification, no scientific repetition")
    start = time.perf_counter()
    points, old, banks = load(out)
    with np.load(out / "likelihoods.npz", allow_pickle=False) as z:
        logs = z["log_joint_mass"]
    likelihood_error = 0.0
    for b in (0, 1):
        for u in range(24):
            actual = literal(banks[b], points[u, 0])
            likelihood_error = max(
                likelihood_error, float(np.max(np.abs(actual - logs[u, 0, :, b])))
            )
    replay_error = float(
        np.max(
            np.abs(
                logs.reshape(216, 2, 2, 32)
                - logsumexp(logs.reshape(216, 2, 2, 32), -1, keepdims=True)
                - old
            )
        )
    )
    rows = pd.read_csv(out / "comparison_certificates.csv")
    error, root_error, checks = 0.0, 0.0, 0
    for row in rows.to_dict("records"):
        u, v, f, law, q = (
            int(row[k]) for k in ("u", "v", "frame_index", "law_index", "q")
        )
        lu, lv = logs[u, f, law], logs[v, f, law]
        mu, mv = np.exp(lu - lu.max()), np.exp(lv - lv.max())
        bu, bv = mu.sum(-1), mv.sum(-1)
        coeff = json.loads(row["quadratic"])
        roots = json.loads(row["crossing_weights"])
        times = {
            0.0,
            0.5,
            1.0,
            *json.loads(row["polynomial_extrema_weights"]),
            *json.loads(row["delta_extrema_weights"]),
            *roots,
        }
        values = []
        for t in times:
            pu, pv = (1 - t) * mu[0] + t * mu[1], (1 - t) * mv[0] + t * mv[1]
            delta = float((pu / pu.sum() - pv / pv.sum()) @ M.BITS[:, q])
            direct = (
                delta * ((1 - t) * bu[0] + t * bu[1]) * ((1 - t) * bv[0] + t * bv[1])
            )
            reconstructed = np.polynomial.polynomial.polyval(t, coeff)
            error = max(error, abs(direct - reconstructed))
            if t in roots:
                root_error = max(root_error, abs(delta))
            values.append(delta)
            checks += 1
        error = max(
            error,
            abs(min(values) - row["delta_min"]),
            abs(max(values) - row["delta_max"]),
        )
    run = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "cold",
            "--out",
            str(out),
            "--input",
            str(out / "cold_input.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    expected, actual = R.read(out / "cold_expected.json"), json.loads(run.stdout)
    cold_equal = expected == actual
    maximum = max(error, likelihood_error, replay_error, root_error)
    receipt = {
        "PASS": bool(maximum <= cfg["arithmetic_tolerance"] and cold_equal),
        "literal_full_history_law_values": 24 * 2 * 2 * 32,
        "literal_log_mass_error": likelihood_error,
        "old_endpoint_replay_error": replay_error,
        "analytic_point_checks": checks,
        "cross_product_or_extrema_error": error,
        "crossing_direct_probability_error": root_error,
        "separate_cold_process_exact_readback": cold_equal,
        "same_agent_not_blind": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if not receipt["PASS"]:
        R.write(out / "verification_failure.json", {**receipt, "no_rerun": True})
        raise ValueError("Reference-law audit failed, preserve result")
    R.write(out / "verification.json", receipt)
    R.seal(out, STAGES[2], ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Execute one explicit exclusive stage or a read-only numeric query."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compute", "verify", "check", "cold")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.stage == "cold":
        check(args.out, STAGES[:2])
        print(json.dumps(M.cold(R.read(args.input or args.out / "cold_input.json"))))
    elif args.stage == "check":
        check(args.out)
        print(json.dumps({"PASS": True, "read_only": True, "no_rerun": True}))
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
