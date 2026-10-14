#!/usr/bin/env python3
"""HM70: material-common event functions and joint author-profile observation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_common_observables as M
from grain_research.vector_mechanisms.navigation import digest, read_json

SOURCES = {
    "HM68": ROOT / "results/grain_math_response_v1/ownership_composition_r1",
    "HM69": ROOT / "results/grain_math_response_v1/author_observation_accounts_r1",
}
OUT = ROOT / "results/grain_math_response_v1/common_observable_transport_r1"
TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
PAIRS = tuple(itertools.permutations(TARGETS, 2))
STAGES = (
    "score",
    "log_odds",
    "centered_logits",
    "body_endpoint",
    "score_link_control",
    "log_odds_link_control",
)
CASES = (
    "choice_f0",
    "choice_f1",
    "expression_f0",
    "expression_f1",
    "law_only",
    "mixed",
    "same_scores_f0",
    "same_scores_f1",
)
CODE = (
    "docs/GRAIN_COMMON_OBSERVABLE_JOINT_TRANSPORT_PLAN_V1.md",
    "grain_research/construction/math_common_observables.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_math_common_observables.py",
    "tests/test_grain_math_common_observables.py",
)


def write(path, value):
    """Save finite local metadata without permissive NaN serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    """Preserve full numerical case tables rather than selected headline cases."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_check():
    """Verify frozen actual readouts/profiles, not upstream completion strings alone."""
    for name, seals in (
        ("HM68", ("01/raw_seal.json", "03/output_seal.json")),
        ("HM69", ("output_seal.json",)),
    ):
        for seal in seals:
            for path, sha in read_json(SOURCES[name] / seal).items():
                if digest(SOURCES[name] / path) != sha:
                    raise ValueError(f"Frozen {name} changed: {path}")
    if read_json(SOURCES["HM68"] / "verification.json")["actual_calls"] != 3456:
        raise ValueError("All actual HM68 observations required")
    if read_json(SOURCES["HM69"] / "01/summary.json")["profile_cases"] != 240:
        raise ValueError("All HM69 profile cases required")


def prepare(out):
    """Freeze the complete bounded design, dependencies and observation identities."""
    if out.exists():
        raise FileExistsError("Never overwrite a frozen experiment")
    source_check()
    inputs = {
        "HM68": (
            "00/manifest.json",
            "01/raw_seal.json",
            "03/output_seal.json",
            "03/stages_and_worlds.npz",
            "verification.json",
        ),
        "HM69": (
            "manifest.json",
            "output_seal.json",
            "01/profiles.npz",
            "01/rows.json",
            "01/summary.json",
            "verification.json",
        ),
    }
    code = {}
    for name in CODE:
        destination = out / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = digest(destination)
    write(
        out / "manifest.json",
        {
            "packet": "HM70",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_hashes": {
                key: {name: digest(SOURCES[key] / name) for name in names}
                for key, names in inputs.items()
            },
            "code_hashes": code,
            "pairs": PAIRS,
            "stages": STAGES,
            "cases": CASES,
            "declared_local_quadratic_global_linear": [180, 6],
            "new_calls_training_API_private_labels": [0, 0, 0, 0, 0],
            "event_identity_is_design_not_semantic_discovery": True,
            "original_scope_complete": False,
        },
    )
    write(out / "checkpoint.json", {"status": "PREPARED_HM70", "active_processes": []})
    print("HM70 prepared: 180 local quadratic charts, six global linear charts")


def bound_check(out):
    """Refuse drift in inputs or frozen executable code."""
    manifest = read_json(out / "manifest.json")
    for source, hashes in manifest["source_hashes"].items():
        for name, sha in hashes.items():
            if digest(SOURCES[source] / name) != sha:
                raise ValueError("Frozen source changed: " + name)
    for name, sha in manifest["code_hashes"].items():
        if digest(ROOT / name) != sha or digest(out / "code" / name) != sha:
            raise ValueError("Frozen executable changed: " + name)


def load_sources():
    """Load complete aligned public numerical tables and probability profiles."""
    observations = {}
    with np.load(
        SOURCES["HM68"] / "03/stages_and_worlds.npz", allow_pickle=False
    ) as data:
        for stage in STAGES:
            observations[stage] = np.array(
                [
                    [data[f"f{f}_{j}_{k}_{stage}"].reshape(8, -1) for f in (0, 1)]
                    for j, k in PAIRS
                ]
            )
    expected = [
        {"pair_id": i, "focal": j, "companion": k, "case": case}
        for i, (j, k) in enumerate(PAIRS)
        for case in CASES
    ]
    if read_json(SOURCES["HM69"] / "01/rows.json") != expected:
        raise ValueError("Profile identities do not match declared cell/pair order")
    with np.load(SOURCES["HM69"] / "01/profiles.npz", allow_pickle=False) as data:
        profiles = data["probabilities"].reshape(30, 8, 2, 8)
    return observations, profiles


def chart_row(chart, stage, pair_id, focal, companion):
    """Summarize structural dimensions and calibration identities, not accuracy."""
    return {
        "stage": stage,
        "pair_id": pair_id,
        "focal": focal,
        "companion": companion,
        "support": len(chart["basis"]),
        "degree": int(chart["degree"]),
        "material0_rank": int(chart["material_ranks"][0]),
        "material1_rank": int(chart["material_ranks"][1]),
        "common_contrasts": int(chart["contrast_dimension"]),
        "moment_rank": int(chart["joint_rank"]),
        "minimum_event_separation": float(chart["minimum_event_separation"]),
        "calibration_max_error": float(chart["calibration_errors"].max()),
        "full_profile_identified": bool(
            int(chart["joint_rank"]) == len(chart["basis"])
        ),
    }


def profile_rows(observed, probabilities, stage, pair_id, focal, companion):
    """Report all fixed profile contrasts including actual reconstruction failures."""
    result = []
    for case_id, case in enumerate(CASES):
        indices = [2 * case_id, 2 * case_id + 1]
        a, b = indices
        result.append(
            {
                "stage": stage,
                "pair_id": pair_id,
                "focal": focal,
                "companion": companion,
                "case": case,
                "probability_gap_norm": float(
                    np.linalg.norm(probabilities[b] - probabilities[a])
                ),
                "common_mean_gap_norm": float(
                    np.linalg.norm(observed["means"][b] - observed["means"][a])
                ),
                "joint_moment_gap_norm": float(
                    np.linalg.norm(
                        observed["joint_moments"][b] - observed["joint_moments"][a]
                    )
                ),
                "product_of_means_gap_norm": float(
                    np.linalg.norm(
                        observed["product_of_means"][b]
                        - observed["product_of_means"][a]
                    )
                ),
                "reconstruction_max_error": float(
                    np.max(
                        abs(observed["reconstructed"][indices] - probabilities[indices])
                    )
                ),
            }
        )
    return result


def fiber_rows(observations, profiles):
    """Certify failure of score-only material transport, including nonlinear maps."""
    result = []
    for pair_id, (j, k) in enumerate(PAIRS):
        for f in (0, 1):
            delta = profiles[pair_id, 6 + f, 1] - profiles[pair_id, 6 + f, 0]
            origin = delta @ observations["score"][pair_id, f]
            other = delta @ observations["score"][pair_id, 1 - f]
            result.append(
                {
                    "pair_id": pair_id,
                    "focal": j,
                    "companion": k,
                    "origin_family": f,
                    "origin_score_gap_norm": float(np.linalg.norm(origin)),
                    "other_score_gap_norm": float(np.linalg.norm(other)),
                    "score_only_worst_case_error_lower_bound": float(
                        np.linalg.norm(other) / 2
                    ),
                }
            )
    return result


def run(out):
    """Execute every declared chart and profile case without selecting a best result."""
    bound_check(out)
    if read_json(out / "checkpoint.json")["status"] != "PREPARED_HM70":
        raise ValueError("Do not overwrite or retry a collected packet")
    start = time.monotonic()
    write(
        out / "checkpoint.json",
        {"status": "RUNNING_HM70", "active_processes": [os.getpid()]},
    )
    observations, profiles = load_sources()
    directory = out / "01"
    directory.mkdir()
    charts, global_charts, rows = [], [], []
    for stage in STAGES:
        for pair_id, (j, k) in enumerate(PAIRS):
            c = M.chart(*observations[stage][pair_id])
            p = profiles[pair_id].reshape(16, 8)
            observed = M.profile_moments(p, c)
            np.savez_compressed(
                directory / f"{stage}_pair{pair_id:02}.npz",
                **c,
                probabilities=p,
                **observed,
            )
            charts.append(chart_row(c, stage, pair_id, j, k))
            rows.extend(profile_rows(observed, p, stage, pair_id, j, k))
        c = M.chart(
            observations[stage][:, 0].reshape(240, -1),
            observations[stage][:, 1].reshape(240, -1),
            joint=False,
        )
        np.savez_compressed(directory / f"{stage}_global.npz", **c)
        global_charts.append(
            chart_row(c, stage, -1, "distinct_30_pairs", "not_same_semantic_state")
        )
        if time.monotonic() - start > 120:
            raise TimeoutError("Declared CPU collector exceeded120 seconds")
    fibers = fiber_rows(observations, profiles)
    write_csv(directory / "local_charts.csv", charts)
    write_csv(directory / "global_charts.csv", global_charts)
    write_csv(directory / "profile_observations.csv", rows)
    write_csv(directory / "score_only_transport_fibers.csv", fibers)
    summaries = []
    for stage in STAGES:
        cs = [r for r in charts if r["stage"] == stage]
        rs = [r for r in rows if r["stage"] == stage]
        fs = [r for r in rs if r["case"].startswith("same_scores")]
        summaries.append(
            {
                "stage": stage,
                "local_charts": len(cs),
                "common_contrast_dimensions": sorted(
                    {r["common_contrasts"] for r in cs}
                ),
                "moment_ranks": sorted({r["moment_rank"] for r in cs}),
                "full_profile_charts": sum(r["full_profile_identified"] for r in cs),
                "maximum_calibration_error": max(
                    r["calibration_max_error"] for r in cs
                ),
                "maximum_reconstruction_error": max(
                    r["reconstruction_max_error"] for r in rs
                ),
                "score_fiber_maximum_common_mean_gap": max(
                    r["common_mean_gap_norm"] for r in fs
                ),
                "score_fiber_minimum_joint_gap": min(
                    r["joint_moment_gap_norm"] for r in fs
                ),
                "score_fiber_joint_gap_RMS_norm": math.sqrt(
                    math.fsum(r["joint_moment_gap_norm"] ** 2 for r in fs) / len(fs)
                ),
                "score_fiber_product_of_means_gap_RMS_norm": math.sqrt(
                    math.fsum(r["product_of_means_gap_norm"] ** 2 for r in fs) / len(fs)
                ),
            }
        )
    write(
        directory / "summary.json",
        {
            "packet": "HM70",
            "local_charts": len(charts),
            "global_linear_charts": global_charts,
            "profile_observation_rows": len(rows),
            "score_only_fibers": len(fibers),
            "stage_summaries": summaries,
            "maximum_origin_score_gap": max(r["origin_score_gap_norm"] for r in fibers),
            "minimum_other_score_gap": min(r["other_score_gap_norm"] for r in fibers),
            "minimum_score_only_error_lower_bound": min(
                r["score_only_worst_case_error_lower_bound"] for r in fibers
            ),
            "CPU_wall_seconds": time.monotonic() - start,
            "new_calls_fits_API_private_labels": [0, 0, 0, 0, 0],
            "original_scope_complete": False,
        },
    )
    write(
        out / "output_seal.json",
        {
            str(path.relative_to(out)): digest(path)
            for path in sorted(directory.iterdir())
            if not path.name.startswith("._")
        },
    )
    write(out / "checkpoint.json", {"status": "COLLECTED_HM70", "active_processes": []})
    print(json.dumps(read_json(directory / "summary.json")))


def independent_rank(a):
    """Alternate numerical dimension computation from equilibrated concatenated columns."""
    scales = np.sqrt(np.sum(a * a, axis=0))
    scales[scales == 0] = 1
    s = np.linalg.svd(a / scales, compute_uv=False)
    return int(np.count_nonzero(s > 64 * np.finfo(float).eps * max(a.shape) * s[0]))


def verify(out):
    """Check all charts and moments with direct sums and union-space dimensions."""
    bound_check(out)
    for name, sha in read_json(out / "output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Frozen output changed: " + name)
    observations, profiles = load_sources()
    checks, max_error = 0, 0.0
    expected_charts, expected_global, expected_rows = [], [], []

    def compare(a, b, tolerance=2e-9):
        nonlocal checks, max_error
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Aligned finite verification required")
        error = float(np.max(abs(a - b))) if a.size else 0.0
        if error > tolerance:
            raise ValueError(f"Alternate calculation mismatch {error}")
        checks += a.size
        max_error = max(max_error, error)

    def check_chart(c, f0, f1, degree):
        n = len(f0)
        a0, a1 = [np.column_stack((np.ones(n), f)) for f in (f0, f1)]
        r0, r1 = independent_rank(a0), independent_rank(a1)
        dimension = r0 + r1 - independent_rank(np.column_stack((a0, a1))) - 1
        z = c["basis"]
        compare(c["material_ranks"], [r0, r1])
        compare(c["contrast_dimension"], dimension)
        compare(c["degree"], degree)
        compare(z.T @ z, np.eye(dimension))
        compare(z.sum(0), np.zeros(dimension))
        compare(c["projector"], z @ z.T)
        errors = []
        for a, key in ((a0, "calibration0"), (a1, "calibration1")):
            mapped = a @ c[key]
            compare(mapped, z)
            errors.append(float(np.max(abs(mapped - z))) if z.size else 0.0)
        compare(c["calibration_errors"], errors)
        products = (
            list(itertools.combinations_with_replacement(range(dimension), 2))
            if degree == 2
            else []
        )
        compare(c["products"], np.array(products, int).reshape(-1, 2))
        h = np.array(
            [
                [1.0, *z[e], *[float(z[e, i]) * float(z[e, j]) for i, j in products]]
                for e in range(n)
            ]
        )
        compare(c["quadratic"], h)
        compare(c["joint_rank"], independent_rank(h))
        compare(h @ c["inverse"] @ h, h)
        compare(
            c["minimum_event_separation"],
            min(np.linalg.norm(z[i] - z[j]) for i in range(n) for j in range(i)),
        )
        return h

    for stage in STAGES:
        for pair_id, (j, k) in enumerate(PAIRS):
            with np.load(
                out / "01" / f"{stage}_pair{pair_id:02}.npz", allow_pickle=False
            ) as archive:
                c = dict(archive)
            h = check_chart(c, *observations[stage][pair_id], 2)
            p = profiles[pair_id].reshape(16, 8)
            compare(c["probabilities"], p)
            expected_moments = np.array(
                [
                    [
                        math.fsum(float(p[row, e]) * float(h[e, q]) for e in range(8))
                        for q in range(h.shape[1])
                    ]
                    for row in range(16)
                ]
            )
            means = expected_moments[:, 1 : 1 + c["basis"].shape[1]]
            products = c["products"]
            separate = np.array(
                [[float(m[i]) * float(m[j]) for i, j in products] for m in means]
            )
            compare(c["means"], means)
            compare(c["joint_moments"], expected_moments)
            compare(c["product_of_means"], separate)
            compare(
                c["reconstructed"],
                np.linalg.lstsq(
                    h.T,
                    expected_moments.T,
                    rcond=64 * np.finfo(float).eps * max(h.shape),
                )[0].T,
            )
            compare(c["reconstructed"] @ h, expected_moments)
            if int(c["joint_rank"]) == 8:
                compare(c["reconstructed"], p)
            expected_charts.append(chart_row(c, stage, pair_id, j, k))
            expected_rows.extend(profile_rows(c, p, stage, pair_id, j, k))
        with np.load(out / "01" / f"{stage}_global.npz", allow_pickle=False) as archive:
            c = dict(archive)
        check_chart(
            c,
            observations[stage][:, 0].reshape(240, -1),
            observations[stage][:, 1].reshape(240, -1),
            1,
        )
        expected_global.append(
            chart_row(c, stage, -1, "distinct_30_pairs", "not_same_semantic_state")
        )
    expected_fibers = []
    for pair_id, (j, k) in enumerate(PAIRS):
        for f in (0, 1):
            p0, p1 = profiles[pair_id, 6 + f]
            outputs = []
            for material in (f, 1 - f):
                table = observations["score"][pair_id, material]
                outputs.append(
                    np.array(
                        [
                            math.fsum(
                                float(p1[e] - p0[e]) * float(table[e, q])
                                for e in range(8)
                            )
                            for q in range(6)
                        ]
                    )
                )
            compare(outputs[0], np.zeros(6))
            expected_fibers.append(
                {
                    "pair_id": pair_id,
                    "focal": j,
                    "companion": k,
                    "origin_family": f,
                    "origin_score_gap_norm": float(np.linalg.norm(outputs[0])),
                    "other_score_gap_norm": float(np.linalg.norm(outputs[1])),
                    "score_only_worst_case_error_lower_bound": float(
                        np.linalg.norm(outputs[1]) / 2
                    ),
                }
            )
    for name, expected in (
        ("local_charts.csv", expected_charts),
        ("global_charts.csv", expected_global),
        ("profile_observations.csv", expected_rows),
        ("score_only_transport_fibers.csv", expected_fibers),
    ):
        with (out / "01" / name).open() as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError("Incomplete actual output table: " + name)
        for a, b in zip(actual, expected, strict=True):
            for key, value in b.items():
                if isinstance(value, (str, bool)):
                    if a[key] != str(value):
                        raise ValueError("Case identity changed")
                else:
                    compare(float(a[key]), value)
    summary = read_json(out / "01/summary.json")
    compare(summary["local_charts"], 180)
    compare(summary["profile_observation_rows"], 1440)
    compare(summary["score_only_fibers"], 60)
    compare(
        summary["maximum_origin_score_gap"],
        max(r["origin_score_gap_norm"] for r in expected_fibers),
    )
    compare(
        summary["minimum_other_score_gap"],
        min(r["other_score_gap_norm"] for r in expected_fibers),
    )
    compare(
        summary["minimum_score_only_error_lower_bound"],
        min(r["score_only_worst_case_error_lower_bound"] for r in expected_fibers),
    )
    if summary["global_linear_charts"] != expected_global:
        raise ValueError("Global chart summary drift")
    if len(summary["stage_summaries"]) != len(STAGES):
        raise ValueError("All stage summaries required")
    for stage, actual in zip(STAGES, summary["stage_summaries"], strict=True):
        cs = [r for r in expected_charts if r["stage"] == stage]
        rs = [r for r in expected_rows if r["stage"] == stage]
        fs = [r for r in rs if r["case"].startswith("same_scores")]
        if actual["stage"] != stage:
            raise ValueError("Stage summary identity drift")
        compare(actual["local_charts"], len(cs))
        compare(
            actual["common_contrast_dimensions"],
            sorted({r["common_contrasts"] for r in cs}),
        )
        compare(actual["moment_ranks"], sorted({r["moment_rank"] for r in cs}))
        compare(
            actual["full_profile_charts"], sum(r["full_profile_identified"] for r in cs)
        )
        compare(
            actual["maximum_calibration_error"],
            max(r["calibration_max_error"] for r in cs),
        )
        compare(
            actual["maximum_reconstruction_error"],
            max(r["reconstruction_max_error"] for r in rs),
        )
        compare(
            actual["score_fiber_maximum_common_mean_gap"],
            max(r["common_mean_gap_norm"] for r in fs),
        )
        compare(
            actual["score_fiber_minimum_joint_gap"],
            min(r["joint_moment_gap_norm"] for r in fs),
        )
        for column, key in (
            ("joint_moment_gap_norm", "score_fiber_joint_gap_RMS_norm"),
            ("product_of_means_gap_norm", "score_fiber_product_of_means_gap_RMS_norm"),
        ):
            compare(
                actual[key], np.linalg.norm([r[column] for r in fs]) / np.sqrt(len(fs))
            )
    result = {
        "status": "VERIFIED_FINITE_HM70_NOT_GLOBAL",
        "actual_local_charts_global_charts_profiles_fibers": [180, 6, 1440, 60],
        "actual_scalar_checks": checks,
        "maximum_alternate_error": max_error,
        "new_neural_calls": 0,
        "same_agent_alternate_arithmetic_not_independent_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    write(out / "checkpoint.json", {"status": result["status"], "active_processes": []})
    print(json.dumps(result))


def main():
    """Explicit finite stages only; retain failure checkpoints and never restart silently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)
    except Exception as exc:
        if args.stage == "run" and args.out.exists():
            write(
                args.out / "checkpoint.json",
                {
                    "status": "FAILED_HM70_RETAINED",
                    "error": repr(exc),
                    "active_processes": [],
                },
            )
        raise


if __name__ == "__main__":
    main()
