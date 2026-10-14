#!/usr/bin/env python3
"""HM78: complete author-comparison fibers and source-only relational observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_score_fiber as M

SPEC = ROOT / "configs/grain_score_fiber_hm78.json"
CODE = (
    "configs/grain_score_fiber_hm78.json",
    "docs/GRAIN_SCORE_FIBER_RELATIONAL_COMPLETION_PLAN_HM78.md",
    "grain_research/construction/math_score_fiber.py",
    "scripts/run_grain_score_fiber_hm78.py",
    "scripts/verify_grain_score_fiber_hm78.py",
    "tests/test_grain_math_score_fiber.py",
)
SOURCE_FILES = {
    "HM68": (
        "00/manifest.json",
        "03/stages_and_worlds.npz",
        "03/output_seal.json",
        "verification.json",
    ),
    "HM69": (
        "manifest.json",
        "01/rows.json",
        "01/profiles.npz",
        "output_seal.json",
        "verification.json",
    ),
    "HM71": (
        "00/manifest.json",
        "03/observations_and_moments.npz",
        "03/output_seal.json",
        "verification.json",
    ),
}


def digest(path):
    """Hash exact local bytes without printing text or credentials."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    """Read structured local numerical provenance."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Write a resumable finite checkpoint or result with no NaN serialization."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    """Save all cases, including zero effects and reversed comparisons."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_roots(spec):
    """Bind only the three declared public-synthetic numeric sources."""
    return {key: ROOT / value for key, value in spec["sources"].items()}


def validate_sources(spec):
    """Verify source-result seals and actual native-reading counts."""
    roots = source_roots(spec)
    for name, seal in (
        ("HM68", "03/output_seal.json"),
        ("HM69", "output_seal.json"),
        ("HM71", "03/output_seal.json"),
    ):
        for relative, sha in read(roots[name] / seal).items():
            if digest(roots[name] / relative) != sha:
                raise ValueError(f"Frozen source changed:{name}/{relative}")
    for name, count in (("HM68", 3456), ("HM71", 2880)):
        if read(roots[name] / "verification.json")["actual_calls"] != count:
            raise ValueError("Complete native-reading source required")


def load_sources(spec):
    """Load aligned scores and original profiles, never raw text or model weights."""
    roots = source_roots(spec)
    pairs = tuple(itertools.permutations(spec["targets"], 2))
    expected = [
        {"pair_id": i, "focal": j, "companion": k, "case": case}
        for i, (j, k) in enumerate(pairs)
        for case in spec["cases"]
    ]
    if read(roots["HM69"] / "01/rows.json") != expected:
        raise ValueError("Pair/case/profile identities changed")
    with np.load(roots["HM68"] / "03/stages_and_worlds.npz", allow_pickle=False) as a:
        sources = np.asarray(
            [[a[f"f{f}_{j}_{k}_score"].reshape(8, 6) for f in (0, 1)] for j, k in pairs]
        )
    with np.load(roots["HM69"] / "01/profiles.npz", allow_pickle=False) as a:
        profiles = a["probabilities"].reshape(30, 8, 2, 8).copy()
    with np.load(
        roots["HM71"] / "03/observations_and_moments.npz", allow_pickle=False
    ) as a:
        cold = {
            bank: np.asarray(
                [
                    a[f"{bank}_score_pair{i:02d}__native"].reshape(8, 6)
                    for i in range(30)
                ]
            )
            for bank in ("paraphrase", "reordered")
        }
    if not np.isfinite(sources).all() or any(
        not np.isfinite(a).all() for a in cold.values()
    ):
        raise ValueError("Finite complete actual response tables required")
    return pairs, sources, profiles, cold


def prepare(out):
    """Freeze scope/code/input hashes before any new diagnostic outputs."""
    if out.exists():
        raise FileExistsError("Never overwrite a frozen or failed experiment")
    spec = read(SPEC)
    if tuple(spec["schemes"]) != M.SCHEMES or spec["max_linear_programs"] != 2160:
        raise ValueError("Fixed complete finite protocol required")
    validate_sources(spec)
    code = {}
    for name in CODE:
        destination = out / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = digest(destination)
    roots = source_roots(spec)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM78",
            "spec": spec,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": code,
            "source_files": {
                str((roots[k] / n).relative_to(ROOT)): digest(roots[k] / n)
                for k, names in SOURCE_FILES.items()
                for n in names
            },
            "old_result_after_no_fresh_material": True,
            "original_scope_complete": False,
            "previous_goal_turn": "read-only direction audit;no code or new experiment;not execution progress",
        },
    )
    write(
        out / "checkpoint.json",
        {"status": "PREPARED_HM78", "active_processes": [], "linear_programs": 0},
    )
    print(
        "HM78 prepared:60 source contexts/all8 profile cases/six fixed evidence schemes",
        flush=True,
    )


def check(out):
    """Reject input or executable drift without touching old sources."""
    m = read(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift:" + name)
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source drift:" + name)
    return m


def run(out):
    """Execute complete finite comparisons; stop instead of adding accuracy searches."""
    m, state = check(out), read(out / "checkpoint.json")
    if state["status"] != "PREPARED_HM78" or (out / "03").exists():
        raise ValueError("Do not retry or overwrite a finite package")
    spec, start = m["spec"], time.monotonic()
    state.update(status="RUNNING_HM78", active_processes=[os.getpid()])
    write(out / "checkpoint.json", state)
    (out / "01").mkdir()
    pairs, sources, profiles, cold = load_sources(spec)
    structure, bounds, comparisons, lp_rows, missing = [], [], [], [], []
    arrays, lp_p, lp_dual, lp_original_dual, nulls = {}, [], [], [], []
    w, orders = M.character_basis()
    try:
        for pair_id, (focal, companion) in enumerate(pairs):
            for family in (0, 1):
                if time.monotonic() - start > spec["max_seconds"]:
                    raise TimeoutError("Finite180 second budget exhausted")
                p = np.vstack((np.full(8, 1 / 8), profiles[pair_id].reshape(16, 8)))
                f = sources[pair_id, family]
                charts = {
                    scheme: M.row_chart(M.observation_design(f, scheme))
                    for scheme in spec["schemes"]
                }
                base = charts["scores_only"]
                if base["rank"] != 7:
                    raise ValueError(
                        "Source six-score rank must be7;no substitute protocol"
                    )
                h = base["null"][:, 0]
                nulls.append(h)
                key = f"p{pair_id:02d}_f{family}"
                apps = {
                    "other_source_family": sources[pair_id, 1 - family],
                    **{bank: value[pair_id] for bank, value in cold.items()},
                }
                arrays[key + "_profiles"] = p
                arrays[key + "_null"] = h
                for bank, v in apps.items():
                    arrays[key + "_app_" + bank] = v
                    for qi, target in enumerate(spec["targets"]):
                        coefficient = float(h @ v[:, qi])
                        missing.append(
                            {
                                "pair_id": pair_id,
                                "focal": focal,
                                "companion": companion,
                                "family": family,
                                "bank": bank,
                                "query": target,
                                "unobserved_target_coefficient": coefficient,
                                "source_coefficient": float(h @ f[:, qi]),
                            }
                        )
                for scheme, chart in charts.items():
                    a = chart["design"]
                    arrays[key + "_design_" + scheme] = a
                    endpoints = [M.fiber_endpoints(v, chart) for v in p]
                    end = np.asarray([[lo, hi] for lo, hi, _ in endpoints])
                    arrays[key + "_endpoints_" + scheme] = end
                    arrays[key + "_limits_" + scheme] = np.asarray(
                        [t for _, _, t in endpoints]
                    )
                    c = w.T @ h
                    structure.append(
                        {
                            "pair_id": pair_id,
                            "focal": focal,
                            "companion": companion,
                            "family": family,
                            "scheme": scheme,
                            "rank": chart["rank"],
                            "remaining_dimension": 8 - chart["rank"],
                            "source_null_first_order_energy": float(
                                np.sum(c[orders == 1] ** 2)
                            ),
                            "source_null_second_order_energy": float(
                                np.sum(c[orders == 2] ** 2)
                            ),
                            "source_null_third_order_energy": float(
                                np.sum(c[orders == 3] ** 2)
                            ),
                            "added_observation_action_on_source_null": float(
                                np.linalg.norm(a[7:] @ h)
                            )
                            if len(a) > 7
                            else 0.0,
                            "equilibration_error": chart["equilibration_error"],
                        }
                    )
                    for bank, v in apps.items():
                        intervals = [M.target_interval(v, e) for e in endpoints]
                        arrays[key + "_interval_" + scheme + "_" + bank] = np.asarray(
                            [[lo, hi] for lo, hi, _ in intervals]
                        )
                        for profile_id, ((lower, upper, _), pv) in enumerate(
                            zip(intervals, p)
                        ):
                            for qi, target in enumerate(spec["targets"]):
                                bounds.append(
                                    {
                                        "pair_id": pair_id,
                                        "family": family,
                                        "scheme": scheme,
                                        "bank": bank,
                                        "profile_id": profile_id,
                                        "case": "uniform"
                                        if profile_id == 0
                                        else spec["cases"][(profile_id - 1) // 2],
                                        "side": -1
                                        if profile_id == 0
                                        else (profile_id - 1) % 2,
                                        "query": target,
                                        "lower": float(lower[qi]),
                                        "upper": float(upper[qi]),
                                        "width": float(upper[qi] - lower[qi]),
                                        "actual_profile_value": float(pv @ v[:, qi]),
                                    }
                                )
                        for case_id, case in enumerate(spec["cases"]):
                            a_id, b_id = 1 + 2 * case_id, 2 + 2 * case_id
                            lower, upper = M.comparison_interval(
                                intervals[a_id], intervals[b_id]
                            )
                            actual = (p[a_id] - p[b_id]) @ v
                            score_gap = float(np.linalg.norm((p[a_id] - p[b_id]) @ f))
                            for qi, target in enumerate(spec["targets"]):
                                comparisons.append(
                                    {
                                        "pair_id": pair_id,
                                        "family": family,
                                        "scheme": scheme,
                                        "bank": bank,
                                        "case": case,
                                        "query": target,
                                        "source_six_score_gap_norm": score_gap,
                                        "actual_comparison": float(actual[qi]),
                                        "lower": float(lower[qi]),
                                        "upper": float(upper[qi]),
                                        "width": float(upper[qi] - lower[qi]),
                                        "both_rankings_attainable": bool(
                                            lower[qi] < -spec["tolerance"]
                                            and upper[qi] > spec["tolerance"]
                                        ),
                                    }
                                )
                # Independent LP check of every source-score collision context.
                profile_id = 1 + 2 * (6 + family)
                pv, end = p[profile_id], M.fiber_endpoints(p[profile_id], base)
                for bank, v in apps.items():
                    direct = M.target_interval(v, end)
                    for qi, target in enumerate(spec["targets"]):
                        if (
                            state["linear_programs"] + 2 > spec["max_linear_programs"]
                            or time.monotonic() - start > spec["max_seconds"]
                        ):
                            raise TimeoutError(
                                "2160LP/180second finite budget exhausted"
                            )
                        result = M.lp_interval(v[:, qi], pv, base)
                        state["linear_programs"] += 2
                        error = max(
                            abs(result["lower"]["value"] - direct[0][qi]),
                            abs(result["upper"]["value"] - direct[1][qi]),
                        )
                        if error > spec["tolerance"]:
                            raise ArithmeticError("LP/analytic sharp interval mismatch")
                        row_id = len(lp_rows)
                        lp_rows.append(
                            {
                                "row_id": row_id,
                                "pair_id": pair_id,
                                "family": family,
                                "bank": bank,
                                "query": target,
                                "profile_id": profile_id,
                                "lower": result["lower"]["value"],
                                "upper": result["upper"]["value"],
                                "analytic_error": float(error),
                                "certificate_error": max(
                                    result[s]["error"] for s in ("lower", "upper")
                                ),
                            }
                        )
                        lp_p.append([result[s]["p"] for s in ("lower", "upper")])
                        lp_dual.append([result[s]["dual"] for s in ("lower", "upper")])
                        lp_original_dual.append(
                            [result[s]["original_dual"] for s in ("lower", "upper")]
                        )
                state.update(
                    completed_contexts=len(nulls),
                    elapsed_seconds=time.monotonic() - start,
                )
                write(out / "checkpoint.json", state)
                if len(nulls) % 10 == 0:
                    print(
                        f"HM78 contexts{len(nulls)}/60;LPs{state['linear_programs']}/2160",
                        flush=True,
                    )
        if (len(bounds), len(comparisons), len(lp_rows), state["linear_programs"]) != (
            110160,
            51840,
            1080,
            2160,
        ):
            raise ArithmeticError("Incomplete fixed finite battery")
        h = np.asarray(nulls)
        arrays.update(
            nulls=h,
            null_gram_squared=(h @ h.T) ** 2,
            lp_primal=np.asarray(lp_p),
            lp_dual=np.asarray(lp_dual),
            lp_original_dual=np.asarray(lp_original_dual),
        )
        width_summary, comparison_summary = [], []
        for scheme, bank, target in itertools.product(
            spec["schemes"], spec["application_banks"], spec["targets"]
        ):
            selected = [
                r
                for r in bounds
                if (r["scheme"], r["bank"], r["query"]) == (scheme, bank, target)
            ]
            values = np.asarray([r["width"] for r in selected])
            width_summary.append(
                {
                    "scheme": scheme,
                    "bank": bank,
                    "query": target,
                    "profiles": len(values),
                    "mean_width": float(values.mean()),
                    "median_width": float(np.median(values)),
                    "max_width": float(values.max()),
                    "positive_width_count": int(np.sum(values > spec["tolerance"])),
                }
            )
            selected = [
                r
                for r in comparisons
                if (r["scheme"], r["bank"], r["query"]) == (scheme, bank, target)
            ]
            collision = [r for r in selected if r["source_six_score_gap_norm"] < 1e-10]
            comparison_summary.append(
                {
                    "scheme": scheme,
                    "bank": bank,
                    "query": target,
                    "pairs": len(selected),
                    "both_rankings_attainable": sum(
                        r["both_rankings_attainable"] for r in selected
                    ),
                    "score_collision_pairs": len(collision),
                    "collision_both_rankings_attainable": sum(
                        r["both_rankings_attainable"] for r in collision
                    ),
                }
            )
        (out / "03").mkdir()
        for name, rows in (
            ("structure", structure),
            ("target_bounds", bounds),
            ("author_comparisons", comparisons),
            ("lp_certificates", lp_rows),
            ("missing_target_actions", missing),
            ("width_summary", width_summary),
            ("comparison_summary", comparison_summary),
        ):
            csv_write(out / "03" / (name + ".csv"), rows)
        np.savez_compressed(out / "03/fibers_and_certificates.npz", **arrays)
        average_projector = h.T @ h / len(h)
        eigenvalues = np.linalg.eigvalsh(average_projector)[::-1]
        write(
            out / "03/summary.json",
            {
                "status": "COLLECTED_FINITE_HM78_NOT_GLOBAL",
                "contexts": 60,
                "profiles": 1020,
                "target_bounds": len(bounds),
                "author_comparisons": len(comparisons),
                "linear_programs": state["linear_programs"],
                "elapsed_wall_seconds": time.monotonic() - start,
                "null_average_projector_eigenvalues": eigenvalues.tolist(),
                "same_pair_family_null_alignment_mean": float(
                    np.mean([abs(h[2 * i] @ h[2 * i + 1]) ** 2 for i in range(30)])
                ),
                "completion_rank_counts": {
                    scheme: {
                        str(rank): sum(
                            r["rank"] == rank
                            for r in structure
                            if r["scheme"] == scheme
                        )
                        for rank in (7, 8)
                    }
                    for scheme in spec["schemes"]
                },
                "null_energy_by_order_mean": {
                    str(order): float(
                        np.mean(
                            [
                                r[f"source_null_{name}_order_energy"]
                                for r in structure
                                if r["scheme"] == "scores_only"
                            ]
                        )
                    )
                    for order, name in ((1, "first"), (2, "second"), (3, "third"))
                },
                "maximum_LP_analytic_error": max(r["analytic_error"] for r in lp_rows),
                "maximum_LP_certificate_error": max(
                    r["certificate_error"] for r in lp_rows
                ),
                "new_calls_fits_training_API_private": [0, 0, 0, 0, 0],
                "result_after_not_independent_replication": True,
                "original_scope_complete": False,
            },
        )
        write(
            out / "03/output_seal.json",
            {
                str(p.relative_to(out)): digest(p)
                for p in (out / "03").iterdir()
                if p.is_file()
                and not p.name.startswith("._")
                and p.name != "output_seal.json"
            },
        )
        state.update(
            status="COLLECTED_HM78",
            active_processes=[],
            elapsed_seconds=time.monotonic() - start,
        )
        write(out / "checkpoint.json", state)
        print(read(out / "03/summary.json"), flush=True)
    except Exception as exc:
        np.savez_compressed(out / "01/partial_arrays.npz", **arrays)
        write(
            out / "01/failure.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "completed_contexts": len(nulls),
                "saved_partial_arrays": True,
            },
        )
        state.update(
            status="FAILED_HM78_PRESERVED",
            active_processes=[],
            error=str(exc),
            elapsed_seconds=time.monotonic() - start,
        )
        write(out / "checkpoint.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=ROOT / read(SPEC)["output"])
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.out)
