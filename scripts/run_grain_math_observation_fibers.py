#!/usr/bin/env python3
"""Freeze and test missing single-comment input information, without tuning."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_observation_fibers as M
from scripts import run_grain_math_author_coupling as H
from scripts import run_grain_math_natural_summary as N
from scripts import run_grain_math_relation_origin as R

OUT = ROOT / "results/grain_math_response_v1/comment_observation_fibers_r1"
PLAN = ROOT / "docs/GRAIN_SINGLE_COMMENT_OBSERVATION_PLAN_V1.md"
SEEDS = (20261009, 20261010, 20261011)
CODE = (
    PLAN,
    Path(__file__).resolve(),
    ROOT / "grain_research/construction/math_observation_fibers.py",
    ROOT / "tests/test_grain_math_observation_fibers.py",
    ROOT / "scripts/run_grain_math_author_coupling.py",
    ROOT / "scripts/run_grain_math_natural_summary.py",
)


def prepare(out):
    """Freeze the exact protocol, prior seals and safe numerical cache list."""
    R.sealed(R.OUT)
    if (
        N.read(R.OUT / "verification.json")["status"]
        != "VERIFIED_FINITE_HM65_NOT_GLOBAL"
    ):
        raise ValueError("Terminal verified HM65 required")
    hashes, raw_hash = N.binding()
    hashes.update({str(p.relative_to(ROOT)): N.sha(p) for p in CODE})
    for name in (
        "manifest.json",
        "summary.json",
        "output_seal.json",
        "verification.json",
    ):
        path = R.OUT / name
        hashes[str(path.relative_to(ROOT))] = N.sha(path)
    out.mkdir(parents=True, exist_ok=False)
    N.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_EXACT_COMMENT_FIBERS_V1",
            "hashes": hashes,
            "source_material_hash_metadata_only": raw_hash,
            "seeds": SEEDS,
            "worlds": M.WORLDS,
            "expected_authors_comments": [276, 9165],
            "expected_natural_risk_rows": 270,
            "max_cpu_wall_seconds": 300,
            "exact_numeric_equality_not_similarity": True,
            "new_neural_calls_training_labels_raw": [0, 0, 0, 0],
            "source_empirical_projections": 10,
            "not_population_noise_floor_or_personality_accuracy": True,
        },
    )
    print({"status": "PREPARED_HM66", "new_model_calls": 0}, flush=True)


def sealed(out):
    """Check only the fixed allowlist; do not open private source text."""
    m = N.read(out / "manifest.json")
    for path, expected in m["hashes"].items():
        if N.sha(ROOT / path) != expected:
            raise ValueError("HM66 sealed input changed: " + path)
    R.sealed(R.OUT)
    return m


def cases():
    """Join exact old numeric addresses, retaining the same map in each case."""
    supports, maps, replay = H.load_numeric()
    cohort = {}
    for group in H.GROUPS:
        ids = [a for a, s in supports.items() if H.cohort(s) == group]
        queries = supports[ids[0]]["queries"]
        c = {
            name: np.concatenate([supports[a][name] for a in ids])
            for name in ("g", "h", "panel", "comment")
        }
        c["author"] = np.concatenate([np.full(len(supports[a]["g"]), a) for a in ids])
        c["fold"] = np.concatenate(
            [np.full(len(supports[a]["g"]), supports[a]["folds"][0]) for a in ids]
        )
        c["queries"] = queries
        c["maps"] = [
            np.column_stack([maps[f"{q}__{f}__coef"] for q in queries])
            for f in range(5)
        ]
        cohort[group] = c
    return cohort, replay


def risk_rows(key, role, g, y, authors, queries):
    """Report all weighted finite risks, with no winner or new predictor."""
    rows = []
    for mode, w in (
        ("author_equal", M.author_weights(authors)),
        ("comment_equal", np.full(len(g), 1 / len(g))),
    ):
        p = M.empirical_projection(g, y, w)
        for q, query in enumerate(queries):
            rows.append(
                {
                    "case": key,
                    "role": role,
                    "weighting": mode,
                    "query": query,
                    "authors": len(np.unique(authors)),
                    "comments": len(g),
                    "classes": len(p["keys"]),
                    "repeated_classes": int(sum(p["counts"] > 1)),
                    "comments_in_repeated_classes": int(
                        p["counts"][p["counts"] > 1].sum()
                    ),
                    "minimum_empirical_squared_error": p["risk"][q],
                    "total_centered_proxy_variance": p["total"][q],
                    "unrepresentable_fraction": p["fraction"][q],
                }
            )
    return rows


def extrema_rows(key, c, y, w):
    """Bind all repeated class extrema and within-author witnesses to addresses."""
    e = M.roundoff_envelope(c["h"], w)
    rows = M.fiber_extrema(c["g"], y, e)
    _, inverse, _ = M.exact_partition(c["g"])
    for row in rows:
        indexes = np.flatnonzero(inverse == row["class"])
        q, lo, hi = row["query_index"], row["low"], row["high"]
        row.update(
            {
                "case": key,
                "query": c["queries"][q],
                "authors": len(np.unique(c["author"][indexes])),
                "panels": len(np.unique(c["panel"][indexes])),
                "low_author": c["author"][lo],
                "high_author": c["author"][hi],
                "low_panel": c["panel"][lo],
                "high_panel": c["panel"][hi],
                "low_comment": c["comment"][lo],
                "high_comment": c["comment"][hi],
                "low_value": y[lo, q],
                "high_value": y[hi, q],
                "embedding_distance": float(np.linalg.norm(c["h"][hi] - c["h"][lo])),
            }
        )
        for column, value in enumerate(c["g"][lo]):
            row[f"g{column}"] = value
        best = 0.0
        certified = 0
        for author in np.unique(c["author"][indexes]):
            local = indexes[c["author"][indexes] == author]
            a, b = local[np.argmin(y[local, q])], local[np.argmax(y[local, q])]
            gap = float(y[b, q] - y[a, q])
            best = max(best, gap)
            certified += gap > e[a, q] + e[b, q]
        row["within_author_max_gap"] = best
        row["within_author_certified_count"] = certified
    return rows


def run(out):
    """One finite scientific attempt; preserve failures, never overwrite."""
    m = sealed(out)
    if (out / "checkpoint.json").exists():
        raise ValueError("Existing scientific attempt retained; no retry")
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM66", "stage": "source"})
    start = time.monotonic()
    try:
        execute(out, m, start)
    except Exception as error:
        N.write(
            out / "checkpoint.json",
            {"status": "FAILED_HM66_RETAINED", "error": repr(error)},
        )
        raise


def execute(out, manifest, start):
    """Seal source class means before evaluating new held diagnostics."""
    cohort, replay = cases()
    sources, frozen = {}, {}
    for group, c in cohort.items():
        for fold, w in enumerate(c["maps"]):
            key = f"{group}_{fold}"
            source = c["fold"] != fold
            p = M.empirical_projection(
                c["g"][source],
                c["h"][source] @ w,
                M.author_weights(c["author"][source]),
            )
            sources[key] = p
            for name in ("keys", "mass", "means", "counts"):
                frozen[key + "__" + name] = p[name]
            frozen[key + "__W"] = w
            frozen[key + "__source_authors"] = np.unique(c["author"][source])
            frozen[key + "__held_authors"] = np.unique(c["author"][~source])
    np.savez_compressed(out / "private_source_projections.npz", **frozen)
    source_hash = N.sha(out / "private_source_projections.npz")
    N.write(
        out / "source_freeze.json",
        {"sha256": source_hash, "held_analysis_started": False},
    )
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM66", "stage": "application"})
    risks, extrema, coverage, author_coverage, outputs, structures = (
        [],
        [],
        [],
        [],
        {},
        [],
    )
    for group, c in cohort.items():
        _, g_classes, g_counts = M.exact_partition(c["g"])
        _, _, h_counts = M.exact_partition(c["h"])
        structures.append(
            {
                "group": group,
                "authors": len(np.unique(c["author"])),
                "comments": len(c["g"]),
                "exact_g_classes": len(g_counts),
                "repeated_g_classes": int(sum(g_counts > 1)),
                "comments_in_repeated_g_classes": int(g_counts[g_counts > 1].sum()),
                "largest_g_class": int(g_counts.max()),
                "exact_h_classes": len(h_counts),
                "repeated_h_classes": int(sum(h_counts > 1)),
            }
        )
        for fold, w in enumerate(c["maps"]):
            if time.monotonic() - start > manifest["max_cpu_wall_seconds"]:
                raise TimeoutError("Finite CPU budget reached")
            key, y = f"{group}_{fold}", c["h"] @ w
            held = c["fold"] == fold
            for role, mask in (
                ("cohort", np.ones(len(y), bool)),
                ("source", ~held),
                ("held", held),
            ):
                risks.extend(
                    risk_rows(
                        key,
                        role,
                        c["g"][mask],
                        y[mask],
                        c["author"][mask],
                        c["queries"],
                    )
                )
            extrema.extend(extrema_rows(key, c, y, w))
            predicted, seen = M.lookup(sources[key], c["g"][held])
            a, wy = c["author"][held], M.author_weights(c["author"][held])
            mass = float(wy[seen].sum())
            for q, query in enumerate(c["queries"]):
                coverage.append(
                    {
                        "case": key,
                        "query": query,
                        "held_authors": len(np.unique(a)),
                        "held_comments": int(sum(held)),
                        "seen_comments": int(sum(seen)),
                        "author_weighted_seen_mass": mass,
                        "seen_mse": float(
                            np.sum(
                                wy[seen] * (y[held][seen, q] - predicted[seen, q]) ** 2
                            )
                            / mass
                        )
                        if mass
                        else np.nan,
                    }
                )
            for author in np.unique(a):
                local = a == author
                author_coverage.append(
                    {
                        "case": key,
                        "author": author,
                        "comments": int(sum(local)),
                        "seen_comments": int(sum(seen[local])),
                        "seen_fraction": float(seen[local].mean()),
                    }
                )
            outputs[key + "__y"] = y
            outputs[key + "__class"] = g_classes
            outputs[key + "__cold_seen"] = seen
            outputs[key + "__cold_prediction"] = predicted
    risk_frame = pd.DataFrame(risks)
    if len(risk_frame) != manifest["expected_natural_risk_rows"]:
        raise ValueError("Incomplete risk comparison scope")
    risk_frame.to_csv(out / "empirical_information_risk.csv", index=False)
    witness_frame = pd.DataFrame(extrema)
    if witness_frame.empty:
        witness_frame = pd.DataFrame(
            columns=("case", "query_index", "low", "high", "gap", "certified")
        )
    witness_frame.to_csv(out / "private_exact_fiber_witnesses.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "cold_support_diagnostics.csv", index=False)
    pd.DataFrame(author_coverage).to_csv(
        out / "private_author_support.csv", index=False
    )
    pd.DataFrame(structures).to_csv(out / "support_structure.csv", index=False)
    np.savez_compressed(out / "private_numeric_readouts.npz", **outputs)
    synthetic, toy_rows = {}, []
    for world, seed in itertools.product(M.WORLDS, SEEDS):
        material = M.synthetic_world(world, seed)
        key = f"{world}_{seed}"
        synthetic.update({key + "__" + name: value for name, value in material.items()})
        p = M.empirical_projection(
            material["g"], material["y"], M.author_weights(material["author"])
        )
        rows = M.fiber_extrema(
            material["g"],
            material["y"],
            M.roundoff_envelope(material["h"], material["W"]),
        )
        s = material["source"]
        ref = M.empirical_projection(
            material["g"][s], material["y"][s], M.author_weights(material["author"][s])
        )
        _, seen = M.lookup(ref, material["g"][~s])
        toy_rows.append(
            {
                "world": world,
                "seed": seed,
                "authors": 40,
                "records": len(material["g"]),
                "classes": len(p["keys"]),
                "certified_fiber_query_witnesses": sum(
                    row["certified"] for row in rows
                ),
                "risk_mean": float(p["risk"].mean()),
                "fraction_mean": float(p["fraction"].mean()),
                "held_exact_coverage": float(seen.mean()),
            }
        )
    np.savez_compressed(out / "synthetic_material.npz", **synthetic)
    pd.DataFrame(toy_rows).to_csv(out / "synthetic_mechanisms.csv", index=False)
    if source_hash != N.sha(out / "private_source_projections.npz"):
        raise ValueError("Source projection changed after held analysis")
    N.write(
        out / "summary.json",
        {
            "status": "FINITE_HM66_COMPLETE_NOT_GLOBAL",
            "authors_comments": [276, 9165],
            "risk_rows": len(risks),
            "fiber_query_rows": len(extrema),
            "new_neural_calls_training_labels_raw": [0, 0, 0, 0],
            "source_projections": 10,
            "synthetic_cases_records": [12, 11520],
            "CPU_wall_seconds": time.monotonic() - start,
            "prior_numeric_replay_error": replay,
            "original_scope_complete": False,
        },
    )
    paths = [
        p
        for p in out.iterdir()
        if p.name not in ("manifest.json", "checkpoint.json", "output_seal.json")
    ]
    N.write(out / "output_seal.json", {p.name: N.sha(p) for p in paths})
    N.write(out / "checkpoint.json", {"status": "COMPLETED_HM66", "new_calls": 0})
    print(N.read(out / "summary.json"), flush=True)


def alternate_projection(g, y, authors, mode="author_equal"):
    """Independent tuple grouping and math.fsum weighted class arithmetic."""
    weights = (
        M.author_weights(authors)
        if mode == "author_equal"
        else np.full(len(g), 1 / len(g))
    )
    groups = {}
    for i, row in enumerate(g):
        groups.setdefault(tuple(row), []).append(i)
    residual = np.empty_like(y)
    for indexes in groups.values():
        mass = math.fsum(weights[i] for i in indexes)
        for q in range(y.shape[1]):
            mean = math.fsum(weights[i] * y[i, q] for i in indexes) / mass
            residual[indexes, q] = y[indexes, q] - mean
    return np.array(
        [
            math.fsum(weights[i] * residual[i, q] ** 2 for i in range(len(y)))
            for q in range(y.shape[1])
        ]
    )


def verify(out):
    """Replay exact witnesses and finite risks without acquisition or selection."""
    sealed(out)
    for name, expected in N.read(out / "output_seal.json").items():
        if N.sha(out / name) != expected:
            raise ValueError("Frozen HM66 output changed: " + name)
    cohort, _ = cases()
    errors = {
        "weighted_fiber_risk": 0.0,
        "rich_input_dot": 0.0,
        "witness_delta_dot": 0.0,
    }
    risks = pd.read_csv(out / "empirical_information_risk.csv")
    witnesses = pd.read_csv(out / "private_exact_fiber_witnesses.csv")
    checked, certified, isolated = 0, 0, 0
    with (
        np.load(out / "private_numeric_readouts.npz", allow_pickle=False) as z,
        np.load(out / "private_source_projections.npz", allow_pickle=False) as source,
    ):
        for group, c in cohort.items():
            for fold, w in enumerate(c["maps"]):
                key, y = f"{group}_{fold}", c["h"] @ w
                np.testing.assert_array_equal(y, z[key + "__y"])
                held = c["fold"] == fold
                np.testing.assert_array_equal(
                    source[key + "__source_authors"], np.unique(c["author"][~held])
                )
                np.testing.assert_array_equal(
                    source[key + "__held_authors"], np.unique(c["author"][held])
                )
                isolated += 1
                for role, mask in (
                    ("cohort", np.ones(len(y), bool)),
                    ("source", ~held),
                    ("held", held),
                ):
                    for mode in ("author_equal", "comment_equal"):
                        direct = alternate_projection(
                            c["g"][mask], y[mask], c["author"][mask], mode
                        )
                        old = risks[
                            (risks.case == key)
                            & (risks.role == role)
                            & (risks.weighting == mode)
                        ]
                        for q, query in enumerate(c["queries"]):
                            error = abs(
                                direct[q]
                                - old[
                                    old["query"] == query
                                ].minimum_empirical_squared_error.iloc[0]
                            )
                            errors["weighted_fiber_risk"] = max(
                                errors["weighted_fiber_risk"], float(error)
                            )
                lookup_table = {
                    tuple(g): value
                    for g, value in zip(
                        source[key + "__keys"], source[key + "__means"], strict=True
                    )
                }
                for i, g in enumerate(c["g"][held]):
                    expected = lookup_table.get(tuple(g))
                    if expected is None:
                        if (
                            z[key + "__cold_seen"][i]
                            or not np.isnan(z[key + "__cold_prediction"][i]).all()
                        ):
                            raise ValueError("Unseen input silently imputed")
                    else:
                        if not z[key + "__cold_seen"][i]:
                            raise ValueError("Seen source input missing")
                        np.testing.assert_array_equal(
                            expected, z[key + "__cold_prediction"][i]
                        )
                for row in witnesses[witnesses.case == key].itertuples():
                    lo, hi, q = int(row.low), int(row.high), int(row.query_index)
                    np.testing.assert_array_equal(c["g"][lo], c["g"][hi])
                    raw = math.fsum(
                        (c["h"][hi, j] - c["h"][lo, j]) * w[j, q]
                        for j in range(w.shape[0])
                    )
                    errors["witness_delta_dot"] = max(
                        errors["witness_delta_dot"], abs(raw - row.gap)
                    )
                    for i in (lo, hi):
                        direct = math.fsum(
                            c["h"][i, j] * w[j, q] for j in range(w.shape[0])
                        )
                        errors["rich_input_dot"] = max(
                            errors["rich_input_dot"], abs(direct - y[i, q])
                        )
                    certified += bool(row.certified)
                    checked += 1
        support = pd.read_csv(out / "private_author_support.csv")
        if len(support) != 276:
            raise ValueError("Held author support scope missing")
    if max(errors.values()) > 1e-9:
        raise ArithmeticError("HM66 alternate arithmetic failed: " + json.dumps(errors))
    toy = pd.read_csv(out / "synthetic_mechanisms.csv")
    with np.load(out / "synthetic_material.npz", allow_pickle=False) as z:
        for row in toy.itertuples():
            key = f"{row.world}_{row.seed}"
            risk = alternate_projection(
                z[key + "__g"], z[key + "__y"], z[key + "__author"]
            )
            if abs(risk.mean() - row.risk_mean) > 1e-10:
                raise ArithmeticError("Synthetic risk replay differs")
    N.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM66_NOT_GLOBAL",
            "errors": errors,
            "exact_fiber_query_rows": checked,
            "certified_rows": certified,
            "source_held_isolation_checks": isolated,
            "source_hash_unchanged": N.sha(out / "private_source_projections.npz")
            == N.read(out / "source_freeze.json")["sha256"],
            "same_agent_alternate_arithmetic_not_independent_replication": True,
        },
    )
    print(N.read(out / "verification.json"), flush=True)


def main():
    """Expose explicit freeze/run/verify stages, no automatic scientific retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
