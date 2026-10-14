#!/usr/bin/env python3
"""Freeze, run and replay HM64 using only previously sealed numerical caches."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_coupling as M
from scripts import run_grain_math_natural_summary as N

OUT = ROOT / "results/grain_math_response_v1/joint_author_coupling_r1"
PLAN = ROOT / "docs/GRAIN_JOINT_AUTHOR_COUPLING_PLAN_V1.md"
CODE = (
    PLAN,
    Path(__file__).resolve(),
    ROOT / "grain_research/construction/math_author_coupling.py",
    ROOT / "tests/test_grain_math_author_coupling.py",
    ROOT / "scripts/run_grain_math_natural_summary.py",
)
GROUPS = ("big5_only", "mbti_only")
PAIRS = tuple(itertools.permutations(range(3), 2))
BOOTSTRAPS, PERMUTATIONS, SEED = 2000, 199, 20261009
CONTRASTS = (
    ("coupling", "shifted_coupling"),
    ("joint_with_coupling", "joint_location"),
    ("joint_with_shifted", "joint_location"),
)


def prepare(out):
    """Bind original numeric seals, old loader and this protocol before outcomes."""
    hashes, raw_hash = N.binding()
    N.sealed(N.OUT)
    hashes.update({str(p.relative_to(ROOT)): N.sha(p) for p in CODE})
    hashes[str((N.OUT / "manifest.json").relative_to(ROOT))] = N.sha(
        N.OUT / "manifest.json"
    )
    out.mkdir(parents=True, exist_ok=False)
    N.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_JOINT_AUTHOR_COUPLING_V1",
            "hashes": hashes,
            "source_material_hash_metadata_only": raw_hash,
            "arms": list(M.ARMS),
            "metrics": list(M.METRICS),
            "expected_authors": 276,
            "expected_comments": 9165,
            "expected_panel_comparison_metric_rows": 23184,
            "seed": SEED,
            "bootstraps": BOOTSTRAPS,
            "permutations": PERMUTATIONS,
            "max_cpu_wall_seconds": 300,
            "new_fits_model_calls_labels_raw": [0, 0, 0, 0],
            "reference_scope": "new units only; inherit old frozen coordinates/maps",
            "panels_are_not_psychological_conditions": True,
        },
    )
    print({"status": "PREPARED_HM64", "new_calls": 0}, flush=True)


def sealed(out):
    """Check exact allowlisted code/numeric inputs without opening raw texts."""
    manifest = N.read(out / "manifest.json")
    for path, expected in manifest["hashes"].items():
        if N.sha(ROOT / path) != expected:
            raise ValueError("HM64 sealed input changed: " + path)
    N.binding()
    return manifest


def load_numeric():
    """Reuse address-checked numeric loading, suppressing known index warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        supports, replay = N.load_support()
    for item in supports.values():
        if len(set(item["folds"])) != 1 or item["folds"][0] not in range(5):
            raise ValueError("All queries/panels must share one original author fold")
    with np.load(N.PRIOR / "private_local_maps.npz", allow_pickle=False) as z:
        maps = {key: z[key] for key in z.files if key.endswith("__coef")}
    return supports, maps, replay


def cohort(item):
    """Identify separate cached query cohorts, never join external labels."""
    return "big5_only" if item["queries"] == list(N.QUERIES[4:]) else "mbti_only"


def statistics_for_fold(supports, maps, group, fold):
    """Use one frozen map per query/fold on ALL source and held authors."""
    keys = [key for key, item in supports.items() if cohort(item) == group]
    queries = supports[keys[0]]["queries"]
    weights = np.column_stack([maps[f"{q}__{fold}__coef"] for q in queries])
    panels, folds = [], []
    for key in keys:
        item = supports[key]
        y = item["h"] @ weights
        for panel in range(3):
            indexes = np.flatnonzero(item["panel"] == panel)
            if not 7 <= len(indexes) <= 12:
                raise ValueError("Incomplete panel support")
            panels.append(M.panel_statistics(item["g"][indexes], y[indexes]))
        folds.append(item["folds"][0])
    return keys, np.array(folds), M.stack_statistics(panels)


def select_statistics(stats, indexes):
    """Select whole panel records using an explicit author-derived mask."""
    return {key: values[indexes] for key, values in stats.items()}


def _save_npz(path, arrays):
    """Persist generated private numerical arrays, never source text."""
    np.savez_compressed(path, **arrays)


def run(out):
    """Execute the frozen geometry study once; no winner selection or tuning."""
    manifest = sealed(out)
    if (out / "checkpoint.json").exists():
        raise ValueError("Existing attempt retained; no automatic rerun")
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM64", "stage": "source"})
    started = time.monotonic()
    try:
        _run(out, manifest, started)
    except Exception as error:
        N.write(
            out / "checkpoint.json",
            {"status": "FAILED_HM64_RETAINED", "error": repr(error)},
        )
        raise


def _run(out, manifest, started):
    """Freeze every source unit before scoring any held-out author pairing."""
    supports, maps, replay = load_numeric()
    units, stored, cases, diagnostics = {}, {}, {}, []
    for group, fold in itertools.product(GROUPS, range(5)):
        key = f"{group}_{fold}"
        authors, folds, stats = statistics_for_fold(supports, maps, group, fold)
        mask = np.repeat(folds != fold, 3)
        held_mask = np.repeat(folds == fold, 3)
        ref = M.fit_reference(select_statistics(stats, mask))
        for name, value in ref.items():
            units[f"{key}__{name}"] = value
        stored[f"{key}__authors"] = np.array(authors)
        stored[f"{key}__folds"] = folds
        for name, value in stats.items():
            stored[f"{key}__{name}"] = value
        units[f"{key}__reference_authors"] = np.array(authors)[folds != fold]
        units[f"{key}__held_authors"] = np.array(authors)[folds == fold]
        cases[key] = (group, fold, authors, folds, stats, held_mask, ref)
        diagnostics.append(
            {
                "group": group,
                "fold": fold,
                "reference_authors": int(sum(folds != fold)),
                "held_authors": int(sum(folds == fold)),
                "condition_g": float(ref["eigen_g"][-1] / ref["eigen_g"][0]),
                "condition_y": float(ref["eigen_y"][-1] / ref["eigen_y"][0]),
                "neutral_blocks": int(ref["neutral_block"].sum()),
            }
        )
    _save_npz(out / "private_reference_units.npz", units)
    _save_npz(out / "private_panel_statistics.npz", stored)
    source_hash = N.sha(out / "private_reference_units.npz")
    N.write(
        out / "source_freeze.json",
        {"units_sha256": source_hash, "held_distances_started": False},
    )
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM64", "stage": "application"})
    comparisons, profiles_saved, records, nulls = {}, {}, [], []
    null_sums = {group: np.zeros((PERMUTATIONS, len(M.ARMS), 2)) for group in GROUPS}
    for key, (group, fold, authors, folds, stats, held_mask, ref) in cases.items():
        if time.monotonic() - started > manifest["max_cpu_wall_seconds"]:
            raise TimeoutError("Finite CPU budget reached; no automatic continuation")
        ids = np.array(authors)[folds == fold]
        n = len(ids)
        p = M.profiles(select_statistics(stats, held_mask), ref)
        for arm, values in p.items():
            profiles_saved[f"{key}__{arm}"] = values.reshape(n, 3, -1)
        rng = np.random.default_rng(SEED + fold + 100 * GROUPS.index(group))
        permutations = np.stack([rng.permutation(n) for _ in range(PERMUTATIONS)])
        comparisons[f"{key}__permutations"] = permutations
        for left, right in PAIRS:
            values = {arm: profiles_saved[f"{key}__{arm}"] for arm in M.ARMS}
            locations = M.squared_distances(
                values["joint_location"][:, left], values["joint_location"][:, right]
            )
            for a, arm in enumerate(M.ARMS):
                distance = M.squared_distances(
                    values[arm][:, left], values[arm][:, right]
                )
                prefix = f"{key}__{left}{right}__{arm}"
                comparisons[prefix + "__distance"] = distance
                wins = M.candidate_wins(distance, locations)
                for m, (metric, score) in enumerate(zip(M.METRICS, wins, strict=True)):
                    comparisons[prefix + "__" + metric] = score
                    null_sums[group][:, a, m] += score[
                        np.arange(n)[None, :], permutations
                    ].sum(1)
                    for i, author in enumerate(ids):
                        records.append(
                            {
                                "group": group,
                                "fold": fold,
                                "author": author,
                                "left": left,
                                "right": right,
                                "arm": arm,
                                "metric": metric,
                                "win_fraction": float(score[i, i]),
                                "held_strangers": n - 1,
                                "hard_strangers": 3,
                            }
                        )
        print({"stage": "held_pairing", "group": group, "fold": fold}, flush=True)
    _save_npz(out / "private_profiles.npz", profiles_saved)
    _save_npz(out / "private_comparisons.npz", comparisons)
    frame = pd.DataFrame(records)
    if len(frame) != manifest["expected_panel_comparison_metric_rows"]:
        raise ValueError("Incomplete directed comparisons")
    frame.to_csv(out / "private_panel_comparisons.csv", index=False)
    author = (
        frame.groupby(["group", "fold", "author", "arm", "metric"], sort=True)
        .win_fraction.agg(["mean", "size"])
        .reset_index()
    )
    if author["size"].ne(6).any():
        raise ValueError("All six dependent panel comparisons required")
    author.to_csv(out / "private_author_summaries.csv", index=False)
    summaries, contrasts = [], []
    for group in GROUPS:
        selected = author[author.group == group]
        ids = sorted(selected.author.unique())
        columns = [(arm, metric) for arm in M.ARMS for metric in M.METRICS]
        wide = selected.pivot(index="author", columns=["arm", "metric"], values="mean")
        matrix = wide.loc[ids, columns].to_numpy()
        mean, ci, boot = M.bootstrap_summary(matrix, SEED, BOOTSTRAPS)
        for c, (arm, metric) in enumerate(columns):
            summaries.append(
                {
                    "group": group,
                    "authors": len(ids),
                    "arm": arm,
                    "metric": metric,
                    "mean_win_fraction": float(mean[c]),
                    "author_bootstrap_lower": float(ci[0, c]),
                    "author_bootstrap_upper": float(ci[1, c]),
                }
            )
        for first, second in CONTRASTS:
            for metric in M.METRICS:
                a, b = columns.index((first, metric)), columns.index((second, metric))
                bounds = np.quantile(boot[:, a] - boot[:, b], [0.025, 0.975])
                contrasts.append(
                    {
                        "group": group,
                        "metric": metric,
                        "first": first,
                        "second": second,
                        "delta": float(mean[a] - mean[b]),
                        "lower": float(bounds[0]),
                        "upper": float(bounds[1]),
                    }
                )
        averages = null_sums[group] / (6 * len(ids))
        for rep, a, m in itertools.product(
            range(PERMUTATIONS), range(len(M.ARMS)), range(2)
        ):
            nulls.append(
                {
                    "group": group,
                    "replicate": rep,
                    "arm": M.ARMS[a],
                    "metric": M.METRICS[m],
                    "mean_win_fraction": float(averages[rep, a, m]),
                }
            )
    pd.DataFrame(summaries).to_csv(out / "summary_by_cohort.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    pd.DataFrame(nulls).to_csv(out / "identity_permutation_null.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "source_diagnostics.csv", index=False)
    elapsed = time.monotonic() - started
    N.write(
        out / "summary.json",
        {
            "status": "FINITE_HM64_COMPLETED_NOT_GLOBAL",
            "authors_comments": [
                len(supports),
                sum(len(s["g"]) for s in supports.values()),
            ],
            "comparison_rows_author_endpoint_rows": [len(frame), len(author)],
            "CPU_wall_seconds": elapsed,
            "old_observation_replay_error": replay,
            "new_calls_fits_labels_raw": [0, 0, 0, 0],
            "units_sha256": source_hash,
            "source_freeze_before_held_distances": True,
            "bootstrap_scope": "authors; conditional on fixed inherited features/maps and source units",
            "hard_null_not_assumed_half": True,
            "original_scope_complete": False,
        },
    )
    files = [
        p for p in out.iterdir() if p.name not in {"manifest.json", "checkpoint.json"}
    ]
    N.write(out / "output_seal.json", {p.name: N.sha(p) for p in files})
    N.write(out / "checkpoint.json", {"status": "COMPLETED_HM64", "stage": "verify"})
    print({"status": "COMPLETED_HM64", "seconds": elapsed}, flush=True)


def verify(out):
    """Recompute raw cross-products and trace norms, not just file existence."""
    sealed(out)
    for name, expected in N.read(out / "output_seal.json").items():
        if N.sha(out / name) != expected:
            raise ValueError("Frozen HM64 output changed: " + name)
    supports, maps, _ = load_numeric()
    errors = {
        "raw_statistics": 0.0,
        "source_reference": 0.0,
        "trace_distances": 0.0,
        "coordinate_rotation": 0.0,
        "author_wins": 0.0,
    }
    total_distances, disjoint_sets = 0, 0
    with (
        np.load(out / "private_reference_units.npz", allow_pickle=False) as units,
        np.load(out / "private_panel_statistics.npz", allow_pickle=False) as stats,
        np.load(out / "private_comparisons.npz", allow_pickle=False) as comparisons,
    ):
        for group, fold in itertools.product(GROUPS, range(5)):
            key = f"{group}_{fold}"
            ids, folds = stats[f"{key}__authors"], stats[f"{key}__folds"]
            held = ids[folds == fold]
            source = ids[folds != fold]
            if set(held) & set(source) or len(held) + len(source) != len(ids):
                raise ValueError("Reference/held author leak")
            np.testing.assert_array_equal(units[f"{key}__held_authors"], held)
            np.testing.assert_array_equal(units[f"{key}__reference_authors"], source)
            disjoint_sets += 1
            qs = supports[str(ids[0])]["queries"]
            weights = np.column_stack([maps[f"{q}__{fold}__coef"] for q in qs])
            for a, author_id in enumerate(ids):
                item = supports[str(author_id)]
                y = item["h"] @ weights
                for panel in range(3):
                    mask = item["panel"] == panel
                    g, q = item["g"][mask], y[mask]
                    n = len(g)
                    mg, mq = np.sum(g, 0) / n, np.sum(q, 0) / n
                    expected = {
                        "mu_g": mg,
                        "mu_y": mq,
                        "second_g": np.einsum("ni,nj->ij", g, g) / n,
                        "second_y": np.einsum("ni,nj->ij", q, q) / n,
                        "cross": np.einsum("ni,nj->ij", g, q) / n - np.outer(mg, mq),
                        "shifted": np.einsum("ni,nj->ij", g, np.roll(q, n // 2, 0)) / n
                        - np.outer(mg, mq),
                    }
                    for name, values in expected.items():
                        error = np.max(
                            abs(values - stats[f"{key}__{name}"][3 * a + panel])
                        )
                        errors["raw_statistics"] = max(
                            errors["raw_statistics"], float(error)
                        )
            sg, sy = units[f"{key}__cov_g"], units[f"{key}__cov_y"]
            source_mask = np.repeat(folds != fold, 3)
            source_g = stats[f"{key}__mu_g"][source_mask].mean(0)
            source_y = stats[f"{key}__mu_y"][source_mask].mean(0)
            expected_units = {
                "mu_g": source_g,
                "mu_y": source_y,
                "cov_g": stats[f"{key}__second_g"][source_mask].mean(0)
                - np.outer(source_g, source_g),
                "cov_y": stats[f"{key}__second_y"][source_mask].mean(0)
                - np.outer(source_y, source_y),
                "cross": stats[f"{key}__cross"][source_mask].mean(0),
                "shifted": stats[f"{key}__shifted"][source_mask].mean(0),
            }
            for name, values in expected_units.items():
                error = float(np.max(abs(values - units[f"{key}__{name}"])))
                errors["source_reference"] = max(errors["source_reference"], error)
            ig, iy = np.linalg.inv(sg), np.linalg.inv(sy)
            scale = units[f"{key}__scales"]
            selection = np.repeat(folds == fold, 3)
            raw = {
                name: stats[f"{key}__{name}"][selection].reshape(
                    len(held), 3, *stats[f"{key}__{name}"].shape[1:]
                )
                for name in ("mu_g", "mu_y", "cross", "shifted")
            }
            rotation_g = N.M.ROTATION
            rotation_y = np.eye(len(sy)) - 2 * np.ones_like(sy) / len(sy)
            ri_g = np.linalg.inv(rotation_g @ sg @ rotation_g.T)
            ri_y = np.linalg.inv(rotation_y @ sy @ rotation_y.T)
            perms = comparisons[f"{key}__permutations"]
            if perms.shape != (PERMUTATIONS, len(held)) or not np.all(
                np.sort(perms, axis=1) == np.arange(len(held))
            ):
                raise ValueError("Invalid identity permutations")
            for left, right in PAIRS:
                gd = raw["mu_g"][:, left, None] - raw["mu_g"][None, :, right]
                qd = raw["mu_y"][:, left, None] - raw["mu_y"][None, :, right]
                dg = np.einsum("ab,ija,ijb->ij", ig, gd, gd) / scale[0] ** 2
                dq = np.einsum("ab,ija,ijb->ij", iy, qd, qd) / scale[1] ** 2
                distances = {
                    "G_location": dg,
                    "Q_location": dq,
                    "joint_location": (dg + dq) / 2,
                }
                for name, arm, index in (
                    ("cross", "coupling", 2),
                    ("shifted", "shifted_coupling", 3),
                ):
                    difference = raw[name][:, left, None] - raw[name][None, :, right]
                    value = np.einsum(
                        "ac,bd,ijab,ijcd->ij", ig, iy, difference, difference
                    )
                    distances[arm] = value / scale[index] ** 2
                    moved = np.einsum(
                        "ab,ijbc,cd->ijad", rotation_g, difference, rotation_y.T
                    )
                    rotated = np.einsum("ac,bd,ijab,ijcd->ij", ri_g, ri_y, moved, moved)
                    errors["coordinate_rotation"] = max(
                        errors["coordinate_rotation"],
                        float(np.max(abs(value - rotated))),
                    )
                distances["joint_with_coupling"] = (dg + dq + distances["coupling"]) / 3
                distances["joint_with_shifted"] = (
                    dg + dq + distances["shifted_coupling"]
                ) / 3
                for arm, value in distances.items():
                    frozen = comparisons[f"{key}__{left}{right}__{arm}__distance"]
                    errors["trace_distances"] = max(
                        errors["trace_distances"], float(np.max(abs(value - frozen)))
                    )
                    total_distances += value.size
                    for i in range(len(held)):
                        eligible = np.delete(np.arange(len(held)), i)
                        hard = sorted(
                            eligible,
                            key=lambda j: (distances["joint_location"][i, j], j),
                        )[:3]
                        for metric, indexes in zip(
                            M.METRICS, (eligible, hard), strict=True
                        ):
                            delta = frozen[i, indexes] - frozen[i, i]
                            actual = (
                                np.count_nonzero(delta > 1e-12)
                                + 0.5 * np.count_nonzero(abs(delta) <= 1e-12)
                            ) / len(indexes)
                            saved = comparisons[
                                f"{key}__{left}{right}__{arm}__{metric}"
                            ][i, i]
                            errors["author_wins"] = max(
                                errors["author_wins"], float(abs(actual - saved))
                            )
    if max(errors.values()) > 1e-7:
        raise ArithmeticError(
            "HM64 independent numerical replay failed: " + json.dumps(errors)
        )
    frame = pd.read_csv(out / "private_panel_comparisons.csv")
    author = pd.read_csv(out / "private_author_summaries.csv")
    if len(frame) != 23184 or len(author) != 3864 or author["size"].ne(6).any():
        raise ValueError("Incomplete author-centered analysis")
    aggregates = pd.read_csv(out / "summary_by_cohort.csv")
    for row in aggregates.itertuples():
        subset = author[
            (author.group == row.group)
            & (author.arm == row.arm)
            & (author.metric == row.metric)
        ]
        if (
            len(subset) != row.authors
            or abs(subset["mean"].mean() - row.mean_win_fraction) > 1e-12
        ):
            raise ValueError("Aggregate author weighting mismatch")
    null = pd.read_csv(out / "identity_permutation_null.csv")
    if (
        len(null) != 5572
        or null.groupby(["group", "arm", "metric"]).size().ne(PERMUTATIONS).any()
    ):
        raise ValueError("Incomplete finite permutation null")
    N.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM64_NOT_GLOBAL",
            "errors": errors,
            "raw_trace_distance_entries": total_distances,
            "disjoint_reference_checks": disjoint_sets,
            "source_units_hash_unchanged": N.sha(out / "private_reference_units.npz")
            == N.read(out / "source_freeze.json")["units_sha256"],
            "same_agent_alternate_arithmetic_not_independent_replication": True,
        },
    )
    print(N.read(out / "verification.json"), flush=True)


def main():
    """Expose explicit finite stages; refuse scientific result overwrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
