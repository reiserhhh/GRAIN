#!/usr/bin/env python3
"""Freeze and distinguish marginal, common-link and residual author mechanisms."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_coupling as C
from grain_research.construction import math_relation_origin as M
from scripts import run_grain_math_author_coupling as H
from scripts import run_grain_math_natural_summary as N

OUT = ROOT / "results/grain_math_response_v1/author_relation_origin_r1"
PLAN = ROOT / "docs/GRAIN_AUTHOR_RELATION_ORIGIN_PLAN_V1.md"
SEEDS = (20261009, 20261010, 20261011)
CONTRASTS = (
    ("residual_link", "shifted_residual_link"),
    ("shared_link", "shared_shift_link"),
    ("residual_link", "coupling"),
    ("residual_link", "shared_link"),
    ("shared_link", "G_shape"),
)
CODE = (
    PLAN,
    Path(__file__).resolve(),
    ROOT / "grain_research/construction/math_relation_origin.py",
    ROOT / "tests/test_grain_math_relation_origin.py",
    ROOT / "grain_research/construction/math_author_coupling.py",
    ROOT / "grain_research/construction/math_natural_summary.py",
    ROOT / "scripts/run_grain_math_author_coupling.py",
    ROOT / "scripts/run_grain_math_natural_summary.py",
)


def prepare(out):
    """Seal code and terminal HM64 numerical outputs before new outcomes."""
    H.sealed(H.OUT)
    old = N.read(H.OUT / "output_seal.json")
    hashes = {str(path.relative_to(ROOT)): N.sha(path) for path in CODE}
    for name in (
        "private_panel_statistics.npz",
        "private_reference_units.npz",
        "private_comparisons.npz",
    ):
        if N.sha(H.OUT / name) != old[name]:
            raise ValueError("HM64 frozen output changed: " + name)
        hashes[str((H.OUT / name).relative_to(ROOT))] = old[name]
    out.mkdir(parents=True, exist_ok=False)
    N.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_AUTHOR_RELATION_ORIGIN_V1",
            "hashes": hashes,
            "arms": list(M.ARMS),
            "metrics": list(M.METRICS),
            "contrasts": CONTRASTS,
            "synthetic_worlds": list(M.WORLDS),
            "seeds": SEEDS,
            "expected_authors_comments": [276, 9165],
            "expected_natural_panel_rows": 49680,
            "expected_synthetic_world_seed_cases": 12,
            "bootstraps": H.BOOTSTRAPS,
            "permutations": H.PERMUTATIONS,
            "max_cpu_wall_seconds": 300,
            "new_neural_calls_training_labels_raw": [0, 0, 0, 0],
            "reference_links_are_calibrated_not_predictor_optimization": True,
            "prior_numerical_panels_not_raw_ID_reaudit": True,
        },
    )
    print({"status": "PREPARED_HM65", "new_model_calls": 0}, flush=True)


def sealed(out):
    """Read only explicitly allowlisted source/code assets and their seals."""
    manifest = N.read(out / "manifest.json")
    for path, expected in manifest["hashes"].items():
        if N.sha(ROOT / path) != expected:
            raise ValueError("HM65 frozen input changed: " + path)
    H.sealed(H.OUT)
    return manifest


def natural_cases():
    """Recover terminal HM64 statistics without reopening source comments."""
    cases = {}
    with np.load(H.OUT / "private_panel_statistics.npz", allow_pickle=False) as old:
        for group, fold in itertools.product(H.GROUPS, range(5)):
            key = f"{group}_{fold}"
            stats = {
                name: old[key + "__" + name].copy()
                for name in (
                    "mu_g",
                    "mu_y",
                    "second_g",
                    "second_y",
                    "cross",
                    "shifted",
                    "n",
                )
            }
            cases[key] = (
                group,
                fold,
                old[key + "__authors"].copy(),
                old[key + "__folds"].copy(),
                stats,
            )
    return cases


def evaluate_case(stats, ref, ids, key, group, fold, saved, permutations):
    """Compute actual pairing wins and null nominations under fixed negatives."""
    n = len(ids)
    arrays = {arm: p.reshape(n, 3, -1) for arm, p in M.profiles(stats, ref).items()}
    records = []
    null_sum = np.zeros((len(permutations), len(M.ARMS), len(M.METRICS)))
    for left, right in H.PAIRS:
        locations = C.squared_distances(
            arrays["joint_location"][:, left], arrays["joint_location"][:, right]
        )
        shapes = C.squared_distances(
            arrays["joint_shape"][:, left], arrays["joint_shape"][:, right]
        )
        for a, arm in enumerate(M.ARMS):
            distance = C.squared_distances(arrays[arm][:, left], arrays[arm][:, right])
            prefix = f"{key}__{left}{right}__{arm}"
            saved[prefix + "__distance"] = distance
            for m, (metric, scores) in enumerate(
                zip(
                    M.METRICS,
                    M.candidate_wins(distance, locations, shapes),
                    strict=True,
                )
            ):
                saved[prefix + "__" + metric] = scores
                if len(permutations):
                    null_sum[:, a, m] += scores[
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
                            "win_fraction": float(scores[i, i]),
                        }
                    )
    return records, null_sum


def author_summaries(frame):
    """Average all six dependent directed comparisons before author analysis."""
    author = (
        frame.groupby(["group", "fold", "author", "arm", "metric"], sort=True)
        .win_fraction.agg(["mean", "size"])
        .reset_index()
    )
    if author["size"].ne(6).any():
        raise ValueError("Missing directed panel comparisons")
    return author


def run(out):
    """Single finite attempt, with failed results/checkpoint retained."""
    manifest = sealed(out)
    if (out / "checkpoint.json").exists():
        raise ValueError("Existing scientific attempt retained; no overwrite/retry")
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM65", "stage": "source"})
    start = time.monotonic()
    try:
        _run(out, manifest, start)
    except Exception as error:
        N.write(
            out / "checkpoint.json",
            {"status": "FAILED_HM65_RETAINED", "error": repr(error)},
        )
        raise


def _run(out, manifest, start):
    """Freeze every natural/synthetic source link before any application score."""
    cases = natural_cases()
    refs, units, diagnostics, energy_rows = {}, {}, [], []
    for key, (group, fold, ids, folds, stats) in cases.items():
        source = H.select_statistics(stats, np.repeat(folds != fold, 3))
        ref = M.fit_reference(source)
        refs[key] = ref
        units.update({key + "__" + name: value for name, value in ref.items()})
        units[key + "__reference_authors"] = ids[folds != fold]
        units[key + "__held_authors"] = ids[folds == fold]
        diagnostics.append(
            {
                "group": group,
                "fold": fold,
                "reference_authors": int(sum(folds != fold)),
                "held_authors": int(sum(folds == fold)),
                "condition_within_g": float(
                    ref["within_g_eigen"][-1] / ref["within_g_eigen"][0]
                ),
                "neutral_blocks": sum(
                    bool(ref[name + "_neutral"]) for name in M.BLOCKS
                ),
            }
        )
        energy = M.centered_energy(stats, ref)
        for i, row in enumerate(energy):
            energy_rows.append(
                {
                    "group": group,
                    "fold": fold,
                    "author": ids[i // 3],
                    "panel": i % 3,
                    "role": "held" if folds[i // 3] == fold else "source",
                    "total": row[0],
                    "shared": row[1],
                    "residual": row[2],
                    "signed_cross": row[3],
                }
            )
    synthetic_material, synthetic = {}, {}
    for world, seed in itertools.product(M.WORLDS, SEEDS):
        key = f"toy_{world}_{seed}"
        material, stats = M.synthetic_world(world, seed)
        synthetic_material.update(
            {key + "__" + name: value for name, value in material.items()}
        )
        ref = M.fit_reference(H.select_statistics(stats, slice(0, 60)))
        synthetic[key] = (world, seed, stats, ref)
        units.update({key + "__" + name: value for name, value in ref.items()})
    np.savez_compressed(out / "private_reference_units.npz", **units)
    np.savez_compressed(out / "synthetic_material.npz", **synthetic_material)
    source_hash = N.sha(out / "private_reference_units.npz")
    N.write(
        out / "source_freeze.json",
        {
            "units_sha256": source_hash,
            "held_distances_started": False,
            "natural_reference_cases": len(cases),
            "synthetic_reference_cases": len(synthetic),
            "common_link_calibrations": 2 * (len(cases) + len(synthetic)),
        },
    )
    N.write(out / "checkpoint.json", {"status": "RUNNING_HM65", "stage": "application"})
    comparisons, records = {}, []
    null_sums = {
        group: np.zeros((H.PERMUTATIONS, len(M.ARMS), len(M.METRICS)))
        for group in H.GROUPS
    }
    for key, (group, fold, ids, folds, stats) in cases.items():
        if time.monotonic() - start > manifest["max_cpu_wall_seconds"]:
            raise TimeoutError("Finite CPU deadline; preserve partial work")
        held = ids[folds == fold]
        rng = np.random.default_rng(H.SEED + fold + 100 * H.GROUPS.index(group))
        perms = np.stack([rng.permutation(len(held)) for _ in range(H.PERMUTATIONS)])
        comparisons[key + "__permutations"] = perms
        batch, null = evaluate_case(
            H.select_statistics(stats, np.repeat(folds == fold, 3)),
            refs[key],
            held,
            key,
            group,
            fold,
            comparisons,
            perms,
        )
        records.extend(batch)
        null_sums[group] += null
        print({"stage": "natural", "group": group, "fold": fold}, flush=True)
    np.savez_compressed(out / "private_comparisons.npz", **comparisons)
    frame = pd.DataFrame(records)
    if len(frame) != manifest["expected_natural_panel_rows"]:
        raise ValueError("Incomplete natural comparison scope")
    frame.to_csv(out / "private_panel_comparisons.csv", index=False)
    author = author_summaries(frame)
    author.to_csv(out / "private_author_summaries.csv", index=False)
    summaries, contrasts, null_rows = [], [], []
    columns = [(arm, metric) for arm in M.ARMS for metric in M.METRICS]
    for group in H.GROUPS:
        selected = author[author.group == group]
        wide = selected.pivot(index="author", columns=["arm", "metric"], values="mean")
        matrix = wide.loc[sorted(wide.index), columns].to_numpy()
        mean, ci, bootstrap = C.bootstrap_summary(matrix, H.SEED, H.BOOTSTRAPS)
        for c, (arm, metric) in enumerate(columns):
            summaries.append(
                {
                    "group": group,
                    "authors": len(wide),
                    "arm": arm,
                    "metric": metric,
                    "mean_win_fraction": mean[c],
                    "lower": ci[0, c],
                    "upper": ci[1, c],
                }
            )
        for first, second in CONTRASTS:
            for metric in M.METRICS:
                a, b = columns.index((first, metric)), columns.index((second, metric))
                bounds = np.quantile(bootstrap[:, a] - bootstrap[:, b], [0.025, 0.975])
                contrasts.append(
                    {
                        "group": group,
                        "metric": metric,
                        "first": first,
                        "second": second,
                        "delta": mean[a] - mean[b],
                        "lower": bounds[0],
                        "upper": bounds[1],
                    }
                )
        values = null_sums[group] / (6 * len(wide))
        for rep, a, m in itertools.product(
            range(H.PERMUTATIONS), range(len(M.ARMS)), range(len(M.METRICS))
        ):
            null_rows.append(
                {
                    "group": group,
                    "replicate": rep,
                    "arm": M.ARMS[a],
                    "metric": M.METRICS[m],
                    "mean_win_fraction": values[rep, a, m],
                }
            )
    pd.DataFrame(summaries).to_csv(out / "summary_by_cohort.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "identity_permutation_null.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "source_diagnostics.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(
        out / "private_energy_decomposition.csv", index=False
    )
    toy_records, toy_comparisons = [], {}
    for key, (world, seed, stats, ref) in synthetic.items():
        batch, _ = evaluate_case(
            H.select_statistics(stats, slice(60, None)),
            ref,
            np.array([f"a{i:03d}" for i in range(20, 40)]),
            key,
            world,
            seed,
            toy_comparisons,
            np.empty((0, 20), int),
        )
        toy_records.extend(batch)
    np.savez_compressed(out / "synthetic_comparisons.npz", **toy_comparisons)
    toy_author = author_summaries(pd.DataFrame(toy_records))
    toy_author.to_csv(out / "synthetic_author_summaries.csv", index=False)
    toy_author.groupby(["group", "fold", "arm", "metric"], sort=True)[
        "mean"
    ].mean().reset_index().to_csv(out / "synthetic_summary.csv", index=False)
    N.write(
        out / "summary.json",
        {
            "status": "FINITE_HM65_COMPLETED_NOT_GLOBAL",
            "natural_authors_comments": [276, 9165],
            "natural_panel_author_rows": [len(frame), len(author)],
            "synthetic_world_seed_cases": len(synthetic),
            "synthetic_author_instances": 480,
            "synthetic_application_author_instances": 240,
            "natural_source_links": 20,
            "synthetic_source_links": 24,
            "CPU_wall_seconds": time.monotonic() - start,
            "new_neural_calls_training_labels_raw": [0, 0, 0, 0],
            "source_units_sha256": source_hash,
            "source_frozen_before_application": True,
            "original_scope_complete": False,
        },
    )
    files = [
        p for p in out.iterdir() if p.name not in {"manifest.json", "checkpoint.json"}
    ]
    N.write(out / "output_seal.json", {p.name: N.sha(p) for p in files})
    N.write(out / "checkpoint.json", {"status": "COMPLETED_HM65", "stage": "verify"})
    print(N.read(out / "summary.json"), flush=True)


def direct_distances(stats, reference):
    """Use trace/inverse algebra independently of whitened profile distances."""
    n = len(stats["mu_g"]) // 3
    matrices = M.components(stats, reference)
    matrices.update({"coupling": stats["cross"], "shifted_coupling": stats["shifted"]})
    ig = np.linalg.inv(reference["base_cov_g"])
    iy = np.linalg.inv(reference["base_cov_y"])
    result = {}
    for left, right in H.PAIRS:
        mg, my = stats["mu_g"].reshape(n, 3, -1), stats["mu_y"].reshape(n, 3, -1)
        dg = mg[:, left, None] - mg[None, :, right]
        dy = my[:, left, None] - my[None, :, right]
        location = (
            np.einsum("ab,ija,ijb->ij", ig, dg, dg) / reference["base_scales"][0] ** 2
            + np.einsum("ab,ija,ijb->ij", iy, dy, dy) / reference["base_scales"][1] ** 2
        ) / 2
        distances = {"joint_location": location}
        for name, block in matrices.items():
            block = block.reshape(n, 3, *block.shape[1:])
            difference = block[:, left, None] - block[None, :, right]
            first = iy if name == "Q_shape" else ig
            second = ig if name == "G_shape" else iy
            if name in {"coupling", "shifted_coupling"}:
                scale = reference["base_scales"][2 if name == "coupling" else 3]
            else:
                scale = reference[name + "_scale"]
            distances[name] = (
                np.einsum("ac,bd,ijab,ijcd->ij", first, second, difference, difference)
                / scale**2
            )
        distances["joint_shape"] = (distances["G_shape"] + distances["Q_shape"]) / 2
        result[left, right] = distances
    return result


def verify(out):
    """Replay raw-row least squares/residuals, trace distances and controls."""
    sealed(out)
    for name, expected in N.read(out / "output_seal.json").items():
        if N.sha(out / name) != expected:
            raise ValueError("HM65 result changed: " + name)
    supports, maps, _ = H.load_numeric()
    errors = {
        "weighted_row_link": 0.0,
        "raw_residual_covariance": 0.0,
        "signed_energy": 0.0,
        "trace_distances": 0.0,
        "HM64_replay": 0.0,
        "author_wins": 0.0,
        "synthetic_replay": 0.0,
    }
    count, isolated = 0, 0
    with (
        np.load(out / "private_reference_units.npz", allow_pickle=False) as units,
        np.load(out / "private_comparisons.npz", allow_pickle=False) as saved,
        np.load(H.OUT / "private_comparisons.npz", allow_pickle=False) as old,
    ):
        for key, (group, fold, ids, folds, statistics) in natural_cases().items():
            ref = {
                name[len(key) + 2 :]: units[name]
                for name in units.files
                if name.startswith(key + "__")
                and not name.endswith(("__reference_authors", "__held_authors"))
            }
            if set(ids[folds == fold]) & set(ids[folds != fold]):
                raise ValueError("Held/source author overlap")
            np.testing.assert_array_equal(
                units[key + "__reference_authors"], ids[folds != fold]
            )
            np.testing.assert_array_equal(
                units[key + "__held_authors"], ids[folds == fold]
            )
            isolated += 1
            query_names = supports[str(ids[0])]["queries"]
            weights = np.column_stack([maps[f"{q}__{fold}__coef"] for q in query_names])
            source_g, source_y, source_shift = [], [], []
            blocks = M.components(statistics, ref)
            for a, author in enumerate(ids):
                item = supports[str(author)]
                y = item["h"] @ weights
                for panel in range(3):
                    indexes = item["panel"] == panel
                    g, q = item["g"][indexes], y[indexes]
                    gc, qc = g - g.mean(0), q - q.mean(0)
                    if folds[a] != fold:
                        source_g.append(gc / np.sqrt(len(g)))
                        source_y.append(qc / np.sqrt(len(g)))
                        source_shift.append(
                            np.roll(qc, len(q) // 2, axis=0) / np.sqrt(len(g))
                        )
                    for name, link, values in (
                        ("residual_link", ref["link"], q),
                        (
                            "shifted_residual_link",
                            ref["shift_link"],
                            np.roll(q, len(q) // 2, axis=0),
                        ),
                    ):
                        raw = values - g @ link
                        expected = np.einsum("ni,nj->ij", gc, raw - raw.mean(0)) / len(
                            g
                        )
                        errors["raw_residual_covariance"] = max(
                            errors["raw_residual_covariance"],
                            float(np.max(abs(expected - blocks[name][3 * a + panel]))),
                        )
            x = np.vstack(source_g)
            for link, ys in (("link", source_y), ("shift_link", source_shift)):
                alternative = np.linalg.lstsq(x, np.vstack(ys), rcond=None)[0]
                errors["weighted_row_link"] = max(
                    errors["weighted_row_link"],
                    float(np.max(abs(alternative - ref[link]))),
                )
            energy = M.centered_energy(statistics, ref)
            errors["signed_energy"] = max(
                errors["signed_energy"],
                float(np.max(abs(energy[:, 0] - energy[:, 1:].sum(1)))),
            )
            held_stats = H.select_statistics(statistics, np.repeat(folds == fold, 3))
            trace = direct_distances(held_stats, ref)
            n = sum(folds == fold)
            for (left, right), distances in trace.items():
                for arm, distance in distances.items():
                    prefix = f"{key}__{left}{right}__{arm}"
                    errors["trace_distances"] = max(
                        errors["trace_distances"],
                        float(np.max(abs(distance - saved[prefix + "__distance"]))),
                    )
                    count += distance.size
                    if arm in {"joint_location", "coupling", "shifted_coupling"}:
                        errors["HM64_replay"] = max(
                            errors["HM64_replay"],
                            float(np.max(abs(distance - old[prefix + "__distance"]))),
                        )
                    for i in range(n):
                        eligible = np.delete(np.arange(n), i)
                        loc = distances["joint_location"][i]
                        both = (loc + distances["joint_shape"][i]) / 2
                        sets = (
                            eligible,
                            sorted(eligible, key=lambda j: (loc[j], j))[:3],
                            sorted(eligible, key=lambda j: (both[j], j))[:3],
                        )
                        for metric, negative in zip(M.METRICS, sets, strict=True):
                            delta = distance[i, negative] - distance[i, i]
                            win = (
                                np.count_nonzero(delta > 1e-12)
                                + 0.5 * np.count_nonzero(abs(delta) <= 1e-12)
                            ) / len(negative)
                            errors["author_wins"] = max(
                                errors["author_wins"],
                                abs(float(win) - saved[prefix + "__" + metric][i, i]),
                            )
        with (
            np.load(out / "synthetic_material.npz", allow_pickle=False) as material,
            np.load(out / "synthetic_comparisons.npz", allow_pickle=False) as saved_toy,
        ):
            for world, seed in itertools.product(M.WORLDS, SEEDS):
                key = f"toy_{world}_{seed}"
                stats = []
                for a, panel in itertools.product(range(40), range(3)):
                    prefix = key + f"__a{a:03d}_p{panel}"
                    stats.append(
                        C.panel_statistics(
                            material[prefix + "__g"], material[prefix + "__y"]
                        )
                    )
                ref = {
                    name[len(key) + 2 :]: units[name]
                    for name in units.files
                    if name.startswith(key + "__")
                }
                held = H.select_statistics(C.stack_statistics(stats), slice(60, None))
                for (left, right), distances in direct_distances(held, ref).items():
                    for arm, distance in distances.items():
                        difference = abs(
                            distance
                            - saved_toy[f"{key}__{left}{right}__{arm}__distance"]
                        )
                        errors["synthetic_replay"] = max(
                            errors["synthetic_replay"], float(np.max(difference))
                        )
    if max(errors.values()) > 1e-7:
        raise ArithmeticError("HM65 numerical replay failed: " + json.dumps(errors))
    summary = pd.read_csv(out / "summary_by_cohort.csv")
    author = pd.read_csv(out / "private_author_summaries.csv")
    if len(author) != 8280 or author["size"].ne(6).any():
        raise ValueError("Incomplete author-centered scope")
    for row in summary.itertuples():
        selected = author[
            (author.group == row.group)
            & (author.arm == row.arm)
            & (author.metric == row.metric)
        ]
        if (
            len(selected) != row.authors
            or abs(selected["mean"].mean() - row.mean_win_fraction) > 1e-12
        ):
            raise ValueError("Author weighting mismatch")
    null = pd.read_csv(out / "identity_permutation_null.csv")
    if (
        len(null) != 11940
        or null.groupby(["group", "arm", "metric"]).size().ne(H.PERMUTATIONS).any()
    ):
        raise ValueError("Incomplete permutation controls")
    N.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM65_NOT_GLOBAL",
            "errors": errors,
            "trace_distance_entries": count,
            "source_held_author_isolation_checks": isolated,
            "source_hash_unchanged": N.sha(out / "private_reference_units.npz")
            == N.read(out / "source_freeze.json")["units_sha256"],
            "same_agent_alternate_arithmetic_not_independent_replication": True,
        },
    )
    print(N.read(out / "verification.json"), flush=True)


def main():
    """Expose explicit freeze/run/verify without automatic experiment reruns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
