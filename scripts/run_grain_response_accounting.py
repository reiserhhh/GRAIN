#!/usr/bin/env python3
"""Use frozen semantic readers to separate time, composition and support changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.response_accounting import (
    PARTS,
    apply_operator,
    chronological_halves,
    comment_average,
    difference_operator,
    direct_difference,
    paired_energy_interval,
)
from grain_research.semantic_link import validate_addresses

BASE = ROOT / "results/grain_vector_mechanisms"
PRIOR = BASE / "all_plan_closure_v1/semantic_link_r2"
MATERIAL = BASE / "predecessor_integration_v1/material"
PLAN = ROOT / "docs/GRAIN_SEMANTIC_RESPONSE_ACCOUNTING_PLAN_V1.md"
READERS = ("teacher", "opportunity", "grain_linear", "joint_linear", "joint_quadratic")
RANDOM_SEEDS = tuple(range(914, 924))


def write(path, obj):
    """Write finite JSON atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temp.replace(path)


def correlation_columns(x, y):
    """Column Pearson correlation with explicit undefined constant columns."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.full(x.shape[1], np.nan)
    a, b = x - x.mean(0), y - y.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide(
        (a * b).sum(0), den, out=np.full(x.shape[1], np.nan), where=den > 1e-12
    )


def inherited_evidence():
    """Reopen two old result bundles; record support, not broad acceptance by title."""
    e2 = ROOT / "results/v7_condition_opportunity/e2_full_20260715_r7"
    w3 = (
        ROOT / "results/v7_temporal_geometry/w3_temporal_random_corrected_full_20260715"
    )
    manifest = json.loads((e2 / "run_manifest.json").read_text())
    condition = pd.read_csv(e2 / "condition_model_metrics.csv")
    selection = pd.read_csv(e2 / "selection_metrics.csv")
    partial = pd.read_csv(e2 / "matched_partial_metrics.csv")
    native = condition[
        (condition.operator == "native") & (condition.arm == "primary")
    ].iloc[0]
    native_partial = partial[
        (partial.operator == "native") & (partial.arm == "primary")
    ].iloc[0]
    inventory = json.loads((w3 / "artifact_inventory.json").read_text())
    for row in inventory["files"]:
        if digest(w3 / row["path"]) != row["sha256"]:
            raise ValueError("Old W3 artifact inventory does not verify")
    temporal = pd.read_csv(w3 / "temporal_random_contrast.csv")
    return {
        "E2": {
            "scope": "retained observed-selection and operator-alignment findings; not fixed-stimulus response",
            "selected_authors": manifest["n_selected_authors"],
            "context_r2": float(native.context_r2),
            "matched_auc": float(native_partial.auc),
            "matched_minimum_jaccard": float(native_partial.minimum_jaccard),
            "decision": manifest["screen_decision"],
            "selection_rows": selection.to_dict(orient="records"),
            "file_hashes": {
                n: digest(e2 / n)
                for n in (
                    "run_manifest.json",
                    "condition_model_metrics.csv",
                    "selection_metrics.csv",
                    "matched_partial_metrics.csv",
                )
            },
        },
        "W3": {
            "scope": "temporal versus random sampling, not verified personality change",
            "inventory_files_verified": len(inventory["files"]),
            "columns": temporal.columns.tolist(),
            "rows": temporal.to_dict(orient="records"),
            "inventory_sha256": digest(w3 / "artifact_inventory.json"),
        },
    }


def prepare(out):
    """Bind original comments and frozen teacher/student scores without refitting."""
    previous = json.loads((PRIOR / "summary.json").read_text())
    contract = json.loads((PRIOR / "contract.json").read_text())
    for name, checksum in previous["artifacts"].items():
        if digest(PRIOR / name) != checksum:
            raise ValueError("Changed C14 input artifact: " + name)
    manifest = json.loads((MATERIAL / "material_manifest.json").read_text())
    if (
        digest(MATERIAL / "private_material.json")
        != manifest["files"]["private_material.json"]
    ):
        raise ValueError("Changed original text panel")
    data = json.loads((MATERIAL / "private_material.json").read_text())
    comments = validate_addresses(data)
    if set(map(len, comments)) != {36}:
        raise ValueError("Complete original 36-comment panels required")
    n = len(comments)
    with np.load(PRIOR / "private_inputs.npz", allow_pickle=False) as z:
        inputs = {k: z[k] for k in ("g", "z", "subreddits", "authors", "target")}
    addresses = pd.read_csv(PRIOR / "private_addresses.csv")
    if not np.array_equal(addresses.author_index, inputs["authors"]):
        raise ValueError("Sentence/coordinate authors differ")
    for i, s in enumerate(data["sentences"]):
        if any(
            addresses.iloc[i][k] != s[k]
            for k in ("author_index", "comment_index", "comment_id", "start", "end")
        ):
            raise ValueError("C14 sentence address changed")
    ix = addresses.author_index.to_numpy(int) * 36 + addresses.comment_index.to_numpy(
        int
    )
    ref = joblib.load(PRIOR / "joint_linear.joblib").y_scaler_
    unit_values, counts = [], None
    for name in READERS:
        if name == "teacher":
            score = inputs["target"]
        else:
            reader = joblib.load(PRIOR / f"{name}.joblib")
            score = reader.predict(inputs["g"], inputs["z"], inputs["subreddits"])
        scores = (score - ref.mean_) / ref.scale_
        averaged, c = comment_average(scores, ix, n * 36)
        if counts is not None and not np.array_equal(c, counts):
            raise ValueError("Readers must share exactly the same supported comments")
        counts = c
        unit_values.append(averaged.reshape(n, 36, -1))
    panels = np.stack(unit_values, axis=1)
    counts = counts.reshape(n, 36)
    folds = np.asarray([b["official_folds"]["big5"] for b in data["bundles"]])
    conditions = np.asarray([[c["subreddit"] for c in row] for row in comments], str)
    times = np.asarray([[c["timestamp"] for c in row] for row in comments], float)
    ids = [[c["comment_id"] for c in row] for row in comments]
    if not np.isfinite(times).all():
        raise ValueError("Missing recorded times")
    halves, metadata = [], []
    for a in range(n):
        assignment, info = chronological_halves(times[a], ids[a])
        halves.append(assignment)
        metadata.append(info)
    assignments = [np.asarray(halves)]
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        assignment = np.zeros((n, 36), int)
        for a in range(n):
            assignment[a, rng.permutation(36)[:18]] = 1
        assignments.append(assignment)
    write(
        out / "contract.json",
        {
            "plan_sha256": digest(PLAN),
            "runner_sha256": digest(Path(__file__)),
            "module_sha256": digest(ROOT / "grain_research/response_accounting.py"),
            "C14_summary_sha256": digest(PRIOR / "summary.json"),
            "material_manifest_sha256": digest(MATERIAL / "material_manifest.json"),
            "reference": "C14 original source-only teacher sentence mean/SD, applied equally to all readers",
            "readers": READERS,
            "target_names": contract["target_names"],
            "random_seeds": RANDOM_SEEDS,
            "bootstrap": {"draws": 500, "seed": 914},
            "external_outcomes_read": False,
            "fits": 0,
            "new_neural_inferences": 0,
            "scope": "observed subreddit/time semantic-affinity accounting, not causal choice, state or trait identity",
            "comment_weighting": "equal supported comments; sentences averaged within comment; no-candidate comments missing",
            "source_authors": int(np.sum(folds != 0)),
            "held_authors": int(np.sum(folds == 0)),
        },
    )
    np.savez_compressed(
        out / "private_panels.npz",
        values=panels,
        counts=counts,
        folds=folds,
        conditions=conditions,
        timestamps=times,
        assignments=np.stack(assignments),
    )
    return panels, counts, folds, conditions, np.stack(assignments), metadata, addresses


def run(args):
    """Execute the fixed accounting and matched material sampling comparison."""
    out = args.output
    if out.exists():
        raise ValueError("Use a new output directory")
    out.mkdir(parents=True)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            values, counts, folds, conditions, assignments, time_meta, _addresses = (
                prepare(out)
            )
            nr, n, _ = assignments.shape
            nq = values.shape[-1]
            operators = np.full((nr, n, 4, 36), np.nan)
            results = np.full((nr, n, len(READERS), 4, nq), np.nan)
            support = []
            direct_error, reverse_error, propagated_error, sum_error = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            common = np.zeros((nr, n), bool)
            repeated = np.zeros((nr, n), bool)
            for r in range(nr):
                for a in range(n):
                    if time.monotonic() - start > 1200:
                        raise TimeoutError("Declared time limit reached")
                    meta = {
                        "partition": r,
                        "author_index": a,
                        "fold": int(folds[a]),
                        **time_meta[a],
                    }
                    try:
                        op, sup = difference_operator(
                            conditions[a], assignments[r, a], counts[a] > 0
                        )
                    except ValueError as exc:
                        support.append(
                            meta | {"status": "MISSING_HALF", "reason": str(exc)}
                        )
                        continue
                    operators[r, a] = op
                    common[r, a], repeated[r, a] = (
                        sup["within_measured"],
                        sup["repeated_shared_conditions"] > 0,
                    )
                    support.append(meta | sup | {"status": "SCORED"})
                    for m in range(len(READERS)):
                        results[r, a, m] = apply_operator(op, values[a, m])
                    sum_error = max(
                        sum_error,
                        float(
                            np.max(
                                abs(results[r, a, :, 0] - results[r, a, :, 1:].sum(1))
                            )
                        ),
                    )
                    if r == 0:
                        reverse, _ = difference_operator(
                            conditions[a], 1 - assignments[r, a], counts[a] > 0
                        )
                        reverse_error = max(
                            reverse_error, float(np.max(abs(op + reverse)))
                        )
                        for m in range(len(READERS)):
                            independent = direct_difference(
                                conditions[a], assignments[r, a], values[a, m]
                            )
                            direct_error = max(
                                direct_error,
                                float(np.max(abs(independent - results[r, a, m]))),
                            )
                        for m in range(1, len(READERS)):
                            residual = apply_operator(op, values[a, m] - values[a, 0])
                            propagated_error = max(
                                propagated_error,
                                float(
                                    np.max(
                                        abs(
                                            residual
                                            - (results[r, a, m] - results[r, a, 0])
                                        )
                                    )
                                ),
                            )
            valid = np.isfinite(results[:, :, 0, 0]).all(2)
            strict = np.asarray([row["strict_time_boundary"] for row in time_meta])
            held = (folds == 0) & valid[0] & strict
            np.savez_compressed(
                out / "private_accounting.npz",
                results=results,
                operators=operators,
                common=common,
                repeated=repeated,
                strict=strict,
                valid=valid,
            )
            support_frame = pd.DataFrame(support)
            support_frame.to_csv(out / "private_support.csv", index=False)
            public_support = []
            for cohort, mask in (("source", folds != 0), ("held", folds == 0)):
                sf = support_frame[
                    (support_frame.partition == 0)
                    & support_frame.author_index.isin(np.flatnonzero(mask))
                ]
                ok = sf[sf.status == "SCORED"]
                public_support.append(
                    {
                        "cohort": cohort,
                        "authors": int(mask.sum()),
                        "scored_authors": len(ok),
                        "strict_time_authors": int(np.sum(mask & valid[0] & strict)),
                        "any_shared_condition": int(np.sum(mask & common[0] & strict)),
                        "any_repeated_shared_condition": int(
                            np.sum(mask & repeated[0] & strict)
                        ),
                        "median_early_supported_comments": float(
                            ok.early_comments.median()
                        ),
                        "median_late_supported_comments": float(
                            ok.late_comments.median()
                        ),
                        "median_shared_conditions": float(
                            ok.shared_conditions.median()
                        ),
                        "median_repeated_shared_conditions": float(
                            ok.repeated_shared_conditions.median()
                        ),
                        "median_common_mass": float(
                            ((ok.early_shared_mass + ok.late_shared_mass) / 2).median()
                        ),
                        "median_span_days": float(ok.span_days.median()),
                        "median_boundary_gap_days": float(
                            ok.boundary_gap_days.median()
                        ),
                    }
                )
            pd.DataFrame(public_support).to_csv(
                out / "support_summary.csv", index=False
            )
            names = json.loads((out / "contract.json").read_text())["target_names"]
            agreements, aggregate, energy, temporal = [], [], [], []
            for part, part_name in enumerate(PARTS):
                base = held & (common[0] if part in (1, 2) else True)
                for stratum in ("common_required", "repeated_cell_author"):
                    eligible = base & (
                        repeated[0] if stratum == "repeated_cell_author" else True
                    )
                    teacher = results[0, eligible, 0, part]
                    for m, model in enumerate(READERS[1:], 1):
                        pred = results[0, eligible, m, part]
                        rvalue = correlation_columns(teacher, pred)
                        mse = (
                            np.mean((teacher - pred) ** 2, axis=0)
                            if len(teacher)
                            else np.full(nq, np.nan)
                        )
                        for q, name in enumerate(names):
                            kind, facet = name.split(":")
                            agreements.append(
                                {
                                    "reader": model,
                                    "part": part_name,
                                    "support": stratum,
                                    "function": kind,
                                    "facet": facet,
                                    "authors": len(teacher),
                                    "reader_teacher_r": rvalue[q],
                                    "mse": mse[q],
                                }
                            )
                        for kind in ("mean", "max"):
                            ix = [
                                q
                                for q, name in enumerate(names)
                                if name.startswith(kind + ":")
                            ]
                            aggregate.append(
                                {
                                    "reader": model,
                                    "part": part_name,
                                    "support": stratum,
                                    "function": kind,
                                    "authors": len(teacher),
                                    "mean_reader_teacher_r": float(
                                        np.nanmean(rvalue[ix])
                                    )
                                    if np.isfinite(rvalue[ix]).any()
                                    else None,
                                    "mean_mse": float(np.mean(mse[ix]))
                                    if len(teacher)
                                    else None,
                                }
                            )
                matched = held & valid.all(0)
                if part in (1, 2):
                    matched &= common.all(0)
                for kind in ("mean", "max"):
                    ix = [
                        q for q, name in enumerate(names) if name.startswith(kind + ":")
                    ]
                    chrono = np.mean(results[0, matched, 0, part][:, ix] ** 2, axis=1)
                    random = np.mean(
                        results[1:, matched, 0, part][:, :, ix] ** 2, axis=(0, 2)
                    )
                    if len(chrono):
                        temporal.append(
                            {
                                "part": part_name,
                                "function": kind,
                                "chronological_energy": float(chrono.mean()),
                                "random_mean_energy": float(random.mean()),
                                **paired_energy_interval(chrono, random),
                            }
                        )
            # Keep all squared terms and signed cross terms in the accounting.
            for m, model in enumerate(READERS):
                for kind in ("mean", "max"):
                    ix = [
                        q for q, name in enumerate(names) if name.startswith(kind + ":")
                    ]
                    r = results[0, held, m][:, :, ix]
                    e = float(np.mean(r[:, 0] ** 2))
                    row = {
                        "reader": model,
                        "function": kind,
                        "authors": int(held.sum()),
                        "total_energy": e,
                    }
                    for p, label in enumerate(PARTS[1:], 1):
                        row[label + "_squared"] = float(np.mean(r[:, p] ** 2))
                        row[label + "_signed_projection"] = (
                            float(np.mean(r[:, p] * r[:, 0]) / e) if e > 0 else None
                        )
                    for p, k in ((1, 2), (1, 3), (2, 3)):
                        row[PARTS[p] + "_x_" + PARTS[k]] = float(
                            2 * np.mean(r[:, p] * r[:, k])
                        )
                    energy.append(row)
            pd.DataFrame(agreements).to_csv(
                out / "reader_agreement_by_facet.csv", index=False
            )
            pd.DataFrame(aggregate).to_csv(
                out / "reader_agreement_summary.csv", index=False
            )
            pd.DataFrame(energy).to_csv(out / "energy_accounting.csv", index=False)
            pd.DataFrame(temporal).to_csv(
                out / "time_random_comparison.csv", index=False
            )
            old_annotations = pd.read_csv(
                PRIOR / "private_annotation_coordinate_join.csv"
            )
            keep = [
                "audit_id",
                "sentence_index",
                "author_index",
                "comment_index",
                "self_attribution",
                "scope",
                "relation",
                "nli",
            ]
            annotation = old_annotations[keep].copy()
            annotation["chronological_half"] = [
                int(assignments[0, row.author_index, row.comment_index])
                for row in annotation.itertuples()
            ]
            annotation.to_csv(out / "private_annotation_time_links.csv", index=False)
            write(out / "historical_evidence.json", inherited_evidence())
            errors = {
                "direct_cell_max_error": direct_error,
                "reverse_max_error": reverse_error,
                "reader_error_propagation_max_error": propagated_error,
                "component_sum_max_error": sum_error,
            }
            if max(errors.values()) > 1e-10:
                raise ArithmeticError("Accounting audit did not pass")
            summary = {
                "status": "OBSERVED_RESPONSE_ACCOUNTING_COMPLETE",
                "support": public_support,
                "teacher_energy": [r for r in energy if r["reader"] == "teacher"],
                "time_random": temporal,
                "audit": errors,
                "wall_seconds": time.monotonic() - start,
                "annotation_rows_linked": len(annotation),
                "goal_complete": False,
                "scope": "affinity changes and reader error, not personality/state labels or causal exposure",
                "files": {
                    p.name: digest(p)
                    for p in out.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                },
            }
            write(out / "summary.json", summary)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "files"}, indent=2
                ),
                flush=True,
            )
    except Exception as exc:
        (out / "error_log.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/response_accounting_r1",
    )
    run(parser.parse_args())
