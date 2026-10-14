#!/usr/bin/env python3
"""Compile four fixed prefix-only observation laws and test cold binding actions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_prefix_compilation as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv

SPEC = ROOT / "configs/grain_math_prefix_compilation_v1.json"
PLAN = ROOT / "docs/GRAIN_PREFIX_ONLY_COMPILATION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/prefix_compilation_r1"
CODE = (
    "grain_research/construction/math_prefix_compilation.py",
    "scripts/run_grain_math_prefix_compilation.py",
    "scripts/run_grain_math_native_joint.py",
    "scripts/run_grain_math_response.py",
    "grain_research/vector_mechanisms/navigation.py",
    "tests/test_grain_math_prefix_compilation.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, models=False):
    """Require the exact source bytes and frozen source functions before cold reading."""
    manifest = read_json(out / "00/manifest.json")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"source file changed: {name}")
    for name, expected in manifest["inputs"].items():
        if digest(out / name) != expected:
            raise ValueError(f"prepared input changed: {name}")
    if models:
        frozen = read_json(out / "01/functions_manifest.json")
        for name, expected in frozen["artifacts"].items():
            if digest(out / name) != expected:
                raise ValueError(f"frozen source function changed: {name}")
    return manifest


def prepare(out):
    """Verify the full intervention ledger, then freeze prefix-only input matrices."""
    if out.exists():
        raise FileExistsError("preserve existing packet, resume a later stage")
    start = time.monotonic()
    spec = read_json(SPEC)
    source, verified = ROOT / spec["source_packet"], ROOT / spec["source_verification"]
    primary = frozen_files(source)
    verification = read_json(verified / "verification.json")
    if (
        verification["status"]
        != "FINITE_PACKET_COMPLETE_CHECKED_NOT_GLOBAL_SCOPE_COMPLETE"
    ):
        raise ValueError("HM10B3 full numeric verification required")
    ledger_path = verified / "frozen_primary_artifact_hashes.json"
    ledger = read_json(ledger_path)
    for name, expected in ledger.items():
        if digest(source / name) != expected:
            raise ValueError(f"primary intervention artifact changed: {name}")
    docs = read_json(source / "00/documents.json")
    designs = read_json(source / "00/design.json")
    if (
        len(docs) != 224
        or len(designs) != 672
        or spec["queries"] != primary["spec"]["queries"]
    ):
        raise ValueError("full original material/query design required")
    by_doc = defaultdict(dict)
    for row in designs:
        if row["query"] in by_doc[row["doc_id"]]:
            raise ValueError("duplicate capture address")
        by_doc[row["doc_id"]][row["query"]] = row
    basis_path = source / "02/basis.npz"
    with np.load(basis_path, allow_pickle=False) as a:
        frame = a[f"L{spec['layer']}_U"].copy()
    metadata, xrows, orders, targets, consumed = [], [], [], [], {}
    for doc in docs:
        addresses = by_doc[doc["doc_id"]]
        if set(addresses) != set(spec["queries"]):
            raise ValueError("all fixed queries required")
        canonical = addresses[spec["queries"][0]]
        ids = canonical["input_ids"][: canonical["question_start"]]
        if canonical["spans"] != doc["spans"]:
            raise ValueError("raw body span addresses changed")
        target = []
        for query in spec["queries"]:
            row = addresses[query]
            if (
                row["input_ids"][: row["question_start"]] != ids
                or row["spans"] != canonical["spans"]
            ):
                raise ValueError("query-dependent body input found")
            path = source / "01/units" / f"unit_{row['unit']:04d}.json"
            unit = read_json(path)
            array = path.with_suffix(".npz")
            if unit["address"] != row or digest(array) != unit["array_sha256"]:
                raise ValueError("capture address/array changed")
            if (
                unit["body_query_error"] != 0
                or unit["canonical_capture_unit"] != canonical["unit"]
            ):
                raise ValueError("canonical query-prefix invariant failed")
            consumed[str(path.relative_to(source))] = digest(path)
            consumed[str(array.relative_to(source))] = digest(array)
            target.append(unit["native"]["conditional_score"])
        path = source / "01/units" / f"unit_{canonical['unit']:04d}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            x, order = M.prefix_coordinates(
                arrays,
                canonical["spans"],
                frame,
                canonical["question_start"],
                spec["layer"],
            )
        metadata.append({k: doc[k] for k in ("doc_id", "family", "role", "mask")})
        xrows.append(x)
        orders.append(order)
        targets.append(target)
    x, order, y = np.asarray(xrows), np.asarray(orders), np.asarray(targets)
    for role, families in spec["families_by_role"].items():
        selected = [i for i, r in enumerate(metadata) if r["role"] == role]
        unique = {metadata[i]["family"] for i in selected}
        if len(unique) != families or len(selected) != families * 14:
            raise ValueError("source/validation/application family separation changed")
        for family in unique:
            masks = {
                metadata[i]["mask"] for i in selected if metadata[i]["family"] == family
            }
            if len(masks) != 14:
                raise ValueError("incomplete family design")
    # Labels for non-source roles live in a separate file, never loaded by compile().
    out.mkdir(parents=True)
    (out / "00").mkdir()
    source_indices = np.array(
        [i for i, r in enumerate(metadata) if r["role"] == "source"]
    )
    held_indices = np.array(
        [i for i, r in enumerate(metadata) if r["role"] != "source"]
    )
    np.savez_compressed(out / "00/prefix_inputs.npz", x=x, order=order, frame=frame)
    np.savez_compressed(
        out / "00/source_targets.npz", indices=source_indices, y=y[source_indices]
    )
    np.savez_compressed(
        out / "00/held_targets.npz", indices=held_indices, y=y[held_indices]
    )
    write_json(out / "00/metadata.json", metadata)
    write_json(out / "00/consumed_primary_hashes.json", consumed)
    manifest = {
        "packet": "HM-11",
        "spec": spec,
        "source_files": {name: digest(ROOT / name) for name in CODE},
        "inputs": {str(p.relative_to(out)): digest(p) for p in (out / "00").iterdir()},
        "primary_manifest_sha256": digest(source / "00/manifest.json"),
        "full_primary_ledger_sha256": digest(ledger_path),
        "full_primary_artifacts_checked": len(ledger),
        "consumed_capture_artifacts_checked": len(consumed),
        "prefix_features": list(M.NAMES),
        "prefix_query_error": 0,
        "question_endpoint_feature_used": False,
        "truth_owner_polarity_feature_used": False,
        "family_identifier_feature_used": False,
        "elapsed_seconds": time.monotonic() - start,
        "new_model_calls": 0,
        "new_neural_training": False,
        "no_private_data": True,
        "result_guided_old_material_followup": True,
    }
    write_json(out / "00/manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PREFIX_INPUTS_FROZEN",
                "documents": len(x),
                "checked_primary_artifacts": len(ledger),
                "seconds": manifest["elapsed_seconds"],
            }
        ),
        flush=True,
    )


def compile(out):
    """Freeze four source-only functions before accessing held target values."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("source functions already frozen")
    spec = manifest["spec"]
    with np.load(out / "00/prefix_inputs.npz", allow_pickle=False) as data:
        x, order = data["x"], data["order"]
    with np.load(out / "00/source_targets.npz", allow_pickle=False) as data:
        indices, y = data["indices"], data["y"]
    models, coefficient_rows = {}, []
    for arm in spec["arms"]:
        phi, names = M.feature_map(x[indices], order[indices], arm)
        function = M.compile_source(phi, y, spec["alpha"])
        function["columns"] = names
        function["queries"] = spec["queries"]
        models[arm] = function
        for j, query in enumerate(spec["queries"]):
            coefficient_rows.append(
                {
                    "arm": arm,
                    "query": query,
                    "term": "intercept",
                    "active": True,
                    "raw_coefficient": function["raw_intercept"][j],
                    "source_mean": None,
                    "source_scale": None,
                }
            )
            for i, name in enumerate(names):
                coefficient_rows.append(
                    {
                        "arm": arm,
                        "query": query,
                        "term": name,
                        "active": function["active"][i],
                        "raw_coefficient": function["raw_coefficient"][i][j],
                        "source_mean": function["mean"][i],
                        "source_scale": function["scale"][i],
                    }
                )
    write_json(out / "01/functions.json", models)
    write_csv(out / "01/coefficients.csv", coefficient_rows)
    write_json(
        out / "01/functions_manifest.json",
        {
            "source_rows": len(indices),
            "source_indices": indices.tolist(),
            "source_target_sha256": digest(out / "00/source_targets.npz"),
            "held_targets_loaded": False,
            "alpha_search": False,
            "arms": spec["arms"],
            "frozen_before_cold_evaluation": True,
            "artifacts": {
                name: digest(out / name)
                for name in ("01/functions.json", "01/coefficients.csv")
            },
        },
    )
    print(
        {
            "status": "SOURCE_FUNCTIONS_FROZEN",
            "source_rows": len(indices),
            "arms": spec["arms"],
            "held_targets_loaded": False,
        },
        flush=True,
    )


def _interval(values, spec):
    """Empirical family mean interval conditional on the frozen fitted functions."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    samples = x[rng.integers(0, len(x), size=(spec["bootstrap_draws"], len(x)))].mean(
        axis=1
    )
    return np.quantile(samples, [0.025, 0.975]).tolist()


def evaluate(out):
    """Preserve all fixed arms, query outputs and both owner contrast orders."""
    manifest = check(out, models=True)
    if (out / "02").exists():
        raise FileExistsError("cold outputs already preserved")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    models = read_json(out / "01/functions.json")
    with np.load(out / "00/prefix_inputs.npz", allow_pickle=False) as data:
        x, order = data["x"], data["order"]
    y = np.zeros((len(x), len(spec["queries"])))
    for name in ("source", "held"):
        with np.load(out / f"00/{name}_targets.npz", allow_pickle=False) as data:
            y[data["indices"]] = data["y"]
    actual = M.owner_contrasts(metadata, y)
    predictions, levels, contrasts, contributions, summaries = [], [], [], [], []
    for arm in spec["arms"]:
        phi, names = M.feature_map(x, order, arm)
        if names != models[arm]["columns"]:
            raise ValueError("predictor column order changed")
        p = M.apply_function(phi, models[arm])
        predicted = M.owner_contrasts(metadata, p)
        for i, doc in enumerate(metadata):
            for j, query in enumerate(spec["queries"]):
                predictions.append(
                    {
                        **doc,
                        "arm": arm,
                        "query": query,
                        "observed": float(y[i, j]),
                        "predicted": float(p[i, j]),
                        "error": float(p[i, j] - y[i, j]),
                    }
                )
                contributions.append(
                    {
                        **doc,
                        "arm": arm,
                        "query": query,
                        "intercept": models[arm]["raw_intercept"][j],
                        "terms": {
                            name: float(
                                phi[i, k] * models[arm]["raw_coefficient"][k][j]
                            )
                            for k, name in enumerate(names)
                        },
                    }
                )
        for role in spec["families_by_role"]:
            index = [i for i, r in enumerate(metadata) if r["role"] == role]
            for j, query in enumerate(spec["queries"]):
                levels.append(
                    {
                        "role": role,
                        "arm": arm,
                        "query": query,
                        **M.compare_values(y[index, j], p[index, j]),
                        "out_of_ordinal_range": int(
                            ((p[index, j] < 0) | (p[index, j] > 1)).sum()
                        ),
                    }
                )
        for obs, pred in zip(actual, predicted, strict=True):
            if any(
                obs[key] != pred[key] for key in ("family", "axis", "order", "masks")
            ):
                raise ValueError("contrast addresses drifted")
            for j, query in enumerate(spec["queries"]):
                contrasts.append(
                    {
                        **{k: v for k, v in obs.items() if k != "values"},
                        "arm": arm,
                        "query": query,
                        "observed": obs["values"][j],
                        "predicted": pred["values"][j],
                        "error": pred["values"][j] - obs["values"][j],
                    }
                )
    for role in spec["families_by_role"]:
        for axis in ("A", "T"):
            for o in (0, 1):
                for arm in spec["arms"]:
                    for query in spec["queries"]:
                        rows = [
                            r
                            for r in contrasts
                            if (r["role"], r["axis"], r["order"], r["arm"], r["query"])
                            == (role, axis, o, arm, query)
                        ]
                        obs, p = (
                            [r["observed"] for r in rows],
                            [r["predicted"] for r in rows],
                        )
                        lo, hi = _interval(np.asarray(p) - obs, spec)
                        summaries.append(
                            {
                                "role": role,
                                "axis": axis,
                                "order": o,
                                "arm": arm,
                                "query": query,
                                **M.compare_values(obs, p),
                                "mean_error_ci_low": lo,
                                "mean_error_ci_high": hi,
                            }
                        )
    if len(predictions) != 2688 or len(contrasts) != 768 or len(summaries) != 144:
        raise ValueError("all declared level/action addresses required")
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/level_metrics.csv", levels)
    write_csv(out / "02/owner_contrasts.csv", contrasts)
    write_csv(out / "02/contrast_metrics.csv", summaries)
    write_json(out / "02/contributions.json", contributions)
    report = {
        "status": "COMPLETE_FINITE_COMPILATION_NOT_THEORY_COMPLETE",
        "documents": len(metadata),
        "source_rows": 112,
        "application_rows": 56,
        "arms": spec["arms"],
        "level_rows": len(predictions),
        "owner_contrast_rows": len(contrasts),
        "source_mean_scale_only": True,
        "unclipped_predictions": True,
        "new_model_calls": 0,
        "new_neural_training": False,
        "prefix_still_transformer_dependent": True,
        "human_personality_truth": False,
        "interval_scope": "empirical lexical-family uncertainty conditional on frozen source frame and functions; not refit or human population uncertainty",
        "old_material_result_guided_followup": True,
        "original_scope_complete": False,
        "application_levels": [r for r in levels if r["role"] == "application"],
        "application_contrasts": [r for r in summaries if r["role"] == "application"],
    }
    write_json(out / "02/summary.json", report)
    artifacts = {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    }
    write_json(out / "03/artifact_hashes.json", artifacts)
    print(
        {
            "status": report["status"],
            "level_rows": len(predictions),
            "owner_contrast_rows": len(contrasts),
            "new_model_calls": 0,
        },
        flush=True,
    )


def verify(out):
    """Recompute scores/actions and independently solve Ridge with sklearn's SVD."""
    from sklearn.linear_model import Ridge

    manifest = check(out, models=True)
    if (out / "03/verification.json").exists():
        raise FileExistsError("preserve completed numerical verification")
    ledger = read_json(out / "03/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(out / name) != expected:
            raise ValueError(f"packet artifact changed: {name}")
    source = ROOT / manifest["spec"]["source_packet"]
    for name, expected in read_json(out / "00/consumed_primary_hashes.json").items():
        if digest(source / name) != expected:
            raise ValueError("consumed capture changed")
    spec, models, metadata = (
        manifest["spec"],
        read_json(out / "01/functions.json"),
        read_json(out / "00/metadata.json"),
    )
    with np.load(out / "00/prefix_inputs.npz", allow_pickle=False) as data:
        x, order = data["x"], data["order"]
    with np.load(out / "00/source_targets.npz", allow_pickle=False) as data:
        indices, source_y = data["indices"], data["y"]
    y = np.zeros((len(x), 3))
    y[indices] = source_y
    with np.load(out / "00/held_targets.npz", allow_pickle=False) as data:
        y[data["indices"]] = data["y"]
    with (out / "02/predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    with (out / "02/owner_contrasts.csv").open() as stream:
        contrasts = list(csv.DictReader(stream))
    lookup = {(r["arm"], int(r["doc_id"]), r["query"]): r for r in predictions}
    cells = {(r["family"], r["mask"]): i for i, r in enumerate(metadata)}
    errors = {
        key: 0.0
        for key in (
            "raw_coefficient_replay",
            "svd_alternate_source_fit",
            "normal_equation",
            "contrast_replay",
            "source_only_mean_scale",
            "prefix_feature_replay",
            "native_target_replay",
        )
    }
    primary_design = read_json(source / "00/design.json")
    primary_by_address = {(r["doc_id"], r["query"]): r for r in primary_design}
    with np.load(out / "00/prefix_inputs.npz", allow_pickle=False) as data:
        frame = data["frame"]
    for i, doc in enumerate(metadata):
        row = primary_by_address[doc["doc_id"], spec["queries"][0]]
        path = source / "01/units" / f"unit_{row['unit']:04d}.npz"
        with np.load(path, allow_pickle=False) as a:
            alternate_x = np.concatenate(
                [
                    (
                        np.sum(a[f"L16_{span}"], axis=0, dtype=float)
                        / len(row["spans"][span]["positions"])
                    )
                    @ frame
                    for span in M.SPANS
                ]
            )
        errors["prefix_feature_replay"] = max(
            errors["prefix_feature_replay"], float(np.max(np.abs(x[i] - alternate_x)))
        )
        alternate_order = int(
            row["spans"]["T_property"]["positions"][0]
            < row["spans"]["A_property"]["positions"][0]
        )
        if alternate_order != order[i]:
            raise ValueError("relation-order replay failed")
        for j, query in enumerate(spec["queries"]):
            capture = primary_by_address[doc["doc_id"], query]
            saved = read_json(source / "01/units" / f"unit_{capture['unit']:04d}.json")
            logits = np.asarray(saved["native"]["candidate_logits"], float)
            weights = np.exp(logits - logits.max())
            value = float(weights @ np.linspace(0, 1, 10) / weights.sum())
            errors["native_target_replay"] = max(
                errors["native_target_replay"], abs(value - y[i, j])
            )
    for arm in spec["arms"]:
        phi, _ = M.feature_map(x, order, arm)
        fn = models[arm]
        weights = np.asarray(fn["raw_coefficient"]).reshape(phi.shape[1], 3)
        predicted = phi @ weights + fn["raw_intercept"]
        mean, scale = np.asarray(fn["mean"]), np.asarray(fn["scale"])
        if len(mean):
            source_scale = phi[indices].std(axis=0)
            source_scale = np.where(np.asarray(fn["active"], bool), source_scale, 1)
            errors["source_only_mean_scale"] = max(
                errors["source_only_mean_scale"],
                float(np.max(np.abs(mean - phi[indices].mean(axis=0)))),
                float(np.max(np.abs(scale - source_scale))),
            )
        z = (phi - mean) / scale
        z[:, ~np.asarray(fn["active"], bool)] = 0
        if phi.shape[1]:
            alternate = Ridge(
                alpha=spec["alpha"], fit_intercept=True, solver="svd"
            ).fit(z[indices], source_y)
            alt = alternate.predict(z)
            coefficient = np.asarray(fn["coefficient"])
            kkt = (
                z[indices].T @ (z[indices] @ coefficient + fn["intercept"] - source_y)
                + spec["alpha"] * coefficient
            )
            errors["normal_equation"] = max(
                errors["normal_equation"], float(np.max(np.abs(kkt)))
            )
        else:
            alt = np.broadcast_to(source_y.mean(axis=0), predicted.shape)
        errors["svd_alternate_source_fit"] = max(
            errors["svd_alternate_source_fit"], float(np.max(np.abs(alt - predicted)))
        )
        for i, doc in enumerate(metadata):
            for j, query in enumerate(spec["queries"]):
                row = lookup[arm, doc["doc_id"], query]
                errors["raw_coefficient_replay"] = max(
                    errors["raw_coefficient_replay"],
                    abs(float(row["predicted"]) - predicted[i, j]),
                    abs(float(row["observed"]) - y[i, j]),
                )
        for row in contrasts:
            if row["arm"] != arm:
                continue
            ids = [cells[int(row["family"]), mask] for mask in json.loads(row["masks"])]
            j = spec["queries"].index(row["query"])
            observed = np.dot([-1, 1, 1, -1], y[ids, j])
            p = np.dot([-1, 1, 1, -1], predicted[ids, j])
            errors["contrast_replay"] = max(
                errors["contrast_replay"],
                abs(observed - float(row["observed"])),
                abs(p - float(row["predicted"])),
            )
    if max(errors.values()) > 1e-8:
        raise ValueError(f"alternate numerical verification failed: {errors}")
    result = {
        "status": "COMPLETE_NUMERIC_CHECKED",
        "artifacts_checked": len(ledger),
        "prediction_values_checked": len(predictions),
        "contrast_values_checked": len(contrasts),
        "maximum_errors": errors,
        "new_model_calls": 0,
        "independent_arithmetic_same_agent": True,
        "original_scope_complete": False,
    }
    write_json(out / "03/verification.json", result)
    print(result, flush=True)


def main():
    """Execute distinct immutable stages; no collector or model loader is needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "evaluate", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compile": compile, "evaluate": evaluate, "verify": verify}[
        args.stage
    ](args.out)


if __name__ == "__main__":
    main()
