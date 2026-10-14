#!/usr/bin/env python3
"""HM14: compile source-only ordered action from frozen actual interventions."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_ordered_moments as M
from grain_research.construction import math_token_mean as TM
from grain_research.construction import math_token_permutation as TP
from grain_research.construction.math_prefix_compilation import compare_values
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_prefix_compilation import check as compilation_check
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_math_token_permutation import verify_source as permutation_check
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_ordered_moments_v1.json"
PLAN = ROOT / "docs/GRAIN_ORDERED_MOMENT_ACTION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/ordered_moment_action_r1"
CODE = (
    "grain_research/construction/math_ordered_moments.py",
    "scripts/run_grain_math_ordered_moments.py",
    "tests/test_grain_math_ordered_moments.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, models=False):
    """Require all frozen inputs and inherited code; verify fitted functions separately."""
    manifest = frozen_files(out)
    source = ROOT / manifest["spec"]["permutation_packet"]
    permutation_check(source)
    for name, expected in read_json(out / "00/permutation_artifacts.json").items():
        if digest(source / name) != expected:
            raise ValueError(f"HM13 artifact changed: {name}")
    if models:
        for name, expected in read_json(out / "01/functions_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"fitted function changed: {name}")
    return manifest


def prepare(out):
    """Freeze state pairs, original readout addresses and separate source/held targets."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; continue a later stage")
    start = time.monotonic()
    spec = read_json(SPEC)
    if spec["models"] != list(M.ARMS) or spec["layer"] != 16 or spec["rank"] != 2:
        raise ValueError("fixed original frame and four models required")
    permutation = ROOT / spec["permutation_packet"]
    primary = permutation_check(permutation)
    ledger = read_json(
        ROOT / spec["permutation_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in ledger.items():
        if digest(permutation / name) != expected:
            raise ValueError(f"HM13 artifact changed: {name}")
    compilation_check(ROOT / spec["compilation_packet"], models=True)
    span, mean = ROOT / spec["span_packet"], ROOT / spec["mean_packet"]
    capture = read_json(span / "00/design.json")
    lookup = {(r["doc_id"], r["query"]): r for r in capture}
    canonical = {r["doc_id"]: r for r in capture if r["query"] == spec["queries"][0]}
    if spec["queries"] != primary["spec"]["queries"]:
        raise ValueError("original queries required")
    with np.load(span / "02/basis.npz") as arrays:
        frame = arrays["L16_U"].copy()
    with np.load(ROOT / spec["compilation_packet"] / "00/prefix_inputs.npz") as arrays:
        if not np.array_equal(frame, arrays["frame"]):
            raise ValueError("original HM11 source frame changed")
    bodies, geometry = {}, []
    for doc_id, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {name: arrays[f"L16_{name}"].copy() for name in M.SPANS}
        for name, h in bodies[doc_id].items():
            positions = row["spans"][name]["positions"]
            if (
                h.shape != (len(positions), frame.shape[0])
                or max(positions) >= row["question_start"]
            ):
                raise ValueError("complete causal body address required")
            if name in M.PROPERTY:
                if len(h) > spec["max_property_tokens"]:
                    raise ValueError(
                        "new span length exceeds frozen complete moment frame"
                    )
                moments = M.ordered_moments(h, len(h) - 1)
                reconstructed = M.reconstruct_moments(moments, len(h))
                full = M.ordered_moments(h, 3)
                reverse = M.ordered_moments(h[::-1], 3)
                v = np.vander(
                    M.normalized_positions(len(h)), N=len(h), increasing=True
                ).T
                geometry.append(
                    {
                        "doc_id": doc_id,
                        "span": name,
                        "tokens": len(h),
                        "full_dimensional_reconstruction_max": float(
                            np.max(np.abs(reconstructed - h))
                        ),
                        "reversal_parity_max": float(
                            np.max(
                                np.abs(reverse - full * (-1.0) ** np.arange(4)[:, None])
                            )
                        ),
                        "vandermonde_condition": float(np.linalg.cond(v)),
                    }
                )
    if max(g["tokens"] for g in geometry) != spec["max_property_tokens"]:
        raise ValueError("source-supported maximum token count changed")
    full_rows = [
        r
        for r in read_json(span / "00/patch_design.json")
        if r["layer"] == spec["layer"] and r["arm"] == "full_property"
    ]
    mean_rows = read_json(mean / "00/design.json")
    permutation_rows = read_json(permutation / "00/design.json")
    blocks = (
        (span, "03/units", 5, full_rows),
        (mean, "01/units", 4, mean_rows),
        (permutation, "01/units", 4, permutation_rows),
    )
    metadata, references, targets, actions, consumed = [], [], [], [], {}
    state_cache = {}
    for packet, folder, width, design in blocks:
        for oldrow in design:
            name = f"{oldrow['axis']}_property"
            addr = lookup[oldrow["receiver_doc"], oldrow["query"]]
            base_path = span / "01/units" / f"unit_{addr['unit']:04d}.json"
            base = read_json(base_path)
            unit_path = packet / folder / f"unit_{oldrow['unit']:0{width}d}.json"
            unit = read_json(unit_path)
            if (
                unit["address"] != oldrow
                or base["address"] != addr
                or base["body_query_error"] != 0
            ):
                raise ValueError("frozen unit/query address mismatch")
            consumed[str(unit_path.relative_to(ROOT))] = digest(unit_path)
            consumed[str(base_path.relative_to(ROOT))] = digest(base_path)
            current = bodies[oldrow["receiver_doc"]][name]
            donor = bodies[oldrow["donor_doc"]][name]
            donor_address = canonical[oldrow["donor_doc"]]["spans"][name]
            receiver_address = addr["spans"][name]
            if (
                current.shape != donor.shape
                or receiver_address["raw"] != donor_address["raw"]
                or receiver_address["token_ids"] != donor_address["token_ids"]
            ):
                raise ValueError("same-word/token donor alignment mismatch")
            key = oldrow["pair_id"], oldrow["arm"]
            if key not in state_cache:
                reference_state = current
                if oldrow["arm"] == "full_property":
                    target_state = donor
                elif oldrow["arm"] in ("mean_only", "centered_only"):
                    target_state = TM.patch_target(current, donor, oldrow["arm"])
                else:
                    reference_state, target_state = TP.reference_and_target(
                        current, donor, oldrow["arm"]
                    )
                reference_body, target_body = (
                    dict(bodies[oldrow["receiver_doc"]]),
                    dict(bodies[oldrow["receiver_doc"]]),
                )
                reference_body[name], target_body[name] = reference_state, target_state
                a, b = (
                    M.observation(reference_body, frame),
                    M.observation(target_body, frame),
                )
                state_cache[key] = (
                    a,
                    b,
                    TP.state_digest(reference_state),
                    TP.state_digest(target_state),
                    reference_state.tobytes() == target_state.tobytes(),
                )
            a, b, reference_sha, target_sha, noop = state_cache[key]
            baseline_score = base["native"]["conditional_score"]
            reference_path = base_path
            reference_score = baseline_score
            if oldrow["arm"] == "centered_reverse":
                reference_row = next(
                    r
                    for r in mean_rows
                    if r["pair_id"] == oldrow["pair_id"]
                    and r["query"] == oldrow["query"]
                    and r["arm"] == "centered_only"
                )
                reference_path = (
                    mean / "01/units" / f"unit_{reference_row['unit']:04d}.json"
                )
                reference_score = read_json(reference_path)["native"][
                    "conditional_score"
                ]
                consumed[str(reference_path.relative_to(ROOT))] = digest(reference_path)
            expected_reference = (
                unit["receiver_native"]
                if oldrow["arm"] == "full_property"
                else unit["baseline"]
                if oldrow["arm"] in ("mean_only", "centered_only")
                else unit["reference"]
            )
            if reference_score != expected_reference or unit["live_body_error"] != 0:
                raise ValueError("actual receiver/reference unit mismatch")
            score = unit["native"]["conditional_score"]
            metadata.append(
                {
                    **{
                        k: oldrow[k]
                        for k in (
                            "pair_id",
                            "family",
                            "role",
                            "axis",
                            "order",
                            "polarity",
                            "receiver_doc",
                            "donor_doc",
                            "query",
                            "arm",
                        )
                    },
                    "row": len(metadata),
                    "span": name,
                    "tokens": len(current),
                    "opportunity": "single" if len(current) == 1 else "multi",
                    "positions": addr["spans"][name]["positions"],
                    "unit_path": str(unit_path.relative_to(ROOT)),
                    "reference_path": str(reference_path.relative_to(ROOT)),
                    "reference_state_sha256": reference_sha,
                    "target_state_sha256": target_sha,
                    "exact_input_noop": noop,
                }
            )
            references.append(a)
            targets.append(b)
            actions.append(score - reference_score)
    if len(metadata) != spec["paired_query_rows"]:
        raise ValueError("complete 1404 paired query interventions required")
    for role, count in spec["families_by_role"].items():
        if len({r["family"] for r in metadata if r["role"] == role}) != count:
            raise ValueError("source/validation/application role scope changed")
    source_indices = np.asarray(
        [i for i, r in enumerate(metadata) if r["role"] == "source"]
    )
    held_indices = np.asarray(
        [i for i, r in enumerate(metadata) if r["role"] != "source"]
    )
    y = np.asarray(actions)
    out.mkdir(parents=True)
    (out / "00").mkdir()
    np.savez_compressed(
        out / "00/observations.npz",
        reference=np.asarray(references),
        target=np.asarray(targets),
        frame=frame,
    )
    np.savez_compressed(
        out / "00/source_targets.npz", indices=source_indices, y=y[source_indices]
    )
    np.savez_compressed(
        out / "00/held_targets.npz", indices=held_indices, y=y[held_indices]
    )
    write_json(out / "00/metadata.json", metadata)
    write_json(out / "00/geometry.json", geometry)
    write_json(out / "00/permutation_artifacts.json", ledger)
    write_json(out / "00/consumed_hashes.json", consumed)
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "model_sha256": primary["model_sha256"],
            "source_files": {
                **primary["source_files"],
                **{name: digest(ROOT / name) for name in CODE},
            },
            "inputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "00").iterdir())
                if p.is_file()
            },
            "frame_sha256": digest(span / "02/basis.npz"),
            "columns": list(M.ALL_NAMES),
            "frozen_before_fit": True,
            "query_endpoint_feature_used": False,
            "owner_polarity_family_label_feature_used": False,
            "new_model_calls": 0,
            "result_guided_old_material_followup": True,
            "prepare_seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "ORDERED_PAIRS_FROZEN",
            "rows": len(metadata),
            "source": len(source_indices),
            "held": len(held_indices),
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def compile(out):
    """Fit source-only RMS/coefficients and freeze before loading held targets."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("source functions already frozen")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    with np.load(out / "00/observations.npz") as arrays:
        delta = arrays["target"] - arrays["reference"]
    with np.load(out / "00/source_targets.npz") as arrays:
        source_indices, y = arrays["indices"], arrays["y"]
    if any(metadata[i]["role"] != "source" for i in source_indices):
        raise ValueError("held row in source fit")
    functions, coefficients = {}, []
    for query in spec["queries"]:
        selected = np.asarray(
            [k for k, i in enumerate(source_indices) if metadata[i]["query"] == query]
        )
        indices = source_indices[selected]
        families = [metadata[i]["family"] for i in indices]
        functions[query] = {}
        for model in spec["models"]:
            columns = M.feature_indices(model)
            function = M.compile_action(
                delta[indices][:, columns], y[selected], families, spec["alpha"]
            )
            function["columns"] = [M.ALL_NAMES[i] for i in columns]
            functions[query][model] = function
            for j, name in enumerate(function["columns"]):
                coefficients.append(
                    {
                        "query": query,
                        "model": model,
                        "term": name,
                        "active": function["active"][j],
                        "source_rms": function["scale"][j],
                        "raw_coefficient": function["raw_coefficient"][j],
                    }
                )
    write_json(out / "01/functions.json", functions)
    write_csv(out / "01/coefficients.csv", coefficients)
    write_json(
        out / "01/functions_manifest.json",
        {
            "source_rows": len(source_indices),
            "source_indices": source_indices.tolist(),
            "source_target_sha256": digest(out / "00/source_targets.npz"),
            "held_targets_loaded": False,
            "source_only_scaling": True,
            "no_intercept": True,
            "alpha_search": False,
            "models": spec["models"],
            "frozen_before_cold_evaluation": True,
            "artifacts": {
                name: digest(out / name)
                for name in ("01/functions.json", "01/coefficients.csv")
            },
        },
    )
    print(
        {
            "status": "ORDERED_SOURCE_FUNCTIONS_FROZEN",
            "source_rows": len(source_indices),
            "held_targets_loaded": False,
        },
        flush=True,
    )


def summarize(rows, keys, spec):
    """Keep family uncertainty and signed action distinct from ordinary row-level fit."""
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, members in sorted(cells.items()):
        actual, predicted = (
            [r[k] for r in members] for k in ("observed_action", "predicted_action")
        )
        family = defaultdict(list)
        for row in members:
            y, p = row["observed_action"], row["predicted_action"]
            family[row["family"]].append((p - y, (p - y) ** 2, y**2 - (p - y) ** 2))
        values = np.asarray([np.mean(v, axis=0) for v in family.values()])
        _, bias_lo, bias_hi = family_interval(values[:, 0], spec)
        gain, gain_lo, gain_hi = family_interval(values[:, 2], spec)
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                **compare_values(actual, predicted),
                "families": len(family),
                "family_rmse": float(np.sqrt(values[:, 1].mean())),
                "mean_error_ci_low": bias_lo,
                "mean_error_ci_high": bias_hi,
                "mse_gain_vs_zero": gain,
                "mse_gain_ci_low": gain_lo,
                "mse_gain_ci_high": gain_hi,
                "noops": sum(r["exact_input_noop"] for r in members),
                "max_observation_difference": max(
                    r["observation_delta_max"] for r in members
                ),
            }
        )
    return result


def evaluate(out):
    """Evaluate every original intervention without promoting a model by held output."""
    manifest = check(out, models=True)
    if (out / "02").exists():
        raise FileExistsError("preserve frozen evaluated outputs")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    functions = read_json(out / "01/functions.json")
    with np.load(out / "00/observations.npz") as arrays:
        delta = arrays["target"] - arrays["reference"]
    y = np.empty(len(metadata))
    for name in ("source", "held"):
        with np.load(out / f"00/{name}_targets.npz") as arrays:
            y[arrays["indices"]] = arrays["y"]
    predictions = []
    for model in spec["models"]:
        columns = M.feature_indices(model)
        for query in spec["queries"]:
            indices = np.asarray(
                [i for i, r in enumerate(metadata) if r["query"] == query]
            )
            function = functions[query][model]
            if function["columns"] != [M.ALL_NAMES[i] for i in columns]:
                raise ValueError("frozen column order changed")
            p = M.apply_action(delta[indices][:, columns], function)
            for i, value in zip(indices, p, strict=True):
                predictions.append(
                    {
                        **metadata[i],
                        "model": model,
                        "observed_action": float(y[i]),
                        "predicted_action": float(value),
                        "error": float(value - y[i]),
                        "observation_delta_max": float(
                            np.max(np.abs(delta[i, columns]), initial=0)
                        ),
                    }
                )
    cells = summarize(predictions, ("role", "query", "model", "arm"), spec)
    details = summarize(
        predictions,
        ("role", "axis", "query", "model", "arm", "order", "opportunity", "tokens"),
        spec,
    )
    family_cells = summarize(
        predictions, ("role", "family", "query", "model", "arm"), spec
    )
    geometry = read_json(out / "00/geometry.json")
    reversal_mean = [
        r
        for r in predictions
        if r["model"] == "span_mean"
        and r["arm"] in ("receiver_reverse", "centered_reverse")
    ]
    noops = [r for r in predictions if r["exact_input_noop"]]
    summary = {
        "status": "FINITE_ORDERED_ACTION_PACKET_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "paired_query_interventions": len(metadata),
        "predictions": len(predictions),
        "source_rows": sum(r["role"] == "source" for r in metadata),
        "validation_rows": sum(r["role"] == "validation" for r in metadata),
        "application_rows": sum(r["role"] == "application" for r in metadata),
        "source_functions": len(spec["queries"]) * len(spec["models"]),
        "new_model_calls": 0,
        "mean_reversal_observation_delta_max": max(
            r["observation_delta_max"] for r in reversal_mean
        ),
        "mean_reversal_predicted_action_max": max(
            abs(r["predicted_action"]) for r in reversal_mean
        ),
        "input_noop_predicted_action_max": max(
            abs(r["predicted_action"]) for r in noops
        ),
        "input_noop_actual_action_max": max(abs(r["observed_action"]) for r in noops),
        "full_dimensional_reconstruction_max": max(
            r["full_dimensional_reconstruction_max"] for r in geometry
        ),
        "reversal_parity_max": max(r["reversal_parity_max"] for r in geometry),
        "vandermonde_condition_max": max(r["vandermonde_condition"] for r in geometry),
        "interval_scope": "lexical families conditional on source-frozen functions, not source-refit or human uncertainty",
        "psychological_accuracy": False,
        "all_action_cells": cells,
    }
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/action_metrics.csv", cells)
    write_csv(out / "02/order_token_metrics.csv", details)
    write_csv(out / "02/family_metrics.csv", family_cells)
    write_json(out / "02/summary.json", summary)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != out / "02/artifact_hashes.json"
        },
    )
    print({k: v for k, v in summary.items() if k != "all_action_cells"}, flush=True)


def main():
    """Run one bounded CPU stage under a packet lock; never invoke an observer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "compile": compile, "evaluate": evaluate}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
