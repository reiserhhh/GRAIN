#!/usr/bin/env python3
"""HM15: full ordered-vector/source-kernel test from frozen actual endpoints."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_ordered_moments as O
from grain_research.construction import math_span_binding as B
from grain_research.construction import math_token_mean as TM
from grain_research.construction import math_token_permutation as TP
from grain_research.construction import math_vector_action_kernel as K
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_ordered_moments import check as ordered_check
from scripts.run_grain_math_ordered_moments import summarize
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_vector_action_kernel_v1.json"
PLAN = ROOT / "docs/GRAIN_FULL_VECTOR_ACTION_KERNEL_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/full_vector_action_kernel_r1"
CODE = (
    "grain_research/construction/math_vector_action_kernel.py",
    "scripts/run_grain_math_vector_action_kernel.py",
    "tests/test_grain_math_vector_action_kernel.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, models=False):
    """Require frozen code, original observations and immutable source functions."""
    manifest = frozen_files(out)
    original = ROOT / manifest["spec"]["ordered_packet"]
    ordered_check(original, models=True)
    for name, expected in read_json(out / "00/ordered_artifacts.json").items():
        if digest(original / name) != expected:
            raise ValueError(f"HM14 artifact changed: {name}")
    if models:
        for name, expected in read_json(out / "01/functions_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"source function changed: {name}")
    return manifest


def prepare(out):
    """Freeze all2172 original query actions and724 unique state differences."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; continue another stage")
    start = time.monotonic()
    spec = read_json(SPEC)
    if spec["models"] != list(K.MODELS) or spec["layer"] != 16 or spec["rank"] != 2:
        raise ValueError("fixed three models/original layer/frame required")
    original, span = ROOT / spec["ordered_packet"], ROOT / spec["span_packet"]
    inherited = ordered_check(original, models=True)
    ledger = read_json(
        ROOT / spec["ordered_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in ledger.items():
        if digest(original / name) != expected:
            raise ValueError(f"HM14 artifact changed: {name}")
    if spec["queries"] != inherited["spec"]["queries"]:
        raise ValueError("query scope changed")
    captures = read_json(span / "00/design.json")
    addresses = {(r["doc_id"], r["query"]): r for r in captures}
    canonical = {r["doc_id"]: r for r in captures if r["query"] == spec["queries"][0]}
    with np.load(span / "02/basis.npz") as arrays:
        frame, random_frame = arrays["L16_U"].copy(), arrays["L16_R"].copy()
    if frame.shape != (spec["hidden_dimension"], spec["rank"]):
        raise ValueError("original full hidden dimension/rank required")
    bodies = {}
    for doc_id, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {name: arrays[f"L16_{name}"].copy() for name in O.SPANS}
        for name, h in bodies[doc_id].items():
            positions = row["spans"][name]["positions"]
            if (
                h.shape != (len(positions), len(frame))
                or max(positions) >= row["question_start"]
            ):
                raise ValueError("fixed complete body-before-question state required")
            if name in O.PROPERTY and len(h) > spec["max_property_tokens"]:
                raise ValueError("fixed small-span position encoding exceeded")
    old_metadata = read_json(original / "00/metadata.json")
    with np.load(original / "00/observations.npz") as arrays:
        old_delta = arrays["target"] - arrays["reference"]
    extra = [
        r
        for r in read_json(span / "00/patch_design.json")
        if r["layer"] == spec["layer"] and r["arm"] in spec["extra_interventions"]
    ]
    metadata = [dict(r) for r in old_metadata]
    for row in extra:
        name = f"{row['axis']}_property"
        addr = addresses[row["receiver_doc"], row["query"]]
        h = bodies[row["receiver_doc"]][name]
        metadata.append(
            {
                **{
                    k: row[k]
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
                "tokens": len(h),
                "opportunity": "single" if len(h) == 1 else "multi",
                "positions": addr["spans"][name]["positions"],
                "unit_path": str(
                    (span / "03/units" / f"unit_{row['unit']:05d}.json").relative_to(
                        ROOT
                    )
                ),
                "reference_path": str(
                    (span / "01/units" / f"unit_{addr['unit']:04d}.json").relative_to(
                        ROOT
                    )
                ),
            }
        )
    if len(metadata) != spec["paired_query_rows"]:
        raise ValueError("all2172 original endpoints required")
    for role, rows in spec["rows_by_role"].items():
        if (
            sum(r["role"] == role for r in metadata) != rows
            or len({r["family"] for r in metadata if r["role"] == role})
            != spec["families_by_role"][role]
        ):
            raise ValueError("inherited source/validation/application roles changed")
    full, projected, states, geometry, consumed, y = [], [], {}, [], {}, []
    records = {}
    old_delta_error = 0.0
    for i, row in enumerate(metadata):
        name = row["span"]
        h, d = (bodies[row[k]][name] for k in ("receiver_doc", "donor_doc"))
        a, b = (canonical[row[k]]["spans"][name] for k in ("receiver_doc", "donor_doc"))
        if (
            a["raw"] != b["raw"]
            or a["token_ids"] != b["token_ids"]
            or h.shape != d.shape
        ):
            raise ValueError("same-word/token donor identity required")
        key = row["pair_id"], row["arm"]
        if key not in states:
            reference = h
            if row["arm"] == "full_property":
                target = d
            elif row["arm"] in ("mean_only", "centered_only"):
                target = TM.patch_target(h, d, row["arm"])
            elif row["arm"] in ("receiver_reverse", "centered_reverse"):
                reference, target = TP.reference_and_target(h, d, row["arm"])
            else:
                delta = B.intervention_delta(
                    h.astype(float), d.astype(float), row["arm"], frame, random_frame
                )
                target = (h.astype(float) + delta).astype(np.float32)
            rb, tb = (
                dict(bodies[row["receiver_doc"]]),
                dict(bodies[row["receiver_doc"]]),
            )
            rb[name], tb[name] = reference, target
            dx = K.full_observation(tb) - K.full_observation(rb)
            pu = K.projected_observation(dx, frame)
            state_index = len(full)
            full.append(dx)
            projected.append(pu)
            states[key] = (
                state_index,
                TP.state_digest(reference),
                TP.state_digest(target),
                reference.tobytes() == target.tobytes(),
            )
            token_projection = (target.astype(float) - reference.astype(float)) @ frame
            geometry.append(
                {
                    "pair_id": row["pair_id"],
                    "arm": row["arm"],
                    "state_index": state_index,
                    "role": row["role"],
                    "axis": row["axis"],
                    "order": row["order"],
                    "tokens": len(h),
                    "full_observation_norm": float(np.linalg.norm(dx)),
                    "projected_observation_norm": float(np.linalg.norm(pu)),
                    "projected_token_displacement_max": float(
                        np.max(np.abs(token_projection))
                    ),
                    "reference_state_sha256": TP.state_digest(reference),
                    "target_state_sha256": TP.state_digest(target),
                    "exact_input_noop": reference.tobytes() == target.tobytes(),
                }
            )
        state_index, reference_sha, target_sha, noop = states[key]
        if i < len(old_metadata):
            old_delta_error = max(
                old_delta_error,
                float(np.max(np.abs(projected[state_index] - old_delta[i]))),
            )
            if (
                row["reference_state_sha256"] != reference_sha
                or row["target_state_sha256"] != target_sha
                or row["exact_input_noop"] != noop
            ):
                raise ValueError("HM14 complete reference/target states changed")
        row.update(
            state_index=state_index,
            reference_state_sha256=reference_sha,
            target_state_sha256=target_sha,
            exact_input_noop=noop,
        )
        for field in ("unit_path", "reference_path"):
            path = ROOT / row[field]
            if path not in records:
                records[path] = read_json(path)
                consumed[row[field]] = digest(path)
        unit, ref = (
            records[ROOT / row["unit_path"]],
            records[ROOT / row["reference_path"]],
        )
        if (
            unit["live_body_error"] != 0
            or unit["address"]["query"] != row["query"]
            or ref["address"]["query"] != row["query"]
            or ref["address"].get("doc_id", ref["address"].get("receiver_doc"))
            != row["receiver_doc"]
        ):
            raise ValueError("actual query/body reference mismatch")
        if i >= len(old_metadata) and (
            unit["address"]["layer"] != spec["layer"]
            or unit["receiver_native"] != ref["native"]["conditional_score"]
        ):
            raise ValueError("original layer/receiver endpoint mismatch")
        if any(
            unit["address"][k] != row[k]
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
        ):
            raise ValueError("actual intervention address mismatch")
        y.append(
            unit["native"]["conditional_score"] - ref["native"]["conditional_score"]
        )
    if len(full) != spec["unique_state_pairs"] or old_delta_error > 1e-10:
        raise ValueError(
            "complete query-independent state scope/projection replay failed"
        )
    source_indices = np.asarray(
        [i for i, r in enumerate(metadata) if r["role"] == "source"]
    )
    held_indices = np.asarray(
        [i for i, r in enumerate(metadata) if r["role"] != "source"]
    )
    y = np.asarray(y)
    out.mkdir(parents=True)
    (out / "00").mkdir()
    np.savez_compressed(
        out / "00/differences.npz",
        full=np.asarray(full),
        projected=np.asarray(projected),
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
    write_json(out / "00/ordered_artifacts.json", ledger)
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
            "model_sha256": inherited["model_sha256"],
            "source_files": {
                **inherited["source_files"],
                **{name: digest(ROOT / name) for name in CODE},
            },
            "inputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "00").iterdir())
                if p.is_file()
            },
            "frozen_before_fit": True,
            "old_projected_delta_replay_error": old_delta_error,
            "feature_dimensions": {
                "ordered_rank2": 24,
                "ordered_full": 12 * spec["hidden_dimension"],
            },
            "moment_rows": [
                *O.SPANS,
                *(f"{name}_position{k}" for k in (1, 2, 3) for name in O.PROPERTY),
            ],
            "query_endpoint_feature_used": False,
            "owner_polarity_family_feature_used": False,
            "new_model_calls": 0,
            "result_guided_old_material_followup": True,
            "prepare_seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "FULL_VECTOR_PAIRS_FROZEN",
            "rows": len(metadata),
            "unique_states": len(full),
            "source": len(source_indices),
            "held": len(held_indices),
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def compile(out):
    """Fit finite source Gram systems without ever loading held target values."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("source functions already frozen")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    with np.load(out / "00/differences.npz") as arrays:
        views = {
            "ordered_rank2": arrays["projected"].copy(),
            "ordered_full": arrays["full"].copy(),
        }
    with np.load(out / "00/source_targets.npz") as arrays:
        source_indices, y = arrays["indices"], arrays["y"]
    if any(metadata[i]["role"] != "source" for i in source_indices):
        raise ValueError("held row entered source fit")
    functions, fitted = {}, []
    for query in spec["queries"]:
        positions = [
            j for j, i in enumerate(source_indices) if metadata[i]["query"] == query
        ]
        indices = source_indices[positions]
        state_indices = np.asarray([metadata[i]["state_index"] for i in indices])
        families = [metadata[i]["family"] for i in indices]
        if len(np.unique(state_indices)) != len(indices):
            raise ValueError("query source contains duplicate state pairs")
        functions[query] = {}
        for model in spec["models"]:
            x = (
                views[model][state_indices]
                if model != "zero_action"
                else np.empty((len(indices), 0))
            )
            function = K.fit_action(x, y[positions], families, spec["alpha"])
            path = out / "01" / f"{query}_{model}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                name: function.pop(name)
                for name in ("weighted_dual", "coefficient", "eigenvalues")
            }
            np.savez_compressed(path, **arrays, source_state_indices=state_indices)
            function.update(
                path=str(path.relative_to(out)),
                source_state_indices=state_indices.tolist(),
            )
            functions[query][model] = function
            fitted.append(
                {
                    "query": query,
                    "model": model,
                    **{
                        k: v
                        for k, v in function.items()
                        if k not in ("source_state_indices", "path")
                    },
                }
            )
    write_json(out / "01/functions.json", functions)
    write_csv(out / "01/source_geometry.csv", fitted)
    write_json(
        out / "01/functions_manifest.json",
        {
            "source_rows": len(source_indices),
            "source_indices": source_indices.tolist(),
            "source_target_sha256": digest(out / "00/source_targets.npz"),
            "held_targets_loaded": False,
            "source_only_scaling": "single_rms_norm",
            "no_intercept": True,
            "alpha_search": False,
            "frozen_before_cold_evaluation": True,
            "artifacts": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "01").iterdir())
                if p.is_file()
            },
        },
    )
    print(
        {
            "status": "FULL_VECTOR_SOURCE_FUNCTIONS_FROZEN",
            "functions": len(fitted),
            "source_rows": len(source_indices),
            "held_targets_loaded": False,
        },
        flush=True,
    )


def evaluate(out):
    """Evaluate all fixed views and actions, not a selected best held model."""
    manifest = check(out, models=True)
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated results")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    functions = read_json(out / "01/functions.json")
    with np.load(out / "00/differences.npz") as arrays:
        views = {
            "ordered_rank2": arrays["projected"].copy(),
            "ordered_full": arrays["full"].copy(),
        }
    y = np.empty(len(metadata))
    for role in ("source", "held"):
        with np.load(out / f"00/{role}_targets.npz") as arrays:
            y[arrays["indices"]] = arrays["y"]
    predictions = []
    for model in spec["models"]:
        for query in spec["queries"]:
            indices = [i for i, r in enumerate(metadata) if r["query"] == query]
            function = functions[query][model]
            with np.load(out / function["path"]) as arrays:
                coefficient = arrays["coefficient"].copy()
            state_indices = [metadata[i]["state_index"] for i in indices]
            x = (
                views[model][state_indices]
                if model != "zero_action"
                else np.empty((len(indices), 0))
            )
            p = K.apply_action(x, coefficient)
            for i, value, dx in zip(indices, p, x, strict=True):
                predictions.append(
                    {
                        **metadata[i],
                        "model": model,
                        "observed_action": float(y[i]),
                        "predicted_action": float(value),
                        "error": float(value - y[i]),
                        "observation_delta_max": float(np.max(np.abs(dx), initial=0)),
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
    by_view = {(r["row"], r["model"]): r for r in predictions}
    gains = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["model"] != "ordered_full":
            continue
        base = by_view[row["row"], "ordered_rank2"]
        for key in (
            (row["role"], "all", row["query"], row["arm"]),
            (row["role"], row["axis"], row["query"], row["arm"]),
        ):
            gains[key][row["family"]].append(base["error"] ** 2 - row["error"] ** 2)
    paired = []
    for key, family in sorted(gains.items()):
        gain, lo, hi = family_interval([np.mean(v) for v in family.values()], spec)
        paired.append(
            {
                **dict(zip(("role", "axis", "query", "arm"), key, strict=True)),
                "families": len(family),
                "mse_gain_full_vs_rank2": gain,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    noops = [r for r in predictions if r["exact_input_noop"]]
    summary = {
        "status": "FINITE_FULL_VECTOR_ACTION_PACKET_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "paired_query_interventions": len(metadata),
        "unique_state_pairs": spec["unique_state_pairs"],
        "predictions": len(predictions),
        "source_functions_including_zero": 9,
        "new_model_calls": 0,
        "feature_dimensions": manifest["feature_dimensions"],
        "old_projected_delta_replay_error": manifest[
            "old_projected_delta_replay_error"
        ],
        "input_noop_actual_action_max": max(abs(r["observed_action"]) for r in noops),
        "input_noop_predicted_action_max": max(
            abs(r["predicted_action"]) for r in noops
        ),
        "interval_scope": "lexical-family bootstrap conditional on frozen source functions; not source-refit or human uncertainty",
        "all_action_cells": cells,
        "psychological_accuracy": False,
    }
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/action_metrics.csv", cells)
    write_csv(out / "02/order_token_metrics.csv", details)
    write_csv(out / "02/family_metrics.csv", family_cells)
    write_csv(out / "02/paired_full_vs_rank2.csv", paired)
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
    """Run one bounded, locked CPU-only stage with no observer/model loading."""
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
