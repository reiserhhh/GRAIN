#!/usr/bin/env python3
"""HM16: fixed receiver-covector/endpoint-potential kernels from frozen states."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_context_covector as C
from grain_research.construction import math_ordered_moments as O
from grain_research.construction import math_span_binding as B
from grain_research.construction import math_token_mean as TM
from grain_research.construction import math_token_permutation as TP
from grain_research.construction import math_vector_action_kernel as K
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_ordered_moments import summarize
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_math_vector_action_kernel import check as vector_check
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_context_covector_v1.json"
PLAN = ROOT / "docs/GRAIN_CONTEXT_COVECTOR_KERNEL_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/context_covector_kernel_r1"
CODE = (
    "grain_research/construction/math_context_covector.py",
    "scripts/run_grain_math_context_covector.py",
    "tests/test_grain_math_context_covector.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, models=False):
    """Require frozen inherited endpoints, code and source-only fitted kernels."""
    manifest = frozen_files(out)
    original = ROOT / manifest["spec"]["vector_packet"]
    vector_check(original, models=True)
    for name, expected in read_json(out / "00/vector_artifacts.json").items():
        if digest(original / name) != expected:
            raise ValueError(f"HM15 artifact changed: {name}")
    if models:
        for name, expected in read_json(out / "01/functions_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"context source function changed: {name}")
    return manifest


def prepare(out):
    """Freeze absolute observations and a target-blind marginal context control."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; continue another stage")
    start, spec = time.monotonic(), read_json(SPEC)
    if spec["models"] != list(C.MODELS) or spec["layer"] != 16:
        raise ValueError("fixed models/original layer required")
    original, span = ROOT / spec["vector_packet"], ROOT / spec["span_packet"]
    inherited = vector_check(original, models=True)
    ledger = read_json(
        ROOT / spec["vector_verification"] / "frozen_primary_artifact_hashes.json"
    )
    for name, expected in ledger.items():
        if digest(original / name) != expected:
            raise ValueError(f"HM15 primary changed: {name}")
    metadata = read_json(original / "00/metadata.json")
    if (
        len(metadata) != spec["paired_query_rows"]
        or spec["queries"] != inherited["spec"]["queries"]
    ):
        raise ValueError("inherited address/query scope changed")
    for role, n in spec["rows_by_role"].items():
        if (
            sum(r["role"] == role for r in metadata) != n
            or len({r["family"] for r in metadata if r["role"] == role})
            != spec["families_by_role"][role]
        ):
            raise ValueError("fixed source/held role scope changed")
    canonical = {
        r["doc_id"]: r
        for r in read_json(span / "00/design.json")
        if r["query"] == spec["queries"][0]
    }
    with np.load(span / "02/basis.npz") as arrays:
        frame, random_frame = arrays["L16_U"].copy(), arrays["L16_R"].copy()
    bodies = {}
    for doc_id, row in canonical.items():
        with np.load(span / "01/units" / f"unit_{row['unit']:04d}.npz") as arrays:
            bodies[doc_id] = {name: arrays[f"L16_{name}"].copy() for name in O.SPANS}
    with np.load(original / "00/differences.npz") as arrays:
        old_delta = arrays["full"].copy()
    reference, target = np.empty_like(old_delta), np.empty_like(old_delta)
    seen = set()
    for row in metadata:
        j, name = row["state_index"], row["span"]
        if j in seen:
            continue
        seen.add(j)
        h, d = (bodies[row[k]][name] for k in ("receiver_doc", "donor_doc"))
        r = h
        if row["arm"] == "full_property":
            t = d
        elif row["arm"] in ("mean_only", "centered_only"):
            t = TM.patch_target(h, d, row["arm"])
        elif row["arm"] in ("receiver_reverse", "centered_reverse"):
            r, t = TP.reference_and_target(h, d, row["arm"])
        elif row["arm"] in ("projection", "complement"):
            delta = B.intervention_delta(
                h.astype(float), d.astype(float), row["arm"], frame, random_frame
            )
            t = (h.astype(float) + delta).astype(np.float32)
        else:
            raise ValueError("undeclared inherited arm")
        if (
            TP.state_digest(r) != row["reference_state_sha256"]
            or TP.state_digest(t) != row["target_state_sha256"]
        ):
            raise ValueError("absolute full-state identity changed")
        rb, tb = dict(bodies[row["receiver_doc"]]), dict(bodies[row["receiver_doc"]])
        rb[name], tb[name] = r, t
        reference[j], target[j] = K.full_observation(rb), K.full_observation(tb)
    difference_error = float(np.max(np.abs(target - reference - old_delta)))
    if len(seen) != spec["unique_state_pairs"] or difference_error > 1e-12:
        raise ValueError("inherited full difference not preserved")
    mapping, pairs = C.shuffled_state_mapping(metadata, spec["shuffle_seed"])
    out.mkdir(parents=True)
    (out / "00").mkdir()
    np.savez_compressed(
        out / "00/endpoints.npz",
        reference=reference,
        target=target,
        delta=old_delta,
        context_mapping=mapping,
    )
    for name in ("source_targets.npz", "held_targets.npz", "metadata.json"):
        (out / "00" / name).write_bytes((original / "00" / name).read_bytes())
    write_csv(out / "00/context_pair_mapping.csv", pairs)
    write_json(out / "00/vector_artifacts.json", ledger)
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "packet": spec["packet"],
            "spec": spec,
            "model_sha256": inherited["model_sha256"],
            "source_files": {
                **inherited["source_files"],
                **{n: digest(ROOT / n) for n in CODE},
            },
            "inputs": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "00").iterdir())
                if p.is_file()
            },
            "frozen_before_fit": True,
            "difference_replay_max_error": difference_error,
            "context_pairs": len(pairs),
            "context_pairs_moved": sum(r["pair_moved"] for r in pairs),
            "context_pairs_fixed": sum(not r["pair_moved"] for r in pairs),
            "context_states_moved": int(np.sum(mapping != np.arange(len(mapping)))),
            "context_control_keeps_available_arm_marginals": True,
            "role_polarity_family_as_feature": False,
            "question_endpoint_as_feature": False,
            "new_model_calls": 0,
            "result_guided_old_material_followup": True,
            "prepare_seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "CONTEXT_ENDPOINTS_FROZEN",
            "states": len(seen),
            "pairs": len(pairs),
            "pairs_moved": sum(r["pair_moved"] for r in pairs),
            "pairs_fixed": sum(not r["pair_moved"] for r in pairs),
            "difference_error": difference_error,
        },
        flush=True,
    )


def load_views(out, parameters=None):
    """Load frozen endpoints; application calls use only frozen source coordinates."""
    with np.load(out / "00/endpoints.npz") as arrays:
        r, t, d = (arrays[k].copy() for k in ("reference", "target", "delta"))
        mapping = arrays["context_mapping"].copy()
    if parameters is None:
        return r, t, d, mapping
    x = d / parameters["delta_scale"]
    c = (r - parameters["origin"]) / parameters["context_scale"]
    m = ((r + t) / 2 - parameters["origin"]) / parameters["context_scale"]
    return x, c, m, c[mapping], mapping


def source_layout(metadata, source_indices, query):
    """Select only source positions and their family-balanced empirical weights."""
    positions = [
        j for j, i in enumerate(source_indices) if metadata[i]["query"] == query
    ]
    indices = source_indices[positions]
    states = np.asarray([metadata[i]["state_index"] for i in indices])
    if len(states) != 360 or len(np.unique(states)) != 360:
        raise ValueError("fixed unique360source states required perquery")
    w = O.family_weights([metadata[i]["family"] for i in indices])
    return positions, indices, states, w


def compile(out):
    """Fit nine finite tensor kernels without opening held target values."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("source context functions already frozen")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    original = ROOT / spec["vector_packet"]
    old_functions = read_json(original / "01/functions.json")
    with np.load(out / "00/source_targets.npz") as arrays:
        source_indices, y = arrays["indices"].copy(), arrays["y"].copy()
    if any(metadata[i]["role"] != "source" for i in source_indices):
        raise ValueError("held row in source fit")
    r, _, _, _ = load_views(out)
    _, _, states, w = source_layout(metadata, source_indices, spec["queries"][0])
    origin, context_scale, context_scale_squared = C.context_reference(r[states], w)
    parameters = {
        "origin": origin,
        "context_scale": context_scale,
        "delta_scale": old_functions[spec["queries"][0]]["ordered_full"]["scale"],
    }
    (out / "01").mkdir()
    np.savez_compressed(
        out / "01/source_coordinates.npz",
        **parameters,
        context_scale_squared=context_scale_squared,
        source_state_indices=states,
    )
    x, c, m, shuffled, _ = load_views(out, parameters)
    context = {
        "receiver_tensor": c,
        "endpoint_potential": m,
        "shuffled_context": shuffled,
    }
    blocks = {
        model: C.kernel_parts(
            x,
            ctx,
            x[states],
            ctx[states],
            "endpoint_potential"
            if model == "endpoint_potential"
            else "receiver_tensor",
        )
        for model, ctx in context.items()
    }
    functions, fitted = {}, []
    for query in spec["queries"]:
        positions, _, query_states, query_w = source_layout(
            metadata, source_indices, query
        )
        if not np.array_equal(query_states, states) or not np.array_equal(query_w, w):
            raise ValueError("source observation/weight layout varies byquery")
        old = old_functions[query]["ordered_full"]
        if (
            old["scale"] != parameters["delta_scale"]
            or old["source_state_indices"] != states.tolist()
        ):
            raise ValueError("inherited fixed linear source law changed")
        functions[query] = {
            "full_linear": {
                "inherited_path": str((original / old["path"]).relative_to(ROOT)),
                "new_fit": False,
                "source_rows": len(states),
            }
        }
        for model, (linear, coupling) in blocks.items():
            raw = (linear + coupling)[states]
            f = C.fit_kernel(raw, y[positions], w, spec["alpha"])
            path = out / "01" / f"{query}_{model}.npz"
            np.savez_compressed(
                path,
                weighted_dual=f.pop("weighted_dual"),
                eigenvalues=f.pop("eigenvalues"),
                source_state_indices=states,
            )
            f.update(
                path=str(path.relative_to(out)),
                new_fit=True,
                source_families=len(
                    np.unique(
                        [metadata[i]["family"] for i in source_indices[positions]]
                    )
                ),
            )
            functions[query][model] = f
            fitted.append({"query": query, "model": model, **f})
    write_json(out / "01/functions.json", functions)
    write_csv(out / "01/source_geometry.csv", fitted)
    write_json(
        out / "01/functions_manifest.json",
        {
            "source_indices": source_indices.tolist(),
            "source_target_sha256": digest(out / "00/source_targets.npz"),
            "source_functions_including_inherited": 12,
            "new_kernel_fits": len(fitted),
            "held_targets_loaded": False,
            "alpha_search": False,
            "action_intercept": False,
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
            "status": "CONTEXT_SOURCE_FUNCTIONS_FROZEN",
            "new_fits": len(fitted),
            "inherited_functions": 3,
            "source_rows": len(source_indices),
            "context_scale": context_scale,
            "held_targets_loaded": False,
        },
        flush=True,
    )


def evaluate(out):
    """Evaluate every fixed kernel/arm and preserve all contribution and control results."""
    manifest = check(out, models=True)
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated context results")
    spec, metadata = manifest["spec"], read_json(out / "00/metadata.json")
    functions = read_json(out / "01/functions.json")
    with np.load(out / "01/source_coordinates.npz") as arrays:
        parameters = {
            k: arrays[k].copy() for k in ("origin", "context_scale", "delta_scale")
        }
        states = arrays["source_state_indices"].copy()
    x, c, m, shuffled, _ = load_views(out, parameters)
    ctx = {"receiver_tensor": c, "endpoint_potential": m, "shuffled_context": shuffled}
    blocks = {
        model: C.kernel_parts(
            x,
            v,
            x[states],
            v[states],
            "endpoint_potential"
            if model == "endpoint_potential"
            else "receiver_tensor",
        )
        for model, v in ctx.items()
    }
    _, _, delta, _ = load_views(out)
    y = np.empty(len(metadata))
    for role in ("source", "held"):
        with np.load(out / f"00/{role}_targets.npz") as arrays:
            y[arrays["indices"]] = arrays["y"]
    predictions = []
    for model in spec["models"]:
        for query in spec["queries"]:
            f = functions[query][model]
            if model == "full_linear":
                with np.load(ROOT / f["inherited_path"]) as arrays:
                    p_linear = delta @ arrays["coefficient"]
                p_coupling = np.zeros(len(delta))
            else:
                with np.load(out / f["path"]) as arrays:
                    p_linear, p_coupling = C.apply_parts(
                        *blocks[model], arrays["weighted_dual"], f["kernel_unit"]
                    )
            for i, row in enumerate(metadata):
                if row["query"] != query:
                    continue
                j = row["state_index"]
                p = float(p_linear[j] + p_coupling[j])
                predictions.append(
                    {
                        **row,
                        "model": model,
                        "observed_action": float(y[i]),
                        "predicted_action": p,
                        "error": float(p - y[i]),
                        "linear_contribution": float(p_linear[j]),
                        "coupling_contribution": float(p_coupling[j]),
                        "observation_delta_max": float(np.max(np.abs(delta[j]))),
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
    by_model = {(r["row"], r["model"]): r for r in predictions}
    gains = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row["model"] == "full_linear":
            continue
        comparisons = ["full_linear"]
        if row["model"] == "receiver_tensor":
            comparisons.append("shuffled_context")
        for comparison in comparisons:
            base = by_model[row["row"], comparison]
            for axis in ("all", row["axis"]):
                gains[
                    row["role"],
                    axis,
                    row["query"],
                    row["arm"],
                    row["model"],
                    comparison,
                ][row["family"]].append(base["error"] ** 2 - row["error"] ** 2)
    paired = []
    for key, family in sorted(gains.items()):
        gain, lo, hi = family_interval([np.mean(v) for v in family.values()], spec)
        paired.append(
            {
                **dict(
                    zip(
                        ("role", "axis", "query", "arm", "model", "comparison"),
                        key,
                        strict=True,
                    )
                ),
                "families": len(family),
                "mse_gain": gain,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    noops = [r for r in predictions if r["exact_input_noop"]]
    summary = {
        "status": "FINITE_CONTEXT_COVECTOR_PACKET_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "paired_query_interventions": len(metadata),
        "unique_state_pairs": len(delta),
        "predictions": len(predictions),
        "new_source_kernel_fits": 9,
        "inherited_source_functions": 3,
        "new_model_calls": 0,
        "context_pairs_moved": manifest["context_pairs_moved"],
        "context_pairs_fixed": manifest["context_pairs_fixed"],
        "actual_noop_max": max(abs(r["observed_action"]) for r in noops),
        "predicted_noop_max": max(abs(r["predicted_action"]) for r in noops),
        "all_action_cells": cells,
        "interval_scope": "lexical-family bootstrap conditional on frozen source laws; not source-refit or human uncertainty",
        "psychological_accuracy": False,
    }
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/action_metrics.csv", cells)
    write_csv(out / "02/order_token_metrics.csv", details)
    write_csv(out / "02/family_metrics.csv", family_cells)
    write_csv(out / "02/paired_mechanism_gains.csv", paired)
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
    """Run one bounded CPU stage, locked and without observer/model loading."""
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
