#!/usr/bin/env python3
"""HM17: decompose frozen context laws; no new fit or observer invocation."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_context_operator_decomposition as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_covector import check as context_check
from scripts.run_grain_math_context_covector import load_views
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_ordered_moments import summarize
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_context_operator_v1.json"
PLAN = ROOT / "docs/GRAIN_CONTEXT_OPERATOR_DECOMPOSITION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/context_operator_decomposition_r2"
CODE = (
    "grain_research/construction/math_context_operator_decomposition.py",
    "scripts/run_grain_math_context_operator_decomposition.py",
    "tests/test_grain_math_context_operator_decomposition.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
    "docs/GRAIN_CONTEXT_ACTION_GEOMETRY_V1.md",
)
KINDS = ("receiver_tensor", "endpoint_potential", "shuffled_context")
COMPARISONS = (
    ("receiver_without_skew", "receiver_tensor"),
    ("receiver_potential", "receiver_tensor"),
    ("receiver_potential", "endpoint_potential"),
    ("shuffled_without_skew", "shuffled_context"),
    ("shuffled_potential", "shuffled_context"),
    ("receiver_tensor", "shuffled_context"),
)


def check(out, geometry=False):
    """Verify inherited functions and every frozen input before reading results."""
    manifest = frozen_files(out)
    old = ROOT / manifest["spec"]["context_packet"]
    context_check(old, models=True)
    for name, expected in read_json(out / "00/context_artifacts.json").items():
        if digest(old / name) != expected:
            raise ValueError(f"HM16 artifact changed: {name}")
    verification = ROOT / manifest["spec"]["context_verification"]
    if digest(verification / "summary.json") != manifest["verification_sha256"]:
        raise ValueError("HM16 alternate verification changed")
    if geometry:
        for name, expected in read_json(out / "01/geometry_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"frozen geometry changed: {name}")
    return manifest


def prepare(out):
    """Freeze fixed coefficient ablations and all borrowed artifacts before analysis."""
    if out.exists():
        raise FileExistsError("preserve prepared packet; continue another stage")
    start, spec = time.monotonic(), read_json(SPEC)
    if spec["models"] != list(M.MODELS) or spec["new_source_fits"] != 0:
        raise ValueError("fixed eight diagnostics with zero new fits required")
    old = ROOT / spec["context_packet"]
    inherited = context_check(old, models=True)
    verification = ROOT / spec["context_verification"]
    verified = read_json(verification / "summary.json")
    verifier_name = "scripts/verify_grain_math_context_covector.py"
    if (
        verified["status"] != "VERIFIED_FINITE_HM16_NOT_GLOBAL_SCOPE_COMPLETE"
        or digest(ROOT / verifier_name) != verified["verifier_sha256"]
    ):
        raise ValueError("verified HM16 arithmetic/code required")
    ledger = read_json(verification / "frozen_primary_artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(old / name) != expected:
            raise ValueError(f"HM16 verified artifact changed: {name}")
    metadata = read_json(old / "00/metadata.json")
    if (
        len(metadata) != spec["paired_query_rows"]
        or spec["queries"] != inherited["spec"]["queries"]
    ):
        raise ValueError("original address/query scope changed")
    for role, n in spec["rows_by_role"].items():
        members = [r for r in metadata if r["role"] == role]
        if (
            len(members) != n
            or len({r["family"] for r in members}) != spec["families_by_role"][role]
        ):
            raise ValueError("original source/held scope changed")
    (out / "00").mkdir(parents=True)
    write_json(out / "00/context_artifacts.json", ledger)
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
                verifier_name: verified["verifier_sha256"],
                **{n: digest(ROOT / n) for n in CODE},
            },
            "inputs": {
                "00/context_artifacts.json": digest(out / "00/context_artifacts.json")
            },
            "verification_sha256": digest(verification / "summary.json"),
            "inherited_manifest_sha256": digest(old / "00/manifest.json"),
            "new_model_calls": 0,
            "new_source_fits": 0,
            "frozen_before_analysis": True,
            "result_after_diagnostic": True,
            "prepare_seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "OPERATOR_DIAGNOSTICS_FROZEN",
            "inherited_files": len(ledger),
            "new_fits": 0,
        },
        flush=True,
    )


def inputs(out):
    """Load coordinates and existing coefficients without loading held target values."""
    spec = read_json(out / "00/manifest.json")["spec"]
    old = ROOT / spec["context_packet"]
    with np.load(old / "01/source_coordinates.npz") as arrays:
        parameters = {
            k: arrays[k].copy() for k in ("origin", "context_scale", "delta_scale")
        }
        states = arrays["source_state_indices"].copy()
    views = load_views(old, parameters)
    functions = read_json(old / "01/functions.json")
    alphas, baselines = {}, {}
    for query in spec["queries"]:
        for kind in KINDS:
            f = functions[query][kind]
            with np.load(old / f["path"]) as arrays:
                if not np.array_equal(states, arrays["source_state_indices"]):
                    raise ValueError("source support changed by query/kind")
                alphas[query, kind] = arrays["weighted_dual"].copy() / f["kernel_unit"]
        with np.load(
            ROOT / functions[query]["full_linear"]["inherited_path"]
        ) as arrays:
            baselines[query] = arrays["coefficient"].copy() * parameters["delta_scale"]
    return spec, old, parameters, states, views, alphas, baselines


def cosine(value, left, right):
    """Leave zero-vector similarities missing rather than reporting perfect sharing."""
    return float(value / np.sqrt(left * right)) if left > 0 and right > 0 else None


def geometry(out):
    """Reconstruct source-supported operators and spectra without refitting or held labels."""
    check(out)
    if (out / "01").exists():
        raise FileExistsError("preserve frozen source geometry")
    spec, _, parameters, states, views, alphas, baselines = inputs(out)
    x, c, m, shuffled, _ = views
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": m,
        "shuffled_context": shuffled,
    }
    support = M.source_basis(x[states], c[states], spec["machine_tolerance_multiplier"])
    q = support["basis"]
    (out / "01").mkdir()
    np.savez_compressed(
        out / "01/source_support.npz", **support, source_state_indices=states
    )
    rows, cross, operators, max_gram_error = [], [], {}, 0.0
    for kind, ctx in contexts.items():
        weights = np.asarray([alphas[query, kind] for query in spec["queries"]])
        grams = M.gram_products(x[states], ctx[states], weights)
        if kind == "endpoint_potential":
            grams = {**grams, "A": grams["S"], "H": np.zeros_like(grams["H"])}
        np.savez_compressed(out / "01" / f"{kind}_cross_grams.npz", **grams)
        for k, query in enumerate(spec["queries"]):
            a, b = M.compact_operator(
                x[states], ctx[states], weights[k], q, kind == "endpoint_potential"
            )
            spectral = M.spectra(a, spec["machine_tolerance_multiplier"])
            operators[query, kind] = (a, b)
            np.savez_compressed(
                out / "01" / f"{query}_{kind}.npz",
                operator=a,
                linear=b,
                alpha=weights[k],
                **spectral["values"],
            )
            for part, value in {
                **spectral["frobenius_squared"],
                "b": float(b @ b),
            }.items():
                max_gram_error = max(max_gram_error, abs(value - grams[part][k, k]))
                rows.append(
                    {
                        "query": query,
                        "model": kind,
                        "part": part,
                        "frobenius_squared": value,
                        "source_gram_squared": float(grams[part][k, k]),
                        "rank": spectral["ranks"].get(part),
                        "support_rank": support["rank"],
                    }
                )
        for i, j in itertools.combinations(range(len(spec["queries"])), 2):
            for part, values in grams.items():
                cross.append(
                    {
                        "model": kind,
                        "query_a": spec["queries"][i],
                        "query_b": spec["queries"][j],
                        "part": part,
                        "inner_product": float(values[i, j]),
                        "cosine": cosine(values[i, j], values[i, i], values[j, j]),
                    }
                )
    for query in spec["queries"]:
        b = baselines[query]
        np.savez_compressed(out / "01" / f"{query}_full_linear.npz", linear=b)
        for kind in ("receiver_tensor", "endpoint_potential"):
            left, _ = operators[query, kind]
            right, _ = operators[query, "shuffled_context"]
            for part, a, v in (
                ("A", left, right),
                ("S", (left + left.T) / 2, (right + right.T) / 2),
                ("H", (left - left.T) / 2, (right - right.T) / 2),
            ):
                inner = float(np.sum(a * v))
                cross.append(
                    {
                        "model": f"{kind}_vs_shuffled_context",
                        "query_a": query,
                        "query_b": query,
                        "part": part,
                        "inner_product": inner,
                        "cosine": cosine(inner, np.sum(a * a), np.sum(v * v)),
                    }
                )
    write_csv(out / "01/source_geometry.csv", rows)
    write_csv(out / "01/cross_target_geometry.csv", cross)
    write_json(
        out / "01/geometry_manifest.json",
        {
            "source_rows_per_query": len(states),
            "ambient_dimension": x.shape[1],
            "support_rank": support["rank"],
            "support_tolerance": support["tolerance"],
            "source_reconstruction_max": support["source_reconstruction_max"],
            "basis_orthogonality_max": support["basis_orthogonality_max"],
            "discarded_positive_eigen_sum": support["discarded_positive_eigen_sum"],
            "compact_vs_gram_norm_max_error": max_gram_error,
            "kappa": float(parameters["delta_scale"] / parameters["context_scale"]),
            "new_source_fits": 0,
            "held_targets_loaded": False,
            "accuracy_selected_rank": False,
            "artifacts": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "01").iterdir())
                if p.is_file()
            },
        },
    )
    print(
        {
            "status": "SOURCE_OPERATOR_GEOMETRY_FROZEN",
            "support_rank": support["rank"],
            "norm_error": max_gram_error,
            "held_targets_loaded": False,
        },
        flush=True,
    )


def correlation(a, b):
    """Keep undefined correlations missing and retain the full signed values."""
    a, b = np.asarray(a), np.asarray(b)
    return (
        float(np.corrcoef(a, b)[0, 1])
        if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0
        else None
    )


def evaluate(out):
    """Replay all inherited actions and evaluate fixed component removals/corrections."""
    check(out, geometry=True)
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated operator diagnostics")
    start = time.monotonic()
    spec, old, parameters, states, views, alphas, baselines = inputs(out)
    x, c, m, shuffled, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    metadata = read_json(old / "00/metadata.json")
    y = np.empty(len(metadata))
    for role in ("source", "held"):
        with np.load(old / f"00/{role}_targets.npz") as arrays:
            y[arrays["indices"]] = arrays["y"]
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": m,
        "shuffled_context": shuffled,
    }
    actions, components, checks = {}, [], []
    for query in spec["queries"]:
        linear = x @ baselines[query]
        actions[query, "full_linear"] = linear
        query_parts = {
            "full_linear": {
                "linear": linear,
                **{k: np.zeros(len(x)) for k in ("symmetric", "skew", "finite_step")},
            }
        }
        for kind, ctx in contexts.items():
            alpha = alphas[query, kind]
            p = M.action_parts(x, ctx, x[states], ctx[states], alpha, kappa)
            if kind == "endpoint_potential":
                actions[query, kind] = p["linear"] + p["symmetric"]
                query_parts[kind] = {
                    **p,
                    "skew": np.zeros(len(x)),
                    "finite_step": np.zeros(len(x)),
                }
                begin, end = c, c + kappa * x
            else:
                actions[query, kind] = p["receiver"]
                stem = "receiver" if kind == "receiver_tensor" else "shuffled"
                actions[query, stem + "_without_skew"] = p["linear"] + p["symmetric"]
                actions[query, stem + "_potential"] = p["potential"]
                query_parts[kind] = p
                begin, end = ctx, ctx + kappa * x
            mid = (begin + end) / 2
            phi = [
                M.potential_value(z, x[states], ctx[states], alpha, kappa)
                for z in (begin, mid, end)
            ]
            expected = (
                actions[query, kind] if kind == "endpoint_potential" else p["potential"]
            )
            reverse = M.action_parts(-x, end, x[states], ctx[states], alpha, kappa)
            checks.append(
                {
                    "query": query,
                    "model": kind,
                    "virtual_shuffled_endpoints": kind == "shuffled_context",
                    "potential_difference_max_error": float(
                        np.max(abs(phi[2] - phi[0] - expected))
                    ),
                    "potential_reversal_max_error": float(
                        np.max(abs(reverse["potential"] + expected))
                    ),
                    "telescoping_max_error": float(
                        np.max(abs((phi[1] - phi[0]) + (phi[2] - phi[1]) - expected))
                    ),
                    "native_cycle_observed": False,
                }
            )
        for kind, p in query_parts.items():
            for row in metadata:
                if row["query"] != query:
                    continue
                j = row["state_index"]
                components.append(
                    {
                        **row,
                        "model": kind,
                        "observed_action": float(y[row["row"]]),
                        **{
                            k: float(p[k][j])
                            for k in ("linear", "symmetric", "skew", "finite_step")
                        },
                        "predicted_action": float(actions[query, kind][j]),
                        "receiver_minus_potential": float(
                            p["skew"][j] - p["finite_step"][j]
                        ),
                    }
                )
    predictions = []
    for model in spec["models"]:
        for row in metadata:
            j = row["state_index"]
            value = float(actions[row["query"], model][j])
            predictions.append(
                {
                    **row,
                    "model": model,
                    "observed_action": float(y[row["row"]]),
                    "predicted_action": value,
                    "error": float(value - y[row["row"]]),
                    "observation_delta_max": float(
                        np.max(abs(x[j])) * parameters["delta_scale"]
                    ),
                }
            )
    import csv

    with (old / "02/predictions.csv").open(newline="") as handle:
        inherited = {(int(r["row"]), r["model"]): r for r in csv.DictReader(handle)}
    max_replay, max_parts = 0.0, 0.0
    for row in components:
        previous = inherited[row["row"], row["model"]]
        max_replay = max(
            max_replay,
            abs(row["predicted_action"] - float(previous["predicted_action"])),
        )
        max_parts = max(
            max_parts,
            abs(row["linear"] - float(previous["linear_contribution"])),
            abs(
                row["symmetric"]
                + row["skew"]
                - float(previous["coupling_contribution"])
            ),
        )
    if max(max_replay, max_parts) > 1e-10:
        raise ValueError("inherited action replay failed")
    cells = summarize(predictions, ("role", "query", "model", "arm"), spec)
    details = summarize(
        predictions,
        ("role", "axis", "query", "model", "arm", "order", "opportunity", "tokens"),
        spec,
    )
    families = summarize(predictions, ("role", "family", "query", "model", "arm"), spec)
    lookup = {(r["row"], r["model"]): r for r in predictions}
    gains = defaultdict(lambda: defaultdict(list))
    for model, comparison in COMPARISONS:
        for row in metadata:
            a, b = lookup[row["row"], model], lookup[row["row"], comparison]
            for axis in ("all", row["axis"]):
                gains[row["role"], axis, row["query"], row["arm"], model, comparison][
                    row["family"]
                ].append(b["error"] ** 2 - a["error"] ** 2)
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
    sharing = []
    for role, arm, kind in sorted(
        {(r["role"], r["arm"], r["model"]) for r in components}
    ):
        group = [
            r
            for r in components
            if (r["role"], r["arm"], r["model"]) == (role, arm, kind)
        ]
        by_query = {
            q: {r["state_index"]: r for r in group if r["query"] == q}
            for q in spec["queries"]
        }
        for qa, qb in itertools.combinations(spec["queries"], 2):
            indices = sorted(by_query[qa])
            if indices != sorted(by_query[qb]):
                raise ValueError("cross-query action addresses not aligned")
            for part in (
                "observed_action",
                "linear",
                "symmetric",
                "skew",
                "finite_step",
                "predicted_action",
            ):
                a, b = ([by_query[q][j][part] for j in indices] for q in (qa, qb))
                sharing.append(
                    {
                        "role": role,
                        "arm": arm,
                        "model": kind,
                        "query_a": qa,
                        "query_b": qb,
                        "part": part,
                        "n": len(indices),
                        "correlation": correlation(a, b),
                    }
                )
    noops = [r for r in predictions if r["exact_input_noop"]]
    (out / "02").mkdir()
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/components.csv", components)
    write_csv(out / "02/action_metrics.csv", cells)
    write_csv(out / "02/order_token_metrics.csv", details)
    write_csv(out / "02/family_metrics.csv", families)
    write_csv(out / "02/paired_mechanism_gains.csv", paired)
    write_csv(out / "02/shared_action_correlations.csv", sharing)
    write_csv(out / "02/potential_checks.csv", checks)
    summary = {
        "status": "FINITE_OPERATOR_DIAGNOSTIC_COMPLETE_NOT_GLOBAL_SCOPE_COMPLETE",
        "paired_query_rows": len(metadata),
        "unique_state_pairs": len(x),
        "predictions": len(predictions),
        "component_rows": len(components),
        "new_source_fits": 0,
        "new_model_calls": 0,
        "api_calls": 0,
        "private_text_reads": 0,
        "inherited_prediction_replay_max_error": max_replay,
        "inherited_part_replay_max_error": max_parts,
        "potential_max_error": max(r["potential_difference_max_error"] for r in checks),
        "actual_noop_max": max(abs(r["observed_action"]) for r in noops),
        "predicted_noop_max": max(abs(r["predicted_action"]) for r in noops),
        "result_after_diagnostic": True,
        "native_cycle_observed": False,
        "interval_scope": "family bootstrap conditional on frozen source laws; not source-refit or human uncertainty",
        "psychological_accuracy": False,
        "evaluate_seconds": time.monotonic() - start,
    }
    write_json(out / "02/summary.json", summary)
    write_json(
        out / "02/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != out / "02/artifact_hashes.json"
        },
    )
    print(summary, flush=True)


def main():
    """Run a bounded CPU stage and preserve all earlier stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "geometry", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "geometry": geometry, "evaluate": evaluate}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
