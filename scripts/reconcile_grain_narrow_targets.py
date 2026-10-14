#!/usr/bin/env python3
"""Replay narrow-target history and make reference-function anchoring explicit."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.affine_reference import (
    combine_affine_references,
    score_affine_reference,
)
from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.mood_catalog import COMPOSITES
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_narrow_covariance import run as replay_covariance
from scripts.analyze_grain_narrow_functions import run as replay_transport
from scripts.build_grain_all_factor_atlas import run as replay_atlas

PLAN = ROOT / "docs/GRAIN_NARROW_TARGET_RECONCILIATION_PLAN_V1.md"
REFERENCE = BASE / "narrow_functions_m30b_v1"
TRANSPORT = BASE / "narrow_function_transport_m30b_v1"
COVARIANCE = BASE / "narrow_covariance_atlas_m31_v1"
ATLAS = BASE / "all_factor_equation_atlas_m31_v1"
INVENTORY = BASE / "all_factor_inventory_m27_v1"
MOOD = BASE / "counseling_mood_m30c_v1"
FOLDERS = (
    BASE / "counseling_external_m30_free_v2",
    BASE / "counseling_external_m30_direct_v2",
    MOOD,
)
KEYS = ("Z", "h", "G", "g", "b")


def comparison(old, new):
    """Compare full tables, including identifiers, missingness and numeric values."""
    left, right = pd.read_csv(old), pd.read_csv(new)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise ValueError("Historical table schema/size changed: " + old.name)
    error = 0.0
    for key in left:
        if pd.api.types.is_numeric_dtype(left[key]) and pd.api.types.is_numeric_dtype(
            right[key]
        ):
            x, y = left[key].to_numpy(float), right[key].to_numpy(float)
            np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
            np.testing.assert_allclose(x, y, rtol=0, atol=2e-12, equal_nan=True)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                error = max(error, float(abs(x[finite] - y[finite]).max()))
        else:
            pd.testing.assert_series_equal(left[key], right[key], check_names=False)
    return {
        "file": old.name,
        "rows": len(left),
        "columns": len(left.columns),
        "byte_identical": sha(old) == sha(new),
        "max_numeric_difference": error,
        "old_sha256": sha(old),
        "new_sha256": sha(new),
    }


def verify_references(out):
    """Rebuild 25 averages from 50 fixed PANDORA source charts and bind tokens."""
    manifest, design = (
        read(REFERENCE / "manifest.json"),
        read(REFERENCE / "private_design.json"),
    )
    if (
        read(REFERENCE / "run_status.json")["status"] != "COMPLETE"
        or sha(REFERENCE / "reference.npz")
        != read(REFERENCE / "reference_manifest.json")["sha256"]
    ):
        raise ValueError("Complete frozen narrow functions required")
    if sha(REFERENCE / "private_design.json") != manifest["design_sha256"]:
        raise ValueError("Changed construction design")
    inputs = {
        str(REFERENCE / name): sha(REFERENCE / name)
        for name in (
            "manifest.json",
            "private_design.json",
            "reference.npz",
            "reference_manifest.json",
            "queries.json",
        )
    }
    for name, expected in manifest["sources"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError("Changed reference-source artifact")
        inputs[str(path)] = expected
    refs = {
        v["author_id"]
        for v in read(BASE / "query_region_functional_v1/private_design.json")
        if v["role"] == "reference"
    }
    if (
        len(refs) != 2
        or manifest["meps_reference_authors"] != 0
        or manifest["human_outcome_fit"]
    ):
        raise ValueError("Only the two original PANDORA construction authors allowed")
    fitted, rows, maximum, units = {}, [], 0.0, 0
    with np.load(REFERENCE / "reference.npz", allow_pickle=False) as frozen:
        for query in manifest["queries"]:
            selected = [r for r in design if r["query"] == query]
            if len(selected) != 2 or {r["author_id"] for r in selected} != refs:
                raise ValueError("Query reference membership changed")
            charts = []
            for row in selected:
                p = REFERENCE / "private_units" / f"unit_{row['index']:03d}.json"
                metadata = read(p)
                if (
                    metadata["index"] != row["index"]
                    or metadata["query"] != query
                    or sha(p.with_suffix(".npz")) != metadata["array_sha256"]
                ):
                    raise ValueError("Reference chart identity/hash mismatch")
                np.testing.assert_array_equal(
                    np.asarray(row["full_ids"])[np.asarray(row["full_mask"], bool)],
                    manifest["question_tokens"][query],
                )
                with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
                    charts.append({k: z[k] for k in KEYS})
                inputs[str(p)] = sha(p)
                inputs[str(p.with_suffix(".npz"))] = metadata["array_sha256"]
                units += 1
            result = combine_affine_references(charts)
            for key in KEYS:
                np.testing.assert_allclose(
                    result[key], frozen[f"{query}__{key}"], rtol=0, atol=2e-12
                )
                maximum = max(
                    maximum, float(abs(result[key] - frozen[f"{query}__{key}"]).max())
                )
            mean_score = np.mean(
                [score_affine_reference(result, c["Z"], c["h"]) for c in charts]
            )
            np.testing.assert_allclose(
                mean_score, result["b_mean_anchor"], rtol=0, atol=2e-12
            )
            fitted[query] = result
            rows.append(
                {
                    "query": query,
                    "reference_authors": 2,
                    "question_tokens": result["Z"].shape[0],
                    "hidden_dim": result["Z"].shape[1],
                    **{
                        k: result[k]
                        for k in (
                            "b_mean_anchor",
                            "b_chart_mean",
                            "mean_anchor_minus_chart_mean",
                            "p_reference_operator_covariance",
                            "q_reference_operator_covariance",
                        )
                    },
                    "source_mean_anchor_error": abs(
                        mean_score - result["b_mean_anchor"]
                    ),
                    "historical_convention": "mean_anchor_not_mean_of_full_charts",
                }
            )
    pd.DataFrame(rows).to_csv(out / "anchor_conventions.csv", index=False)
    return fitted, manifest, inputs, maximum, units


def verify_states(out, functions, reference_manifest):
    """Cold direct-dot scoring before external joins, preserving all state addresses."""
    old = pd.read_csv(TRANSPORT / "private_function_values.csv").set_index(
        ["participant_code", "view", "query"]
    )
    if (
        old.index.has_duplicates
        or sha(TRANSPORT / "private_function_values.csv")
        != read(TRANSPORT / "function_freeze.json")["sha256"]
    ):
        raise ValueError("Historical score addresses changed")
    rows, seen, inputs, maximum = [], set(), {}, 0.0
    for folder in FOLDERS:
        manifest = read(folder / "manifest.json")
        if (
            read(folder / "run_status.json")["status"] != "COMPLETE"
            or manifest["model_sha256"] != reference_manifest["model_sha256"]
            or manifest["alphabet"] != reference_manifest["alphabet"]
        ):
            raise ValueError("Model/state alphabet changed")
        if sha(folder / "private_design.json") != manifest["input_sha256"]:
            raise ValueError("Native design changed")
        for query, tokens in manifest["question_tokens"].items():
            if tokens != reference_manifest["question_tokens"][query]:
                raise ValueError(
                    "Same shape is not enough: question coordinates changed"
                )
        inputs[str(folder / "manifest.json")] = sha(folder / "manifest.json")
        inputs[str(folder / "private_design.json")] = sha(
            folder / "private_design.json"
        )
        for design in read(folder / "private_design.json"):
            path = folder / "private_packets" / f"cell_{design['index']:04d}.json"
            metadata = read(path)
            if (
                any(
                    metadata[k] != design[k]
                    for k in ("participant_code", "query", "view", "index")
                )
                or sha(path.with_suffix(".npz")) != metadata["array_sha256"]
            ):
                raise ValueError("Native packet mismatch")
            inputs[str(path)] = sha(path)
            inputs[str(path.with_suffix(".npz"))] = metadata["array_sha256"]
            key = (metadata["participant_code"], metadata["view"], metadata["query"])
            if key in seen or key not in old.index:
                raise ValueError("Duplicate/unknown participant-query-view")
            seen.add(key)
            ref = functions[metadata["query"]]
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                p, q = z["P"].astype(float), z["Q"].astype(float)
                value = score_affine_reference(ref, p, q)
                alternate = score_affine_reference(ref, p, q, "chart_mean")
                delta = p - ref["Z"]
                mu = delta.mean(0)
                parts = {
                    "mean": float(ref["G"].sum(0) @ mu),
                    "configuration": float(
                        np.sum((ref["G"] - ref["G"].mean(0)) * (delta - mu))
                    ),
                    "final": float(ref["g"] @ (q - ref["h"])),
                }
            expected = old.loc[key]
            for k, v in {**parts, "score": value}.items():
                np.testing.assert_allclose(v, expected[k], rtol=0, atol=2e-12)
                maximum = max(maximum, abs(float(v - expected[k])))
            np.testing.assert_allclose(
                value - alternate,
                ref["mean_anchor_minus_chart_mean"],
                rtol=0,
                atol=2e-12,
            )
            rows.append(
                {
                    "participant_code": key[0],
                    "view": key[1],
                    "query": key[2],
                    "score_mean_anchor": value,
                    "score_chart_mean": alternate,
                    "native_score": metadata["native_score"],
                    **parts,
                }
            )
    if len(seen) != 2300 or set(old.index) != seen:
        raise ValueError("Incomplete 46 x25 x2 cached evaluation")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_direct_function_values.csv", index=False)
    write(
        out / "direct_score_freeze.json",
        {
            "sha256": sha(out / "private_direct_function_values.csv"),
            "external_scores_joined_before_direct_freeze": False,
        },
    )
    return frame, inputs, maximum


def anchor_correlation_check(out, frame):
    """Confirm constant-offset invariance on authorized intended numeric criteria."""
    ext = pd.read_csv(INVENTORY / "private_meps_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    mood = pd.read_csv(MOOD / "private_mood_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    if not ext.index.equals(mood.index) or len(ext) != 46:
        raise ValueError("Validated participant criteria do not match")
    rows = []
    for (view, query), group in frame.groupby(["view", "query"]):
        group = group.set_index("participant_code").sort_index()
        key = (
            f"{'post' if view == 'free_chat' else 'mid'}_{query}"
            if query in COMPOSITES
            else query
        )
        y = (mood if query in COMPOSITES else ext).reindex(group.index)[key]
        original = association(group.score_mean_anchor, y)["pearson_r"]
        alternate = association(group.score_chart_mean, y)["pearson_r"]
        np.testing.assert_allclose(original, alternate, rtol=0, atol=2e-12)
        rows.append(
            {
                "view": view,
                "query": query,
                "criterion": key,
                "n": len(group),
                "mean_anchor_r": original,
                "chart_mean_r": alternate,
                "r_difference": alternate - original,
            }
        )
    pd.DataFrame(rows).to_csv(out / "anchor_correlation_invariance.csv", index=False)
    return rows


def source_lines(path, marker):
    """Record actual current lines rather than inventing provenance ranges."""
    return [
        i for i, line in enumerate(path.read_text().splitlines(), 1) if marker in line
    ]


def run(out):
    """Reconcile the finite narrow-target scope without overwriting original data."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    functions, manifest, inputs, coefficient_error, units = verify_references(out)
    frame, state_inputs, score_error = verify_states(out, functions, manifest)
    write(
        out / "input_manifest.json",
        {
            "plan_sha256": sha(PLAN),
            "runner_sha256": sha(Path(__file__)),
            "algebra_sha256": sha(ROOT / "grain_research/affine_reference.py"),
            "sources": inputs | state_inputs,
            "analyzer_sha256": {
                name: sha(ROOT / name)
                for name in (
                    "scripts/analyze_grain_narrow_functions.py",
                    "scripts/analyze_grain_narrow_covariance.py",
                    "scripts/build_grain_all_factor_atlas.py",
                    "scripts/analyze_grain_all_factor_source.py",
                    "grain_research/vector_mechanisms/functional_profiles.py",
                    "grain_research/vector_mechanisms/factor_catalog.py",
                )
            },
            "new_neural_calls": 0,
            "human_training": False,
            "meps_role": "previously_authorized_cached_numeric_validation_only",
        },
    )
    print(
        "25 coefficients /50 source units /2300 state packets directly verified",
        flush=True,
    )
    with (
        (out / "original_analyzer_replay.log").open("w") as log,
        contextlib.redirect_stdout(log),
    ):
        replay_transport(REFERENCE, list(FOLDERS), INVENTORY, MOOD, out / "transport")
        replay_covariance(out / "transport", INVENTORY, MOOD, out / "covariance")
        replay_atlas(out / "atlas")
    tables = []
    for old, new, names in (
        (
            TRANSPORT,
            out / "transport",
            (
                "private_function_values.csv",
                "native_function_external_metrics.csv",
                "signed_external_r_contributions.csv",
                "full_query_criterion_matrix.csv",
            ),
        ),
        (
            COVARIANCE,
            out / "covariance",
            (
                "all_query_all_criterion_links.csv",
                "cross_query_external_relations.csv",
                "signed_path_covariance.csv",
                *[
                    f"{v}_{n}_matrix.csv"
                    for v in sorted(frame["view"].unique())
                    for n in ("functional_query", "native_query", "intended_human")
                ],
            ),
        ),
        (ATLAS, out / "atlas", ("factor_index.csv",)),
    ):
        for name in names:
            tables.append(comparison(old / name, new / name))
    old_cards, new_cards = (
        read(ATLAS / "factor_equation_ledger.json"),
        read(out / "atlas/factor_equation_ledger.json"),
    )
    if old_cards != new_cards:
        raise ValueError("34 historical equation cards changed")
    invariance = anchor_correlation_check(out, frame)
    pd.DataFrame(tables).to_csv(out / "replay_comparisons.csv", index=False)
    evidence_rows = []
    source_plan = ROOT / "docs/GRAIN_ALL_FACTOR_RESEARCH_GOAL_V5.md"
    for card in new_cards["cards"]:
        narrow = card["query"] in functions
        evidence = (
            out / "transport/private_function_values.csv"
            if narrow
            else BASE
            / "all_plan_closure_v1/all_factor_source_replay_r1/paired_native_function_metrics.csv"
        )
        evidence_rows.append(
            {
                "query": card["query"],
                "family": card["family"],
                "requirement_source": str(source_plan.relative_to(ROOT)),
                "requirement_source_sha256": sha(source_plan),
                "requirement_lines": source_lines(source_plan, "M30B")
                if narrow
                else source_lines(source_plan, "M29"),
                "reconciled_now": narrow,
                "new_human_training": False,
                "artifact": str(evidence.relative_to(ROOT)),
                "artifact_sha256": sha(evidence),
                "neural_prefix_required": True,
                "suffix_required": False,
                "all_34_raw_text_independence": "OPEN_NOT_CLOSED_BY_AFFINE_REPLAY",
                "anchor_convention": "separately_averaged_reference_and_operator_arrays",
                "full_narrow_edit_coverage": "finite_NAV_packet_only_adapted_dejected5"
                if narrow
                else "nine_legacy_edits_plus_finite_NAV_packet",
            }
        )
    write(
        out / "query_evidence_ledger.json",
        {"cards": evidence_rows, "old_nine_replay_reused_not_new_observations": True},
    )
    anchor = pd.read_csv(out / "anchor_conventions.csv")
    summary = {
        "status": "COMPLETE_FINITE_NARROW_RECONCILIATION",
        "queries_replayed": len(functions),
        "reference_units": units,
        "native_state_packets": len(frame),
        "equation_cards_rebuilt": len(new_cards["cards"]),
        "tables_compared": len(tables),
        "byte_identical_tables": sum(t["byte_identical"] for t in tables),
        "coefficient_max_error": coefficient_error,
        "direct_score_max_error": score_error,
        "intended_correlations_recomputed": len(invariance),
        "max_anchor_correlation_difference": max(
            abs(r["r_difference"]) for r in invariance
        ),
        "anchor_offset_abs_median": float(
            anchor.mean_anchor_minus_chart_mean.abs().median()
        ),
        "anchor_offset_abs_max": float(anchor.mean_anchor_minus_chart_mean.abs().max()),
        "new_neural_calls": 0,
        "human_training": False,
        "wall_seconds": time.monotonic() - start,
        "scope": "reconciliation_and_affine_identity_not_independent_participant_replication",
    }
    write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    """Run inside ignored result storage, with no automatic old collector restart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(BASE.resolve()):
        parser.error("Use ignored result storage")
    with threadpool_limits(limits=2):
        run(out)


if __name__ == "__main__":
    main()
