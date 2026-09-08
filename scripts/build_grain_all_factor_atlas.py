#!/usr/bin/env python3
"""M31: index each computable target's frozen equation and evidence, not a model search."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.mood_catalog import COMPOSITES
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def json_safe(value):
    """Keep unavailable statistics null rather than emitting invalid JSON NaN."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(out):
    """Require all planned collectors/reports and bind34 query functions to evidence."""
    core = BASE / "all_factor_function_analysis_m28_v2"
    source = BASE / "all_factor_source_atlas_m29_v1"
    narrow_ref = BASE / "narrow_functions_m30b_v1"
    narrow = BASE / "narrow_function_transport_m30b_v1"
    cross = BASE / "narrow_covariance_atlas_m31_v1"
    state = BASE / "counseling_mood_m30c_v1"
    inventory = BASE / "all_factor_inventory_m27_v1"
    expected = {
        core / "summary.json": "COMPLETE_M28",
        source / "summary.json": "COMPLETE_SOURCE_FUNCTION_ATLAS",
        narrow / "summary.json": "COMPLETE_NARROW_FUNCTION_TRANSPORT",
        cross / "summary.json": "COMPLETE_NARROW_COVARIANCE_ATLAS",
        state / "external_summary.json": "COMPLETE_M30C_EXPLORATORY_STATE_LINK",
    }
    for path, status in expected.items():
        if read(path)["status"] != status:
            raise ValueError(f"Complete packet required: {path.parent.name}")
    core_npz, narrow_npz = (
        core / "nine_query_functions.npz",
        narrow_ref / "reference.npz",
    )
    if (
        sha(core_npz) != read(core / "summary.json")["functions_sha256"]
        or sha(narrow_npz) != read(narrow_ref / "reference_manifest.json")["sha256"]
    ):
        raise ValueError("Frozen coefficient archives changed")
    source_metrics = pd.read_csv(source / "paired_native_function_metrics.csv")
    core_metrics = pd.read_csv(core / "native_function_metrics.csv")
    narrow_metrics = pd.read_csv(narrow / "native_function_external_metrics.csv")
    d1 = pd.read_csv(inventory / "diagonal_metrics.csv")
    definitions = read(narrow_ref / "queries.json")
    cards, index = [], []
    for archive, queries in (
        (core_npz, (*BIG5, *AXES)),
        (narrow_npz, tuple(definitions)),
    ):
        with np.load(archive, allow_pickle=False) as arrays:
            for q in queries:
                Z, h, G, g, b = (
                    arrays[f"{q}__{key}"] for key in ("Z", "h", "G", "g", "b")
                )
                if Z.shape != G.shape or h.shape != g.shape or Z.shape[1:] != h.shape:
                    raise ValueError("Query equation dimensions differ")
                is_core = q in (*BIG5, *AXES)
                family = (
                    "Big5"
                    if q in BIG5
                    else "MBTI"
                    if q in AXES
                    else "adapted_state"
                    if q in COMPOSITES
                    else "relational_session"
                )
                card = {
                    "query": q,
                    "family": family,
                    "equation": "f_q(P,Q)=b_q+sum(G_q*(P-P_Rq))+dot(g_q,Q-Q_Rq)",
                    "partition": "f_q=b_q+M_q+C_q+Q_q; mean tokens and centered configuration",
                    "archive": str(archive.relative_to(ROOT)),
                    "archive_sha256": sha(archive),
                    "array_keys": {
                        name: f"{q}__{key}"
                        for name, key in (
                            ("P_reference", "Z"),
                            ("Q_reference", "h"),
                            ("G", "G"),
                            ("g", "g"),
                            ("b", "b"),
                        )
                    },
                    "query_tokens": int(Z.shape[0]),
                    "hidden_dim": int(Z.shape[1]),
                    "anchor": float(b),
                    "human_training": False,
                    "neural_prefix_required": True,
                    "neural_suffix_required_for_scoring": False,
                    "reference_authors": "two independently fixed M7 PANDORA authors",
                    "definition": definitions.get(
                        q,
                        {
                            "positive_orientation": "E/N/F/J"
                            if q in AXES
                            else "higher domain score"
                        },
                    ),
                    "native_fidelity_evidence": core_metrics[
                        core_metrics["query"] == q
                    ].to_dict("records")
                    if is_core
                    else narrow_metrics[narrow_metrics["query"] == q][
                        ["view", "native_fidelity_r"]
                    ].to_dict("records"),
                    "human_external_evidence": (
                        source_metrics[source_metrics["query"] == q]
                        if is_core
                        else narrow_metrics[narrow_metrics["query"] == q]
                    ).to_dict("records"),
                    "broader_native_source_baseline": d1[
                        (d1["query"] == q) & (d1.observer == "native_query")
                    ].to_dict("records")
                    if is_core
                    else [],
                }
                cards.append(card)
                index.append(
                    {
                        k: card[k]
                        for k in (
                            "query",
                            "family",
                            "query_tokens",
                            "hidden_dim",
                            "archive_sha256",
                            "neural_prefix_required",
                            "human_training",
                        )
                    }
                )
    if len(cards) != 34 or len({c["query"] for c in cards}) != 34:
        raise ValueError("All9 core plus25 narrower equations required")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "factor_equation_ledger.json",
        json_safe({"version": "GRAIN_M31_V1", "cards": cards}),
    )
    pd.DataFrame(index).to_csv(out / "factor_index.csv", index=False)
    write(
        out / "evidence_manifest.json",
        {
            "status": "COMPLETE_ALL_FACTOR_EQUATION_ATLAS",
            "functions": 34,
            "core": 9,
            "relational_session_queries": 22,
            "adapted_state_queries": 3,
            "distinct_human_criteria_with_occasions": 43,
            "all_construction_uses_one_program": True,
            "all_targets_significant": False,
            "source_populations": "213 Big5,64 MBTI functional source;2328 MBTI broader D1;46 MEPS external",
            "scope": "shared mathematical construction with finite target- and opportunity-dependent external evidence; not universal psychometric equivalence",
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (*expected, core_npz, narrow_npz)
            },
            "ledger_sha256": sha(out / "factor_equation_ledger.json"),
        },
    )
    print(
        {
            "status": "COMPLETE_ALL_FACTOR_EQUATION_ATLAS",
            "functions": len(cards),
            "human_training": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use ignored local output")
    run(args.output_dir.resolve())
