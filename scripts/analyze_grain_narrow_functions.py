#!/usr/bin/env python3
"""M30B/M31: frozen narrow-function transport, full criterion links and terms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import (
    association,
    bh_adjust,
    scales,
)
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.mood_catalog import COMPOSITES
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_all_factor_source import paired_quality


def score_states(functions, folders):
    """Score every saved P/Q packet before reading any human criteria."""
    rows = []
    for folder in folders:
        if read(folder / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Complete each native-state view first")
        for d in read(folder / "private_design.json"):
            path = folder / "private_packets" / f"cell_{d['index']:04d}.json"
            meta = read(path)
            if (
                any(
                    meta[k] != d[k]
                    for k in ("participant_code", "query", "view", "index")
                )
                or sha(path.with_suffix(".npz")) != meta["array_sha256"]
            ):
                raise ValueError("Native state provenance mismatch")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                parts = functions[d["query"]].parts(z["P"], z["Q"])
            rows.append(
                {
                    k: meta[k]
                    for k in ("participant_code", "query", "view", "native_score")
                }
                | parts
            )
    data = pd.DataFrame(rows)
    if (
        data.duplicated(["participant_code", "query", "view"]).any()
        or len(data) != 2300
    ):
        raise ValueError("Expected46 participants x25 queries x2 views")
    return data


def run(reference, folders, inventory, mood, out):
    """Evaluate transport without optimizing references, directions or thresholds."""
    if read(reference / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("All narrow functions must be fixed first")
    digest = sha(reference / "reference.npz")
    if digest != read(reference / "reference_manifest.json")["sha256"]:
        raise ValueError("Frozen reference functions changed")
    definitions = read(reference / "queries.json")
    with np.load(reference / "reference.npz", allow_pickle=False) as z:
        functions = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in definitions
        }
    expected_tokens = read(reference / "manifest.json")["question_tokens"]
    for folder in folders:
        m = read(folder / "manifest.json")
        if any(
            tokens != expected_tokens[q] for q, tokens in m["question_tokens"].items()
        ):
            raise ValueError("Native/reference query coordinates differ")
    data = score_states(functions, folders)
    out.mkdir(parents=True, exist_ok=False)
    data.to_csv(out / "private_function_values.csv", index=False)
    write(
        out / "function_freeze.json",
        {
            "sha256": sha(out / "private_function_values.csv"),
            "reference_sha256": digest,
            "criterion_joined_before_freeze": False,
        },
    )
    criterion_path = inventory / "private_meps_scores.csv"
    state_path = mood / "private_mood_scores.csv"
    for folder in folders:
        expected = read(folder / "manifest.json")["criterion_cache_sha256"]
        actual = sha(state_path if folder == mood else criterion_path)
        if actual != expected:
            raise ValueError("Audited criterion bytes changed")
    external = pd.read_csv(criterion_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    state = pd.read_csv(state_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    if not external.index.equals(state.index):
        raise ValueError("Relational/mood participant sets differ")
    scale_specs = {s.name: s for s in scales()}
    qualities, accounting, links = [], [], []
    for (view, q), group in data.groupby(["view", "query"]):
        group = group.set_index("participant_code").sort_index()
        is_state = q in COMPOSITES
        criteria = (state if is_state else external).reindex(group.index)
        intended = f"{'post' if view == 'free_chat' else 'mid'}_{q}" if is_state else q
        y = criteria[intended]
        if is_state:
            lo, hi, family = 1, 6, "state"
        else:
            lo, hi, family = (
                scale_specs[q].low,
                scale_specs[q].high,
                scale_specs[q].phase,
            )
        quality = paired_quality(
            group.score, group.native_score, (y - lo) / (hi - lo), draws=2000
        )
        qualities.append(
            {
                "view": view,
                "query": q,
                "criterion": intended,
                "family": family,
                **quality,
                "external_p": association(group.score, y)["p"],
            }
        )
        denominator = float(np.std(group.score, ddof=1) * np.std(y, ddof=1))
        pieces = {
            part: float(np.cov(group[part], y)[0, 1] / denominator)
            for part in ("mean", "configuration", "final")
        }
        total = association(group.score, y)["pearson_r"]
        accounting.append(
            {
                "view": view,
                "query": q,
                "total_external_r": total,
                **pieces,
                "accounting_error": abs(total - sum(pieces.values())),
            }
        )
        for k in criteria:
            links.append(
                {
                    "view": view,
                    "query": q,
                    "criterion": k,
                    "kind": "state" if is_state else "relational_session",
                    "intended": intended == k,
                    **association(group.score, criteria[k]),
                }
            )
    quality, terms, full = (
        pd.DataFrame(qualities),
        pd.DataFrame(accounting),
        pd.DataFrame(links),
    )
    quality["q_bh_view_family"] = quality.groupby(
        ["view", "family"]
    ).external_p.transform(bh_adjust)
    full["q_bh_view_kind"] = full.groupby(["view", "kind"]).p.transform(bh_adjust)
    quality.to_csv(out / "native_function_external_metrics.csv", index=False)
    terms.to_csv(out / "signed_external_r_contributions.csv", index=False)
    full.to_csv(out / "full_query_criterion_matrix.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_NARROW_FUNCTION_TRANSPORT",
            "queries": len(functions),
            "views": 2,
            "native_states_scored": len(data),
            "external_cells": len(full),
            "new_neural_calls": 0,
            "meps_training_or_label_fit": False,
            "reference_sha256": digest,
            "maximum_accounting_error": float(terms.accounting_error.max()),
            "criterion_sources": {
                str(p.relative_to(ROOT)): sha(p) for p in (criterion_path, state_path)
            },
        },
    )
    print(
        quality[
            [
                "view",
                "query",
                "native_fidelity_r",
                "function_external_r",
                "native_external_r",
                "q_bh_view_family",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-dir",
        "free-dir",
        "direct-dir",
        "mood-dir",
        "inventory-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Local ignored outputs only")
    run(
        a.reference_dir.resolve(),
        [a.free_dir.resolve(), a.direct_dir.resolve(), a.mood_dir.resolve()],
        a.inventory_dir.resolve(),
        a.mood_dir.resolve(),
        a.output_dir.resolve(),
    )
