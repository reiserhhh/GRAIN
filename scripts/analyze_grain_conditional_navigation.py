#!/usr/bin/env python3
"""CON00: read frozen navigation caches and locate conditional-response failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_navigation import (
    error_accounting,
    projected_response_error,
    response_metrics,
    scalar_gain_oracle,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.state_geometry import (
    linear_reader,
    residual_basis,
)
from grain_research.vector_mechanisms.state_transport import pack_state
from scripts.analyze_grain_evidence_packets import read_packet, response_rows
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_source_span_extension import OLD, load_extension
from scripts.run_grain_state_transport import load_operators

NATURAL = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
ROUTES = BASE / "navigation_v1/NAV-10/reference_routes_r1"
EARLY = BASE / "navigation_v1/NAV-10/state_natural_r1"
DEFAULT = BASE / "conditional_navigation_v1/CON-00/diagnosis_r1"
PAIRS = (
    ("self_report", "other_report", "actor"),
    ("endorsed", "denied", "endorsement"),
    ("today", "past", "time"),
    ("original", "remove", "removal"),
)


def support_inventory():
    """Inspect generated intervention support, not inferred semantics of real texts."""
    if (
        digest(OUTPUT / "worlds.json")
        != read_json(OUTPUT / "manifest.json")["worlds_sha256"]
    ):
        raise ValueError("Controlled world bytes changed")
    worlds = pd.DataFrame(read_json(OUTPUT / "worlds.json"))
    construction = worlds[worlds.split == "construction"]
    rows = []
    for field in ("actor", "opportunity", "negation", "event_time", "realization"):
        for value, count in construction[field].astype(str).value_counts().items():
            rows.append(
                {
                    "scope": "synthetic_construction",
                    "variable": field,
                    "value": value,
                    "n": count,
                }
            )
    natural = pd.DataFrame(read_json(NATURAL / "private_worlds.json"))
    for value, count in natural.condition.value_counts().items():
        rows.append(
            {
                "scope": "natural_evaluation",
                "variable": "condition",
                "value": value,
                "n": count,
            }
        )
    return pd.DataFrame(rows)


def response_tables():
    """Join same-world responses after checking hashes, never touch psychological labels."""
    frame = read_packet(NATURAL)
    path = ROUTES / "natural/predictions.csv"
    if digest(path) != read_json(path.parent / "prediction_freeze.json")["sha256"]:
        raise ValueError("Route predictions changed")
    routes = pd.read_csv(path)
    metadata = frame.drop_duplicates("world_id")[
        ["world_id", "scene_id", "split", "condition"]
    ]
    routes = routes.merge(metadata, on="world_id", validate="many_to_one")
    routes["readout"] = routes.reference + "/" + routes.method
    effects = response_rows(pd.concat([frame, routes], ignore_index=True), "evidence")
    keys = ["query", "scene_id", "split", "axis"]
    responses = effects.pivot(
        index=keys, columns="readout", values="score_delta"
    ).reset_index()
    metrics, accounting, oracles = [], [], []
    for (q, axis), group in responses.groupby(["query", "axis"]):
        for ref in ("R_real", "R_synth"):
            for method in ("A_linear", "A_source", "B_token", "B_context4"):
                program = ref + "/" + method
                ident = {"query": q, "axis": axis, "reference": ref, "method": method}
                for target in ("native", ref):
                    metrics.append(
                        ident
                        | {"target": target}
                        | response_metrics(group[target], group[program])
                    )
                accounting.append(
                    ident | error_accounting(group.native, group[ref], group[program])
                )
    for (q, author), group in responses.groupby(["query", "scene_id"]):
        for ref in ("R_real", "R_synth"):
            for method in ("A_linear", "A_source", "B_token", "B_context4"):
                oracles.append(
                    {
                        "query": q,
                        "source_unit": str(author),
                        "reference": ref,
                        "method": method,
                    }
                    | scalar_gain_oracle(group.native, group[ref + "/" + method])
                )
    return (
        responses,
        pd.DataFrame(metrics),
        pd.DataFrame(accounting),
        pd.DataFrame(oracles),
    )


def geometry_table():
    """Read cached paired states; project response errors, not pooled level errors."""
    _, old, _ = load_operators(OLD)
    new = load_extension(SPAN)
    models = {"A_linear": old, "A_source": new}
    bases = {m: {q: residual_basis(z[q]) for q in QUERIES} for m, z in models.items()}
    readers = {
        ref: load_functions(OUTPUT / file)
        for ref, file in (
            ("R_real", "real_reference.npz"),
            ("R_synth", "synthetic_reference.npz"),
        )
    }
    worlds = {w["world_id"]: w for w in read_json(NATURAL / "private_worlds.json")}
    design = read_json(NATURAL / "private_design.json")
    state = {}
    for d in design:
        i, q = d["index"], d["query"]
        meta, upper = checked_arrays(NATURAL / "private_packets" / f"unit_{i:04d}.json")
        emeta, early = checked_arrays(EARLY / "private_units" / f"unit_{i:04d}.json")
        nmeta, expanded = checked_arrays(
            SPAN / "natural/private_units" / f"unit_{i:04d}.json"
        )
        for ident in (meta, emeta["identity"], nmeta["identity"]):
            if any(ident[k] != d[k] for k in ("index", "query", "world_id")):
                raise ValueError("Cached response state identity mismatch")
        w = worlds[d["world_id"]]
        state[q, w["scene_id"], w["condition"]] = {
            "x": early["residual_mean"] - old[q]["mean_r"],
            "y": pack_state(upper["P"], upper["Q"]),
            "A_linear": early["transport1"],
            "A_source": expanded["estimated"],
        }
    rows = []
    for q in QUERIES:
        for author in sorted({w["scene_id"] for w in worlds.values()}):
            for low, high, axis in PAIRS:
                a, b = state[q, author, low], state[q, author, high]
                for method in models:
                    for ref, functions in readers.items():
                        reader, _ = linear_reader(functions[q])
                        values = projected_response_error(
                            b["x"] - a["x"],
                            b["y"] - a["y"],
                            b[method] - a[method],
                            bases[method][q],
                            reader.sum(1),
                        )
                        rows.append(
                            {
                                "query": q,
                                "source_unit": str(author),
                                "axis": axis,
                                "method": method,
                                "reference": ref,
                            }
                            | values
                        )
    return pd.DataFrame(rows)


def main():
    """Save aggregate diagnostics with input/code provenance and no neural work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(
        (BASE / "conditional_navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored conditional-navigation output")
    with threadpool_limits(limits=2):
        responses, metrics, accounting, oracle = response_tables()
        support, geometry = support_inventory(), geometry_table()
    out.mkdir(parents=True)
    responses.to_csv(out / "private_responses.csv", index=False)
    oracle.to_csv(out / "private_gain_oracle.csv", index=False)
    geometry.to_csv(out / "private_response_geometry.csv", index=False)
    keys = ["query", "axis", "method", "reference"]
    numeric = [k for k in geometry if k not in keys + ["source_unit"]]
    geometry.groupby(keys)[numeric].mean().reset_index().to_csv(
        out / "response_geometry.csv", index=False
    )
    oracle.groupby(["query", "reference", "method"])[
        ["raw_sse", "oracle_sse", "oracle_relative_sse_reduction"]
    ].mean().reset_index().to_csv(out / "gain_oracle_summary.csv", index=False)
    metrics.to_csv(out / "response_metrics.csv", index=False)
    accounting.to_csv(out / "error_accounting.csv", index=False)
    support.to_csv(out / "support_inventory.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "new_neural_calls": 0,
        "human_labels_used": False,
        "queries": list(QUERIES),
        "evaluation_authors": 12,
        "paired_edits_per_query": 48,
        "response_metric_rows": len(metrics),
        "geometry_rows": len(geometry),
        "max_error_accounting_difference": float(accounting.accounting_error.max()),
        "max_reader_geometry_accounting_difference": float(
            geometry.reader_accounting_error.max()
        ),
        "support_finding": "No systematically varied denial/time in the48 synthetic construction worlds; natural reference level snapshots are not matched edit support.",
        "oracle_policy": "Evaluation-author gain and upper-state geometry are descriptive only, never runtime inputs.",
        "next": "CON01: add supported source-reference endpoints; compare fixed objective x kernel family.",
    }
    write_json(out / "summary.json", summary)
    inputs = [
        OUTPUT / "manifest.json",
        OUTPUT / "worlds.json",
        NATURAL / "manifest.json",
        NATURAL / "prediction_freeze.json",
        ROUTES / "construction_freeze.json",
        ROUTES / "natural/prediction_freeze.json",
        SPAN / "construction_freeze.json",
        ROOT / "docs/GRAIN_CONDITIONAL_NAVIGATION_PLAN_V1.md",
    ]
    write_json(
        out / "manifest.json",
        {
            "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "grain_research/vector_mechanisms/conditional_navigation.py",
                )
            },
            "outputs": {p.name: digest(p) for p in out.glob("*.csv")},
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
