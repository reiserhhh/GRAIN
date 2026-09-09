#!/usr/bin/env python3
"""Check finite NAV execution evidence without equating completion to model success."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)

NAV = BASE / "navigation_v1"
EVIDENCE = {
    "NAV-01": [
        "NAV-01/controlled_r1/run_status.json",
        "NAV-01/controlled_r1/native_and_function.csv",
    ],
    "NAV-02": [
        "NAV-02/query_sites_r1/summary.json",
        "NAV-02/query_sites_r1/patched_effects.csv",
        "NAV-02/state_transport_r1/program.json",
        "NAV-02/source_span_r1/natural/analysis/metrics.csv",
        "NAV-02/source_span_r1/center_span_analysis/summary.csv",
    ],
    "NAV-03": [
        "NAV-03/input_pullback_r1/evaluation.json",
        "NAV-03/context_pullback_r2/program.json",
        "NAV-10/context_external_r2/metrics.csv",
    ],
    "NAV-04": [
        "NAV-04/natural_evidence_r1/run_status.json",
        "NAV-04/natural_evidence_r1/analysis/response_metrics.csv",
    ],
    "NAV-05": [
        "NAV-05/query_identity_r1/run_status.json",
        "NAV-05/query_identity_r1/analysis/framing_coding_terms.csv",
        "NAV-05/query_identity_r1/analysis/coordinate_compatibility.json",
    ],
    "NAV-06": [
        "NAV-06/endpoint_exchange_r1/run_status.json",
        "NAV-06/endpoint_exchange_r1/shared_specific_edges.csv",
        "NAV-06/author_groups_r1/analysis/summary.json",
        "NAV-06/author_groups_r1/analysis/group_condition_effects.csv",
    ],
    "NAV-07": [
        "NAV-07/state_opportunity_r1/summary.json",
        "NAV-07/state_opportunity_r1/cross_occasion_links.csv",
        "NAV-07/temporal_order_r1/run_status.json",
        "NAV-07/temporal_order_r1/analysis/time_order_effects.csv",
    ],
    "NAV-08": [
        "NAV-08/author_panel_r1/analysis/batch_invariance.json",
        "NAV-08/author_panel_r1/analysis/reference_sensitivity.csv",
        "NAV-08/author_panel_r1/analysis/cross_view_stability.csv",
    ],
    "NAV-09": [
        "NAV-09/shared_prefix_r1/summary.json",
        "NAV-09/shared_prefix_r1/paired_timings.csv",
    ],
    "NAV-10": [
        "NAV-10/reference_routes_r1/analysis/summary.json",
        "NAV-10/reference_routes_r1/analysis/function_metrics.csv",
        "NAV-10/reference_routes_r1/analysis/response_metrics.csv",
        "NAV-10/reference_routes_r1/analysis/external_metrics.csv",
        "NAV-02/source_span_r1/external/analysis/summary.json",
    ],
    "NAV-11": [
        "NAV-11/runtime_export_r1/replay_token/summary.json",
        "NAV-11/runtime_export_r1/replay_context4/summary.json",
        "NAV-11/runtime_export_r1/replay_state1/summary.json",
        "NAV-11/runtime_export_r1/replay_state2/summary.json",
        "NAV-11/source_affine_r1/replay_source_affine/summary.json",
        "NAV-11/source_affine_r1/export.json",
    ],
    "NAV-12": [
        "NAV-12/context_atlas_r1/summary.json",
        "NAV-12/context_atlas_r1/conditional_signed_terms.csv",
        "NAV-12/context_atlas_r1/overlap_consistency.csv",
    ],
}


def run(out):
    """Verify packet counts, input/prediction freezes and actual runtime replays."""
    if out.exists():
        raise FileExistsError("Preserve prior delivery audit")
    ledger, sources = [], {}
    for packet, paths in EVIDENCE.items():
        for name in paths:
            path = NAV / name
            sources[name] = digest(path)
            if path.suffix == ".csv":
                frame = pd.read_csv(path)
                if frame.empty:
                    raise ValueError("Empty result: " + name)
                rows = len(frame)
            else:
                value = read_json(path)
                if not value:
                    raise ValueError("Empty evidence: " + name)
                status = value.get("status", "NO_STATUS_FIELD")
                if status != "NO_STATUS_FIELD" and not (
                    status.startswith("COMPLETE")
                    or status in ("CONSTRUCTION_COMPLETE", "PASS", "EXPORTED")
                ):
                    raise ValueError("Unfinished packet: " + name)
                rows = None
            ledger.append(
                {
                    "packet": packet,
                    "artifact": name,
                    "rows": rows,
                    "sha256": sources[name],
                }
            )
    if (
        read_json(NAV / "NAV-01/controlled_r1/run_status.json")["completed_units"]
        != 288
    ):
        raise ValueError("World packet incomplete")
    if read_json(NAV / "NAV-02/query_sites_r1/summary.json")["patches"] != 300:
        raise ValueError("Actual upstream interventions incomplete")
    if (
        read_json(NAV / "NAV-03/input_pullback_r1/evaluation.json")[
            "runtime_neural_calls"
        ]
        != 0
    ):
        raise ValueError("Token program is not Transformer-free")
    invariance = read_json(
        NAV / "NAV-08/author_panel_r1/analysis/batch_invariance.json"
    )
    if (
        invariance["maximum_old_author_shift"] != 0
        or not invariance["norm_never_refit_on_evaluation_batch"]
    ):
        raise ValueError("Reference invariance failed")
    ref = NAV / "NAV-10/reference_routes_r1"
    for mode, units in (("worlds", 288), ("natural", 288), ("external", 369)):
        folder = ref / mode
        frame = pd.read_csv(folder / "predictions.csv")
        if (
            len(frame) != units * 8
            or frame.duplicated(["index", "reference", "method"]).any()
        ):
            raise ValueError("Incomplete same-unit A/B x reader factorial")
        if (
            digest(folder / "predictions.csv")
            != read_json(folder / "prediction_freeze.json")["sha256"]
        ):
            raise ValueError("Route predictions changed")
        if read_json(folder / "prediction_freeze.json")["labels_read_before_freeze"]:
            raise ValueError("Human labels accessed before freeze")
    replay = []
    for name in EVIDENCE["NAV-11"][:-1]:
        r = read_json(NAV / name)
        if (
            r["status"] != "PASS"
            or not r["dependency_boundary_pass"]
            or r["rows"] != 15
        ):
            raise ValueError("Raw-text dependency replay failed")
        replay.append(r)
    out.mkdir(parents=True)
    pd.DataFrame(ledger).to_csv(out / "artifact_ledger.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "PASS_FINITE_EXECUTION_CONTRACT",
            "packets": 12,
            "evidence_files": len(sources),
            "evidence_sha256": sources,
            "raw_text_replay_rows": sum(r["rows"] for r in replay),
            "largest_raw_replay_error": max(r["max_abs_error"] for r in replay),
            "comparison_factorial": "4 independent candidates x2 reader origins x3 input families",
            "completion_meaning": "Every finite NAV card has implemented experiments, observations and a decision; negative results retained.",
            "universal_personality_program_proven": False,
            "all_34_targets_independent": False,
            "future_mechanism": "conditional derivative/response construction, not a new human-label grid",
            "script_sha256": digest(Path(__file__)),
        },
    )
    print(
        {"status": "PASS_FINITE_EXECUTION_CONTRACT", "packets": 12, "replay_rows": 75},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NAV / "delivery_audit_r1")
    run(parser.parse_args().out)
