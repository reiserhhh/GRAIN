#!/usr/bin/env python3
"""Replay frozen token/context programs on real-author evidence interventions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_program import score_context_charts
from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.projection_program import score_token_ids
from scripts.run_grain_context_program import load_program as load_context
from scripts.run_grain_projection_program import load_program as load_static
from scripts.run_grain_upstream_routes import collector_guard


def run(packet, out, seconds=1200):
    """Score fixed independent candidates without reading native/PQ packet outputs."""
    m = read_json(packet / "manifest.json")
    if (
        m["kind"] != "evidence"
        or digest(packet / "private_design.json")
        != m["input_sha256"]["private_design.json"]
    ):
        raise ValueError("Natural intervention design changed")
    static_dir = BASE / "navigation_v1/NAV-03/input_pullback_r1"
    context_dir = BASE / "navigation_v1/NAV-03/context_pullback_r2"
    static, table, static_charts = load_static(static_dir)
    context, runtime, context_charts = load_context(context_dir)
    expected = {
        "protocol": "NAV04_INDEPENDENT_RESPONSE_V1",
        "design_sha256": digest(packet / "private_design.json"),
        "program_sha256": {
            str(p.relative_to(BASE)): digest(p / "program.json")
            for p in (static_dir, context_dir)
        },
        "code_sha256": {
            p: digest(ROOT / p)
            for p in (
                "scripts/evaluate_grain_evidence_programs.py",
                "grain_research/vector_mechanisms/context_program.py",
                "grain_research/vector_mechanisms/projection_program.py",
            )
        },
        "methods": [
            "token_semantic_order",
            "context1_mean",
            "context4_mean",
            "context8_mean",
        ],
        "native_outputs_read": False,
        "full_PQ_inputs_read": False,
        "human_labels_read": False,
        "context_execution": "8 extracted blocks once, read cuts1/4/8",
    }
    if out.exists():
        if read_json(out / "manifest.json") != expected:
            raise ValueError("Frozen independent evaluation changed")
        if read_json(out / "run_status.json")["status"] == "COMPLETE":
            print(read_json(out / "run_status.json"), flush=True)
            return
    else:
        out.mkdir(parents=True)
        (out / "private_units").mkdir()
        write_json(out / "manifest.json", expected)
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    prior, start = read_json(out / "run_status.json"), time.monotonic()
    rows, design = [], read_json(packet / "private_design.json")
    with threadpool_limits(limits=2):
        for d in design:
            identity = {k: d[k] for k in ("index", "world_id", "query")}
            path = out / "private_units" / f"unit_{d['index']:04d}.json"
            if path.exists():
                stored = read_json(path)
                if stored["identity"] != identity:
                    raise ValueError("Prediction identity changed")
                rows.extend(stored["rows"])
                continue
            if time.monotonic() - start >= seconds:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "calls": prior["calls"] + runtime.calls,
                        "seconds": prior["seconds"] + time.monotonic() - start,
                    },
                )
                return
            q, ids, mask = d["query"], d["input_ids"], d["mask"]
            if any(
                np.asarray(ids)[mask].tolist() != p["question_tokens"][q]
                for p in (static, context)
            ):
                raise ValueError("Candidate query coordinates changed")
            result = score_token_ids(
                ids,
                mask,
                table,
                static_charts[q],
                static["anchors"][q],
                "semantic_order",
            )
            current = [
                identity
                | {
                    "readout": "token_semantic_order",
                    **{k: result[k] for k in ("score", "projection1", "projection2")},
                }
            ]
            states = runtime.capture(ids)
            for cut, h in states.items():
                value = score_context_charts(
                    context_charts[q][cut], h, mask, context["anchors"][q]
                )["mean"]
                current.append(
                    identity | {"readout": f"context{cut + 1}_mean", **value}
                )
            write_json(path, {"identity": identity, "rows": current})
            rows.extend(current)
            if (d["index"] + 1) % 30 == 0:
                print(
                    {
                        "status": "SCORING",
                        "units": d["index"] + 1,
                        "total": len(design),
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "predictions.csv"),
            "native_outputs_read": False,
            "full_PQ_inputs_read": False,
            "human_labels_read": False,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "units": len(design),
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--packet", type=Path, default=BASE / "navigation_v1/NAV-04/natural_evidence_r1"
    )
    p.add_argument(
        "--out", type=Path, default=BASE / "navigation_v1/NAV-04/natural_programs_r1"
    )
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    with collector_guard(a.out):
        run(a.packet, a.out, a.seconds)
