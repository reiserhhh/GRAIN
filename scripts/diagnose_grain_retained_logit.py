#!/usr/bin/env python3
"""NEXT24 cache-only reference diagnostics; no evaluation endpoints or Y reads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.retained_logit import path_account
from scripts.run_grain_action_geometry import PARENT, verify
from scripts.run_grain_joint_carrier_readout import check, donors_and_design
from scripts.run_grain_real_author_path import BASE, read, sha, write

ACTION = BASE / "logic_action_geometry_v1"
PLAN = ROOT / "docs/GRAIN_RETAINED_LOGIT_CURVATURE_PLAN_V1.md"


def reference_paths():
    """Enumerate planned cached nodes, never glob through AppleDouble sidecars."""
    frozen = read(PARENT / "carrier_nodes_freeze.json")
    check(PARENT / "private_integration_audit.json", frozen["audit_sha256"])
    audits = {
        (r["index"], r["body_case"]): r
        for r in read(PARENT / "private_integration_audit.json")
    }
    _, design = donors_and_design()
    check(
        ACTION / "private_displacements.npz",
        read(ACTION / "prediction_freeze.json")["files"]["private_displacements.npz"],
    )
    with np.load(ACTION / "private_displacements.npz") as z:
        displacements = z["displacements"].reshape(24, 136, 1536)
    rows = []
    for j, row in enumerate(design):
        if row["role"] != "reference":
            continue
        for body in (0, 1):
            audit = audits[row["index"], body]
            t = np.linspace(0, 1, audit["nodes"])
            p, g = [], []
            for fraction in t:
                name = f"private_nodes/ref_{row['index']:03d}_m{body}_t{int(fraction * 100):03d}.npz"
                check(PARENT / name, frozen["files"][name])
                with np.load(PARENT / name) as z:
                    if float(z["fraction"]) != fraction:
                        raise ValueError("Cached fraction disagrees with planned node")
                    p.append(float(z["score"]))
                    g.append(z["gradient"].astype(float))
            rows.append(
                {
                    "index": row["index"],
                    "body_case": body,
                    "fractions": t,
                    "probabilities": np.array(p),
                    "gradients": np.array(g),
                    "displacement": displacements[j],
                    "old_integration_error": audit["final_error"],
                }
            )
    if len(rows) != 8:
        raise ValueError("Exactly eight reference paths required")
    return rows


def diagnose(name):
    """Save private path details and aggregate identity/curvature evidence."""
    verify(ACTION)
    out = BASE / name
    out.mkdir(exist_ok=False)
    inputs = [
        PARENT / "carrier_nodes_freeze.json",
        ACTION / "prediction_freeze.json",
        ACTION / "manifest.json",
    ]
    code = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/retained_logit.py",
    ]
    write(
        out / "manifest.json",
        {
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "code": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    rows = []
    for path in reference_paths():
        account, _ = path_account(
            path["fractions"],
            path["probabilities"],
            path["gradients"],
            path["displacement"],
        )
        rows.append(
            {
                "index": path["index"],
                "body_case": path["body_case"],
                "old_integration_error": path["old_integration_error"],
                **account,
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(out / "private_path_accounts.csv", index=False)
    numeric = [
        "probability_change",
        "probability_tangent",
        "total_remainder",
        "link_curvature",
        "internal_remainder",
        "logit_change",
        "logit_tangent",
        "logit_remainder",
    ]
    summaries = []
    groups = [
        ("body0", data[data.body_case == 0]),
        ("body1", data[data.body_case == 1]),
        ("author_path_average", data.groupby("index")[numeric].mean()),
    ]
    for group, frame in groups:
        summaries.append(
            {
                "group": group,
                "authors": len(frame),
                **{
                    key + "_rms": float(np.sqrt(np.mean(frame[key] ** 2)))
                    for key in numeric
                },
            }
        )
    pd.DataFrame(summaries).to_csv(out / "reference_curvature_summary.csv", index=False)
    files = {
        p.name: sha(p)
        for p in (
            out / "private_path_accounts.csv",
            out / "reference_curvature_summary.csv",
        )
    }
    summary = {
        "status": "COMPLETE",
        "paths": 8,
        "reference_authors": 4,
        "new_neural_calls": 0,
        "human_labels_read": False,
        "evaluation_endpoints_read": False,
        "identity_max": float(data.identity_error.max()),
        "probability_integration_error_max": float(
            data.probability_integration_error.max()
        ),
        "old_error_replay_max": float(
            abs(data.probability_integration_error - data.old_integration_error).max()
        ),
        "logit_integration_error_max": float(data.logit_integration_error.max()),
        "logit_gradient_relative_change_mean": float(
            (data.logit_gradient_delta_norm / data.logit_gradient_origin_norm).mean()
        ),
        "probability_min": float(data.probability_min.min()),
        "probability_max": float(data.probability_max.max()),
        "files": files,
    }
    write(out / "summary.json", summary)
    print(pd.DataFrame(summaries).to_string(index=False))
    print({k: v for k, v in summary.items() if k != "files"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="logic_retained_logit_v1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Simple output name required")
    diagnose(args.name)
