#!/usr/bin/env python3
"""Replay NEXT37 numeric assets and locate missing-action versus reading error."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_material_reader import direction
from grain_research.vector_mechanisms.reader_ensemble import unpack_models
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_reader_new_author_transport import (
    QUERIES,
    packet_arrays,
    verify_prediction_freeze,
)
from scripts.run_grain_real_author_path import BASE, sha, write


def audit(out):
    """Use current gradients only as explicitly marked evaluator oracles, never fit."""
    verify_prediction_freeze(out)
    dest = out / "audit"
    dest.mkdir(exist_ok=False)
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_grain_reader_ensemble.py"),
        "--jobs",
        str(out / "private_jobs.json"),
        "--output",
        str(dest / "private_replay.npz"),
        "--forbid",
        str(out / "private_packets/a00_TF_cont.npz"),
        "--forbid",
        str(out / "analysis/private_actions.csv"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    with (
        np.load(dest / "private_replay.npz") as replay,
        np.load(out / "private_predictions.npz") as original,
    ):
        if set(replay.files) != set(original.files):
            raise ValueError("Replay key set differs")
        for key in original.files:
            np.testing.assert_array_equal(replay[key], original[key])
        replay_count = len(original.files)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_grain_reader_new_author_transport.py"),
            "--name",
            out.name,
            "--analysis",
            "analysis_replay",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compared = []
    for file in sorted((out / "analysis").glob("*.csv")):
        if file.name.startswith("."):
            continue
        if sha(file) != sha(out / "analysis_replay" / file.name):
            raise ValueError("Numerical analysis replay differs: " + file.name)
        compared.append(file.name)
    oracle_rows, accounts, prefixes = [], [], []
    actual = pd.read_csv(out / "analysis/private_actions.csv")
    for q in QUERIES:
        with np.load(out / "private_models" / f"{q}__{q}.npz") as z:
            components = unpack_models(dict(z))
        allv = np.concatenate([m["basis"] for m in components])
        _, singular, vt = np.linalg.svd(allv, full_matrices=False)
        v = vt[singular > singular[0] * 1e-10]
        for i in range(8):
            with np.load(out / "private_inputs" / f"a{i:02d}.npz") as z:
                h, g = z["hidden"], z["grain"]
            _, arrays = packet_arrays(out / "private_packets" / f"a{i:02d}_{q}.json")
            j = arrays["body_gradient"].astype(float)
            projected = (j @ v.T) @ v
            fixed = actual[
                (actual["query"] == q) & (actual.arm == q) & (actual["index"] == i)
            ].sort_values("factor")
            dirs = np.stack([direction(h, g, k)[0] for k in range(4)])
            full = np.einsum("kth,th->k", dirs, j)
            oracle = np.einsum("kth,th->k", dirs, projected)
            np.testing.assert_allclose(full, fixed.actual, rtol=1e-12, atol=1e-12)
            for k in range(4):
                oracle_rows.append(
                    {
                        "query": q,
                        "index": i,
                        "factor": f"whole:F{k + 1:02d}",
                        "actual": full[k],
                        "oracle": oracle[k],
                        "predicted": fixed.predicted.iloc[k],
                    }
                )
        frame = pd.DataFrame([r for r in oracle_rows if r["query"] == q])
        y, p, o = (
            frame.actual.to_numpy(),
            frame.predicted.to_numpy(),
            frame.oracle.to_numpy(),
        )
        missing, reading = o - y, p - o
        energy = float(y @ y)
        total = float(np.sum((p - y) ** 2) / energy)
        db, dc, cross = (
            float(missing @ missing / energy),
            float(reading @ reading / energy),
            float(2 * missing @ reading / energy),
        )
        if abs(total - db - dc - cross) > 1e-10:
            raise ValueError("Action error identity failed")
        accounts.append(
            {
                "query": q,
                "union_rank": len(v),
                "oracle_uses_held_gradient": True,
                **action_metrics(y, o),
                "own_total_error": total,
                "missing_action_error": db,
                "reading_action_error": dc,
                "signed_cross": cross,
                "identity_error": total - db - dc - cross,
            }
        )
    pd.DataFrame(oracle_rows).to_csv(dest / "private_oracle_actions.csv", index=False)
    table = pd.DataFrame(accounts)
    table.to_csv(dest / "oracle_action_account.csv", index=False)
    # AppleDouble ._ files are filesystem metadata, not experimental units.
    for folder in ("private_prefix", "private_packets"):
        for i in range(8):
            for q in QUERIES:
                packet_arrays(out / folder / f"a{i:02d}_{q}.json")
        prefixes.append(
            {
                "folder": folder,
                "verified_units": 24,
                "appledouble_sidecars": len(list((out / folder).glob("._*.json"))),
            }
        )
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "script_sha256": sha(Path(__file__)),
            "exact_prediction_arrays": replay_count,
            "exact_csv_files": compared,
            "runtime": json.loads(result.stdout),
            "packet_counts": prefixes,
            "counter_note": "Frozen run_status glob counters include AppleDouble metadata; use explicit verified24+24 here. Actual54 calls/24 backwards unchanged.",
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "oracle_action_account_sha256": sha(dest / "oracle_action_account.csv"),
        },
    )
    print(table.to_string(index=False))
    print(
        {
            "exact_prediction_arrays": replay_count,
            "exact_csv_files": len(compared),
            "verified_prefix_units": 24,
            "verified_gradient_units": 24,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="logic_reader_new_author_transport_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        audit(BASE / args.name)
