#!/usr/bin/env python3
"""Verify NEXT43 target lineage, source exclusion and numerical replay."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.compiled_gated_readout import mean_readout
from scripts.run_grain_complete_transport_jvp import verify
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Recompute key metrics and component accounting without new native model calls."""
    out = BASE / "logic_gated_update_v1"
    verify(out)
    capture = read(out / "capture_freeze.json")
    frozen = read(out / "prediction_freeze.json")
    source = read(out / "source_freeze.json")
    for f in (capture, frozen, source):
        verify_hashes(out, f["files"])
    result = read(out / "analysis/summary.json")
    verify_hashes(ROOT, result["target_hashes"])
    for name, digest in result["files"].items():
        if any(
            sha(out / folder / name) != digest
            for folder in ("analysis", "analysis_replay")
        ):
            raise ValueError("Numerical analysis CSV replay differs")
    with np.load(out / "private_weights.npz", allow_pickle=False) as z:
        weights = {k: z[k].astype(float) for k in z.files}
    with np.load(out / "private_source_bank.npz", allow_pickle=False) as z:
        bank = dict(z)
    compiled, compiled_keys, compiled_modes, compile_error = [], [], [], 0.0
    for j in read(out / "private_jobs.json"):
        rec, pred = packet_arrays(out / "private_predictions" / (j["key"] + ".json"))
        expected = set(range(4)) - ({j["index"]} if j["cohort"] == "source4" else set())
        if set(rec["source_indices"]) != expected or len(rec["source_indices"]) != len(
            expected
        ):
            raise ValueError("Source exclusion differs")
        keep = np.asarray(
            [
                i in expected and q == j["query"]
                for i, q in zip(bank["index"], bank["query"], strict=True)
            ]
        )
        _, boundary = packet_arrays(out / "private_capture" / (j["key"] + ".json"))
        with np.load(
            out / "private_models" / (j["key"] + ".npz"), allow_pickle=False
        ) as z:
            means = dict(z)
        for mode in ("own", "pooled"):
            w = mean_readout(bank["x"][keep], means[mode + "_t"], weights)
            difference = boundary["d"].astype(float) @ w - pred[mode + "_source_BN"]
            compile_error = max(compile_error, float(abs(difference).max()))
            compiled.append(w)
            compiled_keys.append(j["key"])
            compiled_modes.append(mode)
    if compile_error > 1e-12:
        raise ValueError("Post-outcome source compilation differs from frozen operator")
    np.savez_compressed(
        out / "private_compiled_readouts.npz",
        w=np.asarray(compiled),
        key=np.asarray(compiled_keys),
        mode=np.asarray(compiled_modes),
    )
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    table = pd.read_csv(out / "analysis/comparison.csv")
    if (
        len(actions) != 768
        or actions.duplicated(["key", "mode", "arm", "factor"]).any()
    ):
        raise ValueError("Action coverage differs")
    metric_error = 0.0
    for _, r in table.iterrows():
        a = actions
        for k in ("cohort", "query", "mode", "arm"):
            a = a[a[k] == r[k]]
        y, p = a.actual.to_numpy(), a.predicted.to_numpy()
        recalculated = {
            "action_r": np.corrcoef(y, p)[0, 1],
            "action_mse_over_zero": np.square(y - p).sum() / np.square(y).sum(),
        }
        metric_error = max(
            metric_error, *(abs(r[k] - v) for k, v in recalculated.items())
        )
    components = pd.read_csv(out / "analysis/components.csv")
    share_error = float(
        abs(
            components.input_projected_share + components.gate_projected_share - 1
        ).max()
    )
    scalar_error = float(
        abs(actions.input_term + actions.gate_term - actions.predicted).max()
    )
    if metric_error > 1e-12 or share_error > 1e-12 or scalar_error > 1e-12:
        raise ValueError("Independent scalar accounting differs")
    if capture["runtime"]["jvps"] != 96 or any(
        f["runtime"][k]
        for f in (capture, frozen)
        for k in ("current_scores", "backwards", "fits", "extra_denials")
    ):
        raise ValueError("Execution boundary differs")
    report = {
        "capture_packets": 24,
        "prediction_packets": 24,
        "exact_analysis_csvs": len(result["files"]),
        "source_exclusions_verified": 24,
        "metric_replay_max": metric_error,
        "signed_share_sum_error": share_error,
        "scalar_sum_max_error": scalar_error,
        "analytic_max_relative_l2": result["analytic_max_relative_l2"],
        "analytic_max_absolute": result["analytic_max_absolute"],
        "capture_seconds": capture["runtime"]["seconds"],
        "numeric_seconds": frozen["runtime"]["seconds"],
        "jvps": 96,
        "new_target_scores_or_backwards": 0,
        "post_outcome_compilation_max_error": compile_error,
        "compiled_source_readouts": len(compiled),
        "compiled_readouts_sha256": sha(out / "private_compiled_readouts.npz"),
    }
    write(out / "replay_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
