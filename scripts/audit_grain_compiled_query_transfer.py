#!/usr/bin/env python3
"""Verify NEXT44 and derive an explicitly post-outcome equal shared reader."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_compiled_query_transfer import (
    BASE,
    MODES,
    QUERIES,
    arrays,
    fidelity,
    read,
    sha,
    validate_sources,
    verify,
    verify_hashes,
    write,
)


def main():
    """Recompute predictions, fixed-recipient metrics and signed components independently."""
    out = BASE / "logic_compiled_query_transfer_v1"
    verify(out)
    frozen = read(out / "prediction_freeze.json")
    verify_hashes(out, frozen["files"])
    summary = read(out / "analysis/summary.json")
    verify_hashes(ROOT, summary["target_hashes"])
    for folder in ("analysis", "analysis_replay"):
        verify_hashes(out / folder, summary["files"])
    alignment = read(out / "private_alignment.json")
    for a in alignment:
        assert a["same_body"] and a["same_grain_positions"]
        cohort, index = a["key"].split("_a")
        for members in a["source_indices"]:
            validate_sources(cohort, int(index), members)
    jobs = read(out / "private_jobs.json")
    packet = arrays(out / "private_predictions.npz")
    con = pd.read_csv(out / "analysis/private_contrasts.csv")
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    max_prediction, max_components, max_shared = 0.0, 0.0, 0.0
    shared_rows, shared, specific, slot_keys, modes = [], [], [], [], []
    for i, j in enumerate(jobs):
        a = arrays(out / "private_inputs" / (j["key"] + ".npz"))
        p = packet["predicted"][i]
        for m, mode in enumerate(MODES):
            w = a["w"][:, m].mean(0)
            e = (a["w"][0, m] - a["w"][1, m]) / 2
            shared.append(w)
            specific.append(e)
            slot_keys.append(j["key"])
            modes.append(mode)
            for r, recipient in enumerate(QUERIES):
                for donor in range(2):
                    direct = a["d"][r] @ a["w"][donor, m]
                    max_prediction = max(
                        max_prediction, float(abs(direct - p[r, donor, m]).max())
                    )
                equal = p[r, :, m].mean(0)
                max_shared = max(max_shared, float(abs(a["d"][r] @ w - equal).max()))
                subset = actions[
                    (actions.key == j["key"])
                    & (actions.recipient == recipient)
                    & (actions.donor == recipient)
                    & (actions["mode"] == mode)
                ].sort_values("factor")
                for f, target in enumerate(subset.actual):
                    shared_rows.append(
                        {
                            "key": j["key"],
                            "cohort": j["cohort"],
                            "index": j["index"],
                            "recipient": recipient,
                            "mode": mode,
                            "factor": f"whole:F{f + 1:02d}",
                            "actual": target,
                            "predicted": equal[f],
                        }
                    )
            c = con[(con.key == j["key"]) & (con["mode"] == mode)].sort_values("factor")
            reader = (p[0, 0, m] - p[0, 1, m] + p[1, 0, m] - p[1, 1, m]) / 2
            transport = (p[0, 0, m] + p[0, 1, m] - p[1, 0, m] - p[1, 1, m]) / 2
            max_components = max(
                max_components,
                float(abs(reader - c.reader).max()),
                float(abs(transport - c.transport).max()),
            )
    table = pd.read_csv(out / "analysis/comparison.csv")
    metric_error = 0.0
    for _, row in table.iterrows():
        g = actions
        for k in ("cohort", "recipient", "donor", "mode"):
            g = g[g[k] == row[k]]
        y, p = g.actual.to_numpy(), g.predicted.to_numpy()
        metric_error = max(
            metric_error,
            abs(row.action_r - np.corrcoef(y, p)[0, 1]),
            abs(row.action_mse_over_zero - np.square(y - p).sum() / np.square(y).sum()),
        )
    if max(max_prediction, max_components, max_shared, metric_error) > 1e-12:
        raise ValueError("Independent numerical replay differs")
    cs = pd.read_csv(out / "analysis/contrast_summary.csv")
    covariance_error = float(
        abs(
            cs.contrast_variance
            - cs.reader_variance
            - cs.transport_variance
            - 2 * cs.covariance
        ).max()
    )
    if covariance_error > 1e-12 or frozen["runtime"]["extra_denials"]:
        raise ValueError("Accounting or target-denial boundary differs")
    shared_frame = pd.DataFrame(shared_rows)
    shared_summary = []
    keys = ["cohort", "recipient", "mode"]
    for key, g in shared_frame.groupby(keys, sort=True):
        centered = g[["actual", "predicted"]] - g.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        shared_summary.append(
            dict(zip(keys, key, strict=True))
            | {
                "derivation": "post_outcome_equal_mean_not_selected_weights",
                "authors": g["index"].nunique(),
                **fidelity(g.actual, g.predicted),
                "factor_centered_action_r": fidelity(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
    dest = out / "derived_shared_reader"
    dest.mkdir(exist_ok=True)
    shared_frame.to_csv(dest / "private_actions.csv", index=False)
    pd.DataFrame(shared_summary).to_csv(dest / "comparison.csv", index=False)
    np.savez_compressed(
        dest / "private_readouts.npz",
        shared=np.asarray(shared),
        specific=np.asarray(specific),
        key=np.asarray(slot_keys),
        mode=np.asarray(modes),
    )
    report = {
        "pairs": len(jobs),
        "source_exclusions": len(alignment) * 2,
        "primary_predictions": 384,
        "prediction_replay_max": max_prediction,
        "component_replay_max": max_components,
        "independent_metric_max": float(metric_error),
        "covariance_replay_max": covariance_error,
        "exact_analysis_csvs": len(summary["files"]),
        "post_outcome_shared_rows": len(shared_frame),
        "shared_compile_error": max_shared,
        "shared_reader_sha256": sha(dest / "private_readouts.npz"),
        "neural_calls": 0,
        "fits": 0,
    }
    write(out / "replay_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
