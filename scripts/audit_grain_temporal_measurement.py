#!/usr/bin/env python3
"""Independent Kalman, evidence-address and summary replay for M08/M09."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_moment_operator import read, sha, write
from scripts.run_grain_temporal_measurement import BANK, verify


def scientific_files(mapping):
    """Ignore macOS sidecars, not scientific files, in portable output inventories."""
    return {
        name: value
        for name, value in mapping.items()
        if not Path(name).name.startswith("._")
    }


def recursive_endpoint(t, y, z, u, cfg, arm):
    """Independent chronological filtering; no batch helper is called."""
    n = len(t)
    t = np.arange(n, dtype=float) if arm == "order_clock" else t
    u = np.zeros(n) if arm == "no_partner" else u
    a = -1e6 * np.eye(2) if arm == "independent_state" else np.asarray(cfg["drift"])
    stationary = np.asarray(cfg["stationary"])
    m = np.r_[cfg["profile_mean"], 0.0, 0.0]
    p = np.zeros((8, 8))
    p[:6, :6] = np.diag(np.square(cfg["profile_sd"]))
    p[6:, 6:] = stationary
    for i in range(n):
        if i:
            phi = expm(a * (t[i] - t[i - 1]))
            f = np.eye(8)
            f[6:, 6:] = phi
            process = np.zeros((8, 8))
            process[6:, 6:] = stationary - phi @ stationary @ phi.T
            m, p = f @ m, f @ p @ f.T + process
        h = np.zeros((2, 8))
        for k in range(2):
            h[k, 3 * k : 3 * k + 3] = [1, z[i, k], u[i]]
            h[k, 6 + k] = 1
        gain = np.linalg.solve(
            h @ p @ h.T + np.eye(2) / cfg["events_per_occasion"], h @ p
        ).T
        m = m + gain @ (y[i] - h @ m)
        # Joseph covariance update, independent of batch subtraction.
        r = np.eye(8) - gain @ h
        p = r @ p @ r.T + gain @ gain.T / cfg["events_per_occasion"]
    return m, p


def endpoint_replay(obs, pred, config):
    """All worlds/seeds/arms on fixed author0, not favorable cases."""
    error, checked = 0.0, 0
    for panel, group in obs[obs.author.eq(0)].groupby("panel"):
        group = group.sort_values("occasion")
        for n in config["prefixes"]:
            for arm in config["arms"]:
                frame = group.iloc[:n]
                m, p = recursive_endpoint(
                    frame.time.to_numpy(),
                    frame[["y0", "y1"]].to_numpy(),
                    frame[["zh0", "zh1"]].to_numpy(),
                    frame.partner.to_numpy(),
                    config,
                    arm,
                )
                rows = pred[
                    (pred.panel == panel) & (pred.prefix == n) & (pred.arm == arm)
                ]
                for row in rows.itertuples():
                    if row.target not in (
                        "level",
                        "condition",
                        "partner",
                        "current_state",
                        "current_response",
                        "conditional_profile",
                    ):
                        continue
                    k = row.channel
                    w = np.zeros(8)
                    if row.target in ("level", "condition", "partner"):
                        w[
                            3 * k + ("level", "condition", "partner").index(row.target)
                        ] = 1
                    elif row.target == "current_state":
                        w[6 + k] = 1
                    else:
                        z = (
                            frame.iloc[-1][f"zh{k}"]
                            if row.target == "current_response"
                            else 0.5
                        )
                        u = (
                            frame.partner.iloc[-1]
                            if row.target == "current_response" and arm != "no_partner"
                            else 0
                        )
                        w[3 * k : 3 * k + 3] = [1, z, u]
                        if row.target == "current_response":
                            w[6 + k] = 1
                    error = max(
                        error, abs(w @ m - row.score), abs(w @ p @ w - row.variance)
                    )
                    checked += 1
    if error > 1e-10:
        raise ArithmeticError("Independent endpoint replay failed")
    return {"checked_scores": checked, "max_error": error}


def evidence_replay(out, obs):
    """Reconstruct every decoded observation using immutable public text addresses."""
    source = pd.read_csv(BANK)
    addresses = pd.read_csv(out / "public_event_addresses.csv")
    source = source[["case_id", "partition", "evidence_hat", "condition_hat"]]
    joined = addresses.merge(source, on="case_id", validate="many_to_one")
    if len(joined) != len(addresses) or not joined.partition.eq("held").all():
        raise ValueError("Missing or source-wording substitution")
    agg = joined.groupby(["panel", "occasion", "channel"])[
        ["evidence_hat", "condition_hat"]
    ].mean()
    original = []
    for row in obs[obs.modality.eq("decoded_text")].itertuples():
        for k in range(2):
            original.append(
                {
                    "panel": row.panel,
                    "occasion": row.occasion,
                    "channel": k,
                    "evidence_hat": getattr(row, f"y{k}"),
                    "condition_hat": getattr(row, f"zh{k}"),
                }
            )
    ref = (
        pd.DataFrame(original).set_index(["panel", "occasion", "channel"]).sort_index()
    )
    error = float(abs(agg.sort_index() - ref).max().max())
    if error > 1e-12:
        raise ArithmeticError("Text observation aggregation changed")
    return {
        "event_addresses": len(addresses),
        "observations": len(agg),
        "max_error": error,
    }


def summarize(out, pred):
    """Whole-author paired bootstrap; channels and prefixes are not new people."""
    rows = []
    rng = np.random.default_rng(91379)
    f = pred[pred.prefix.eq(32)].copy()
    f["sq"] = (f.score - f.truth) ** 2
    for (world, modality, target), group in f.groupby(["world", "modality", "target"]):
        table = group.groupby(["panel", "arm"]).sq.mean().unstack()
        for ref in table.columns:
            if ref == "joint":
                continue
            old, new = table[ref].to_numpy(), table.joint.to_numpy()
            draws = rng.integers(0, len(table), size=(999, len(table)))
            delta = np.sqrt(old[draws].mean(1)) - np.sqrt(new[draws].mean(1))
            rows.append(
                {
                    "world": world,
                    "modality": modality,
                    "target": target,
                    "reference": ref,
                    "authors": len(table),
                    "reference_rmse": np.sqrt(old.mean()),
                    "joint_rmse": np.sqrt(new.mean()),
                    "rmse_reduction": np.sqrt(old.mean()) - np.sqrt(new.mean()),
                    "lower": np.quantile(delta, 0.025),
                    "upper": np.quantile(delta, 0.975),
                    "positive_authors": int((old > new).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "paired_comparisons.csv", index=False)
    macro = pred.copy()
    macro["squared_error"] = (macro.score - macro.truth) ** 2
    macro["covered"] = np.where(
        macro.lower.notna(),
        (macro.truth >= macro.lower) & (macro.truth <= macro.upper),
        np.nan,
    )
    keys = ["world", "modality", "arm", "prefix", "target"]
    summary = (
        macro.groupby(keys)
        .agg(
            mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            authors=("panel", "nunique"),
            cells=("score", "size"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary.mse)
    summary.to_csv(out / "macro_metrics.csv", index=False)
    macro.groupby(["seed", *keys]).agg(
        mse=("squared_error", "mean"), coverage=("covered", "mean")
    ).to_csv(out / "metrics_by_seed.csv")


def main():
    """Audit frozen outputs and write a public result, without any model refit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    config, contract = verify(out)
    for name in ("simulation_summary.json", "natural_summary.json"):
        for path, expected in scientific_files(read(out / name)["files"]).items():
            if sha(out / path) != expected:
                raise ValueError("Frozen output changed: " + path)
    obs, pred = (
        pd.read_csv(out / "simulation_observations.csv"),
        pd.read_csv(out / "predictions.csv"),
    )
    if len(obs) != 1152 * 32 or len(pred) != 248832:
        raise ValueError("Incomplete fixed simulation")
    with threadpool_limits(limits=1):
        numeric = endpoint_replay(obs, pred, config)
        text = evidence_replay(out, obs)
        summarize(out, pred)
    # The report's status does not convert declared working parameters into learned ones.
    write(
        out / "audit.json",
        {
            "status": "PASS",
            "inputs": len(contract["files"]),
            "endpoint": numeric,
            "text_observations": text,
            "mechanism_parameters_source_learned": False,
            "frozen_scores_refitted": False,
            "human_labels_read": False,
            "meps_scope": "Existing role/turn metadata only, no psychological scoring",
            "sidecar_inventory_correction": "Original natural manifest includes AppleDouble metadata; portable science inventory excludes it without changing results",
        },
    )
    names = [
        "contract.json",
        "config.json",
        "simulation_summary.json",
        "natural_summary.json",
        "audit.json",
        "paired_comparisons.csv",
        "macro_metrics.csv",
        "metrics_by_seed.csv",
    ]
    for name in ("simulation_summary.json", "natural_summary.json"):
        names.extend(scientific_files(read(out / name)["files"]))
    write(
        out / "delivery_manifest.json",
        {
            "files": {name: sha(out / name) for name in sorted(set(names))},
            "private_artifacts_local_only": True,
        },
    )
    print(read(out / "audit.json"), flush=True)


if __name__ == "__main__":
    main()
