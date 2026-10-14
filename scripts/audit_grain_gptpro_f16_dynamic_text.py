#!/usr/bin/env python3
"""Audit dynamic text artifacts and compare ADF to non-Gaussian state grids."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as primary
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f16_t1_text import numeric_difference, save


def grid_filter(parameters, logs, z, categories, mask, channel, nodes):
    """Propagate probability mass on a stationary-SD grid; no Gaussian projection."""
    a, b, eta = parameters
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(logs))
    sd = np.sqrt(q / (1 - rho * rho))
    grid = np.linspace(-12 * sd, 12 * sd, nodes)
    dx = grid[1] - grid[0]
    mass = norm.pdf(grid, 0, sd) * dx
    mass /= mass.sum()
    transition = norm.pdf(grid[:, None], rho * grid[None, :], np.sqrt(q)) * dx
    ll, max_mass_loss = 0.0, 0.0
    for t in range(len(z)):
        if t:
            mass = transition @ mass
            total = mass.sum()
            max_mass_loss = max(max_mass_loss, abs(1 - total))
            mass /= total
        if mask[t]:
            c = int(categories[t])
            pole = ndtr((a + b * z[t] + grid) / np.sqrt(r))
            likelihood = 0.5 * (
                (1 - pole) * channel[0, 1, c] + pole * channel[1, 1, c]
            ) + 0.25 * (channel[0, 0, c] + channel[1, 0, c])
            mass *= likelihood
            total = mass.sum()
            ll += np.log(total)
            mass /= total
    mean = mass @ grid
    return {
        "mean": float(mean),
        "variance": float(mass @ ((grid - mean) ** 2)),
        "loglik": float(ll),
        "max_transition_mass_loss": float(max_mass_loss),
        "nodes": nodes,
    }


def projected_targets(score):
    """Explicit scalar summation independent of scorer einsum target contraction."""
    m = score["joint_mean"]
    c = score["joint_covariance"]
    return {
        "T2": (m[0], c[0][0]),
        "T3": (
            sum(m[1:]) / 4,
            sum(c[i][j] for i in range(1, 5) for j in range(1, 5)) / 16,
        ),
    }


def metric_record(rows):
    """Rebuild loss, correlation and interval summaries from saved scalar scores."""
    y, x = rows.truth.to_numpy(), rows.score.to_numpy()
    centered_y, centered_x = y - y.mean(), x - x.mean()
    den = np.sqrt(np.dot(centered_y, centered_y) * np.dot(centered_x, centered_x))
    return {
        "mse": float(sum((x - y) ** 2) / len(x)),
        "bias": float(sum(x - y) / len(x)),
        "r": float(np.dot(centered_y, centered_x) / den),
        "coverage": float(sum((rows.lower <= y) & (y <= rows.upper)) / len(x)),
        "width": float(sum(rows.upper - rows.lower) / len(x)),
    }


def main():
    """Retain all approximation discrepancies; arithmetic PASS is not theory PASS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.run / "protocol_before_run.json").read_text())
    completion = args.run / "completion_protocol.json"
    if completion.exists():
        repair = json.loads(completion.read_text())
        runner = Path(__file__).with_name("complete_grain_gptpro_f16_dynamic_text.py")
        if (
            sha256_file(runner) != repair["code_hash"]
            or sha256_file(args.run / "protocol_before_run.json")
            != repair["original_protocol_hash"]
        ):
            raise ValueError("completion adapter/protocol changed")
    for kind in ("source_hashes", "code_hashes"):
        for path, value in protocol[kind].items():
            if sha256_file(path) != value:
                raise ValueError(f"changed {kind}: {path}")
    manifest = json.loads((args.run / "output_manifest.json").read_text())
    for path, value in manifest.items():
        if sha256_file(args.run / path) != value:
            raise ValueError(f"changed output: {path}")
    full = pd.read_csv(args.run / "scores.csv")
    table = pd.read_csv(args.run / "metrics_by_seed.csv")
    base = Path(__file__).resolve().parents[1] / "results/grain_gptpro_foundation_v3"
    panel, counter, max_error, target_rows, lineage = [], 0, 0.0, 0, 0
    for seed in protocol["seeds"]:
        root = args.run / str(seed)
        bundle = json.loads((root / "bundle.json").read_text())
        inputs = [
            json.loads(x)
            for x in (root / "application_inputs.jsonl").read_text().splitlines()
        ]
        arrays, _ = primary.prepare(inputs[:2], bundle)
        z, categories, _weights, mask = arrays
        components = [0] if protocol["smoke"] else [0, 100, 199]
        for b in components:
            model = bundle["source_models"][b]
            psi = np.asarray(model["prior"]["mu"])
            channel = np.asarray(bundle["channel_draws"]["hybrid_joint"][b])
            adf = primary.filter_nodes(
                psi[None], *arrays, model["log_sigma"], channel, "hybrid_joint", 47
            )
            for i, raw in enumerate(inputs[:2]):
                coarse = grid_filter(
                    psi, model["log_sigma"], z[i], categories[i], mask[i], channel, 301
                )
                fine = grid_filter(
                    psi, model["log_sigma"], z[i], categories[i], mask[i], channel, 601
                )
                vals = {
                    k: float(v[0, i])
                    for k, v in zip(("mean", "variance", "loglik"), adf, strict=True)
                }
                panel.append(
                    {
                        "seed": seed,
                        "author_id": raw["author_id"],
                        "source_component": b,
                        "adf": vals,
                        "grid301": coarse,
                        "grid601": fine,
                        "adf_minus_grid601": {k: vals[k] - fine[k] for k in vals},
                        "grid301_minus601": {k: coarse[k] - fine[k] for k in vals},
                    }
                )
        source = json.loads((root / "source_scores.json").read_text())
        for target in ("T2", "T3"):
            values = np.array([s["targets"][target]["score"] for s in source])
            ref = bundle["reference"][target]
            max_error = max(
                max_error,
                abs(values.mean() - ref["mu"]),
                abs(values.std(ddof=1) - ref["sigma"]),
            )
        scored = {
            a: json.loads((root / f"{a}_scores.json").read_text()) for a in primary.ARMS
        }
        cold = [
            json.loads(x)
            for x in (root / "application_cold_scores.jsonl").read_text().splitlines()
        ]
        max_error = max(max_error, numeric_difference(cold, scored["hybrid_joint"]))
        for a, items in scored.items():
            for i, score in enumerate(items):
                cov = np.asarray(score["joint_covariance"])
                reconstructed = sum(
                    np.asarray(c) for c in score["joint_components"].values()
                )
                max_error = max(max_error, float(abs(cov - reconstructed).max()))
                if np.linalg.eigvalsh(cov).min() < -1e-10:
                    raise ValueError("negative joint covariance")
                for t, (m, v) in projected_targets(score).items():
                    s = score["targets"][t]
                    max_error = max(
                        max_error,
                        abs(m - s["score"]),
                        abs(v - s["variance"]),
                        abs(v - sum(s["components"].values())),
                        abs(s["lower"] - (m - 1.96 * np.sqrt(v))),
                        abs(s["upper"] - (m + 1.96 * np.sqrt(v))),
                    )
                    record = full[
                        (full.seed == seed)
                        & (full.arm == a)
                        & (full.author_id == inputs[i]["author_id"])
                        & (full.target == t)
                    ].iloc[0]
                    max_error = max(
                        max_error, abs(record.score - m), abs(record.variance - v)
                    )
                    target_rows += 1
                counter += 1
        # Reconnect numeric targets and rendered observations without feeding them to scorer.
        archive = base / f"F05/source_full_r2/core_{seed}"
        observed = [
            json.loads(x)
            for x in (archive / "application.jsonl").read_text().splitlines()
        ]
        original = {(x["author_id"], x["event_id"]): x for x in observed}
        truth = {
            x["author_id"]: x
            for x in json.loads((archive / "truth.json").read_text())
            if x["role"] == "application"
        }
        ev = json.loads((root / "application_rendering_evaluator.json").read_text())
        for x in ev:
            o = original[x["author_id"], x["event_id"]]
            if any(x[k] != o[k] for k in ("y", "z", "t", "author_id")):
                raise ValueError("rendered event lineage mismatch")
            if x["qualification"] and x["rendered_pole"] != int(o["y"] >= 0):
                raise ValueError("eligible polarity mismatch")
            lineage += 1
        for x in inputs:
            author = x["author_id"]
            rows = sorted(
                (r for r in observed if r["author_id"] == author), key=lambda r: r["t"]
            )
            actual = truth[author]
            state = actual["states"][47]
            future = [
                actual["a"]
                + actual["b"] * primary.PLAN[j]
                + actual["states"][48 + j]
                + (
                    rows[48 + j]["y"]
                    - actual["a"]
                    - actual["b"] * rows[48 + j]["z"]
                    - actual["states"][48 + j]
                )
                for j in range(4)
            ]
            for t, y in [("T2", state), ("T3", sum(future) / 4)]:
                target = full[
                    (full.seed == seed)
                    & (full.author_id == author)
                    & (full.target == t)
                ]
                max_error = max(max_error, float(abs(target.truth - y).max()))
        print(
            json.dumps(
                {
                    "stage": "seed_audited",
                    "seed": seed,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    for row in table.to_dict("records"):
        subset = full[
            (full.seed == row["seed"])
            & (full.arm == row["arm"])
            & (full.target == row["target"])
        ]
        rec = metric_record(subset)
        max_error = max(max_error, max(abs(row[k] - v) for k, v in rec.items()))
    paired = pd.read_csv(args.run / "paired_contrasts.csv")
    for row in paired.to_dict("records"):
        subset = full[full.target == row["target"]]
        pair = subset[subset.arm == "hybrid_joint"].merge(
            subset[subset.arm == row["reference"]],
            on=["seed", "author_id"],
            suffixes=("_p", "_r"),
            validate="one_to_one",
        )
        delta = (pair.score_p - pair.truth_p) ** 2 - (pair.score_r - pair.truth_r) ** 2
        rng, replicates = np.random.default_rng(165), np.zeros(2000)
        for seed in protocol["seeds"]:
            values = delta[pair.seed == seed].to_numpy()
            replicates += values[
                rng.integers(0, len(values), (2000, len(values)))
            ].mean(1) / len(protocol["seeds"])
        lo, hi = np.quantile(replicates, [0.025, 0.975])
        max_error = max(
            max_error,
            abs(delta.mean() - row["delta_mse"]),
            abs(lo - row["ci_low"]),
            abs(hi - row["ci_high"]),
        )
    if max_error > 1e-10:
        raise ValueError(f"arithmetic audit discrepancy {max_error}")
    save(args.output / "adf_grid_comparison.json", panel)
    save(
        args.output / "summary.json",
        {
            "status": "SAVED_ARITHMETIC_LINEAGE_AND_FIXED_PARAMETER_GRID_AUDIT_COMPLETE",
            "score_objects": counter,
            "target_rows": target_rows,
            "rendered_events": lineage,
            "hashes": len(manifest),
            "metric_rows": len(table),
            "paired_intervals": len(paired),
            "max_arithmetic_error": max_error,
            "grid_cases": len(panel),
            "max_adf_error": {
                k: max(abs(x["adf_minus_grid601"][k]) for x in panel)
                for k in ("mean", "variance", "loglik")
            },
            "max_grid_resolution_difference": {
                k: max(abs(x["grid301_minus601"][k]) for x in panel)
                for k in ("mean", "variance", "loglik")
            },
            "max_grid_mass_loss": max(
                x["grid601"]["max_transition_mass_loss"] for x in panel
            ),
            "scope": "same-agent scalar/matrix arithmetic and fixed-parameter grid;not whole independent integrated posterior or human validation",
            "seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
