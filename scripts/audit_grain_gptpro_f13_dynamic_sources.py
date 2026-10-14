#!/usr/bin/env python3
"""Audit bootstrap identities, source moments and vectorized likelihoods."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def batch_nll(parameters, observations, conditions, logs):
    """Sum independent stationary AR-filter likelihoods across balanced authors."""
    p = np.asarray(parameters)
    y, z = np.asarray(observations), np.asarray(conditions)
    if y.shape != z.shape or len(p) != len(y):
        raise ValueError("aligned author panels required")
    rho = 0.99 * np.tanh(p[:, 2])
    q, r = np.exp(2 * np.asarray(logs))
    variance = q / (1 - rho * rho)
    mean = np.zeros(len(p))
    value = np.zeros(len(p))
    for t in range(y.shape[1]):
        if t:
            mean *= rho
            variance = rho**2 * variance + q
        residual = y[:, t] - p[:, 0] - p[:, 1] * z[:, t] - mean
        total = variance + r
        value += 0.5 * (np.log(2 * np.pi * total) + residual**2 / total)
        gain = variance / total
        mean += gain * residual
        variance *= 1 - gain
    return float(value.sum())


def main():
    """Audit every currently durable fit; unfinished grid remains explicitly open."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    for item in config["source_files_and_code"]:
        if (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("frozen source/code hash mismatch")
    checked, missing, failed, records = 0, [], [], []
    max_nll = max_prior = 0.0
    personal_failures = noise_failures = 0
    for wi, world in enumerate(config["worlds"]):
        for seed in config["seeds"]:
            groups = {}
            for line in (
                (args.source / f"{world}_{seed}" / "source.jsonl")
                .read_text()
                .splitlines()
            ):
                row = json.loads(line)
                assert row["role"] == "source"
                groups.setdefault(row["author_id"], []).append(row)
            ids = sorted(groups)
            panels = [sorted(groups[a], key=lambda x: x["t"])[:48] for a in ids]
            assert all([r["t"] for r in panel] == list(range(48)) for panel in panels)
            y = np.array([[r["y"] for r in panel] for panel in panels])
            z = np.array([[r["z"] for r in panel] for panel in panels])
            for b in range(config["outer"]):
                path = args.input / f"{world}_{seed}" / f"fit_{b:03d}.json"
                if not path.exists():
                    missing.append([world, seed, b])
                    continue
                raw = path.read_bytes()
                saved = json.loads(raw)
                assert (
                    saved["seed"] == seed
                    and saved["world"] == world
                    and saved["bootstrap"] == b
                )
                assert saved["source_ids"] == ids
                selected = np.random.default_rng(
                    np.random.SeedSequence([seed, 13, 5, wi, b, 0])
                ).integers(0, len(ids), len(ids))
                np.testing.assert_array_equal(selected, saved["indices"])
                if saved["status"] != "FITTED":
                    failed.append([world, seed, b, saved.get("error")])
                    continue
                model = saved["fit"]
                assert (
                    model["source_authors"] == len(ids)
                    and model["source_events_used"] == len(ids) * 48
                )
                assert len(model["rounds"]) == 2
                for round_ in model["rounds"]:
                    fits = round_["personal_fits"]
                    assert set(fits) == {f"copy_{i:04d}" for i in range(len(ids))}
                    psi = np.array(
                        [fits[f"copy_{i:04d}"]["psi"] for i in range(len(ids))]
                    )
                    success = np.array(
                        [fits[f"copy_{i:04d}"]["success"] for i in range(len(ids))]
                    )
                    personal_failures += int((~success).sum())
                    assert int(success.sum()) == round_["successful_authors"]
                    if success.sum() >= 2:
                        beta = psi[success]
                        mu = beta.sum(axis=0) / len(beta)
                        delta = beta - mu
                        cov = delta.T @ delta / (len(beta) - 1) + np.eye(3) * 1e-4
                        max_prior = max(
                            max_prior,
                            float(np.max(abs(mu - round_["prior"]["mu"]))),
                            float(np.max(abs(cov - round_["prior"]["Sigma"]))),
                        )
                    if round_["noise_success"]:
                        nll = batch_nll(
                            psi, y[selected], z[selected], round_["log_sigma"]
                        )
                        max_nll = max(max_nll, abs(nll - round_["noise_objective"]))
                    else:
                        noise_failures += 1
                assert model["prior"] == model["rounds"][-1]["prior"]
                assert model["log_sigma"] == model["rounds"][-1]["log_sigma"]
                checked += 1
                records.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    summary = {
        "status": "SOURCE_IDENTITIES_MOMENTS_LIKELIHOODS_AUDITED",
        "fits_checked": checked,
        "missing": missing,
        "failed_tasks": failed,
        "personal_optimizer_failures": personal_failures,
        "noise_optimizer_failures": noise_failures,
        "max_prior_error": max_prior,
        "max_noise_objective_error": max_nll,
        "scope": "does not refit optimizers or establish global optimality; shared-shock bootstrap remains episode-conditional",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps(records) + "\n")
    if max_prior > 1e-9 or max_nll > 1e-6:
        raise ValueError("source audit mismatch")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "missing"}
            | {"missing_count": len(missing)}
        )
    )


if __name__ == "__main__":
    main()
