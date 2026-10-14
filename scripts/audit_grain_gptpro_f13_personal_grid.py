#!/usr/bin/env python3
"""Bounded replay of personal draws with explicit target covariance formulas."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def reference_mixture(parameters, y, z, times, logs, future):
    """Scalar filter over parameter draws, then explicit cross-target formulas."""
    a, b, eta = np.asarray(parameters).T
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.array(logs))
    steady = q / (1 - rho * rho)
    m, v = np.zeros(len(a)), steady.copy()
    for j in range(len(y)):
        if j:
            transition = rho ** int(times[j] - times[j - 1])
            m = transition * m
            v = transition**2 * v + steady * (1 - transition**2)
        # Information-weighted scalar update instead of gain recursion.
        precision = 1 / v + 1 / r
        m = (m / v + (y[j] - a - b * z[j]) / r) / precision
        v = 1 / precision
    g = sum(rho**k for k in range(1, 5)) / 4
    innovation_average = [sum(rho**k for k in range(5 - j)) / 4 for j in range(1, 5)]
    means = np.column_stack(
        [m, a + b * future[-1] + rho**4 * m, a + b * np.mean(future) + g * m]
    )
    covariance = np.zeros((len(a), 3, 3))
    covariance[:, 0, 0] = v
    covariance[:, 1, 1] = rho**8 * v + q * sum(rho ** (2 * k) for k in range(4)) + r
    covariance[:, 2, 2] = g * g * v + q * sum(c * c for c in innovation_average) + r / 4
    covariance[:, 0, 1] = covariance[:, 1, 0] = rho**4 * v
    covariance[:, 0, 2] = covariance[:, 2, 0] = g * v
    covariance[:, 1, 2] = covariance[:, 2, 1] = (
        rho**4 * g * v
        + q * sum(rho ** (3 - j) * c for j, c in enumerate(innovation_average))
        + r / 4
    )
    mean = means.sum(axis=0) / len(a)
    event = covariance.sum(axis=0) / len(a)
    personal = means.T @ means / len(a) - np.outer(mean, mean)
    return {
        "mean": mean,
        "event_state": event,
        "parameter": personal,
        "total": event + personal,
    }


def main():
    """Audit completed tasks without touching producer/consumer files or processes."""
    parser = argparse.ArgumentParser()
    for name in ("input", "source", "bootstrap", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    start = time.monotonic()
    checked = prefixes = draws = unsupported = 0
    maximum = 0.0
    missing = []
    for world in ("core", "near_root", "shared_shock"):
        for seed in (20260915, 20260916, 20260917):
            groups = {}
            for line in (
                (args.source / f"{world}_{seed}" / "application.jsonl")
                .read_text()
                .splitlines()
            ):
                r = json.loads(line)
                groups.setdefault(r["author_id"], []).append(r)
            for values in groups.values():
                values.sort(key=lambda r: r["t"])
            for index in range(200):
                root = args.input / f"{world}_{seed}" / f"b{index:03d}"
                out = args.output / f"{world}_{seed}_b{index:03d}.json"
                if out.exists():
                    result = json.loads(out.read_text())
                    if (
                        result["status"] != "DRAW_AND_TARGET_REPLAY"
                        or result["max_error"] > 1e-8
                    ):
                        raise ValueError(
                            "previous failed audit retained; not resumable as pass"
                        )
                    if (
                        hashlib.sha256(
                            (root / "personal_mixtures.json").read_bytes()
                        ).hexdigest()
                        != result["mixture_sha256"]
                    ):
                        raise ValueError("audited mixture changed")
                elif not (root / "summary.json").exists() or time.monotonic() - start >= args.max_seconds:
                    missing.append([world, seed, index])
                    continue
                else:
                    config = json.loads((root / "protocol_before_run.json").read_text())
                    for item in config["files"]:
                        if (
                            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
                            != item["sha256"]
                        ):
                            raise ValueError("input/code hash mismatch")
                    source = json.loads(
                        (
                            args.bootstrap / f"{world}_{seed}" / f"fit_{index:03d}.json"
                        ).read_text()
                    )
                    if (
                        Path(config["source_fit"]).resolve()
                        != (
                            args.bootstrap / f"{world}_{seed}" / f"fit_{index:03d}.json"
                        ).resolve()
                    ):
                        raise ValueError("wrong source fit")
                    payload = (root / "personal_mixtures.json").read_bytes()
                    records = json.loads(payload)
                    keys = set()
                    local_max, local_draws, failed = 0.0, 0, 0
                    for record in records:
                        key = (record["author_id"], record["budget"])
                        if key in keys:
                            raise ValueError("duplicate personal prefix")
                        keys.add(key)
                        posterior = record["posterior"]
                        if record["mixture"] is None:
                            if posterior["status"] == "LAPLACE_APPROXIMATION":
                                raise ValueError("missing supported mixture")
                            failed += 1
                            continue
                        if posterior["status"] != "LAPLACE_APPROXIMATION":
                            raise ValueError("unsupported mixture scored")
                        hessian = np.array(posterior["hessian"])
                        cov = np.array(posterior["parameter_covariance"])
                        np.testing.assert_allclose(hessian @ cov, np.eye(3), atol=1e-9)
                        expected_recipe = [
                            seed,
                            13,
                            5,
                            ("core", "near_root", "shared_shock").index(world),
                            index,
                            (16, 32, 48).index(key[1]),
                            sorted(groups).index(key[0]),
                            1,
                        ]
                        if record["rng_recipe"] != expected_recipe:
                            raise ValueError("personal random stream identity mismatch")
                        rng = np.random.default_rng(
                            np.random.SeedSequence(record["rng_recipe"])
                        )
                        normal = rng.normal(size=(config["draws"], 3))
                        parameters = (
                            np.array(posterior["fit"]["psi"])
                            + normal @ np.linalg.cholesky(cov).T
                        )
                        events = groups[key[0]]
                        n = key[1]
                        reference = reference_mixture(
                            parameters,
                            np.array([r["y"] for r in events[:n]]),
                            np.array([r["z"] for r in events[:n]]),
                            np.array([r["t"] for r in events[:n]]),
                            source["fit"]["log_sigma"],
                            [r["z"] for r in events[n : n + 4]],
                        )
                        for name, value in reference.items():
                            local_max = max(
                                local_max,
                                float(np.max(abs(value - record["mixture"][name]))),
                            )
                        local_draws += config["draws"]
                    if keys != {(a, n) for a in groups for n in (16, 32, 48)}:
                        raise ValueError("personal grid incomplete")
                    result = {
                        "status": "DRAW_AND_TARGET_REPLAY"
                        if local_max <= 1e-8
                        else "REPLAY_FAILED",
                        "prefixes": len(keys),
                        "parameter_draws": local_draws,
                        "unsupported": failed,
                        "max_error": local_max,
                        "mixture_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                    out.write_text(json.dumps(result) + "\n")
                    if local_max > 1e-8:
                        raise ValueError("personal mixture mismatch")
                checked += 1
                prefixes += result["prefixes"]
                draws += result["parameter_draws"]
                unsupported += result["unsupported"]
                maximum = max(maximum, result["max_error"])
                if checked % 10 == 0:
                    progress = {
                        "tasks": checked,
                        "prefixes": prefixes,
                        "draws": draws,
                        "max_error": maximum,
                    }
                    (args.output / "checkpoint.json").write_text(
                        json.dumps(progress) + "\n"
                    )
                    print(json.dumps(progress), flush=True)
    summary = {
        "status": "AVAILABLE_PERSONAL_DRAWS_REPLAYED",
        "tasks": checked,
        "prefixes": prefixes,
        "parameter_draws": draws,
        "unsupported": unsupported,
        "max_error": maximum,
        "missing": missing,
        "scope": "saved parameter draws and target calculations; not independent MAP refit or validation of Laplace approximation",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
