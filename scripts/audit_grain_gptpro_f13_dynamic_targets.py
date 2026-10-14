#!/usr/bin/env python3
"""Replay dynamic draws by scalar state recursion, not joint loading matrices."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import grouped
from grain_research.gptpro_foundation_v3.f05_recovery import kalman_loglik
from scripts.audit_grain_gptpro_f13_nested import wilson


def main():
    """Check source linkage, every prediction and every reported coverage cell."""
    p = argparse.ArgumentParser()
    for name in ("input", "source", "application", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    link = json.loads((args.application / "input_manifest.json").read_text())
    assert (
        link["source_manifest_sha256"]
        == hashlib.sha256(
            (args.source / "output_manifest.json").read_bytes()
        ).hexdigest()
    )
    for item in json.loads((args.input / "input_manifest.json").read_text()):
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    errors = []
    all_rows = []
    vectors = 0
    for wi, world in enumerate(config["worlds"]):
        for seed in config["seeds"]:
            root = args.source / f"{world}_{seed}"
            people = grouped(
                [
                    json.loads(s)
                    for s in (root / "application.jsonl").read_text().splitlines()
                ]
            )
            source_ids = {
                json.loads(s)["author_id"]
                for s in (root / "source.jsonl").read_text().splitlines()
            }
            assert not source_ids & set(people)
            assert all(
                row["role"] == "application"
                for events in people.values()
                for row in events
            )
            model = json.loads((root / "source_model.json").read_text())
            truth = {
                r["author_id"]: r for r in json.loads((root / "truth.json").read_text())
            }
            cache = {
                (r["author_id"], r["n"]): r
                for r in json.loads(
                    (
                        args.application
                        / f"{world}_{seed}"
                        / "observed_predictions.json"
                    ).read_text()
                )
                if r["method"] == "personal_MAP" and r["phase"] == "prefix"
            }
            with (args.input / f"predictions_{world}_{seed}.csv").open() as f:
                rows = list(csv.DictReader(f))
            lookup = {
                (
                    r["author_id"],
                    int(r["budget"]),
                    r["method"],
                    r["arm"],
                    r["target"],
                ): r
                for r in rows
            }
            assert len(lookup) == len(rows)
            visited = set()
            for ni, n in enumerate(config["budgets"]):
                for i, author in enumerate(
                    sorted(people)[:6] if config["draws"] == 10 else sorted(people)
                ):
                    actual = truth[author]
                    events = people[author]
                    z = np.array([r["z"] for r in events[n : n + 4]])
                    true_values = [
                        actual["states"][n - 1],
                        events[n + 3]["y"],
                        np.mean([r["y"] for r in events[n : n + 4]]),
                    ]
                    for mi, method in enumerate(("plugin", "oracle")):
                        old = cache[(author, n)]
                        psi = np.array(
                            old["psi"]
                            if method == "plugin"
                            else [
                                actual["a"],
                                actual["b"],
                                np.arctanh(actual["rho"] / 0.99),
                            ]
                        )
                        logs = np.array(
                            model["log_sigma"]
                            if method == "plugin"
                            else np.log([0.5, 0.5])
                        )
                        filtered = kalman_loglik(psi, events[:n], logs)
                        m, v = (
                            filtered["state_mean"][-1],
                            filtered["state_variance"][-1],
                        )
                        rho = 0.99 * np.tanh(psi[2])
                        q, r = np.exp(2 * logs)
                        normal = np.random.default_rng(
                            np.random.SeedSequence([seed, 13, 5, wi, ni, i, mi])
                        ).normal(size=(config["draws"], 9))
                        state = m + np.sqrt(v) * normal[:, 0]
                        samples = [state.copy()]
                        for k in range(1, 5):
                            state = rho * state + np.sqrt(q) * normal[:, k]
                            samples.append(
                                psi[0]
                                + psi[1] * z[k - 1]
                                + state
                                + np.sqrt(r) * normal[:, 4 + k]
                            )
                        sampled = np.column_stack(
                            [samples[0], samples[4], np.mean(samples[1:], axis=0)]
                        )
                        exact_mean = [
                            m,
                            psi[0] + psi[1] * z[-1] + rho**4 * m,
                            psi[0]
                            + psi[1] * z.mean()
                            + m * sum(rho**k for k in range(1, 5)) / 4,
                        ]
                        diagonal = [
                            rho ** (2 * k) * v
                            + q * sum(rho ** (2 * j) for j in range(k))
                            + r
                            for k in range(1, 5)
                        ]
                        average_var = (
                            v * (sum(rho**k for k in range(1, 5)) / 4) ** 2
                            + q
                            / 16
                            * sum(
                                sum(rho**l for l in range(5 - j)) ** 2
                                for j in range(1, 5)
                            )
                            + r / 4
                        )
                        exact_var = [v, diagonal[-1], average_var]
                        for ti, target in enumerate(config["targets"]):
                            variants = [
                                ("joint_analytic", exact_mean[ti], exact_var[ti]),
                                (
                                    "joint_mc",
                                    sampled[:, ti].mean(),
                                    sampled[:, ti].var(),
                                ),
                            ]
                            if ti == 2:
                                variants.append(
                                    (
                                        "incorrect_diagonal_only",
                                        exact_mean[ti],
                                        sum(diagonal) / 16,
                                    )
                                )
                            for arm, mean, variance in variants:
                                key = (author, n, method, arm, target)
                                row = lookup[key]
                                visited.add(key)
                                errors.extend(
                                    [
                                        abs(mean - float(row["mean"])),
                                        abs(variance - float(row["variance"])),
                                        abs(true_values[ti] - float(row["truth"])),
                                    ]
                                )
                        vectors += config["draws"]
            assert visited == set(lookup)
            all_rows.extend(rows)
    groups = {}
    for row in all_rows:
        key = tuple(
            row[k] for k in ("world", "seed", "budget", "method", "arm", "target")
        )
        groups.setdefault(key, []).append(row)
    with (args.input / "metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    assert len(metrics) == len(groups)
    for row in metrics:
        key = tuple(
            row[k] for k in ("world", "seed", "budget", "method", "arm", "target")
        )
        group = groups[key]
        means, variances, targets = (
            np.array([float(r[k]) for r in group])
            for k in ("mean", "variance", "truth")
        )
        covered = abs(means - targets) <= 1.96 * np.sqrt(variances)
        low, high = wilson(int(covered.sum()), len(group))
        vals = [
            np.mean((means - targets) ** 2),
            covered.mean(),
            np.mean(3.92 * np.sqrt(variances)),
            low,
            high,
        ]
        for name, value in zip(
            ("mse", "coverage", "mean_width", "wilson_low", "wilson_high"),
            vals,
            strict=True,
        ):
            errors.append(abs(float(row[name]) - value))
        decision = (
            "DEPENDENT_AUTHORS_DESCRIPTIVE_ONLY"
            if key[0] == "shared_shock"
            else ("PASS" if low <= 0.95 <= high else "FAIL")
        )
        assert row["calibration_decision"] == decision
    result = {
        "status": "SCALAR_RECURSION_AND_ALL_TABLES_REPLAYED",
        "rows": len(all_rows),
        "metrics": len(metrics),
        "simulation_vectors": vectors,
        "max_error": max(errors),
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if max(errors) > 1e-9:
        raise ValueError("dynamic replay mismatch")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
