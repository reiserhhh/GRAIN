#!/usr/bin/env python3
"""Replay F06 B physical paths, selected gains and terminal error accounting."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from scripts.audit_grain_gptpro_f06_policy_sensitivity import dense_gaussian


def main():
    """Check all completed authored records without rerunning selection or fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier audit")
    args.output.mkdir(parents=True)
    protocol = json.loads((args.input / "protocol.json").read_text())
    checkpoint = json.loads((args.input / "checkpoint.json").read_text())
    candidates = [(-1.0, 0.0), (-1.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    indices = {"a": 0, "b": 1, "beta": 3, "state": 4}
    largest = 0.0
    steps_checked, metrics_checked, hashes = 0, 0, 0
    nonconverged = 0
    source_cache = {}
    for filename, expected_hash in checkpoint["completed"].items():
        path = args.input / "authors" / filename
        assert sha256(path) == expected_hash
        hashes += 1
        with gzip.open(path, "rt") as handle:
            packet = json.load(handle)
        truth = packet["truth"]
        if packet["seed"] not in source_cache:
            source_cache[packet["seed"]] = json.loads(
                (
                    Path(protocol["source"])
                    / f"core_{packet['seed']}"
                    / "source_model.json"
                ).read_text()
            )
        q, r = np.exp(2 * np.asarray(source_cache[packet["seed"]]["log_sigma"]))
        last_key, selected_z = None, None
        saved = {}
        for step in packet["steps"]:
            key = step["reactivity"], step["arm"]
            t = step["t"]
            if key != last_key:
                last_key = key
                state = reference = truth["initial"]
                last_h, perturbation = 0.0, 0.0
                observed = []
                fit_successes = 0
                selected_z = None
                if key[1] == "random":
                    initial_z = candidates[truth["random_choices"][0]][0]
                elif key[1] in ("fixed", "coverage"):
                    initial_z = -1.0
                else:
                    source = source_cache[packet["seed"]]
                    prior_mean = np.r_[source["prior"]["mu"], 0.4, 0.0]
                    prior_rho = 0.99 * np.tanh(prior_mean[2])
                    prior_covariance = np.zeros((5, 5))
                    prior_covariance[:3, :3] = source["prior"]["Sigma"]
                    prior_covariance[3, 3] = 0.04
                    prior_covariance[4, 4] = q / (1 - prior_rho**2)
                    transition = np.eye(5)
                    transition[4] = [0, 0, 0, 0, prior_rho]
                    predictive = transition @ prior_covariance @ transition.T
                    predictive[4, 4] += q
                    index = indices[step["design_target"]]
                    first_gains = []
                    for z in (-1.0, 1.0):
                        x = np.array([1.0, z, 0.0, 0.0, 1.0])
                        first_gains.append(
                            (predictive[index] @ x) ** 2 / (x @ predictive @ x + r)
                        )
                    initial_z = (-1.0, 1.0)[
                        int(
                            np.flatnonzero(
                                np.isclose(
                                    first_gains, max(first_gains), atol=1e-14, rtol=0
                                )
                            )[0]
                        )
                    ]
                assert step["observation"]["z"] == initial_z
            if selected_z is not None:
                assert step["observation"]["z"] == selected_z
            effect = 0.0 if key[0] == "zero" else truth["beta"]
            state = truth["rho"] * state + effect * last_h + truth["innovations"][t]
            reference = truth["rho"] * reference + truth["innovations"][t]
            row = step["observation"]
            expected_y = (
                truth["a"] + truth["b"] * row["z"] + state + truth["measurement"][t]
            )
            np.testing.assert_allclose(
                [state, reference, state - reference, expected_y],
                [
                    step["state_true"],
                    step["state_no_action"],
                    step["delta_state"],
                    row["y"],
                ],
                atol=1e-10,
                rtol=1e-8,
            )
            observed.append(row)
            mean = np.array(step["posterior_mean"])
            covariance = np.array(step["posterior_covariance"])
            rho = 0.99 * np.tanh(mean[2])
            drho = 0.99 * (1 - np.tanh(mean[2]) ** 2)
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-10, rtol=0)
            assert np.linalg.eigvalsh(covariance).min() >= -1e-10
            _, reconstructed_state = dense_gaussian(
                observed, mean[0], mean[1], rho, mean[3], q, r
            )
            largest = max(largest, abs(reconstructed_state - mean[4]))
            np.testing.assert_allclose(
                reconstructed_state, mean[4], atol=1e-10, rtol=1e-8
            )
            target = step["design_target"] or "a"
            gains = []
            for z, h in candidates:
                transformation = np.eye(5)
                transformation[4] = [0, 0, drho * mean[4], h, rho]
                predicted = transformation @ covariance @ transformation.T
                predicted[4, 4] += q
                observation = np.array([1.0, z, 0.0, 0.0, 1.0])
                cross = predicted @ observation
                joint = np.block(
                    [
                        [predicted, cross[:, None]],
                        [cross[None, :], np.array([[observation @ cross + r]])],
                    ]
                )
                conditioned = joint[:5, :5] - joint[:5, 5:] @ np.linalg.solve(
                    joint[5:, 5:], joint[5:, :5]
                )
                j = indices[target]
                gains.append(predicted[j, j] - conditioned[j, j])
            largest = max(
                largest,
                float(np.max(np.abs(np.asarray(gains) - step["candidate_gains"]))),
            )
            np.testing.assert_allclose(
                gains, step["candidate_gains"], atol=1e-10, rtol=1e-8
            )
            if t + 1 == protocol["events"]:
                selected_z, selected_h = row["z"], 0.0
            elif key[1] == "fixed":
                selected_z, selected_h = (-1.0 if (t + 1) % 2 == 0 else 1.0), 0.0
            elif key[1] == "random":
                selected_z, selected_h = candidates[truth["random_choices"][t + 1]]
            elif key[1] == "coverage":
                selected_z, selected_h = candidates[(t + 1) % 4]
            else:
                assert key[1] == "scalar_" + target
                selected_z, selected_h = candidates[
                    int(
                        np.flatnonzero(
                            np.isclose(gains, max(gains), atol=1e-14, rtol=0)
                        )[0]
                    )
                ]
            assert (step["next_z"], row["h"]) == (selected_z, selected_h)
            last_h = row["h"]
            perturbation += (state - reference) ** 2
            fit_successes += int(step["fit_success"])
            nonconverged += int(not step["fit_success"])
            saved[(key[0], key[1], t + 1)] = (
                step,
                perturbation,
                fit_successes,
                sum(x["h"] for x in observed[:-1]),
            )
            steps_checked += 1
        for metric in packet["metrics"]:
            step, perturbation, successes, actions = saved[
                (metric["reactivity"], metric["arm"], metric["budget"])
            ]
            j = indices[metric["target"]]
            effect = 0.0 if metric["reactivity"] == "zero" else truth["beta"]
            value = {
                "a": truth["a"],
                "b": truth["b"],
                "beta": effect,
                "state": step["state_true"],
            }[metric["target"]]
            estimate = step["posterior_mean"][j]
            variance = step["posterior_covariance"][j][j]
            expected = [
                value,
                estimate,
                variance,
                (estimate - value) ** 2,
                estimate - value,
                perturbation,
                actions,
                successes / metric["budget"],
            ]
            actual = [
                metric[k]
                for k in (
                    "truth",
                    "estimate",
                    "variance",
                    "MSE",
                    "bias",
                    "sum_squared_perturbation",
                    "effective_actions",
                    "prefix_fit_rate",
                )
            ]
            largest = max(largest, float(np.max(np.abs(np.array(expected) - actual))))
            np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=1e-8)
            metrics_checked += 1
    result = {
        "status": "PHYSICAL_PATH_DENSE_STATE_GAIN_METRIC_REPLAY_PASS",
        "completed_authors_audited": hashes,
        "expected_authors": checkpoint.get(
            "total_expected", sum(protocol["counts"].values()) * len(protocol["seeds"])
        ),
        "steps": steps_checked,
        "metrics": metrics_checked,
        "nonconverged_prefixes": nonconverged,
        "maximum_numeric_difference": float(largest),
        "scope": "same-agent alternate formulas; analytic parameter/Fisher gradients tested separately; aggregated intervals and final production rules still pending",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
