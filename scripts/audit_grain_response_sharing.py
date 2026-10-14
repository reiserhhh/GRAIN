#!/usr/bin/env python3
"""Independent covariance, observation-address, scoring and matching replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_joint_conditional_observation import dump, read, sha
from scripts.run_grain_response_sharing import DEFAULT, TARGETS


def positive(matrix):
    """Spectral projection, without calling the production fit helper."""
    eig, vec = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vec * np.maximum(eig, 0)) @ vec.T


def program(model, observed, arm):
    """Joint covariance transform of the prediction error, independent of core."""
    obs = np.asarray(observed, int)
    m = np.asarray(model["mean"])
    k = np.asarray(
        model["geometry_kernel"] if arm == "condition_geometry" else model["kernel"]
    )
    r = np.asarray(model["noise"])
    s = (k + r)[np.ix_(obs, obs)]
    eig, vec = np.linalg.eigh(s)
    inv = (vec / eig) @ vec.T
    w = k[:, obs] @ inv
    if arm == "source_mean":
        w[:] = 0
    elif arm == "level_only":
        w = np.tile(w.sum(0) / len(m), (len(m), 1))
    joint = np.block([[k, k[:, obs]], [k[obs], s]])
    transform = np.c_[np.eye(len(m)), -w]
    covariance = transform @ joint @ transform.T
    return w, m - w @ m[obs], covariance


def main(out):
    """Check every saved source model, person function and final target metric."""
    cfg = read(out / "config.json")
    for name, digest in read(out / "freeze.json").items():
        assert sha(ROOT / name) == digest
    for name, digest in read(out / "generate/seal.json").items():
        assert sha(out / "generate" / name) == digest
    assert sha(out / "fit/models.json") == read(out / "fit/seal.json")["models"]
    models = read(out / "fit/models.json")
    metrics = pd.read_csv(out / "evaluate/metrics_by_seed.csv")
    bank = pd.read_csv(out / "generate/bank_values.csv").sort_values("case_id")
    lookup = bank.value.to_numpy()
    meter = read(out / "calibrate/meter.json")
    neural_pairs = 0
    for phase in ["cold_source", "cold_held"]:
        pairs = pd.read_csv(out / phase / "nli_pairs.csv")
        readings = pd.read_csv(out / phase / "readings.csv")
        a = pairs[pairs.polarity == 0].set_index("case_id").p_entailment
        b = pairs[pairs.polarity == 1].set_index("case_id").p_entailment
        np.testing.assert_allclose(
            readings.raw_value, readings.case_id.map(a - b), atol=1e-12
        )
        neural_pairs += len(pairs)
        for row in readings.itertuples():
            m = meter[str(row.condition_index)]
            np.testing.assert_allclose(
                lookup[row.case_id],
                (row.raw_value - m["offset"]) / m["scale"],
                atol=1e-12,
            )
    source_checks, address_checks = 0, 0
    with np.load(out / "generate/source.npz", allow_pickle=False) as source:
        for seed in cfg["seeds"]:
            for world in cfg["worlds"]:
                key = f"s{seed}_{world}"
                y, ids = source[key + "_halves"], source[key + "_cases"]
                np.testing.assert_allclose(lookup[ids].mean(-1), y, atol=1e-12)
                assert bank.iloc[ids.ravel()].variant.isin(cfg["source_variants"]).all()
                address_checks += ids.size
                a, b = y[:, :, 0], y[:, :, 1]
                for suffix in ["", "_unpaired"]:
                    m = models[key + suffix]
                    second = (
                        b
                        if not suffix
                        else b[np.random.default_rng(seed + 400000).permutation(len(b))]
                    )
                    cross = (a - a.mean(0)).T @ (second - second.mean(0)) / (len(a) - 1)
                    np.testing.assert_allclose(positive(cross), m["kernel"], atol=1e-12)
                    np.testing.assert_allclose(y.mean((0, 2)), m["mean"], atol=1e-12)
                    np.testing.assert_allclose(
                        positive(np.cov(a - b, rowvar=False) / 2)
                        + cfg["noise_floor"] * np.eye(5),
                        m["noise"],
                        atol=1e-12,
                    )
                    z = np.array(cfg["conditions"])
                    d = abs(z[:, None] - z)
                    width = np.median(d[d > 0])
                    bases = np.array(
                        [np.ones((5, 5)), np.exp(-0.5 * (d / width) ** 2), np.eye(5)]
                    )
                    coef = np.array(m["geometry_coefficients"])
                    geometry = np.einsum("k,kij->ij", coef, bases)
                    np.testing.assert_allclose(
                        geometry, m["geometry_kernel"], atol=1e-12
                    )
                    grad = (
                        bases.reshape(3, -1)
                        @ (geometry - (cross + cross.T) / 2).ravel()
                    )
                    assert np.min(grad) > -1e-9
                    np.testing.assert_allclose(grad[coef > 1e-10], 0, atol=1e-9)
                    source_checks += 1
    checks, values_checked, metric_checks, worst = 0, 0, 0, 0.0
    with np.load(out / "generate/truth.npz", allow_pickle=False) as truth:
        for stage, obs in cfg["stages"].items():
            programs = read(out / f"apply_{stage}/programs.json")
            for name, digest in read(out / f"apply_{stage}/seal.json").items():
                assert sha(out / f"apply_{stage}/{name}") == digest
            with (
                np.load(
                    out / f"generate/application_{stage}.npz", allow_pickle=False
                ) as data,
                np.load(
                    out / f"apply_{stage}/predictions.npz", allow_pickle=False
                ) as saved,
            ):
                for seed in cfg["seeds"]:
                    for world in cfg["worlds"]:
                        key = f"s{seed}_{world}"
                        for arm in cfg["arms"]:
                            m = models[
                                key + ("_unpaired" if arm == "unpaired_source" else "")
                            ]
                            w, offset, covariance = program(m, obs, arm)
                            p = programs[key + "_" + arm]
                            np.testing.assert_allclose(p["weights"], w, atol=1e-10)
                            np.testing.assert_allclose(
                                p["covariance"], covariance, atol=1e-10
                            )
                            for style in ["source_style", "held_style"]:
                                y, ids = (
                                    data[key + "_" + style],
                                    data[key + "_" + style + "_cases"],
                                )
                                np.testing.assert_allclose(
                                    lookup[ids].mean(-1), y, atol=1e-12
                                )
                                assert set(
                                    bank.iloc[ids.ravel()].condition_index
                                ) == set(obs)
                                address_checks += ids.size
                                value = y @ w.T + offset
                                actual = saved[f"{key}_{arm}_{style}_value"]
                                worst = max(worst, float(np.max(abs(value - actual))))
                                np.testing.assert_allclose(value, actual, atol=1e-10)
                                personal = (y - np.array(m["mean"])[obs]) @ w.T
                                level = np.tile(personal.mean(1, keepdims=True), (1, 5))
                                response = personal - level
                                np.testing.assert_allclose(
                                    level,
                                    saved[f"{key}_{arm}_{style}_personal_level"],
                                    atol=1e-10,
                                )
                                np.testing.assert_allclose(
                                    response,
                                    saved[f"{key}_{arm}_{style}_personal_response"],
                                    atol=1e-10,
                                )
                                for target, c in TARGETS.items():
                                    c = np.array(c)
                                    t, v = truth[key + "_application"] @ c, value @ c
                                    err = v - t
                                    variance = float(c @ covariance @ c)
                                    expected = {
                                        "mse": np.mean(err**2),
                                        "bias": err.mean(),
                                        "correlation": np.corrcoef(v, t)[0, 1]
                                        if np.std(v) > 1e-12
                                        else np.nan,
                                        "working_variance": variance,
                                        "coverage": np.mean(
                                            abs(err)
                                            <= norm.ppf(0.975)
                                            * np.sqrt(max(variance, 0))
                                        ),
                                        "personal_response_rms": np.sqrt(
                                            np.mean(response**2)
                                        ),
                                    }
                                    row = metrics[
                                        (metrics.seed == seed)
                                        & (metrics.world == world)
                                        & (metrics.stage == stage)
                                        & (metrics.arm == arm)
                                        & (metrics["style"] == style)
                                        & (metrics.target == target)
                                    ].iloc[0]
                                    for name, val in expected.items():
                                        np.testing.assert_allclose(
                                            val, row[name], atol=1e-10, equal_nan=True
                                        )
                                        metric_checks += 1
                                values_checked += value.size
                            checks += 1
            assert (
                str(out / "generate/truth.npz")
                in read(out / f"apply_{stage}/guard.json")["denied"]
            )
        neighbor_rows = pd.read_csv(out / "evaluate/neighbor_comparison.csv")
        neighbor_checks = 0
        with (
            np.load(out / "generate/source.npz", allow_pickle=False) as source,
            np.load(
                out / "generate/application_response.npz", allow_pickle=False
            ) as app,
        ):
            for row in neighbor_rows.itertuples():
                key = f"s{row.seed}_{row.world}"
                a = source[key + "_halves"].mean(-1)[:, cfg["stages"]["response"]]
                b = app[key + "_" + row.style]
                if row.rule == "random":
                    ids = np.random.default_rng(row.seed + 500000).integers(
                        len(a), size=len(b)
                    )
                else:
                    if row.rule == "level":
                        a, b = a[:, [1]], b[:, [1]]
                    else:
                        a, b = (
                            a - a.mean(1, keepdims=True),
                            b - b.mean(1, keepdims=True),
                        )
                    ids = cdist(b, a, metric="sqeuclidean").argmin(1)
                delta = (
                    truth[key + "_source"][ids] - truth[key + "_application"]
                ) @ np.array(TARGETS["finite_change"])
                np.testing.assert_allclose(
                    np.mean(delta**2), row.hidden_response_difference_mse, atol=1e-12
                )
                neighbor_checks += 1
    assert str(out / "generate/truth.npz") in read(out / "fit/guard.json")["denied"]
    report = {
        "status": "PASS",
        "source_models": source_checks,
        "programs": checks,
        "prediction_values": values_checked,
        "metric_checks": metric_checks,
        "bank_address_checks_with_repetition": address_checks,
        "neighbor_checks": neighbor_checks,
        "neural_pairs": neural_pairs,
        "worst_prediction_error": worst,
        "same_agent_formula_audit_not_D02": True,
    }
    dump(out / "evaluate/independent_audit.json", report)
    structure = []
    p = np.eye(5) - np.ones((5, 5)) / 5
    for seed in cfg["seeds"]:
        for world in cfg["worlds"]:
            model = models[f"s{seed}_{world}"]
            eig, vec = np.linalg.eigh(np.array(model["response_kernel"]))
            if world == "independent_response":
                alignment = None
            else:
                v = np.array(
                    cfg["smooth_pattern"]
                    if world == "smooth_shared"
                    else cfg["crossed_pattern"]
                )
                alignment = float(abs(vec[:, -1] @ v) / np.linalg.norm(v))
            structure.append(
                {
                    "seed": seed,
                    "world": world,
                    "response_eigenvalues": eig,
                    "leading_direction_absolute_cosine_to_generator": alignment,
                }
            )
    source = np.array([[-1, 0, 1], [3, 2, 1]])
    person = np.array([1, 0, -1])
    assert source[0].mean() == person.mean()
    np.testing.assert_allclose(source[1] - source[1].mean(), person - person.mean())
    dump(
        out / "evaluate/response_structure.json",
        {
            "post_result_description_not_model_selection": True,
            "source_fitted_axes_generator_only_for_evaluation": structure,
            "same_mean_opposite_response": [source[0], person],
            "different_mean_same_response": [source[1], person],
            "independent_equal_variance_centering_identity": p @ np.eye(5) @ p,
            "centering_induced_off_diagonal": -0.2,
        },
    )
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    main(parser.parse_args().output.resolve())
