#!/usr/bin/env python3
"""Four-cell uncertainty tests, including the frozen source-estimator equivalence."""

import copy

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_t1_source as t1
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_moments,
    generate_response_world,
)


@pytest.fixture
def example():
    rows, _ = generate_response_world(
        "L",
        20260915,
        counts={"source": 12, "validation": 0, "application": 0},
        events=24,
    )
    groups = {}
    for row in rows:
        groups.setdefault(row["author_id"], []).append(row)
    people = [groups[author] for author in sorted(groups)]
    stats = tuple(
        np.stack(values)
        for values in zip(*(t1.sufficient(row) for row in people), strict=True)
    )
    model = t1.fit_source(stats, np.arange(12))
    return rows, people, stats, model


def test_original_four_cell_estimator_and_duplicated_author_copies(example):
    rows, people, stats, model = example
    original = fit_source_moments(rows, "four_cell")
    for key in ("mu", "Sigma", "sigma2", "raw_eigenvalues"):
        np.testing.assert_allclose(model[key], original[key], atol=1e-12, rtol=0)
    chosen = [0, 0, 0, 1, 2, 4, 5, 5, 8, 9, 10, 10]
    copies = [
        {**row, "author_id": str(i), "event_id": f"{i}/{j}"}
        for i, index in enumerate(chosen)
        for j, row in enumerate(people[index])
    ]
    original = fit_source_moments(copies, "four_cell")
    fitted = t1.fit_source(stats, chosen)
    for key in ("mu", "Sigma", "sigma2", "raw_eigenvalues"):
        np.testing.assert_allclose(fitted[key], original[key], atol=1e-12, rtol=0)


@pytest.mark.parametrize("positions", [[], [0], [0, 1, 5], list(range(24))])
def test_full_observation_gaussian_equivalence(example, positions):
    _, people, _, model = example
    rows = [people[0][i] for i in positions]
    counts, sums, _ = t1.sufficient(rows)
    mean, covariance = t1.posterior(counts, sums, model)
    prior = np.asarray(model["Sigma"])
    mu = np.asarray(model["mu"])
    if rows:
        x = np.eye(4)[[t1.CONDITIONS.index(row["z"]) for row in rows]]
        y = np.asarray([row["y"] for row in rows])
        observation = x @ prior @ x.T + model["sigma2"] * np.eye(len(rows))
        m = mu + prior @ x.T @ np.linalg.solve(observation, y - x @ mu)
        v = prior - prior @ x.T @ np.linalg.solve(observation, x @ prior)
    else:
        m, v = mu, prior
    np.testing.assert_allclose(mean, m, atol=1e-12, rtol=0)
    np.testing.assert_allclose(covariance, v, atol=1e-12, rtol=0)


def test_joint_draws_and_total_variance_decomposition(example):
    _, people, stats, model = example
    other = t1.fit_source(stats, [0, 0, 0, 1, 2, 4, 5, 5, 8, 9, 10, 10])
    bundle = {"source_models": [model, other], "seed": 20260915, "personal_draws": 200}
    score = t1.raw_score(people[0], bundle)
    assert score["variance"] == pytest.approx(sum(score["components"].values()))
    assert score["mc_variance"] == pytest.approx(sum(score["mc_components"].values()))
    renamed = [{**row, "author_id": "different"} for row in people[0]]
    assert t1.raw_score(renamed, bundle) == score
    assert t1.raw_score([], bundle)["prior_only"] is True
    assert t1.raw_score(people[0][:1], bundle)["support"] == "model_borrowed"


def test_extra_truth_duplicates_wrong_world_rejected(example):
    _, people, _, _ = example
    row = people[0][0]
    for rows in (
        [{**row, "truth": 1}],
        [row, row],
        [{**row, "world": "T"}],
        [{**row, "z": 0.2}],
        [{**row, "y": float("nan")}],
        [{**row, "t": True}],
    ):
        with pytest.raises(ValueError):
            t1.sufficient(rows)


def test_bundle_reference_and_reader_validation(example):
    _, people, _, model = example
    bundle = {
        "target_id": "T1",
        "world_contract": "F04_L_four_condition",
        "reader_hash": t1.READER_HASH,
        "formula_version": "T1_four_cell_source_mixture_v1",
        "scorer_code_hash": t1.code_hash(),
        "seed": 20260915,
        "source_models": [model],
        "source_bootstraps": 1,
        "personal_draws": 200,
        "reference_mu": 0.2,
        "reference_sigma": 0.5,
    }
    raw = {
        "world_contract": bundle["world_contract"],
        "reader_hash": t1.READER_HASH,
        "records": people[0],
    }
    value = t1.cold_score(raw, bundle)
    from grain_research.gptpro_foundation_v3.f16_factory import (
        cold_score as public_score,
    )

    assert public_score(raw, bundle) == value
    assert value["reference_score"] == pytest.approx((value["score"] - 0.2) / 0.5)
    changed = copy.deepcopy(bundle)
    changed["source_models"] = []
    with pytest.raises(ValueError):
        t1.cold_score(raw, changed)
    with pytest.raises(ValueError):
        t1.cold_score({**raw, "reader_hash": "other"}, bundle)
