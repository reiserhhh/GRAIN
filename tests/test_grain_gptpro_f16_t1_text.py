#!/usr/bin/env python3
"""Observation-law, unit, information and text-history safeguards for T1."""

import copy
import hashlib

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import norm, qmc

from grain_research.gptpro_foundation_v3 import f16_t1_text as text


def fixture():
    """Provide a tiny fully declared bundle without external files or model calls."""
    channels = text.channel_draws(np.ones((2, 2, 4)), 2, 2)
    material = {
        "text": "I supported the plan.",
        "pole_readout": 0.9,
        "nli_qualification": 0.8,
        "grain_qualification": 0.9,
    }
    material["text_sha256"] = hashlib.sha256(material["text"].encode()).hexdigest()
    bundle = {
        "version": text.VERSION,
        "scorer_code_hash": text.code_hash(),
        "world_contract": "F04_L_thresholded_text_eligibility_v1",
        "source_components": 2,
        "qualification_rate": 0.5,
        "seed": 13,
        "qmc_power": 8,
        "source_models": [
            {"mu": [0.0, 0.0, 0.0, 0.0], "Sigma": np.eye(4).tolist(), "sigma2": 0.25}
        ]
        * 2,
        "channel_draws": {
            key: channels.tolist() for key in ("nli_joint", "hybrid_joint")
        },
        "materials": {"positive": material},
        "reference_mu": 0.0,
        "reference_sigma": 1.0,
    }
    obs = {
        "author_id": "A",
        "event_id": "e",
        "t": 0,
        "z": 1.0,
        "material_id": "positive",
    }
    raw = {
        "author_id": "A",
        "world_contract": bundle["world_contract"],
        "target_time": 3,
        "history": [
            {"op": "new", "record_id": "r", "recorded_at": 0, "observation": obs}
        ],
    }
    return bundle, raw


def test_likelihood_matches_explicit_latent_enumeration():
    e = text.channel_draws(np.ones((2, 2, 4)), 5, 1)[0]
    m = np.array([-0.6, 0.2, 1.0])
    actual = text.event_probabilities(m, 0.3, e)
    expected = np.zeros_like(actual)
    for i, value in enumerate(m):
        for q in (0, 1):
            p = ndtr(value / np.sqrt(0.3)) if q else 0.5
            for b in (0, 1):
                expected[i] += 0.5 * (p if b else 1 - p) * e[b, q]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(actual.sum(-1), 1)


def test_factorization_discards_real_joint_information():
    e = np.zeros((2, 2, 4))
    e[0, :, [0, 3]] = 0.5
    e[1, :, [1, 2]] = 0.5
    f = text.factorize_channel(e)
    np.testing.assert_allclose(f, 0.25)
    assert not np.allclose(
        text.event_probabilities(np.array([-1.0, 1.0]), 1, e),
        text.event_probabilities(np.array([-1.0, 1.0]), 1, f),
    )


def test_scale_is_not_identified_by_binary_observations():
    e = text.oracle_channel()
    m = np.array([[-1.0, -0.4, 0.3, 1.2]])
    np.testing.assert_allclose(
        text.event_probabilities(m, 0.25, e),
        text.event_probabilities(m * 3, 0.25 * 9, e),
        atol=1e-15,
    )
    assert (m * 3 @ text.CONTRAST)[0] == pytest.approx(3 * (m @ text.CONTRAST)[0])


def test_posterior_nodes_match_independent_scalar_integral():
    u = qmc.Sobol(1, scramble=True, seed=18).random_base2(16)[:, 0]
    nodes = np.zeros((len(u), 4))
    nodes[:, 3] = ndtri(u)
    counts = np.zeros((1, 4, 4))
    counts[0, 3, 3] = 3
    counts[0, 3, 1] = 1
    m, v, ess = text.posterior_from_nodes(counts, nodes, 0.25, text.oracle_channel())

    def density(x):
        return norm.pdf(x) * ndtr(2 * x) ** 3 * ndtr(-2 * x)

    z = quad(density, -10, 10, epsabs=1e-11)[0]
    mean = quad(lambda x: x * density(x), -10, 10, epsabs=1e-11)[0] / z
    variance = (
        quad(lambda x: x * x * density(x), -10, 10, epsabs=1e-11)[0] / z - mean**2
    )
    np.testing.assert_allclose([m[0], v[0]], [mean, variance], atol=1e-4)
    assert ess[0] > 1000


def test_no_information_channel_cannot_create_update():
    rng = np.random.default_rng(42)
    nodes = rng.normal(size=(1024, 4))
    counts = np.zeros((1, 4, 4))
    counts[0, 0, 0] = 10
    m, v, ess = text.posterior_from_nodes(counts, nodes, 1, np.full((2, 2, 4), 0.25))
    d = nodes @ text.CONTRAST
    np.testing.assert_allclose(
        [m[0], v[0], ess[0]], [d.mean(), d.var(), len(d)], atol=1e-12
    )


def test_calibration_does_not_accept_application_or_ridge_source():
    rows = [
        {
            "material_id": str(b) + str(q),
            "role": "validation",
            "pole": b,
            "qualification": q,
            "pole_readout": float(b),
            "grain_qualification": float(q),
            "nli_qualification": float(q),
        }
        for b in (0, 1)
        for q in (0, 1)
    ]
    assert text.channel_counts(rows, "hybrid_joint").sum() == 4
    for role in ("source", "application"):
        wrong = copy.deepcopy(rows)
        wrong[0]["role"] = role
        with pytest.raises(ValueError, match="validation"):
            text.channel_counts(wrong, "hybrid_joint")


def test_history_alias_summary_revision_and_retraction():
    bundle, raw = fixture()
    before = text.counts_from_input(raw, bundle)[0]
    raw["history"] += [
        {"op": "alias", "record_id": "a", "recorded_at": 1, "parent_id": "r"},
        {"op": "summary", "record_id": "s", "recorded_at": 1, "parents": ["r"]},
    ]
    np.testing.assert_array_equal(before, text.counts_from_input(raw, bundle)[0])
    raw["history"].append(
        {
            "op": "revise",
            "record_id": "r2",
            "recorded_at": 2,
            "parent_id": "r",
            "observation": raw["history"][0]["observation"],
        }
    )
    after, support = text.counts_from_input(raw, bundle)
    np.testing.assert_array_equal(before, after)
    assert support["stale_annotations"] == ["a", "s"]
    raw["history"].append(
        {"op": "retract", "record_id": "r3", "recorded_at": 3, "parent_id": "r2"}
    )
    assert text.counts_from_input(raw, bundle)[0].sum() == 0


def test_reject_leakage_duplicates_future_and_other_author():
    bundle, raw = fixture()
    for field, value in [("qualification", 1), ("y", 0.3), ("latent_state", 1)]:
        bad = copy.deepcopy(raw)
        bad["history"][0]["observation"][field] = value
        with pytest.raises(ValueError, match="truth"):
            text.counts_from_input(bad, bundle)
    for field, value in [("author_id", "B"), ("t", 5)]:
        bad = copy.deepcopy(raw)
        bad["history"][0]["observation"][field] = value
        with pytest.raises(ValueError):
            text.counts_from_input(bad, bundle)
    bad = copy.deepcopy(raw)
    op = copy.deepcopy(bad["history"][0])
    op["record_id"] = "duplicate"
    bad["history"].append(op)
    with pytest.raises(ValueError, match="duplicate event"):
        text.counts_from_input(bad, bundle)


def test_batch_companions_and_id_do_not_set_person_score():
    bundle, raw = fixture()
    a = text.cold_scores([raw], bundle)[0]
    b = copy.deepcopy(raw)
    b["author_id"] = "B"
    b["history"][0]["observation"]["author_id"] = "B"
    joint = text.cold_scores([raw, b], bundle)
    for v in joint:
        assert v["score"] == pytest.approx(a["score"], abs=1e-12)
        assert v["variance"] == pytest.approx(a["variance"], abs=1e-12)


def test_bundle_cannot_store_material_truth():
    bundle, _ = fixture()
    bundle["materials"]["positive"]["qualification"] = 1
    with pytest.raises(ValueError, match="truth"):
        text.verify_bundle(bundle)
