#!/usr/bin/env python3
"""Exact small-world checks of overlap-qualified evidence composition."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_overlap_chain as M


def toy():
    """A fixed nonuniform two-profile/five-position complete SOURCE."""
    paths = np.array(list(product(range(2), repeat=5)), int)
    targets = paths[:, [0]] ^ paths[:, [2]]
    # Replicate a SOURCE pattern, without changing the test's profile alphabet.
    paths = np.concatenate((paths, paths[:7]))[:, None]
    targets = np.concatenate((targets, targets[:7]))
    return paths, targets, M.compile_source(paths, targets, states=2)


def record(events):
    """Build one label-free, identity-bound observation record."""
    return {
        "case_key": "case",
        "author_key": "author",
        "window_size": 5,
        "events": [
            {"event_id": f"event/{t}", "position": t, "cell": c} for t, c in events
        ],
    }


def enumerate_law(initial, factors):
    """Independent literal summation, not message passing."""
    g, k, _ = initial.shape
    values = np.zeros((g,) + (k,) * (len(factors) + 2))
    for profile in range(g):
        for path in product(range(k), repeat=len(factors) + 2):
            value = initial[profile, path[0], path[1]]
            for t, factor in enumerate(factors):
                value *= factor[profile, *path[t : t + 3]]
            values[(profile, *path)] = value
    return values


def test_compatible_cliques_and_prior():
    _, _, b = toy()
    law = enumerate_law(b["initial"], b["transitions"])
    assert law.sum() == pytest.approx(1)
    np.testing.assert_allclose(law.sum(axis=(1, 2, 3, 4, 5)), b["prior"])
    np.testing.assert_allclose(
        M.clique_marginals(b["initial"], b["transitions"]), b["cliques"], atol=1e-14
    )


@pytest.mark.parametrize("mode", ["chain", "product"])
def test_all_masks_match_direct_sums(mode):
    _, _, b = toy()
    initial = b["initial"] if mode == "chain" else b["product_initial"]
    factors = b["transitions"] if mode == "chain" else b["product_factors"]
    law = enumerate_law(initial, factors)
    for observation in product((-1, 0, 1), repeat=5):
        slices = (slice(None),) + tuple(
            slice(None) if i < 0 else i for i in observation
        )
        exact = law[slices].reshape(2, -1).sum(1)
        posterior, logmass = M.propagate(initial, factors, np.array(observation))
        if exact.sum() > 0:
            np.testing.assert_allclose(posterior, exact / exact.sum(), atol=1e-13)
            assert logmass == pytest.approx(np.log(exact.sum()), abs=1e-13)
        else:
            assert not np.isfinite(logmass)


def test_hidden_position_is_not_a_splice():
    _, _, b = toy()
    evidence = record([(0, 0), (2, 0), (4, 1)])
    observations = M.observed_vector(evidence, 2, 5)
    np.testing.assert_array_equal(observations, [0, -1, 0, -1, 1])
    assert M.score(b, evidence)["observed_events"] == 3


def test_alias_does_not_multiply_evidence():
    _, _, b = toy()
    r = record([(1, 0), (3, 1)])
    expected = M.score(b, r)
    r["events"].append(r["events"][0].copy())
    assert M.score(b, r) == expected
    r["events"][-1]["cell"] = 1
    with pytest.raises(ValueError, match="Conflicting alias"):
        M.score(b, r)


def test_position_and_label_guards():
    _, _, b = toy()
    r = record([(1, 0), (1, 0)])
    r["events"][-1]["event_id"] = "different"
    with pytest.raises(ValueError, match="same event position"):
        M.score(b, r)
    r = record([])
    r["label"] = 1
    with pytest.raises(ValueError, match="no labels"):
        M.score(b, r)


def test_empty_prior_and_entropy():
    x, y, b = toy()
    for mode in ("chain", "normalized_product"):
        np.testing.assert_allclose(
            M.score(b, record([]))[mode]["profile"], b["prior"], atol=1e-14
        )
    account = M.source_entropy_accounts(x, y, b)
    assert account["KL_empirical_chain"] >= -1e-14
    assert account["entropy_identity_error"] < 1e-14


def test_impossible_observations_are_not_neutral():
    x = np.array([[[0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1]]])
    y = np.array([[0], [1]])
    b = M.compile_source(x, y, states=2)
    for mode in ("chain", "normalized_product"):
        result = M.score(b, record([(0, 0), (1, 1)]))[mode]
        assert result == {
            "supported": False,
            "log_evidence": None,
            "profile": None,
            "targets": None,
        }
