#!/usr/bin/env python3
"""Small synthetic tests; no public outcomes or model calls in tests."""

import numpy as np
import pytest

from grain_research.construction import math_author_position_response as M


def bank():
    """Provide every fixed profile with synthetic finite vector prototypes."""
    return {
        "source_means": np.random.default_rng(3).normal(size=(32, 2, 4)),
        "codes": np.arange(32),
        "variance": 2.0,
    }


def test_reference_and_batch_independence():
    b = bank()
    origin = M.reference(b)
    x = np.random.default_rng(7).normal(size=(12, 4))
    args = (b, x, [str(i) for i in range(12)], [0, 1, 2, 3], [4, 5, 6, 7])
    a = M.score(*args, origin)
    shift = np.stack((np.arange(5), np.arange(5) / 3, np.arange(5) * 2 / 3))
    shifted = M.score(*args, origin + shift)
    np.testing.assert_allclose(a["response"], shifted["response"], atol=0, rtol=0)
    np.testing.assert_allclose(
        a["positions"] - shifted["positions"], np.broadcast_to(shift, (3, 3, 5))
    )
    np.testing.assert_allclose(
        a["positions"][:, 0], a["positions"][:, 1] + a["positions"][:, 2], atol=1e-14
    )
    M.score(b, x * 9, [str(i) for i in range(12)], [0, 1, 2, 3], [4, 5, 6, 7], origin)
    np.testing.assert_array_equal(a["positions"], M.score(*args, origin)["positions"])


@pytest.mark.parametrize(
    "bad", ["duplicates", "overlap", "out_of_range", "float_indices"]
)
def test_event_qualification(bad):
    ids, s, c = [str(i) for i in range(12)], [0, 1, 2, 3], [4, 5, 6, 7]
    if bad == "duplicates":
        ids[-1] = ids[0]
    elif bad == "overlap":
        c[-1] = 0
    elif bad == "out_of_range":
        c[-1] = 12
    else:
        s = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        M.score(bank(), np.ones((12, 4)), ids, s, c, M.reference(bank()))


@pytest.mark.parametrize("bad", ["order", "outcomes", "roles", "duplicates"])
def test_author_join_fail_closed(bad):
    ids = np.array(["a", "b", "c", "d"])
    other, y, same = ids.copy(), np.full((4, 5), 0.5), np.full((4, 5), 0.5)
    panel, source = np.array([2, 3]), np.array([0, 1])
    if bad == "order":
        other = other[::-1]
    elif bad == "outcomes":
        same[0, 0] = 0.2
    elif bad == "roles":
        source = np.array([0, 2])
    else:
        ids[-1] = ids[0]
        other = ids.copy()
    with pytest.raises(ValueError):
        M.join(ids, other, y, same, panel, source)


def test_covariance_and_constant_missing():
    rng = np.random.default_rng(11)
    x, z, y = (
        rng.normal(size=(24, 5)),
        rng.normal(size=(24, 5)),
        rng.normal(size=(24, 5)),
    )
    np.testing.assert_allclose(
        M.covariance(x + z, y), M.covariance(x, y) + M.covariance(z, y), atol=1e-14
    )
    assert np.isnan(M.correlations(np.ones((24, 1)), y)).all()
    assert np.max(abs(M.correlations(x, x).diagonal() - 1)) < 1e-14


def test_joint_row_null_and_paired_resamples():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(24, 10, 5)), rng.normal(size=(24, 5))
    x[:, 0, 2] = y[:, 2]
    r = M.associations(x, y, 31, 39, 225)
    assert abs(r["pearson"][0, 2, 2] - 1) < 1e-14
    assert r["point_tail"][0, 2, 2] == 1 / 40
    assert np.all(r["family_tail"] >= r["point_tail"])
    for ix in r["permutation_rows"]:
        np.testing.assert_array_equal(np.sort(ix), np.arange(24))
        np.testing.assert_allclose(
            M.covariance(y[ix], y[ix]), M.covariance(y, y), atol=1e-14
        )
    ix = r["bootstrap_rows"][0]
    np.testing.assert_allclose(
        r["bootstrap"][0],
        M.correlations(x[ix].reshape(24, -1), y[ix]).reshape(10, 5, 5),
    )


def test_objects_do_not_erase_common_or_source_association():
    rng = np.random.default_rng(2)
    pos, response, native = (
        rng.normal(size=(24, 3, 3, 5)),
        rng.normal(size=(24, 3, 5)),
        rng.normal(size=(24, 3, 5)),
    )
    a = M.objects(pos, response, native)
    assert a.shape == (24, 10, 5)
    np.testing.assert_array_equal(a[:, 0], pos[:, 0, 0])
    np.testing.assert_array_equal(a[:, 6], native[:, 0])
    np.testing.assert_allclose(a[:, 8].mean(1), 0, atol=1e-14)
