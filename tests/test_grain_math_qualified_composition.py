#!/usr/bin/env python3
"""Finite composition and known nulls can fail independently of real captures."""

import numpy as np
import pytest

from grain_research.construction import math_qualified_composition as M


def test_complete_support_has_no_duplicate_evidence_configurations():
    rows = list(M.design())
    assert len(rows) == 146 and len({M.key(r) for r in rows}) == 146
    assert [sum(r["kind"] == k for r in rows) for k in ("base", "single", "pair")] == [
        2,
        24,
        120,
    ]


def test_additive_qualified_events_have_no_xy_or_xyo_component():
    x, y, o = M.SIGNS.T
    f = 3 + 2 * x - y + 4 * o + 0.8 * x * o - 0.7 * y * o
    c = M.coefficients(f)
    assert abs(c[6]) < 1e-12 and abs(c[7]) < 1e-12
    assert np.allclose(M.CHARACTERS @ c, f)
    g = f + 0.6 * x * y - 0.2 * x * y * o
    cg = M.coefficients(g)
    assert cg[6] == pytest.approx(0.6) and cg[7] == pytest.approx(-0.2)


def test_two_composition_mechanisms_are_separable_and_sum_exactly():
    rng = np.random.default_rng(2)
    b, i, j = [rng.normal(size=(3, 1, 4)) for _ in range(3)]
    q, d = [rng.normal(size=(2, 2, 4)) for _ in range(2)]
    constructed = M.compose(b, i, j)
    a = M.attention(i, i, q, d) + M.attention(j, j, q, d) - M.attention(b, b, q, d)
    state = M.attention(constructed, constructed, q, d)
    observed = M.attention(constructed + 0.2, constructed - 0.1, q, d)
    split = M.account(observed, a, state)
    assert np.allclose(split["defect"], split["reading"] + split["encoding"])
    assert (
        np.linalg.norm(split["reading"]) > 0 and np.linalg.norm(split["encoding"]) > 0
    )
    assert np.allclose(M.attention_loop(i, j, q, d), M.attention(i, j, q, d))


def test_positive_author_laws_match_marginals_but_preserve_joint_evidence():
    signs = M.SIGNS[M.AUTHOR_CORNERS, :2]
    p = M.LAW_COUNTS / 20
    assert np.all(p > 0) and np.array_equal(p.sum(1), [1, 1])
    assert np.allclose((p[0] - p[1]) @ signs, [0, 0])
    f = 2 + signs[:, 0] + 0.3 * signs[:, 1] + 0.7 * signs.prod(1)
    assert (p[0] - p[1]) @ f == pytest.approx(0.8 * 0.7)


def test_cold_numeric_channel_excludes_other_owner_and_equals_event_sum():
    obs = np.arange(48, dtype=float).reshape(2, 4, 2, 3)
    owners = np.ones((2, 4), int)
    scores = M.cold_scores(obs, M.LAW_COUNTS, owners)
    for pair in range(2):
        for law, counts in enumerate(M.LAW_COUNTS):
            explicit = sum(obs[pair, c] * n for c, n in enumerate(counts)) / 20
            assert np.allclose(scores[pair, law], explicit)
    owners[0, 0] = -1
    with pytest.raises(ValueError):
        M.cold_scores(obs, M.LAW_COUNTS, owners)


def test_invalid_or_incomplete_support_is_not_promoted():
    with pytest.raises(ValueError):
        M.coefficients(np.ones((7, 2)))
    with pytest.raises(ValueError):
        M.compose([1, 2], [2], [3])
    with pytest.raises(ValueError):
        M.attention(
            np.ones((1, 1, 3)),
            np.ones((1, 1, 2)),
            np.ones((1, 1, 3)),
            np.ones((1, 1, 3)),
        )
    with pytest.raises(ValueError):
        M.cold_scores(np.ones((1, 4, 1, 1)), np.zeros((2, 4)), np.ones((1, 4)))
