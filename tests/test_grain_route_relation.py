#!/usr/bin/env python3
"""Two competing mechanism worlds plus controls for the new observation law."""

import numpy as np
import pytest

from grain_research.construction import math_route_relation as M


def pair_world(route_only):
    """Designed target relation from routes OR expressions, never both implicitly."""
    j = np.zeros((32, 8, 8))
    cells = (0, 4) if route_only else (0, 1)
    for c in range(32):
        if c & 1:
            j[c, cells[0], cells[0]] = j[c, cells[1], cells[1]] = 1 / 64
        else:
            j[c, cells[0], cells[1]] = j[c, cells[1], cells[0]] = 1 / 64
    p = j.sum(2)[:, :, None] * j.sum(1)[:, None, :] * 32
    return np.stack((p, j))


def test_route_only_relation_has_no_within_route_target_action():
    laws = M.three_laws(pair_world(True))
    np.testing.assert_allclose(laws[1], laws[2])
    a = M.actions(laws)
    supported = a["supported"]
    np.testing.assert_allclose(a["fields"][1][:, supported], 0)
    assert np.max(np.abs(a["fields"][0][:, supported])) == 0.5
    assert M.divergences(laws)["within"] == 0


def test_expression_relation_in_fixed_route_is_not_erased_as_context():
    laws = M.three_laws(pair_world(False))
    np.testing.assert_allclose(laws[0], laws[1])
    a = M.actions(laws)
    np.testing.assert_allclose(a["fields"][0][:, a["supported"]], 0)
    assert np.max(np.abs(a["fields"][1][:, a["supported"]])) == 0.5
    assert M.divergences(laws)["routing"] == 0


def random_laws():
    """A nonfactorizing positive law with all profile and route supports."""
    j = np.random.default_rng(193).uniform(0.1, 2, size=(32, 8, 8))
    j /= j.sum()
    mass = j.sum((1, 2))
    p = j.sum(2)[:, :, None] * j.sum(1)[:, None, :] / mass[:, None, None]
    return M.three_laws(np.stack((p, j)))


def test_all_singletons_and_route_conditional_singletons_are_preserved():
    p, r, j = random_laws()
    for axis in (1, 2):
        np.testing.assert_allclose(p.sum(axis), r.sum(axis))
        np.testing.assert_allclose(r.sum(axis), j.sum(axis))
    for a in (0, 1):
        for b in (0, 1):
            sl = (slice(None), slice(a * 4, a * 4 + 4), slice(b * 4, b * 4 + 4))
            for axis in (1, 2):
                np.testing.assert_allclose(r[sl].sum(axis), j[sl].sum(axis))
    assert np.max(np.abs(r - j)) > 1e-5


def test_kl_pythagoras_and_probability_update_accounts():
    laws = random_laws()
    d = M.divergences(laws)
    np.testing.assert_allclose(d["total"], d["within"] + d["routing"], atol=1e-14)
    a = M.actions(laws)
    np.testing.assert_allclose(a["fields"][2], a["fields"][:2].sum(0), atol=1e-15)
    np.testing.assert_allclose(a["before"], np.broadcast_to(a["before"][0], (3, 5, 8)))


def test_independence_null_has_no_target_pair_action():
    laws = random_laws()
    x = M.three_laws(np.stack((laws[0], laws[0])))
    np.testing.assert_allclose(M.actions(x)["fields"], 0, atol=1e-15)


def test_structural_missing_routes_are_not_smoothed_or_renormalized():
    laws = M.three_laws(pair_world(False))
    assert laws[1, :, 4:, :].sum() == 0
    a = M.actions(laws)
    assert np.isnan(a["fields"][:, :, 7, 7]).all()
    assert not a["supported"][7, 7]


def test_signed_transport_cross_accounts_not_independent_channels():
    a = M.actions(random_laws())
    fields = np.broadcast_to(a["fields"], (2, 2, 3, 5, 8, 8)).copy()
    fields[1, :, 0] *= 0.9
    fields[1, :, 1] *= -0.7
    fields[:, :, 2] = fields[:, :, :2].sum(2)
    support = np.ones((2, 2, 8, 8), bool)
    v = M.transport(fields, support, np.full((8, 8), 1 / 64))
    np.testing.assert_allclose(
        v["crossed_accounts"].sum((1, 2)), v["components"][2]["by_target"]
    )
    assert v["crossed_accounts"][:, 1, 1].max() < 0


def test_pair_bundle_keeps_both_components_and_rejects_copied_events():
    bundle = M.make_bundle(M.three_laws(pair_world(True)))
    events = [
        {"event_id": "early", "window": 0, "cell": 0},
        {"event_id": "late", "window": 1, "cell": 0},
    ]
    q = M.query_pair(bundle, events)
    assert q["supported"] and q["route"] == [0, 0]
    np.testing.assert_allclose(q["components"][2], q["components"][:2].sum(0))
    events[1]["event_id"] = "early"
    with pytest.raises(ValueError, match="same event"):
        M.query_pair(bundle, events)


def test_invalid_probability_input_is_rejected_not_fixed():
    j = random_laws()[2]
    j[0, 0, 0] = -1
    with pytest.raises(ValueError, match="Normalized"):
        M.route_product(j)
