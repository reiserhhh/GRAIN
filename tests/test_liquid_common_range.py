#!/usr/bin/env python3
import numpy as np
import pytest
from grain_core import liquid_common_range as m
from scipy.optimize import minimize


def toy():
    rng = np.random.default_rng(37)
    a = rng.normal(size=(48, 8, 3))
    b = rng.normal(size=(48, 8, 4))
    b[:, :, :2] = a[:, :, :2] + 0.05 * b[:, :, :2]
    x = np.concatenate([a, b], 2)
    return x, m.fit(x, list("abcdefg"), 3, "toy", "event")


def test_unknown_source_and_private_null():
    x, ref = toy()
    assert ref["rank"] == 2
    a, b = x[2, 3, :3], x[2, 3, 3:]
    loose = m.estimate(ref, [a, b], None)
    assert loose["status"] == "unrestricted_shared"
    for shared, privacy in [
        (loose["center"], "minimum_private_witness"),
        ([0, 0], "entirely_private_witness"),
    ]:
        for raw, w, mu, p in zip(
            (a, b), ref["loadings"], ref["means"], loose[privacy], strict=True
        ):
            decoded = (np.pad(shared, (0, len(mu) - 2)) + p) @ np.linalg.inv(w) + mu
            np.testing.assert_allclose(decoded, raw, atol=1e-12)
    cost = loose["minimum_private_norm"]
    assert m.estimate(ref, [a, b], cost / 2)["status"] == "outside_private_budget"
    fit = m.estimate(ref, [a, b], cost + 0.4)
    z = [np.array(v) for v in fit["canonical_observations"]]
    for sign in [-1, 1]:
        sol = minimize(
            lambda s, sign=sign: sign * s[0],
            fit["center"],
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda s: (
                        (cost + 0.4) ** 2
                        - sum(
                            np.square(v - np.pad(s, (0, len(v) - 2))).sum() for v in z
                        )
                    ),
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-11},
        )
        assert sol.success
        np.testing.assert_allclose(
            sol.x[0], fit["axis_bounds"][0][int(sign < 0)], atol=2e-6
        )


def test_missing_observation_and_motion_witness():
    x, ref = toy()
    a = x[1, 2, :3]
    b = x[1, 2, 3:]
    for B in (2.0, 100.0):
        left = m.estimate(ref, [a, None], B)
        right = m.estimate(ref, [None, b + 4], B)
        v = m.motion(left, right)
        if v["status"] == "bounded":
            for field, amount in [
                ("witness_min", "min_move"),
                ("witness_max", "max_move"),
            ]:
                lo, hi = np.array(v[field])
                assert np.linalg.norm(lo - left["center"]) <= left["radius"] + 1e-9
                assert np.linalg.norm(hi - right["center"]) <= right["radius"] + 1e-9
                np.testing.assert_allclose(
                    np.linalg.norm(lo - hi), v[amount], atol=1e-9
                )
    doc = {
        "axes": ref["axes"],
        "basis": "toy",
        "quantity": "event",
        "budget": None,
        "observations": [[a, None], None, [a, b]],
    }
    assert m.score(ref, doc)["results"][2]["from_previous"] is None
    with pytest.raises(ValueError):
        m.score(ref, {**doc, "quantity": "move"})
    with pytest.raises(ValueError):
        m.fit(x * 0, ref["axes"], 3, "zero", "level")
