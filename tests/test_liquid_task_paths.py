#!/usr/bin/env python3
import numpy as np
import pytest
from grain_core import liquid_event_time as timed
from grain_core import liquid_task_paths as t


def fake_ref(paths):
    return {"training_count": len(paths)}


def fake_gate(p, sizes, ref):
    np.testing.assert_allclose(p.sum(1), 1)
    return np.tile(np.array([0.25, 0.75]), (len(p), 1)), 0.0


def test_fit_once_per_person_and_reset():
    data = [
        ("u", "b", [[0], [2]]),
        ("v", "a", [[1], [3]]),
        ("u", "c", [[4], [1]]),
        ("skip", "d", [[4]]),
    ]
    a = t.freeze_clock(data, ["x"], "signed", fake_ref)
    assert a == t.freeze_clock(data[::-1], ["x"], "signed", fake_ref)
    assert a["training_count"] == 2
    paths = [[[0], [2], [0]], [[9]], [[2], [5]]]
    vals, _ = t.read_paths(paths, a, ["x"], "signed", fake_gate)
    for raw, (fixed, liquid, matched) in zip(paths, vals, strict=True):
        np.testing.assert_array_equal(liquid[0], raw[0])
        np.testing.assert_allclose(liquid, matched)
    with pytest.raises(ValueError, match="coordinates"):
        t.read_paths(paths, a, ["other"], "signed", fake_gate)
    with pytest.raises(ValueError, match="repeated"):
        t.freeze_clock(data + data[:1], ["x"], "signed", fake_ref)


def test_rotated_reference_and_first_passage():
    r = np.random.default_rng(7)
    x = r.normal(size=(14, 3))
    g = np.repeat(["a", "b"], 7)
    x[:7, 0] += 1
    u = [np.stack([v - 1, v + 1]) for v in x]
    ids = np.arange(14)
    Q = np.linalg.qr(r.normal(size=(3, 3)))[0]
    c, d, h = t.pull_axis(x, u, g, ids)["a"]
    cc, dd, hh = t.pull_axis(x @ Q, [v @ Q for v in u], g, ids)["a"]
    np.testing.assert_allclose((x - c) @ d - h, (x @ Q - cc) @ dd - hh, atol=1e-10)
    assert t.crossing([0.3, 1.1, 0.8], 1) == {
        "ever": True,
        "end": False,
        "first": 1,
        "steps": 3,
    }
    assert t.pull_axis(x, u, g, np.zeros(14, int)) == {}


def test_two_shutters_vs_middle_observation():
    x = np.array([[2.0, -4.0], [-2.0, 4.0], [1.0, 3.0]])
    ref = t.freeze_clock([("a", 1, x)], ["x", "y"], "sensor", fake_ref)

    def varying(p, sizes, ref):
        gates = np.array([[0.2, 0.3], [0.3, 0.7], [0.8, 0.9]])
        return np.vstack([gates[:n] for n in sizes]), 0.0

    v, _ = t.read_paths([x[:2], x], ref, ["x", "y"], "sensor", varying)
    _, y, fixed_equivalent = v[0]
    np.testing.assert_allclose(y, fixed_equivalent)
    np.testing.assert_allclose(y[1] - x[0], 0.5 * (x[1] - x[0]))
    _, y, fixed_equivalent = v[1]
    assert not np.allclose(y[-1], fixed_equivalent[-1])


def test_missing_is_not_a_negative():
    assert t.observed_episode({"presented": False, "events": []}) == (
        "not_presented",
        [],
    )
    assert t.observed_episode({"presented": True, "events": []}) == (
        "presented_no_record",
        [],
    )
    e = [
        {"index": 1, "value": [0]},
        {"index": 3, "value": [2]},
        {"index": 4, "value": None},
    ]
    assert t.observed_episode({"presented": True, "events": e}) == (
        "partially_observed",
        [[e[0]], [e[1]]],
    )
    with pytest.raises(ValueError):
        t.observed_episode({"presented": True, "events": e[::-1]})


def test_measured_gaps_are_not_interchanged_with_observations():
    # An author with more observations does not dominate the clock calibration.
    assert timed.clock_unit([[1] * 21, [3]]) == 2
    s = timed.schedules([10, 12, 12, 17], 2)
    np.testing.assert_array_equal(s["wall"], [0, 1, 0, 2.5])
    np.testing.assert_array_equal(s["reversed_gaps"], [0, 2.5, 0, 1])
    np.testing.assert_array_equal(s["equal"], [0, 1, 0, 1])
    with pytest.raises(ValueError):
        timed.schedules([9, 8], 1)
    with pytest.raises(ValueError):
        timed.clock_unit([[1, np.nan]])
