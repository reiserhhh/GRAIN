#!/usr/bin/env python3
"""Synthetic arithmetic controls for HM92's separate result-after verifier."""

import copy

import numpy as np

from scripts.run_grain_event_history_observation_hm92 import observe
from scripts.verify_grain_event_history_observation_hm92 import reading


def fixture():
    """Make GQA-shaped finite arrays, not synthetic model-performance claims."""
    rng = np.random.default_rng(92)
    raw = {
        "k": rng.normal(size=(14, 2, 4)).astype(np.float32),
        "v": rng.normal(size=(14, 2, 4)).astype(np.float32),
        "q_body": rng.normal(size=(12, 12, 4)).astype(np.float32),
    }
    row = {
        "body_indices": list(range(2, 14)),
        "action_groups": [[2, 3], [6, 7]],
        "groups": [list(range(2, 6)), list(range(6, 10)), list(range(10, 14))],
        "event_order": [0, 1, 2],
        "root_labels": [2, 0, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],
    }
    bundle = {
        "raw_queries": rng.normal(size=(6, 12, 4)),
        "directions": rng.normal(size=(6, 12, 4)),
        "gaps": [1] * 6,
        "frequencies": [1.0, 0.01],
        "scaling": 0.5,
    }
    return raw, row, bundle


def test_alternate_local_and_full_reading_agree():
    raw, row, bundle = fixture()
    first, factors, means = observe(raw, row, bundle)
    second, alternative, alt_means, roundoff = reading(raw, row, bundle)
    np.testing.assert_allclose(first, second, atol=1e-12)
    np.testing.assert_array_equal(means, alt_means)
    assert roundoff < 1e-6
    for qi in range(6):
        for event in range(2):
            for component in range(3):
                np.testing.assert_allclose(
                    factors[qi][event][component],
                    alternative[qi][event][component],
                    atol=1e-12,
                )


def test_future_array_change_keeps_earlier_local_unit():
    raw, row, bundle = fixture()
    baseline = reading(raw, row, bundle)[0]
    changed = copy.deepcopy(raw)
    changed["k"][6:] += 2
    changed["v"][6:] -= 3
    changed["q_body"][4:] *= 2
    value = reading(changed, row, bundle)[0]
    np.testing.assert_array_equal(baseline[2:5], value[2:5])
    assert np.max(np.abs(baseline[0] - value[0])) > 1e-5
