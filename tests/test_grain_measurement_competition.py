"""Synthetic tests for GP02's finite-history estimators."""

import numpy as np

from grain_research.measurement_discovery.estimators import (
    fit_b1,
    fit_b2,
    fit_m,
    metric_rows,
    transitions,
)


def rows(author, role, values):
    return {
        author: [
            {"author_id": author, "event_id": f"{author}-{i}", "t": i, "observation": value, "role": role}
            for i, value in enumerate(values)
        ]
    }


def test_transition_design_is_second_order():
    x, y = transitions(rows("a", "source", [0, 1, 1, 0])["a"])
    np.testing.assert_array_equal(x, [[1, 1, 0, 0], [1, 1, 1, 1]])
    np.testing.assert_array_equal(y, [1, 0])


def test_estimators_fit_finite_source_data():
    source = {**rows("a", "source", [0, 1, 1, 0, 1]), **rows("b", "source", [1, 0, 1, 0, 1])}
    assert fit_b1(source)["source_p"].shape == (4,)
    assert fit_b2(source)["coef"].shape == (4,)
    assert fit_m(source)["right"].shape == (2, 4)


def test_metrics_are_finite_and_bounded():
    predictions = [{"candidate": "b1", "author_id": "a", "p": .25, "y": 0}]
    item = metric_rows(predictions)[0]
    assert 0 <= item["brier"] <= 1
    assert np.isfinite(item["log_loss"])
