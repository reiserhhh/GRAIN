"""Synthetic tests for the bounded GP03 text channel."""

import numpy as np

from grain_research.measurement_discovery.text_channel import (
    FrozenTextReader,
    render_observation,
    second_order_soft_probability,
)


def test_renderer_is_stateless_and_split_specific():
    assert render_observation(1, "source", 0) != render_observation(0, "source", 0)
    assert render_observation(1, "source", 0) != render_observation(1, "application", 0)


def test_reader_fits_and_bounds_predictions():
    texts = [render_observation(i % 2, "source", i % 3) for i in range(12)]
    reader = FrozenTextReader().fit(texts, [i % 2 for i in range(12)])
    predictions = reader.predict(texts[:3])
    assert predictions.shape == (3,)
    assert np.all((predictions > 0) & (predictions < 1))


def test_soft_four_cell_integration():
    table = np.array([0.1, 0.2, 0.7, 0.9])
    assert np.isclose(second_order_soft_probability(0, 0, table), 0.1)
    assert np.isclose(second_order_soft_probability(1, 1, table), 0.9)
