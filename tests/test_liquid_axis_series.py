#!/usr/bin/env python3
import numpy as np
import pytest
from suica_core import liquid_axis_series as ax


def panel():
    return {
        "clock": "observed_opportunity",
        "coordinate_system": "sample",
        "axes": ["a", "b"],
        "series": [{"id": "left", "time": ["01", "02"], "values": [[-1, 2], [3, 3]]}],
    }


def test_signed_coordinates_round_trip():
    vals = np.array([[-3.0, 1.0], [0.0, -4.0], [2.0, 2.0]])
    mean = np.array([1.0, -2.0])
    scale = np.array([2.0, 0.5])
    p = ax.encode(vals, {"mean": mean, "scale": scale})
    np.testing.assert_allclose(
        np.log(p[:, :-1] / p[:, [-1]]) * scale + mean, vals, atol=1e-13
    )


def test_refuse_invalid_opportunities():
    d = panel()
    d["series"][0]["time"].reverse()
    with pytest.raises(ValueError, match="monotonic"):
        ax.inputs(d)
    d = panel()
    d["series"][0]["values"][1][0] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        ax.inputs(d)


def test_no_in_sample_scoring_or_axis_swap(monkeypatch):
    monkeypatch.setattr(ax.li, "call", lambda *args, **kw: {"weights": {}})
    ref = ax.fit(panel(), "local")
    with pytest.raises(ValueError, match="temporal"):
        ax.score(panel(), ref, "local")
    d = panel()
    d["axes"].reverse()
    with pytest.raises(ValueError, match="incompatible"):
        ax.score(d, ref, "local")
