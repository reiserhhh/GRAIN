#!/usr/bin/env python3
"""Mechanism/range tests, not psychological validity or predictor accuracy."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_conditional_chart as M


def example():
    """Two conditions with different reader offsets/gains."""
    offset = np.array([[3.0, -1.0], [7.0, 2.0]])
    gain = np.array([[1.0, 2.0], [-2.0, 1.0]])
    return offset[:, None] + np.array([-1, 1])[None, :, None] * gain[:, None]


def test_constant_behavior_can_have_raw_condition_difference():
    k = example()
    chart = M.compile_chart(k)
    y = M.integrate(k, M.event_counts(0.2, 0, 0.25, 400))
    assert np.linalg.norm(y[1] - y[0]) > 1
    score = M.score_chart(chart, y, [300, 100], "own_actual")
    np.testing.assert_allclose(score["coordinates"], [0.2, 0], atol=1e-12)
    assert score["choice"] == [0.75, 0.25]


@pytest.mark.parametrize("a,b", [(-0.2, -0.4), (0.0, 0.4), (0.2, 0.0)])
def test_integer_integrals_and_response(a, b):
    k = example()
    counts = M.event_counts(a, b, 0.5, 400)
    y = M.integrate(k, counts)
    explicit = [np.repeat(k[c], counts[c], axis=0).mean(0) for c in range(2)]
    np.testing.assert_allclose(y, explicit)
    score = M.score_chart(M.compile_chart(k), y, [200, 200], "own_actual")
    np.testing.assert_allclose(score["coordinates"], [a, b], atol=1e-12)


def test_source_chart_transport_not_automatic():
    source = example()
    changed = source + np.array([[[0.2, 0.1]], [[-0.3, 0.5]]])
    y = M.integrate(changed, M.event_counts(0, 0, 0.5, 400))
    old = M.score_chart(M.compile_chart(source), y, [200, 200], "own_actual")
    assert np.linalg.norm(old["coordinates"]) > 0.01
    new = M.score_chart(M.compile_chart(changed), y, [200, 200], "own_actual")
    np.testing.assert_allclose(new["coordinates"], 0, atol=1e-12)


def test_missing_condition_and_foreign_owner_not_zero():
    chart = M.compile_chart(example())
    assert M.score_chart(chart, None, [200, 0], "own_actual")["coordinates"] is None
    assert M.score_chart(chart, None, [200, 200], "other_actual")["coordinates"] is None


def test_rank_deficiency_not_fabricated_coordinates():
    k = example()
    k[0] = k[0].mean(0)
    chart = M.compile_chart(k)
    assert chart["rank"] == 1
    y = M.integrate(k, M.event_counts(0.2, 0.4, 0.5, 400))
    score = M.score_chart(chart, y, [200, 200], "own_actual")
    assert score["status"] == "PARTIAL_IDENTIFICATION"
    np.testing.assert_allclose(
        np.asarray(chart["design"]) @ np.asarray(chart["null_basis"]).T, 0, atol=1e-12
    )


def test_zero_instrument():
    k = np.ones((2, 2, 3))
    chart = M.compile_chart(k)
    assert chart["rank"] == 0
    assert len(chart["null_basis"]) == 2


def test_illegal_event_law():
    with pytest.raises(ValueError):
        M.event_counts(1, 1, 0.5, 400)


def test_all_signed_components_reconstruct():
    rng = np.random.default_rng(114)
    y = rng.normal(size=(8, 6))
    beta = M.decompose(y)
    chars = np.array(
        [
            np.prod(M.SIGNS[:, [j for j in range(3) if mask & (1 << j)]], axis=1)
            for mask in range(8)
        ]
    ).T
    np.testing.assert_allclose(chars @ beta, y, atol=1e-12)


@pytest.mark.parametrize("missing", [False, True])
def test_alternate_projection_includes_partial_support(missing):
    k = example()
    if missing:
        k[0] = k[0].mean(0)
    chart = M.compile_chart(k)
    y = M.integrate(k, M.event_counts(0.2, 0.4, 0.5, 400))
    score = M.score_chart(chart, y, [200, 200], "own_actual")
    np.testing.assert_allclose(
        score["coordinates"], M.condition_projection(chart, y), atol=1e-12
    )
    np.testing.assert_allclose(
        score["coordinates"],
        np.asarray(chart["identified_projector"]) @ [0.2, 0.4],
        atol=1e-12,
    )


def test_coupling_separates_shared_and_distinct_readout_directions():
    distinct = M.target_coupling(np.stack([np.eye(6), np.eye(6)]))
    shared = M.target_coupling(np.ones((2, 6, 6)))
    assert distinct["rank"] == 12
    assert shared["rank"] == 2
    np.testing.assert_allclose(shared["family_cosines_by_condition"], 1)
    np.testing.assert_allclose(
        np.asarray(shared["joint_design"]) @ np.asarray(shared["null_basis"]).T,
        0,
        atol=1e-12,
    )


def test_cold_cli_cannot_read_truth_or_import_encoder(tmp_path):
    chart = M.compile_chart(example())
    y = M.integrate(example(), M.event_counts(0.2, 0.4, 0.5, 400))
    calibration, observations, output = [
        tmp_path / n for n in ("cal.json", "obs.json", "out.json")
    ]
    calibration.write_text(json.dumps({"0:0:0": chart}))
    observations.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "bank": 0,
                    "family": 0,
                    "unit": 0,
                    "means": y.tolist(),
                    "counts": [200, 200],
                    "qualification": "own_actual",
                },
                {
                    "id": 8,
                    "bank": 0,
                    "family": 0,
                    "unit": 0,
                    "means": None,
                    "counts": [200, 0],
                    "qualification": "own_actual",
                },
            ]
        )
    )
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_grain_conditional_chart_hm114.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
            "--output",
            str(output),
            "--mode",
            "source",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(output.read_text())
    assert not cold["truth_input"] and not cold["neural_runtime_loaded"]
    assert cold["results"][1]["coordinates"] is None
    np.testing.assert_allclose(
        cold["results"][0]["coordinates"], [0.2, 0.4], atol=1e-12
    )
