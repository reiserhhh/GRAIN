#!/usr/bin/env python3
"""Exact response crossing, symmetry and error accounting witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.response_repair import allocate, summarize


def test_additive_and_interaction_allocation():
    a = np.arange(1, 6, dtype=float)
    f = np.array([[a, a + 3], [a + 2, a + 7]])
    r = allocate(f, a - 1)
    np.testing.assert_array_equal(r["state"], 3)
    np.testing.assert_array_equal(r["tangent"], 4)
    np.testing.assert_array_equal(r["interaction"], 2)
    np.testing.assert_array_equal(r["state"] + r["tangent"], r["change"])
    swapped = allocate(f.swapaxes(0, 1), a - 1)
    np.testing.assert_array_equal(swapped["state"], r["tangent"])
    s = summarize(r)
    assert s["mse_accounting_error"] < 1e-12
    assert (
        abs(
            s["state_attributed_error_gain"]
            + s["tangent_attributed_error_gain"]
            - s["total_relative_error_gain"]
        )
        < 1e-12
    )


def test_identity_and_single_branch_change():
    y = np.arange(1, 6, dtype=float)
    r = allocate(np.array([[y, y], [y, y]]), y)
    np.testing.assert_array_equal(r["change"], 0)
    for f, branch in (
        (np.array([[y, y], [y + 1, y + 1]]), "tangent"),
        (np.array([[y, y + 1], [y, y + 1]]), "state"),
    ):
        np.testing.assert_array_equal(allocate(f, y)[branch], 0)


def test_error_gain_is_not_correction_correlation():
    y = np.arange(1, 6, dtype=float)
    f = np.array([[2 * y, 2.5 * y], [2.5 * y, 3 * y]])
    s = summarize(allocate(f, y))
    assert s["total_relative_error_gain"] == -3
    assert s["mse_accounting_error"] == 0
    with pytest.raises(ValueError):
        allocate(np.zeros((2, 3, 5)), y)
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        allocate(f, y)


def test_grouped_cells_preserve_pairing_and_reject_duplicates():
    import pandas as pd

    from scripts.run_grain_response_repair import effects

    cells = ("VV", "VJ", "JV", "JJ")
    frame = pd.DataFrame(
        [
            {"author": u, "query": q, "cell": c, "actual": y, "predicted": p}
            for u in ("a", "b")
            for q, y, values in (("q1", 2, (1, 3, 4, 8)), ("q2", 1, (0, 1, 3, 5)))
            for c, p in zip(cells, values, strict=True)
        ]
    ).sample(frac=1, random_state=7)
    result = effects(frame, ["author", "query"])
    np.testing.assert_allclose(result["state"] + result["tangent"], result["change"])
    index = ["author", "cell"]
    contrast = (
        frame[frame["query"] == "q1"].set_index(index)[["actual", "predicted"]]
        - frame[frame["query"] == "q2"].set_index(index)[["actual", "predicted"]]
    ).reset_index()
    paired = effects(contrast, ["author", "actual"])
    np.testing.assert_allclose(paired["error_change"], 4)
    # Squared errors must be calculated after pairing, not subtracted across queries.
    naive = (
        result[result["query"] == "q1"]["error_change"].to_numpy()
        - result[result["query"] == "q2"]["error_change"].to_numpy()
    )
    assert not np.allclose(naive, paired["error_change"])
    with pytest.raises(ValueError, match="Four exactly paired"):
        effects(pd.concat([frame, frame.iloc[:1]]), ["author", "query"])
