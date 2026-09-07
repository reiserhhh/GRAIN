"""Fixed vector-position bins and transparent held-setting operator fitting."""

import numpy as np
import pytest

from scripts.run_grain_query_transport import MASKS, analyze, bin_indices, design


def test_bins_are_disjoint_and_exhaustive():
    bins = bin_indices(list(range(5, 16)))
    assert [j for b in bins for j in b] == list(range(5, 16))
    assert max(map(len, bins)) - min(map(len, bins)) <= 1
    with pytest.raises(ValueError):
        bin_indices([1, 2, 3])


def test_design_has_zero_no_patch_row_and_full_rank():
    x, _cols = design(MASKS, 4)
    assert x.shape == (16, 15)
    assert not x[0].any()
    assert np.linalg.matrix_rank(x) == 15


def test_operator_predicts_constructed_held_settings_without_refit():
    rows = []
    for g in range(6):
        for variant in ("event_target", "trait_code1"):
            for level in (-1, 1):
                gap = -2 * level * (1 + g / 10)
                for mask in MASKS:
                    shift = gap * (0.2 * mask[0] + 0.8 * mask[1] * mask[2])
                    rows.append(
                        {
                            "case_id": f"{g}_{variant}_{level}",
                            "group_id": f"g{g}",
                            "variant": variant,
                            "mask": "".join(map(str, mask)),
                            "shift": shift,
                            "donor_gap": gap,
                        }
                    )
    result = analyze(rows)["operator_scores"]
    assert result.query("degree == 2 and split == 'heldout'").rmse.max() < 1e-12
    assert result.query("degree == 1 and split == 'heldout'").rmse.min() > 0.05
