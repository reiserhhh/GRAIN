#!/usr/bin/env python3
"""Control-span accounting tests, not proof of psychological specificity."""

import numpy as np
import pytest

from grain_research.construction import math_control_overlap as M
from grain_research.construction import math_semantic_anchor as S
from scripts import run_grain_control_overlap_hm149 as R


def fixtures():
    """Return declared numeric test observations, unrelated to research results."""
    rng = np.random.default_rng(149)
    b = rng.normal(size=(7, 20))
    w = rng.normal(size=(3, 2, 20))
    x = rng.normal(size=(9, 20))
    return b, w, x


def test_exact_covector_and_score_sums():
    b, w, x = fixtures()
    split = M.response_projection(b, w)
    np.testing.assert_allclose(split["parts"].sum(axis=-2), w, atol=1e-14)
    np.testing.assert_allclose(
        M.score_features(x, split["parts"]).sum(axis=-1),
        np.einsum("nf,akf->nak", x, w),
        atol=1e-13,
    )


def test_source_control_null_is_constructed_not_validation():
    b, w, _ = fixtures()
    split = M.response_projection(b, w)
    np.testing.assert_allclose(
        np.einsum("if,akf->iak", b, split["parts"][:, :, 1]), 0, atol=1e-13
    )


def test_new_control_can_act_in_complement():
    b, w, _ = fixtures()
    split = M.response_projection(b, w)
    direction = split["parts"][0, 0, 1][None]
    assert M.score_features(direction, split["parts"])[0, 0, 0, 1] > 1


def test_target_coupling_exact_signed_sum():
    b, w, _ = fixtures()
    gram = M.target_couplings(M.response_projection(b, w)["parts"])
    np.testing.assert_allclose(
        gram["full"], gram["overlap"] + gram["complement"], atol=1e-13
    )
    np.testing.assert_allclose(gram["cross"], 0, atol=1e-13)


def test_control_basis_row_relabel_and_scale_invariance():
    b, w, _ = fixtures()
    split = M.response_projection(b, w)
    transformed = b[::-1] * np.linspace(1, 2, len(b))[:, None]
    np.testing.assert_allclose(
        split["parts"], M.response_projection(transformed, w)["parts"], atol=1e-13
    )


def test_common_orthogonal_coordinate_change():
    b, w, x = fixtures()
    q = np.linalg.qr(np.random.default_rng(151).normal(size=(20, 20)))[0]
    old = M.score_features(x, M.response_projection(b, w)["parts"])
    new = M.score_features(x @ q, M.response_projection(b @ q, w @ q)["parts"])
    np.testing.assert_allclose(old, new, atol=1e-13)


def test_dual_and_independent_product_kernel_replay():
    rng = np.random.default_rng(152)
    source, cold = rng.normal(size=(8, 4, 4)), rng.normal(size=(5, 4, 4))
    design = np.eye(8)[:3] - np.eye(8)[3:6]
    b = design @ S.tensor_features(source)
    w = rng.normal(size=(3, 2, 625))
    split = M.response_projection(b, w)
    cold_kernel = design @ S.product_kernel(source, cold)
    kernel_scores = np.einsum("aki,in->nak", split["dual"], cold_kernel)
    np.testing.assert_allclose(
        kernel_scores,
        M.score_features(S.tensor_features(cold), split["parts"])[:, :, :, 0],
        atol=1e-11,
    )


def test_zero_and_full_rank_limits():
    _, w, x = fixtures()
    zero = M.response_projection(np.zeros((2, 20)), w)
    assert len(zero["basis"]) == 0
    np.testing.assert_allclose(zero["parts"][:, :, 0], 0)
    full = M.response_projection(np.eye(20), w)
    np.testing.assert_allclose(full["parts"][:, :, 1], 0)
    np.testing.assert_allclose(
        M.score_features(x, zero["parts"]).sum(axis=-1),
        M.score_features(x, full["parts"]).sum(axis=-1),
    )


def test_duplicate_control_rows_are_not_extra_dimension():
    b, w, _ = fixtures()
    duplicate = np.concatenate((b, b, b))
    old, new = M.response_projection(b, w), M.response_projection(duplicate, w)
    assert len(old["basis"]) == len(new["basis"])
    np.testing.assert_allclose(old["parts"], new["parts"], atol=1e-13)


def test_rank_loss_and_zero_coverage_are_explicit():
    result = M.response_projection(np.asarray([[1, 0], [2, 0]]), np.ones((3, 2, 2)))
    assert len(result["basis"]) == 1
    covered = M.coverage(np.asarray([[1, 0], [0, 1], [0, 0]]), result["basis"])
    np.testing.assert_allclose(covered[:2], [1, 0])
    assert np.isnan(covered[2])


def metadata():
    return [
        {"situation": "task", "seed": 11, "condition": condition, "axis": "none"}
        for condition in ("neutral", "nonpersona_newsletter", "nonpersona_spelling")
    ]


def test_design_preserves_control_neutral_pairing():
    weights, rows = M.control_design(metadata())
    np.testing.assert_array_equal(weights, [[-1, 1, 0], [-1, 0, 1]])
    assert len(rows) == 2


def test_duplicate_or_missing_control_rejected():
    with pytest.raises(ValueError):
        M.control_design(metadata() + [metadata()[0]])
    with pytest.raises(ValueError):
        M.control_design(metadata()[:2])


@pytest.mark.parametrize("bad", [[], [[np.nan, 0]], np.zeros((2, 3, 4))])
def test_invalid_matrix_rejected(bad):
    with pytest.raises(ValueError):
        M.response_projection(bad, np.ones((3, 2, 20)))


def test_scoring_shape_and_rank_rule_guards():
    b, w, x = fixtures()
    with pytest.raises(ValueError):
        M.response_projection(b, w[:, :, :19])
    with pytest.raises(ValueError):
        M.response_projection(b, w, 0)
    with pytest.raises(ValueError):
        M.score_features(x, w)


def test_parent_artifact_contract_if_present():
    """Check the actual parent seal schema before freezing a scientific runner."""
    if not (R.PARENT / "source_seal.json").exists():
        pytest.skip("Historical parent artifacts are not available")
    for name in R.PARENT_FILES:
        assert (R.PARENT / name).is_file()
    for name, value in R.H.load(R.PARENT / "source_seal.json").items():
        assert R.H.digest(R.PARENT / name) == value
