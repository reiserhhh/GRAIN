#!/usr/bin/env python3
"""Synthetic address, alignment and frame tests before local observer calls."""

import numpy as np
import pytest

from grain_research.construction import math_span_binding as B


def test_four_corner_union_and_independent_owner_pairs():
    masks = B.selected_masks()
    assert len(masks) == 14
    pairs = B.owner_pairs()
    assert len(pairs) == 8
    assert len({(p["axis"], p["receiver_mask"]) for p in pairs}) == 8
    for p in pairs:
        bit = 4 if p["axis"] == "A" else 8
        assert p["receiver_mask"] ^ p["donor_mask"] == bit
        assert p["receiver_mask"] in masks and p["donor_mask"] in masks


def test_raw_actor_property_spans_follow_order_not_a_keyword_guess():
    a = "I felt worried while my colleague copied the heading."
    t = "My colleague judged AI systems generally useful while I arranged the folders."
    for text in (f"During a break, {a} {t}", f"During a break, {t} {a}"):
        spans = B.raw_spans(text)
        assert spans["A_property"]["raw"] == "worried"
        assert spans["T_property"]["raw"] == "useful"
        assert spans["A_actor"]["raw"] == "I"
        assert spans["T_actor"]["raw"] == "My colleague"
        for span in spans.values():
            assert text[span["lo"] : span["hi"]] == span["raw"]
    with pytest.raises(ValueError, match="both explicit"):
        B.raw_spans(a)


def test_token_alignment_permits_only_whitespace_overlap():
    rendered = "before worried after"
    span = {"lo": 7, "hi": 14, "raw": "worried"}
    encoding = {"input_ids": [1, 2, 3], "offset_mapping": [(0, 6), (6, 14), (14, 20)]}
    aligned = B.token_span(rendered, 0, span, encoding, 3)
    assert aligned["positions"] == [1]
    assert aligned["token_ids"] == [2]
    bad = {**encoding, "offset_mapping": [(0, 6), (5, 14), (14, 20)]}
    with pytest.raises(ValueError, match="semantic"):
        B.token_span(rendered, 0, span, bad, 3)
    with pytest.raises(ValueError, match="query"):
        B.token_span(rendered, 0, span, encoding, 1)


def test_source_rank_and_projector_are_coordinate_invariant():
    rng = np.random.default_rng(42)
    a, t = rng.normal(size=(32, 8)), rng.normal(size=(32, 8))
    u, directions, singular = B.shared_basis(a, t)
    assert u.shape == (8, 2)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-14)
    q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    v, _, _ = B.shared_basis(a @ q, t @ q)
    np.testing.assert_allclose(v @ v.T, q.T @ u @ u.T @ q, atol=1e-14)
    np.testing.assert_allclose(u @ u.T @ directions, directions, atol=1e-14)
    assert singular.min() > 0
    one, _, _ = B.shared_basis(a, 2 * a)
    assert one.shape[1] == 1


def test_projection_complement_reconstruct_and_random_norms_match():
    rng = np.random.default_rng(43)
    x, y = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
    u, _, _ = B.shared_basis(rng.normal(size=(32, 8)), rng.normal(size=(32, 8)))
    r = B.random_basis(8, 2, 44)
    p = B.intervention_delta(x, y, "projection", u, r)
    c = B.intervention_delta(x, y, "complement", u, r)
    z = B.intervention_delta(x, y, "random_matched", u, r)
    np.testing.assert_allclose(p + c, y - x, atol=1e-14)
    np.testing.assert_allclose(np.sum(p * c, axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(z, axis=1), np.linalg.norm(p, axis=1), atol=1e-14
    )
    np.testing.assert_array_equal(
        B.intervention_delta(x, y, "noop", u, r), np.zeros_like(x)
    )
    with pytest.raises(ValueError, match="orthonormal"):
        B.intervention_delta(x, y, "projection", u * 2, r)


@pytest.mark.parametrize(
    "arm", ["full_property", "projection", "complement", "random_matched", "noop"]
)
def test_equal_donor_is_always_zero(arm):
    x = np.arange(24.0).reshape(3, 8)
    u = B.random_basis(8, 2, 45)
    r = B.random_basis(8, 2, 46)
    np.testing.assert_array_equal(
        B.intervention_delta(x, x, arm, u, r), np.zeros_like(x)
    )


def test_runner_address_count_and_noop_design():
    from scripts.run_grain_math_span_binding import intervention_design

    spec = {
        "queries": ["a", "b", "c"],
        "layers": [4, 8, 12, 16, 20],
        "arms": [
            "full_property",
            "neutral",
            "projection",
            "complement",
            "random_matched",
        ],
        "noop_layer": 16,
    }
    pairs = [{**p, "family": f} for f in range(16) for p in B.owner_pairs()]
    rows = intervention_design(pairs, spec)
    assert len(rows) == 9696
    assert len({r["unit"] for r in rows}) == 9696
    noop = [r for r in rows if r["arm"] == "noop"]
    assert len(noop) == 96
    assert all(
        r["polarity"] == 1 and r["order"] == 0 and r["layer"] == 16 for r in noop
    )


def test_failed_noop_is_not_accepted_on_resume(tmp_path):
    from scripts.run_grain_math_native_joint import write_json
    from scripts.run_grain_math_span_binding import checked_patch

    path = tmp_path / "unit.json"
    row = {"arm": "noop", "unit": 0}
    write_json(
        path,
        {
            "address": row,
            "receiver_native": 0.2,
            "native": {"conditional_score": 0.3},
            "patch_delta": 0.1,
        },
    )
    with pytest.raises(ValueError, match="no-op"):
        checked_patch(path, row, 1e-5)


def test_mixed_difference_exposes_nonlinearity_not_vector_energy():
    from scripts.diagnose_grain_math_span_binding import mixed_difference

    # F(x)=x^2 at x=0, p=2, c=3 yields cross-effect 2pc, not zero.
    assert mixed_difference(25.0, 4.0, 9.0) == 12.0
    assert mixed_difference(5.0, 2.0, 3.0) == 0.0
