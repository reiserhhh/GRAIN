#!/usr/bin/env python3
"""Same marginals, different diagnostic relations and addressed cold inputs."""

import numpy as np
import pytest

from grain_research.construction import math_relational_diagnostic_update as M
from grain_research.construction.math_joint_target_atlas import characters


def material(kind="target"):
    """Use two prototype authors/profile with equal window marginals."""
    g = np.repeat(np.arange(32), 2)
    a = np.tile([-1.0, 1.0], 32)
    sign = 2 * (g % 2) - 1 if kind == "target" else np.ones(len(g))
    z = np.stack((a, a * sign), 1)[..., None]
    return M.compile_bank(z, g, 0.7)


def test_same_marginals_relation_can_change_target():
    b = material()
    x = np.array([[[0.8], [0.8]], [[0.8], [-0.8]]])
    r = M.read(b, x)
    np.testing.assert_allclose(r["targets"][:, 1], 0.5)
    assert r["targets"][0, 0, 0] > 0.7
    assert r["targets"][1, 0, 0] < 0.3
    np.testing.assert_allclose(r["targets"][:, :, 1:], 0.5)


def test_common_nonzero_relation_changes_density_not_target():
    r = M.read(material("common"), np.array([[[0.8], [0.8]]]))
    assert abs(r["log_evidence_ratio"][0]) > 0.1
    np.testing.assert_allclose(r["targets"], 0.5)
    np.testing.assert_allclose(r["target_log_odds_update"], 0, atol=1e-15)
    np.testing.assert_allclose(r["factor_characters"][:, 1:], 0, atol=1e-15)


def test_profile_common_factor_cancels_but_is_retained():
    li = np.arange(64).reshape(2, 32) / 100
    d = np.sin(li)
    r, shifted = M.update(li, d), M.update(li, d + 4.5)
    np.testing.assert_allclose(r["profile"], shifted["profile"], atol=1e-15)
    np.testing.assert_allclose(
        shifted["log_evidence_ratio"] - r["log_evidence_ratio"], 4.5
    )
    np.testing.assert_allclose(
        shifted["factor_characters"][:, 0] - r["factor_characters"][:, 0], 4.5
    )


def test_update_identity_and_all_character_inversion():
    r = M.read(material(), np.array([[[0.3], [-0.9]], [[0.2], [0.4]]]))
    d = r["log_factor"]
    np.testing.assert_allclose(r["factor_characters"] @ characters().T, d, atol=1e-15)
    np.testing.assert_allclose(
        r["log_profile"][:, 0] - r["log_profile"][:, 1],
        d - r["log_evidence_ratio"][:, None],
        atol=1e-15,
    )


def test_literal_full_pair_product_and_both_laws():
    b = material()
    x = np.array([[0.3], [-0.9]])
    r, v = M.read(b, x[None]), M.literal(b, x)
    for k in ("profile", "targets", "log_factor"):
        np.testing.assert_allclose(r[k][0], v[k], atol=1e-15)


def test_gradients_are_target_indexed_not_density_gradients():
    b, x, eps = material(), np.array([[[0.3], [-0.9]]]), 1e-5
    r = M.read(b, x, fields=True)
    for view in (0, 1):
        plus, minus = x.copy(), x.copy()
        plus[:, view, 0] += eps
        minus[:, view, 0] -= eps
        fd = (M.read(b, plus)["log_odds"] - M.read(b, minus)["log_odds"]) / (2 * eps)
        np.testing.assert_allclose(r["target_fields"][..., view, 0], fd, atol=1e-9)
    np.testing.assert_allclose(
        r["relational_target_field"],
        r["target_fields"][:, 0] - r["target_fields"][:, 1],
    )


def test_single_author_profile_has_zero_relation_update():
    g = np.arange(32)
    z = np.arange(64).reshape(32, 2, 1) / 100
    b = M.compile_bank(z, g, 0.7)
    r = M.read(b, np.array([[[0.3], [-0.9]]]))
    np.testing.assert_allclose(r["log_factor"], 0, atol=1e-15)
    np.testing.assert_allclose(r["target_log_odds_update"], 0, atol=1e-15)


def test_input_pairing_permutation_preserves_profile_marginal_product():
    b = material()
    z = b["prototypes"].copy()
    z[:, 1] = z[np.arange(len(z)) ^ 1, 1]
    other = M.compile_bank(z, b["codes"], b["variance"])
    x = np.array([[[0.3], [-0.9]]])
    r, s = M.read(b, x), M.read(other, x)
    np.testing.assert_allclose(r["profile"][:, 1], s["profile"][:, 1])
    assert np.linalg.norm(r["targets"][:, 0] - s["targets"][:, 0]) > 0.1


def test_cold_binding_rejects_labels_duplicates_or_missing_window():
    record = {
        "frame_key": "actual",
        "windows": [
            [{"event_id": "a", "vector": [0.3], "condition": 0}],
            [{"event_id": "b", "vector": [-0.9], "condition": 1}],
        ],
    }
    assert len(M.cold(material(), record)["target_log_odds_update"]) == 5
    record["label"] = 1
    with pytest.raises(ValueError):
        M.cold(material(), record)
    del record["label"]
    record["windows"][1][0]["event_id"] = "a"
    with pytest.raises(ValueError):
        M.cold(material(), record)
    record["windows"][1] = []
    with pytest.raises(ValueError):
        M.cold(material(), record)


def test_rejects_missing_profile_nonfinite_or_zero_variance():
    b = material()
    for z, g, s2 in (
        (b["prototypes"][2:], b["codes"][2:], 0.7),
        (b["prototypes"] * np.nan, b["codes"], 0.7),
        (b["prototypes"], b["codes"], 0),
    ):
        with pytest.raises(ValueError):
            M.compile_bank(z, g, s2)


def test_runner_all_bit_log_score_axes_match_literal_labels():
    from scripts.run_grain_relational_diagnostic_update_hm209 import log_scores

    r = M.read(material(), np.tile(np.array([[[0.3], [-0.9]]]), (32, 1, 1)))
    bits, joint = log_scores(r, np.arange(32))
    assert bits.shape == (32, 2, 5)
    assert joint.shape == (32, 2)
    for code, law, q in np.ndindex(32, 2, 5):
        np.testing.assert_allclose(joint[code, law], r["log_profile"][code, law, code])
        probability = sum(
            r["profile"][code, law, g]
            for g in range(32)
            if ((g >> q) & 1) == ((code >> q) & 1)
        )
        np.testing.assert_allclose(bits[code, law, q], np.log(probability))
