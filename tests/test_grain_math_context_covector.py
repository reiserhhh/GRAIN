"""Synthetic HM16 kernel, endpoint and context-marginal tests."""

import numpy as np
import pytest

from grain_research.construction.math_context_covector import (
    apply_parts,
    context_reference,
    fit_kernel,
    kernel_parts,
    shuffled_state_mapping,
)


def explicit(x, c, kind):
    """Build small explicit tensors for an independent kernel reference."""
    tensor = np.einsum("ni,nj->nij", c, x)
    if kind == "endpoint_potential":
        tensor = (tensor + tensor.transpose(0, 2, 1)) / 2
    return np.column_stack((x, tensor.reshape(len(x), -1)))


@pytest.mark.parametrize("kind", ["receiver_tensor", "endpoint_potential"])
def test_kernel_matches_explicit_tensor(kind):
    rng = np.random.default_rng(17)
    x, c = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    a, b = kernel_parts(x, c, x, c, kind)
    z = explicit(x, c, kind)
    np.testing.assert_allclose(a + b, z @ z.T, atol=1e-12)
    assert np.linalg.eigvalsh(a + b)[0] > -1e-12


@pytest.mark.parametrize("kind", ["receiver_tensor", "endpoint_potential"])
def test_weighted_dual_matches_augmented_primal(kind):
    rng = np.random.default_rng(18)
    x, c, y = rng.normal(size=(9, 3)), rng.normal(size=(9, 3)), rng.normal(size=9)
    w = np.linspace(0.3, 1.5, 9)
    linear, coupling = kernel_parts(x, c, x, c, kind)
    f = fit_kernel(linear + coupling, y, w)
    z = explicit(x, c, kind) / np.sqrt(f["kernel_unit"])
    coef = np.linalg.lstsq(
        np.vstack((np.sqrt(w)[:, None] * z, np.eye(z.shape[1]))),
        np.concatenate((np.sqrt(w) * y, np.zeros(z.shape[1]))),
        rcond=None,
    )[0]
    parts = apply_parts(linear, coupling, f["weighted_dual"], f["kernel_unit"])
    np.testing.assert_allclose(sum(parts), z @ coef, atol=1e-12)


def test_potential_endpoint_reversal_and_telescoping():
    rng = np.random.default_rng(19)
    r, u, t = rng.normal(size=(3, 5))

    def phi(a, b):
        return explicit((b - a)[None], ((a + b) / 2)[None], "endpoint_potential")[0]

    np.testing.assert_allclose(phi(r, t), -phi(t, r), atol=1e-12)
    np.testing.assert_allclose(phi(r, u) + phi(u, t), phi(r, t), atol=1e-12)


@pytest.mark.parametrize("kind", ["receiver_tensor", "endpoint_potential"])
def test_zero_displacement_has_no_action(kind):
    rng = np.random.default_rng(20)
    x, c = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    a, b = kernel_parts(np.zeros((2, 3)), c[:2], x, c, kind)
    assert np.array_equal(a + b, np.zeros((2, 6)))


@pytest.mark.parametrize("kind", ["receiver_tensor", "endpoint_potential"])
def test_orthogonal_gauge_invariance(kind):
    rng = np.random.default_rng(21)
    x, c = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    before = sum(kernel_parts(x, c, x, c, kind))
    after = sum(kernel_parts(x @ q, c @ q, x @ q, c @ q, kind))
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_context_reference_is_source_only_and_translation_covariant():
    rng = np.random.default_rng(22)
    r, w, shift = rng.normal(size=(7, 4)), np.ones(7), rng.normal(size=4)
    a, s, _ = context_reference(r, w)
    b, t, _ = context_reference(r + shift, w)
    np.testing.assert_allclose(a + shift, b, atol=1e-12)
    assert np.isclose(s, t)
    np.testing.assert_allclose((r - a) / s, (r + shift - b) / t, atol=1e-12)


def test_pair_cycle_retains_arm_marginals_and_source_roles():
    rows = []
    for p in range(7):
        arms = ("a", "b") if p < 6 else ("a",)
        for arm in arms:
            i = len(rows)
            for query in ("q1", "q2"):
                rows.append(
                    {
                        "pair_id": p,
                        "arm": arm,
                        "state_index": i,
                        "query": query,
                        "role": "source" if p < 3 else "application",
                        "axis": "A",
                        "order": 0,
                        "tokens": 1,
                    }
                )
    # State indices need not use row indices; give each pair-arm one contiguous ID.
    keys = {(r["pair_id"], r["arm"]): j for j, r in enumerate(rows[::2])}
    for r in rows:
        r["state_index"] = keys[r["pair_id"], r["arm"]]
    mapping, pairs = shuffled_state_mapping(rows)
    assert sum(r["pair_moved"] for r in pairs) == 6
    assert sorted(mapping) == list(range(len(mapping)))
    state = {r["state_index"]: r for r in rows}
    for i, j in enumerate(mapping):
        assert state[i]["arm"] == state[j]["arm"]
        assert state[i]["role"] == state[j]["role"]
    np.testing.assert_array_equal(mapping, shuffled_state_mapping(rows)[0])


def test_indefinite_or_invalid_kernel_fails_closed():
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_kernel(np.array([[1, 2], [2, 1.0]]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="positive weights"):
        fit_kernel(np.eye(2), np.ones(2), np.array([0, 1]))


def test_existing_packet_is_not_replaced(tmp_path):
    from scripts.run_grain_math_context_covector import prepare

    with pytest.raises(FileExistsError):
        prepare(tmp_path)
