#!/usr/bin/env python3
"""Independent full-space solves and signed output-extension accounts."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_observable_span as f


def toy():
    """Three target dimensions plus two genuine observable directions in R9."""
    rng = np.random.default_rng(840)
    x, v = rng.normal(size=(2, 9, 4))
    y = np.zeros((9, 9))
    y[:, :3] = rng.normal(size=(9, 3))
    r = np.zeros((9, 5, 9))
    r[:, :, :5] = rng.normal(size=(9, 5, 5))
    g = np.repeat(np.arange(3), 3)
    m = f.prior.base.fit(x, v, g > 0, y, g, rng.normal(size=9), rng.normal(size=9))
    m.update(target=y, groups=g)
    return rng, m, r


def test_basis_retains_old_and_rulers_without_roundoff_directions():
    rng = np.random.default_rng(84)
    u, _ = np.linalg.qr(rng.normal(size=(30, 4)))
    r = rng.normal(size=(20, 4)) @ u.T
    out, info = f.extend_basis(u, r)
    np.testing.assert_array_equal(out, u)
    assert info["ruler_coverage_relative"] < 1e-12
    r[0, 0] += 0.3
    out, _ = f.extend_basis(u, r)
    assert out.shape == (30, 5)
    np.testing.assert_array_equal(out[:, :4], u)
    np.testing.assert_allclose(r @ out @ out.T, r, atol=1e-12)
    with pytest.raises(ValueError):
        f.extend_basis(u, r * np.nan)


def full_lstsq(m, r):
    """Solve augmented full-output least squares, without the normal-system helper."""
    a, h = m["a"], m["target"].shape[1]
    z = [np.kron(a * m["weights"][:, None], np.eye(h))]
    b = [(m["target"] * m["weights"][:, None]).ravel()]
    for rr in (r[:, :1], r[:, 1:]):
        zz, bb, _ = f.prior.functional_rows(a, np.eye(h), m["target"], rr, m["groups"])
        z.append(zz)
        b.append(bb)
    z.append(np.sqrt(m["regularizer"]) * np.eye(a.shape[1] * h))
    b.append(np.zeros(a.shape[1] * h))
    return np.linalg.lstsq(np.concatenate(z), np.concatenate(b), rcond=None)[0].reshape(
        a.shape[1], h
    )


def test_expanded_objective_matches_full_space_and_control():
    rng, raw, r = toy()
    old = [f.prior.fit(raw, r, flag) for flag in (False, True)]
    new = [
        f.fit(m, r.reshape(-1, 9), flag)
        for m, flag in zip(old, (False, True), strict=True)
    ]
    assert all(m["output_basis"].shape[1] == 5 for m in new)
    np.testing.assert_allclose(new[0]["beta"], old[0]["beta"], atol=1e-10)
    full = full_lstsq(old[1], r)
    np.testing.assert_allclose(
        new[1]["theta"] @ new[1]["output_basis"].T, full, atol=1e-9
    )
    np.testing.assert_allclose(full[:, 5:], 0, atol=1e-10)
    assert new[1]["new_objective"] < new[1]["old_objective"]
    assert new[1]["missing_gradient_norm"] > 1e-3
    assert new[1]["old_gradient_norm"] < 1e-10
    # The omitted complement changes no functional prediction but adds only cost.
    full_error = old[1]["a"] @ full - raw["target"]
    extra = np.zeros_like(full)
    extra[:, 5:] = rng.normal(size=extra[:, 5:].shape)
    delta = old[1]["a"] @ extra
    np.testing.assert_allclose(np.einsum("ih,ifh->if", delta, r), 0, atol=1e-10)
    increase = np.sum(((full_error + delta) * raw["weights"][:, None]) ** 2) - np.sum(
        (full_error * raw["weights"][:, None]) ** 2
    )
    np.testing.assert_allclose(
        increase, np.sum((delta * raw["weights"][:, None]) ** 2), atol=1e-9
    )


def test_missing_gradient_matches_objective_difference():
    _, raw, r = toy()
    m = f.prior.fit(raw, r, True)
    u, _ = f.extend_basis(m["output_basis"], r.reshape(-1, 9))
    h, b = f.normal_system(m, u, True)
    t = np.zeros((m["a"].shape[1], u.shape[1]))
    t[:, :3] = m["theta"]
    direction = np.zeros_like(t)
    direction[:, 3:] = 1
    gradient = 2 * (h @ t.ravel() - b)

    def objective(theta):
        e = m["a"] @ theta @ u.T - m["target"]
        losses, _ = f.prior.losses(e, m["target"], r, m["groups"])
        return losses.sum() + m["regularizer"] * np.sum(theta**2)

    eps = 1e-5
    actual = (objective(t + eps * direction) - objective(t - eps * direction)) / (
        2 * eps
    )
    np.testing.assert_allclose(actual, gradient @ direction.ravel(), atol=1e-8)


def test_unique_rulers_and_mismatch():
    from scripts.run_grain_source_observable_span import previous, unique_rulers

    g = np.repeat(np.arange(3), [8, 16, 4])
    blocks = np.arange(4 * 5 * 9).reshape(4, 5, 9)
    m = {"groups": g, "source_rulers": blocks[previous.observation_authors(g)]}
    np.testing.assert_array_equal(unique_rulers(m), blocks.reshape(20, 9))
    m["source_rulers"][10, 0, 0] += 1
    with pytest.raises(AssertionError):
        unique_rulers(m)


def test_ten_field_inventory_adapter_and_new_direction_account():
    from scripts.run_grain_source_observable_span import (
        COUNTS,
        VARIANT_KEYS,
        previous,
        summarize,
    )

    rng, raw, r = toy()
    old = [f.prior.fit(raw, r, flag) for flag in (False, True)]
    new = [
        f.fit(m, r.reshape(-1, 9), flag)
        for m, flag in zip(old, (False, True), strict=True)
    ]
    rows = []
    for _ in range(8):
        readers = rng.normal(size=(10, 2, 9))
        for vi, m in enumerate([*old, *new], 6):
            d = (
                m["d0"]
                + rng.normal(size=m["output_basis"].shape[1]) @ m["output_basis"].T
            )
            readers[vi] = np.stack((m["m"] + d / 2, m["m"] - d / 2))
        dr = rng.normal(size=(10, 2, 4, 9))
        ps, vs = rng.normal(size=(2, 4, 9)), rng.normal(size=(2, 4, 3, 4, 9))
        packets = [
            previous.previous.projections(ps, vs, rr, dd)
            for rr, dd in zip(readers, dr, strict=True)
        ]
        rows.append(
            {k: np.stack([p[k] for p in packets]) for k in packets[0]}
            | {
                "readers": readers,
                "reader_derivatives": dr,
                "native_r": rng.normal(size=(2, 9)),
                "native_primal": rng.normal(size=(2, 9)),
                "native_vectors": rng.normal(size=(2, 3, 4, 9)),
                "target_scalar": rng.normal(size=(2, 2)),
                "target_actions": rng.normal(size=(2, 3, 2, 4)),
            }
        )
    view = [
        {k: v[:8] if k in VARIANT_KEYS else v for k, v in row.items()} for row in rows
    ]
    before = previous.summarize(view, old)
    tables = summarize(rows, new, before)
    assert [len(t) for t in tables] == COUNTS
    assert set(tables[2]["variant"]) == set(previous.VARIANTS) | set(f.VARIANTS)
    assert tables[9].closure_error.max() < 1e-8
    t = tables[-1]
    np.testing.assert_allclose(
        t.remaining_missing_error2,
        t.native_missing_energy + t.added_energy - 2 * t.added_native_dot,
    )
