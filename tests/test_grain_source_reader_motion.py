#!/usr/bin/env python3
"""Secant field derivatives, new output range and exact intervention boundaries."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_reader_motion as f
from tests.test_grain_context_differential_reader import toy


def model():
    _, m, rng = toy()
    v, eps = f.directions(m)
    return m | {
        "v": v,
        "epsilon": np.asarray(eps),
        "j": rng.normal(size=m["d"].shape),
    }, rng


def test_source_directions_and_no_motion_limit():
    m, rng = model()
    np.testing.assert_allclose(np.linalg.norm(m["v"], axis=1), 1.0)
    c = rng.normal(size=9)
    z = dict(m, j=np.zeros_like(m["j"]))
    np.testing.assert_allclose(f.read(z, c), f.read(z, c, "context_kernel"), atol=1e-12)
    for variant in f.base.VARIANTS:
        np.testing.assert_allclose(
            f.read(m, c, variant), f.base.read(m, c, variant == "context_kernel")
        )


def test_full_motion_derivative():
    m, rng = model()
    c, dc = rng.normal(size=(2, 9))
    eps = 1e-5
    minus, plus = [f.read(m, c + e * dc) for e in (-eps, eps)]
    np.testing.assert_allclose(
        (plus - minus) / (2 * eps), f.derivative(m, c, dc), atol=1e-9
    )
    p, dp = rng.normal(size=(2, 2, 9))
    actual = (
        (plus * (p + eps * dp)).sum(axis=1) - (minus * (p - eps * dp)).sum(axis=1)
    ) / (2 * eps)
    expected = (f.read(m, c) * dp + f.derivative(m, c, dc) * p).sum(axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_array_equal(f.derivative(m, c, np.zeros(9)), np.zeros((2, 9)))


def test_secant_query_sign_and_step():
    _m, rng = model()
    r, dr = rng.normal(size=(2, 2, 9))
    eps = 0.003
    captures = np.stack((r - eps * dr, r + eps * dr), axis=1)
    np.testing.assert_allclose(f.secant(captures, eps), dr[0] - dr[1], atol=1e-12)
    np.testing.assert_allclose(f.secant(captures[::-1], eps), dr[1] - dr[0], atol=1e-12)
    with pytest.raises(ValueError):
        f.secant(captures, 0.0)


def test_motion_expands_range_with_valid_lower_bound():
    m, rng = model()
    assert len(f.basis(m, True)) > len(f.basis(m))
    c, target = rng.normal(size=(2, 9))
    r = f.read(m, c)
    a = f.range_account(m, target, r[0] - r[1], True)
    b = f.range_account(m, target, m["d"].mean(axis=0), False)
    assert a["orthogonal_lower_bound2"] <= b["orthogonal_lower_bound2"] + 1e-12
    assert a["orthogonal_lower_bound2"] <= a["vector_error2"] + 1e-12
    assert f.support_account(m, c)["source_motion_outside_fraction"] > 0


def test_exact_body_retention_and_equal_query_translation():
    m, rng = model()
    a, b = [
        {"state": rng.normal(size=(7, 9)), "positions": np.arange(5)} for _ in range(2)
    ]
    aa, bb = [f.shifted(x, m["v"][0], float(m["epsilon"])) for x in (a, b)]
    np.testing.assert_array_equal(aa["state"][:-1], a["state"][:-1])
    np.testing.assert_allclose(
        aa["state"][-1] - bb["state"][-1], a["state"][-1] - b["state"][-1], atol=1e-14
    )


def test_zero_context_support():
    m, _ = model()
    m = dict(m, c=np.ones_like(m["c"]), h2=np.asarray(0.0))
    v, eps = f.directions(m)
    assert not v.any() and eps == 0
    m.update(v=v, epsilon=np.asarray(eps), j=np.zeros_like(m["j"]))
    assert not f.derivative(m, np.ones(9), np.ones(9)).any()


def test_native_hook_is_at_post22_not_input():
    from types import SimpleNamespace

    import torch

    class Layer(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.post_attention_layernorm = torch.nn.Identity()

        def forward(self, x, **_kwargs):
            return self.post_attention_layernorm(x * self.scale)

    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2"),
        model=SimpleNamespace(
            layers=torch.nn.ModuleList([Layer(i) for i in (2, 3, 4, 5)]),
            norm=torch.nn.Identity(),
        ),
        lm_head=torch.nn.Identity(),
    )
    x = torch.ones((1, 3, 9))
    _, r = f.native_reader(
        model, x, {}, torch.tensor([0]), torch.tensor([1.0]), start=0, read_block=2
    )
    np.testing.assert_array_equal(r, np.array([5.0, *([0.0] * 8)]))
    assert len(model.model.layers[2].post_attention_layernorm._forward_pre_hooks) == 0


def test_three_variant_aggregate_inventory():
    from scripts.run_grain_source_reader_motion import COUNTS, summarize

    m, rng = model()
    groups = {}
    for cohort, n in (("source4", 4), ("new8", 8)):
        rows = []
        for _ in range(n):
            c = rng.normal(size=9)
            r = np.stack([f.read(m, c, v) for v in f.VARIANTS])
            p, u = rng.normal(size=(2, 4, 9)), rng.normal(size=(2, 4, 3, 4, 9))
            ps = [f.base.project(p, u, rv) for rv in r]
            row = {k: np.stack([x[k] for x in ps]) for k in ps[0]}
            rc = rng.normal(size=(2, 9))
            t = f.base.project(p[:, :1], u[:, :1], rc)
            row.update(
                {
                    "target_" + k: (v[:, 0] if k in ("scalar", "actions") else v[0])
                    for k, v in t.items()
                }
            )
            rd = r[:, 0] - r[:, 1]
            row["projected_scalar"] = rd @ p.mean(axis=0)[0]
            row["projected_actions"] = np.einsum("cfh,vh->vcf", u.mean(axis=0)[0], rd)
            row["ranges"] = [
                f.range_account(m, rc[0] - rc[1], d, vi == 2) for vi, d in enumerate(rd)
            ]
            row["diagnostics"] = f.support_account(m, c)
            rows.append(row)
        groups[cohort] = rows
    tables = summarize(groups)
    assert [len(t) for t in tables] == COUNTS[:8]
    assert set(tables[0].variant) == set(f.VARIANTS)
