#!/usr/bin/env python3
"""Partition derivatives, non-identifiability from marginals and old action match."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import query_partition_geometry as f
from grain_research.vector_mechanisms.source_query_selection import key_actions


def toy(seed=72):
    """Construct finite independent GQA evidence, with one masked position."""
    r = np.random.default_rng(seed)
    t, kv, hd, heads = 9, 2, 4, 4
    angle = r.normal(size=(t, hd // 2))
    angle = np.concatenate((angle, angle), axis=1)
    e = {
        "k": r.normal(size=(t, kv, hd)),
        "v": r.normal(size=(t, kv, hd)),
        "dk": r.normal(size=(4, t, kv, hd)),
        "g": r.normal(size=(2, heads, hd)),
        "cos": np.cos(angle),
        "sin": np.sin(angle),
        "mask": np.zeros(t),
        "scale": np.array(1 / np.sqrt(hd)),
    }
    e["mask"][2] = -np.inf
    return r.normal(size=(3, heads, hd)), e


def test_partition_derivatives_and_hessian_psd():
    r = np.random.default_rng(720)
    l, x, t = r.normal(size=11), r.normal(size=(11, 3)), r.normal(size=3) * 0.1
    l[2] = -np.inf
    a = f.partition(l, x, t)
    eps = 1e-5
    gradient, hessian = [], []
    for delta in eps * np.eye(3):
        plus, minus = f.partition(l, x, t + delta), f.partition(l, x, t - delta)
        gradient.append((plus["log_mass"] - minus["log_mass"]) / (2 * eps))
        hessian.append((plus["mean"] - minus["mean"]) / (2 * eps))
    np.testing.assert_allclose(gradient, a["mean"], atol=1e-8)
    np.testing.assert_allclose(hessian, a["covariance"], atol=1e-8)
    assert np.linalg.eigvalsh(a["covariance"]).min() > -1e-12
    shifted = f.partition(l + 19, x, t)
    assert shifted["log_mass"] - a["log_mass"] == pytest.approx(19)
    np.testing.assert_allclose(shifted["covariance"], a["covariance"], atol=1e-12)
    x[2] = 1e7
    np.testing.assert_array_equal(f.partition(l, x, t)["mean"], a["mean"])


def test_constant_statistics_zero_response_and_same_marginals_opposite_response():
    l = np.zeros(4)
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    a = f.partition(l, np.column_stack((x, x)))
    b = f.partition(l, np.column_stack((x, -x)))
    np.testing.assert_array_equal(a["mean"], b["mean"])
    np.testing.assert_array_equal(np.diag(a["covariance"]), np.diag(b["covariance"]))
    assert a["covariance"][0, 1] == -b["covariance"][0, 1] == 2.5
    c = f.partition(l, np.column_stack((x, np.full(4, 0.7))))
    np.testing.assert_array_equal(c["covariance"][1], 0)


@pytest.mark.parametrize("seed", [72, 73])
def test_gqa_rope_head_sum_reconstructs_old_actions_and_scalar_derivative(seed):
    q, e = toy(seed)
    a = f.compile_geometry(q, e)
    np.testing.assert_allclose(a["response"].sum(-1), key_actions(q, e), atol=1e-12)
    scale = np.sqrt(a["var_v"] * a["var_d"])
    np.testing.assert_allclose(a["rho"] * scale, a["response"], atol=1e-12)

    # Perturb only keys along one GRAIN direction, with Q/value/readout frozen.
    def score(delta):
        k = f.rotary(e["k"] + delta * e["dk"][0], e["cos"][:, None], e["sin"][:, None])
        qr = f.rotary(q, e["cos"][-1], e["sin"][-1])
        result = np.zeros((3, 2))
        for i in range(3):
            for h in range(4):
                l = k[:, h // 2] @ qr[i, h] * float(e["scale"]) + e["mask"]
                result[i] += f.partition(l, e["v"][:, h // 2] @ e["g"][:, h].T)["mean"]
        return result

    np.testing.assert_allclose(
        (score(1e-5) - score(-1e-5)) / 2e-5, a["response"].sum(-1)[:, :, 0], atol=1e-8
    )
    out = f.summary({k: v[0, 0][None] for k, v in a.items()})
    assert 0 <= out["spread_weighted_abs_alignment"] <= 1
    assert 0 <= out["head_sign_retention"] <= 1


def test_undefined_alignment_not_imputed_and_bad_support_rejected():
    q, e = toy()
    e["v"][:] = 0
    a = f.compile_geometry(q, e)
    assert np.isnan(a["rho"]).all()
    np.testing.assert_array_equal(a["response"], 0)
    for l, x in [
        (np.full(4, -np.inf), np.ones((4, 2))),
        (np.zeros(4), np.ones((3, 2))),
        (np.zeros(4), np.full((4, 2), np.nan)),
    ]:
        with pytest.raises(ValueError):
            f.partition(l, x)


def test_public_inventory_excludes_filesystem_sidecars(tmp_path):
    from scripts.run_grain_query_partition_geometry import TABLES, table_hashes

    for name in TABLES:
        (tmp_path / name).write_text("cohort,value\nsource4,1\n")
        (tmp_path / ("._" + name)).write_bytes(b"\x00\xb0sidecar")
    assert set(table_hashes(tmp_path)) == set(TABLES)
