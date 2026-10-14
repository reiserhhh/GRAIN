"""P05 native-action packet: solver identities, gradient/hook binding, accounting."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.native_action import (
    Accounting,
    apply_rank1_dual,
    euclid_objective,
    fit_euclid,
    fit_rank1_dual,
    metric_scale,
    rank1_objective,
    source_scale,
)


def test_identity_metric_reproduces_euclid():
    rng = np.random.default_rng(0)
    n, p, d = 9, 3, 5
    Z, Y = rng.standard_normal((n, p)), rng.standard_normal((n, d))
    W = fit_euclid(Z, Y, 1.0)
    item = np.repeat(np.arange(n), d)
    U = np.tile(np.eye(d), (n, 1))
    T = Y.reshape(-1)
    c, K = fit_rank1_dual(Z, U, T, 1.0, item=item)
    znew = rng.standard_normal((4, p))
    np.testing.assert_allclose(apply_rank1_dual(c, Z, U, znew, item=item), znew @ W, atol=1e-10)
    Wd = sum(c[k] * np.outer(Z[item[k]], U[k]) for k in range(len(c)))
    np.testing.assert_allclose(euclid_objective(Wd, Z, Y, 1.0), rank1_objective(c, K, T, 1.0), rtol=1e-10)


def test_metric_scale_isotropic_expectation():
    rng = np.random.default_rng(1)
    J = rng.standard_normal((20000, 6)) * 0.3
    s = metric_scale(J)
    M = s * (J.T @ J) / len(J)
    np.testing.assert_allclose(M, np.eye(6), atol=0.05)


def test_rank1_dual_matches_primal_ridge():
    rng = np.random.default_rng(2)
    n, p, d = 7, 2, 4
    Z, U, T = rng.standard_normal((n, p)), rng.standard_normal((n, d)), rng.standard_normal(n)
    c, _K = fit_rank1_dual(Z, U, T, 1.0)
    X = np.stack([np.outer(Z[i], U[i]).ravel() for i in range(n)])
    w = np.linalg.solve(X.T @ X + np.eye(p * d), X.T @ T)
    Wp = w.reshape(p, d)
    znew = rng.standard_normal(p)
    np.testing.assert_allclose(apply_rank1_dual(c, Z, U, znew)[0], znew @ Wp, atol=1e-10)


def test_zero_variance_coordinate_dropped():
    de = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    scale, keep = source_scale(de)
    assert keep.tolist() == [True, False] and scale[1] == 1.0


def test_hook_gradient_matches_finite_difference():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    dim, hidden, seq = 4, 8, 6
    o_proj = torch.nn.Linear(hidden, hidden)
    head = torch.nn.Linear(hidden, 3)
    x0 = torch.randn(1, seq, hidden)
    pos = [3, 4, 5]
    vals = torch.tensor([0.0, 0.5, 1.0])

    def score(delta):
        box = {}

        def hook(_m, xs):
            x = xs[0].clone()
            x[0, pos, :dim] = x[0, pos, :dim] + delta
            box["x"] = x
            return (x,)

        h = o_proj.register_forward_pre_hook(hook)
        try:
            out = head(torch.tanh(o_proj(x0)))[0, -1]
        finally:
            h.remove()
        q = torch.softmax(out, 0)
        return (q * vals).sum()

    delta = torch.zeros(len(pos), dim, requires_grad=True)
    s = score(delta)
    s.backward()
    g = delta.grad.clone()
    unit = g / g.norm()
    h = 1e-3
    with torch.no_grad():
        fd = (score(h * unit) - score(-h * unit)) / (2 * h)
    assert abs(float(fd) - float(g.norm())) / float(g.norm()) < 1e-3
    with torch.no_grad():
        base = score(torch.zeros(len(pos), dim))
        moved_other = x0.clone()
    assert torch.isfinite(base)
    assert moved_other.shape == (1, seq, hidden)


def test_accounting_cap_and_no_retry(tmp_path):
    acc = Accounting(tmp_path / "a.jsonl", cap=3)
    acc.start("x", "s", 2)
    acc.finish("x")
    with pytest.raises(RuntimeError):
        acc.start("y", "s", 2)
    acc.start("z", "s", 1)
    acc.fail("z", "boom")
    with pytest.raises(RuntimeError):
        acc.start("w", "s", 0)
