#!/usr/bin/env python3
"""Compatible pooled interaction, source normalization and independently checked fits."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import coupled_pooling as f
from grain_research.vector_mechanisms.token_potential import hidden_gradient


def test_stage_guard_denies_teacher_paths_in_a_dedicated_process(tmp_path):
    """The application process cannot open a teacher file, even if it exists."""
    import subprocess
    import sys

    source = tmp_path / "input.dat"
    target = tmp_path / "teacher.dat"
    source.write_text("input")
    target.write_text("target")
    code = (
        "import sys; from pathlib import Path; "
        "from grain_research.vector_mechanisms.stage_file_guard import install; "
        "s,t=map(Path,sys.argv[1:]); d=install([s],[],[t],[]); "
        "assert s.read_text()=='input'; assert len(d)==1"
    )
    subprocess.run([sys.executable, "-c", code, str(source), str(target)], check=True)


def toy():
    """Four source authors with four views and nonzero last/cross responses."""
    from test_grain_full_context_geometry import toy as source

    rng, bodies, lasts, gradients, _, frozen = source()
    records = []
    for i, (body, last, g) in enumerate(zip(bodies, lasts, gradients, strict=True)):
        d = np.zeros_like(body)
        d[: len(body) // 2] = 0.02
        for view, b, l, grad in (
            ("original", body, last, g),
            ("shifted", body, last + 0.2, g + 0.003),
            ("plus", body + d, last, g + 0.001),
            ("minus", body - d, last, g - 0.001),
        ):
            records.append(
                {
                    "author": i,
                    "view": view,
                    "body": b.copy(),
                    "last": l.copy(),
                    "gradient": grad.copy(),
                    "score": rng.normal(),
                }
            )
    frozen["source_delta_energy"] = np.array(f.energies(records, frozen)[0])
    return rng, records, frozen


def test_nested_scalar_full_gradient_and_fixed_reference():
    rng, records, frozen = toy()
    original = [r for r in records if r["view"] == "original"]
    m = f.reference(
        [r["body"] for r in original], [r["last"] for r in original], frozen
    )
    r = original[0]
    width = 2 * m["basis"].shape[1]
    m |= {
        "arm": np.array(f.ARMS[1]),
        "coef": np.r_[frozen["coef"], rng.normal(size=width * (width + 1) // 2)],
    }
    d, dl = rng.normal(size=r["body"].shape), rng.normal(size=r["last"].shape)
    fd = (
        f.scalar(r["body"] + 1e-5 * d, r["last"] + 1e-5 * dl, m)
        - f.scalar(r["body"] - 1e-5 * d, r["last"] - 1e-5 * dl, m)
    ) / 2e-5
    expected = (
        np.sum(hidden_gradient(f.predict(r["body"], r["last"], m)[1], m) * d)
        + f.last_gradient(r["body"], r["last"], m) @ dl
    )
    assert fd == pytest.approx(expected, abs=1e-8)
    local = dict(m, arm=np.array(f.ARMS[0]), coef=frozen["coef"])
    zero = dict(m, coef=np.r_[frozen["coef"], np.zeros(width * (width + 1) // 2)])
    for a, b in zip(
        f.predict(r["body"], r["last"], local), f.predict(r["body"], r["last"], zero)
    ):
        np.testing.assert_allclose(a, b, atol=1e-12)
    center = m["pool_center"].copy()
    f.predict(r["body"] + 9, r["last"], m)
    np.testing.assert_array_equal(center, m["pool_center"])


def test_cross_token_mixed_derivative_symmetry_and_local_zero():
    rng, records, frozen = toy()
    r = records[0]
    originals = records[::4]
    m = f.reference(
        [v["body"] for v in originals], [v["last"] for v in originals], frozen
    )
    n = len(m["pool_center"])
    m |= {
        "arm": np.array(f.ARMS[1]),
        "coef": np.r_[frozen["coef"], rng.normal(size=n * (n + 1) // 2)],
    }
    a, b = np.zeros_like(r["body"]), np.zeros_like(r["body"])
    a[: len(a) // 2] = rng.normal(size=a[: len(a) // 2].shape)
    b[len(b) // 2 :] = rng.normal(size=b[len(b) // 2 :].shape)
    grad = lambda x: hidden_gradient(f.predict(x, r["last"], m)[1], m)
    x = r["body"]
    ab = np.sum((grad(x + 1e-5 * a) - grad(x - 1e-5 * a)) * b) / 2e-5
    ba = np.sum((grad(x + 1e-5 * b) - grad(x - 1e-5 * b)) * a) / 2e-5
    assert abs(ab) > 1e-6
    assert ab == pytest.approx(ba, abs=1e-8)
    m |= {"arm": np.array(f.ARMS[0]), "coef": frozen["coef"]}
    np.testing.assert_array_equal(
        grad(x + 0.01 * a)[len(a) // 2 :], grad(x - 0.01 * a)[len(a) // 2 :]
    )


def test_matched_normal_equations_and_direct_objective():
    _, records, frozen = toy()
    models, diagnostics = f.fit(records, frozen)
    assert diagnostics[0]["source_cross_loss"] == pytest.approx(1, abs=1e-12)
    assert (
        diagnostics[1]["source_objective"] <= diagnostics[0]["source_objective"] + 1e-10
    )
    for m, diag in zip(models, diagnostics, strict=True):
        assert diag["matched_prefix_error"] < 1e-10
        assert diag["objective_reconstruction_error"] < 1e-10
        scales = f.energies(records, m)
        loss = f.objective(records, m, scales)
        assert loss.sum() == pytest.approx(diag["source_objective"])
        d = np.arange(len(m["coef"]), dtype=float) / len(m["coef"])
        step = 0.001
        other = dict(m, coef=m["coef"] + step * d)
        penalty = np.eye(len(d)) * 0.01
        penalty[0, 0] = 0
        assert f.objective(records, other, scales).sum() - loss.sum() == pytest.approx(
            step**2 * d @ (m["gram"] + penalty) @ d, abs=1e-9
        )


def test_source_views_membership_and_energy_not_silently_changed():
    _, records, frozen = toy()
    with pytest.raises(ValueError, match="unique views"):
        f.fit(records[:-1], frozen)
    with pytest.raises(ValueError, match="normalization"):
        f.fit(records, dict(frozen, source_delta_energy=np.array(10.0)))
    changed = [dict(r) for r in records]
    changed[2]["gradient"] = changed[3]["gradient"]
    for i in range(4):
        changed[4 * i + 2]["gradient"] = changed[4 * i + 3]["gradient"]
    with pytest.raises(ValueError, match="Unsupported"):
        f.fit(changed, frozen)
