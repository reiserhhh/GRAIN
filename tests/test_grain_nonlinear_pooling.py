#!/usr/bin/env python3
"""Frozen nonlinear summary, compatible fields, source-only objectives and shape."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import coupled_pooling as old
from grain_research.vector_mechanisms import full_context_geometry as base
from grain_research.vector_mechanisms import nonlinear_pooling as f
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    source_descriptor,
)
from scripts.run_grain_nonlinear_pooling import shape_account


def toy():
    """Four source authors with the real64-center contract at a small hidden width."""
    rng = np.random.default_rng(690)
    bodies = [rng.normal(size=(n, 6)) for n in (40, 44, 48, 52)]
    gradients = [rng.normal(size=b.shape) / len(b) for b in bodies]
    lasts = rng.normal(size=(4, 6))
    m = source_descriptor(bodies, gradients, list(range(4)), rank=2, landmarks=16)
    m |= base.source_context(lasts, m)
    m |= {
        "coef": rng.normal(size=389),
        "alpha": np.array(0.01),
        "score_scale": np.array(1.2),
        "density_scale": np.array(0.8),
        "origin": np.array(0.1),
        "arm": np.array(old.ARMS[0]),
    }
    records = []
    for i, (b, g, last) in enumerate(zip(bodies, gradients, lasts, strict=True)):
        d = np.zeros_like(b)
        d[: len(b) // 2] = rng.normal(size=d[: len(b) // 2].shape) * 0.03
        for view, bb, ll, gg in (
            ("original", b, last, g),
            ("shifted", b, last + 0.2, g + 0.004),
            ("plus", b + d, last, g + 0.002),
            ("minus", b - d, last, g - 0.002),
        ):
            records.append(
                {
                    "author": i,
                    "view": view,
                    "body": bb,
                    "last": ll,
                    "gradient": gg,
                    "score": rng.normal(),
                }
            )
    scales = old.energies(records, m)
    m |= {
        "source_delta_energy": np.array(scales[0]),
        "source_cross_energy": np.array(scales[1]),
    }
    return rng, records, m


def random_model():
    rng, records, local = toy()
    m = f.reference(records, local)
    m["coef"] = np.r_[local["coef"], rng.normal(size=136)]
    return rng, records, local, m


def test_full_scalar_gradient_and_source_reference():
    rng, records, local, m = random_model()
    b, last = records[0]["body"], records[0]["last"]
    d, dl = rng.normal(size=b.shape), rng.normal(size=last.shape)
    eps = 1e-5
    observed = (
        f.scalar(b + eps * d, last + eps * dl, m)
        - f.scalar(b - eps * d, last - eps * dl, m)
    ) / (2 * eps)
    expected = (
        np.sum(hidden_gradient(f.predict(b, last, m)[1], m) * d)
        + f.last_gradient(b, last, m) @ dl
    )
    assert observed == pytest.approx(expected, abs=1e-8)
    nested = dict(m, coef=np.r_[local["coef"], np.zeros(136)])
    for a, c in zip(
        f.predict(b, last, nested), base.predict(b, last, local), strict=True
    ):
        np.testing.assert_allclose(a, c, atol=1e-12)
    center = m["pool_center"].copy()
    f.predict(b + 10, last, m)
    np.testing.assert_array_equal(center, m["pool_center"])
    z, p, _ = base.design(b, last, m)
    phi, jac = f.token_features(z, p, m)
    assert phi.shape == (len(b), 16) and jac.shape == (len(b), 16, 2)


def test_symmetric_mixed_derivative_and_non_affine_B_shape():
    rng, records, _, m = random_model()
    b, last = records[0]["body"], records[0]["last"]
    cut = len(b) // 2
    a, c = np.zeros_like(b), np.zeros_like(b)
    a[:cut] = rng.normal(size=a[:cut].shape)
    c[cut:] = rng.normal(size=c[cut:].shape)
    grad = lambda x: hidden_gradient(f.predict(x, last, m)[1], m)
    ab = np.sum((grad(b + 1e-5 * a) - grad(b - 1e-5 * a)) * c) / 2e-5
    ba = np.sum((grad(b + 1e-5 * c) - grad(b - 1e-5 * c)) * a) / 2e-5
    assert abs(ab) > 1e-9
    assert ab == pytest.approx(ba, abs=1e-8)
    response = (
        f.predict(b + 0.01 * a, last, m)[1] - f.predict(b - 0.01 * a, last, m)[1]
    )[cut:] / 2
    energy, outside = shape_account(response, np.linspace(-1, 1, len(b))[cut:])
    assert outside / energy > 0.01


def test_one_fit_matches_frozen_local_normal_prefix_and_direct_objective():
    _, records, m = toy()
    # Construct a toy NEXT68 local control once; the new real experiment reuses its cache.
    local = old.fit(records, m)[0][0]
    model, diag = f.fit(records, local)
    assert len(model["coef"]) == len(local["coef"]) + 136
    assert diag["matched_prefix_error"] < 1e-9
    assert diag["objective_reconstruction_error"] < 1e-9
    assert diag["normal_equation_error"] < 1e-9
    scales = old.energies(records, model)
    loss = f.objective(records, model, scales).sum()
    d = np.linspace(-0.1, 0.1, len(model["coef"]))
    other = dict(model, coef=model["coef"] + 0.001 * d)
    penalty = np.eye(len(d)) * 0.01
    penalty[0, 0] = 0
    delta = f.objective(records, other, scales).sum() - loss
    assert delta == pytest.approx(1e-6 * d @ (model["gram"] + penalty) @ d, abs=1e-8)
    with pytest.raises(ValueError, match="normalization"):
        f.fit(records, dict(local, source_cross_energy=np.array(10.0)))


def test_contract_missing_view_or_missing_centers_stops():
    _, records, m = toy()
    with pytest.raises(ValueError, match="four unique"):
        f.reference(records[:-1], m)
    with pytest.raises(ValueError, match="64 frozen"):
        f.reference(records, dict(m, landmarks=m["landmarks"][:32]))


def test_shape_account_has_zero_for_affine_positions():
    p = np.linspace(0, 1, 30)
    y = np.column_stack((1 + 2 * p, 3 - 4 * p))
    energy, outside = shape_account(y, p)
    assert energy > 0 and outside < 1e-25


def test_guard_denies_teacher_file_in_separate_process(tmp_path):
    """The trusted-process audit guard still denies an existing teacher packet."""
    import subprocess
    import sys

    source, target = tmp_path / "input.bin", tmp_path / "teacher.bin"
    source.write_bytes(b"input")
    target.write_bytes(b"teacher")
    code = (
        "import sys; from pathlib import Path; "
        "from grain_research.vector_mechanisms.stage_file_guard import install; "
        "s,t=map(Path,sys.argv[1:]); d=install([s],[],[t],[]); "
        "assert s.read_bytes()==b'input'; assert len(d)==1"
    )
    subprocess.run([sys.executable, "-c", code, str(source), str(target)], check=True)
