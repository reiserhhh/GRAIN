#!/usr/bin/env python3
"""Native differential parity, source exclusion and paired-operator algebra."""

import numpy as np
import pytest
import torch
import subprocess
import sys
from pathlib import Path

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.gated_update import (
    alternatives,
    apply_coefficients,
    capture_jvp,
    coefficients,
    export_weights,
    rms_direction,
    source_members,
)
from tests.test_grain_complete_transport import tiny_model


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("epsilon", [1e-6, 0.2])
def test_exact_rms_and_swiglu_derivative(device, epsilon):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = tiny_model(device)
    layer = model.model.layers[1]
    layer.post_attention_layernorm.variance_epsilon = epsilon
    weights = export_weights(model, block=1)
    x = torch.randn(1, 7, 16, device=device)
    d = torch.randn_like(x)
    fn = lambda v: layer.mlp(layer.post_attention_layernorm(v))
    _, native = torch.func.jvp(fn, (x,), (d,))
    a, b = x[0, -1].cpu().numpy(), d[0, -1].cpu().numpy()
    analytic = apply_coefficients(
        coefficients(a, weights), rms_direction(a, b, weights), weights
    ).sum(0)
    np.testing.assert_allclose(
        analytic, native[0, -1].cpu().numpy(), rtol=1e-4, atol=1e-7
    )
    eps = 0.002
    fd = ((fn(x + eps * d) - fn(x - eps * d)) / (2 * eps))[0, -1].cpu().numpy()
    np.testing.assert_allclose(analytic, fd, rtol=0.015, atol=2e-6)
    zero = apply_coefficients(
        coefficients(a, weights), rms_direction(a, np.zeros_like(b), weights), weights
    )
    np.testing.assert_array_equal(zero, 0)
    for terms in alternatives(a, b[None], a[None], weights).values():
        np.testing.assert_allclose(terms.sum(0)[0], analytic)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_capture_cannot_run_selected_mlp_or_later_blocks(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = tiny_model(device)
    x, d = torch.randn(1, 9, 16, device=device), torch.randn(1, 9, 16, device=device)
    kwargs = cached_suffix_kwargs(model, x)

    def forbidden(*_):
        raise AssertionError("A downstream operation was called")

    handles = [
        m.register_forward_pre_hook(forbidden)
        for m in (model.model.layers[1].mlp, model.model.layers[2], model.lm_head)
    ]
    try:
        value, tangent = capture_jvp(model, x, kwargs, d, start=0, block=1)
    finally:
        for h in handles:
            h.remove()
    assert value.shape == tangent.shape == (1, 16)
    assert torch.isfinite(tangent).all()


def test_source_exclusion_and_paired_jacobians():
    rng = np.random.default_rng(43)
    indices = np.tile(np.arange(4), 2)
    queries = np.repeat(["q1", "q2"], 4)
    np.testing.assert_array_equal(
        source_members(indices, queries, "q1", held=2), [0, 1, 3]
    )
    np.testing.assert_array_equal(source_members(indices, queries, "q2"), [4, 5, 6, 7])
    with pytest.raises(ValueError, match="Duplicate"):
        source_members([0, 0], ["q1", "q1"], "q1")
    with pytest.raises(ValueError, match="No independent"):
        source_members([0], ["q1"], "q1", held=0)
    weights = export_weights(tiny_model(), block=1)
    x, d = rng.normal(size=16), rng.normal(size=(4, 16))
    source = rng.normal(size=(3, 16)) * np.asarray([0.15, 1.0, 3.0])[:, None]
    arms = alternatives(x, d, source, weights)
    paired = np.mean(
        [
            apply_coefficients(
                coefficients(s, weights), rms_direction(s, d, weights), weights
            )
            for s in source
        ],
        axis=0,
    )
    wrong = apply_coefficients(
        np.mean([coefficients(s, weights) for s in source], axis=0),
        np.mean([rms_direction(s, d, weights) for s in source], axis=0),
        weights,
    )
    np.testing.assert_allclose(arms["source_BN"], paired)
    assert np.linalg.norm(paired - wrong) > 1e-4


def test_target_guard_allows_only_declared_data(tmp_path):
    allowed, denied, output = (
        tmp_path / n for n in ("input.txt", "target.txt", "out.txt")
    )
    allowed.write_text("input")
    denied.write_text("withheld")
    code = """
from pathlib import Path
import sys
from grain_research.vector_mechanisms.stage_file_guard import install
p=Path(sys.argv[1])
log=install([p/'input.txt'],[p/'out.txt'],[p/'target.txt'],[])
assert len(log)==1
assert (p/'input.txt').read_text()=='input'
(p/'out.txt').write_text('valid')
try:
    (p/'unregistered.txt').write_text('invalid')
except PermissionError:
    pass
else:
    raise AssertionError('Undeclared write accepted')
assert len(log)==2
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "valid"
    assert denied.read_text() == "withheld"
