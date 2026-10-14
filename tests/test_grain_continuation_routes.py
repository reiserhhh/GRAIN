#!/usr/bin/env python3
"""Address, full-token and signed-meaning tests without pretrained inference."""

import numpy as np
import pytest

from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_late_condition as L
from scripts.score_grain_continuation_routes_cold import score


def test_first_target_prediction_is_direct():
    """The last suffix row, not the first input target row, predicts b0."""
    width, rows = C.addresses([3, 1], 3)
    assert width == 5
    assert rows == [([0, 1], [2, 3, 4]), ([], [2, 3, 4])]


def test_mask_only_changes_registered_edges():
    """Left pads and prefix/header/continuation keys remain untouched."""
    import torch

    original = torch.zeros(2, 1, 5, 9)
    changed = C.cut_mask(original, 4, 2, [3, 1], 3, "relay")
    expected = original.clone()
    expected[:, :, 2:, 2:4] = -torch.inf
    assert torch.equal(changed, expected)
    assert torch.isfinite(original).all()
    assert C.cut_mask(original, 4, 2, [3, 1], 3, "native") is original


@pytest.mark.parametrize("route,body", [("missing", 2), ("closed", 0), ("direct", 4)])
def test_bad_support_rejected(route, body):
    """Missing context/support must not become an implicit closed control."""
    import torch

    with pytest.raises(ValueError):
        C.cut_mask(torch.zeros(2, 1, 5, 9), 4, body, [3, 1], 3, route)


def test_signed_meaning_recombines_even_when_amplitude_misleads():
    """A larger native action may have a negative SOURCE contact."""
    x = np.arange(12, dtype=float).reshape(6, 2)
    values = {"closed": np.ones_like(x), "direct": x, "relay": 2 * x, "native": -3 * x}
    parts = C.decompose(values)
    total = sum(
        parts[k]
        for k in ("closed", "direct_increment", "relay_increment", "interaction")
    )
    assert np.max(abs(total - values["native"])) < 1e-12
    assert np.all(np.diag(C.contacts(x, parts["interaction"])) < 0)
    assert np.all(np.diag(C.contacts(x, values["direct"])) > 0)
    assert np.max(abs(C.contacts(x, np.ones_like(x)))) == 0
    assert C.diagonal_intervals(x, parts["interaction"], 7, 30).shape == (2, 2)


def test_missing_route_rejected():
    """No replacing failed scientific arms by zero."""
    with pytest.raises(ValueError):
        C.decompose({"native": np.zeros((3, 2))})


def test_cold_account_preserves_negative_interaction():
    """Numeric output retains every graph, including cancelling interactions."""
    zeros = np.zeros((2, 11))
    routes = {}
    for route, number in zip(C.ROUTES, (0, 1, 2, -1), strict=True):
        z = zeros.copy()
        z[0, 1] = number
        routes[route] = {"chosen": z.tolist(), "partition": zeros.tolist()}
    result = score({"version": "HM259-v1", "units": [{"unit": 0, "routes": routes}]})
    assert result["rows"][0]["terms"]["interaction"][0] == -4


def tiny():
    """Construct random CPU weights only; no downloads or model training."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(259)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=39,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    ).eval()
    return type("Observer", (), {"model": model})()


def test_restored_cache_full_routes_and_equal_length_closed():
    """All-open preserves native; closed ignores only equal-length body identity."""
    observer = tiny()
    cache = L.encode_context(observer, [2, 3, 4, 5])
    before = L.cache_arrays(cache)
    restored = C.restore_cache(observer, before)
    suffixes, target = [[6, 7, 8], [9, 10]], [11, 12, 13]
    native = L.observe_suffixes(observer, cache, suffixes, target)
    opened = C.observe(observer, restored, suffixes, target, 2, "native")
    assert np.max(abs(native["chosen"] - opened["chosen"])) < 1e-6
    for route in ("closed", "direct", "relay"):
        result = C.observe(observer, restored, suffixes, target, 2, route)
        assert len(result["receipts"]) == 2
        assert all(r["prefix_error"] == 0 for r in result["receipts"])
    altered = L.encode_context(observer, [2, 3, 21, 22])
    closed = C.observe(observer, restored, suffixes, target, 2, "closed")
    other = C.observe(observer, altered, suffixes, target, 2, "closed")
    assert np.max(abs(closed["chosen"] - other["chosen"])) < 1e-6
    for key, array in L.cache_arrays(restored).items():
        assert np.array_equal(array, before[key])


def test_global_hook_restored_after_forward_failure(monkeypatch):
    """A partial packet cannot contaminate a later model's attention function."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer = tiny()
    cache = L.encode_context(observer, [2, 3, 4])
    original = qwen.eager_attention_forward

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(L, "observe_suffixes", fail)
    with pytest.raises(RuntimeError):
        C.observe(observer, cache, [[6, 7]], [8, 9], 2, "direct")
    assert qwen.eager_attention_forward is original
