#!/usr/bin/env python3
"""Fast deterministic checks for the reverse-trace design and effect arithmetic."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.reverse_trace import (
    effect_summary,
    equal_norm_random,
    select_discovery,
    splice_context,
)


def test_splice_keeps_only_article_difference():
    h, l, pos = splice_context([7, 8], [9], list(range(40)), list(range(50)), cap=32)
    assert len(h) == len(l) == 35
    assert pos == list(range(2, 34))
    assert h[:2] == l[:2] and h[-1:] == l[-1:]
    assert h[2:34] == list(range(32))


def test_short_pair_rejected():
    with pytest.raises(ValueError):
        splice_context([1], [2], [3] * 4, [3] * 40)


def test_random_norm_and_replay():
    low, high = np.ones((5, 9)), np.arange(45).reshape(5, 9)
    value = equal_norm_random(low, high, "fixed")
    np.testing.assert_allclose(
        np.linalg.norm(value - low, axis=1), np.linalg.norm(high - low, axis=1)
    )
    np.testing.assert_array_equal(value, equal_norm_random(low, high, "fixed"))


def test_selection_does_not_use_application():
    base = {
        "axis": "extraversion",
        "query": "extraversion",
        "kind": "donor",
        "role": "discovery",
    }
    rows = [
        {**base, "site": "a", "effect": 0.9, "contrast": 1},
        {**base, "site": "b", "effect": 0.5, "contrast": 1},
        {**base, "role": "application", "site": "b", "effect": 100, "contrast": 100},
    ]
    assert select_discovery(rows, "site") == "a"


def test_effect_summary_small_denominator():
    rows = [{"effect": 0.2, "contrast": 0, "family": "named", "situation": "opinion"}]
    result = effect_summary(rows, repeats=10)
    assert result["slope"] is None
    assert result["mean_effect"] == 0.2
    assert result["donor_mse"] == pytest.approx(0.04)


def test_effect_summary_keeps_seed_blocks():
    rows = [
        {"effect": i, "contrast": i, "family": f, "situation": "opinion"}
        for f in ("named", "described")
        for i in (1, 2)
    ]
    result = effect_summary(rows, repeats=20)
    assert result["blocks"] == 2
    assert result["slope"] == 1
    assert result["donor_mse"] == 0


def test_head_hook_cleanup_on_failure(monkeypatch):
    from types import SimpleNamespace

    import torch

    import scripts.run_grain_persona_reverse_trace as runner

    module = torch.nn.Linear(4, 4)
    observer = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=4, num_attention_heads=2),
            model=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=module))]
            ),
        )
    )

    def fail(*args, **kwargs):
        raise RuntimeError("intentional")

    monkeypatch.setattr(runner, "read_score", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        runner.head_score(observer, [1, 2], {}, 0, torch.zeros(1, 2, 4), [1], 0)
    assert len(module._forward_pre_hooks) == 0


def test_head_patch_changes_only_declared_rows_and_head():
    import torch

    from scripts.run_grain_persona_generation_trace import make_row_replace

    x = torch.zeros(1, 3, 4)
    donor = torch.ones_like(x)
    updated = make_row_replace(donor, [2], head=1, dim=2)(None, (x,))[0]
    assert updated.sum().item() == 2
    assert x.sum().item() == 0
    assert updated[0, 2, 2:].tolist() == [1, 1]


def test_address_audit_ignores_only_appledouble(tmp_path):
    from scripts.audit_grain_persona_reverse_trace import address_paths

    (tmp_path / "valid.json").write_text("{}")
    (tmp_path / "._valid.json").write_bytes(b"\x00\xb0")
    (tmp_path / "malformed.json").write_text("not JSON")
    assert [p.name for p in address_paths(tmp_path)] == ["malformed.json", "valid.json"]
