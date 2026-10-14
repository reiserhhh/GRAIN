"""Synthetic checks for the source-only evidence-action construction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_operator import (
    fit_action,
    make_pairs,
    metrics,
    predict_action,
    split_name,
)


def test_group_partition():
    assert split_name("named", "opinion") == "source"
    assert split_name("named", "email") == "new_task"
    assert split_name("anchored", "opinion") == "new_wording"
    assert split_name("anchored", "email") == "new_both"


def test_pair_integrity():
    row = {
        "axis": "extraversion",
        "family": "named",
        "situation": "opinion",
        "seed": 11,
    }
    rows = [dict(row, pole=1), dict(row, pole=-1)]
    assert len(make_pairs(rows)) == 1
    with pytest.raises(ValueError):
        make_pairs(rows + [rows[0]])
    with pytest.raises(ValueError):
        make_pairs(rows[:1])


def test_rank_zero_and_source_frozen():
    rng = np.random.default_rng(15)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 12))
    model = fit_action(x, y, rank=2)
    assert np.linalg.matrix_rank(model["weight"], tol=1e-9) == 2
    assert np.all(predict_action(model, np.zeros((2, 4))) == 0)
    before = model["weight"].copy()
    predict_action(model, np.full((9, 4), 1e9))
    np.testing.assert_array_equal(model["weight"], before)


def test_ridge_equation_and_undefined_correlation():
    x = np.arange(40).reshape(10, 4)
    y = np.arange(20).reshape(10, 2)
    m = fit_action(x, y)
    z = x / m["scale"]
    np.testing.assert_allclose((z.T @ z + np.eye(4)) @ m["weight"], z.T @ y)
    assert metrics([0, 0, 0], [1, 2, 3])["r"] is None


def test_native_action_scope_and_cleanup(monkeypatch):
    """The action must not modify article rows, other heads, or leave hooks."""
    from types import SimpleNamespace

    import torch

    from scripts import run_grain_conditional_operator_native as native

    module = torch.nn.Identity()
    observer = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=8, num_attention_heads=2),
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=SimpleNamespace(o_proj=module))
                    for _ in range(13)
                ]
            ),
        )
    )
    x = torch.zeros((1, 5, 8))

    def fake_score(*_):
        result = module(x)
        assert torch.all(result[0, :3] == 0)
        assert torch.all(result[0, 3:, 4:] == 0)
        assert torch.all(result[0, 3:, :4] == 1)
        raise RuntimeError("test cleanup")

    monkeypatch.setattr(native, "read_score", fake_score)
    with pytest.raises(RuntimeError, match="test cleanup"):
        native.intervene(observer, [1] * 5, [3, 4], {}, np.ones((2, 4)))
    assert not module._forward_pre_hooks
    assert torch.all(x == 0)


def test_remainder_accounting():
    """Four corners allow nonlinear interaction without assuming additivity."""
    from scripts.run_grain_operator_remainder import account

    interaction, a_share, r_share = account(0.1, 0.2, 0.5)
    assert interaction == pytest.approx(0.2)
    assert a_share + r_share == pytest.approx(0.5)
    interaction, _, _ = account(0.1, 0.2, 0.3)
    assert interaction == pytest.approx(0)
