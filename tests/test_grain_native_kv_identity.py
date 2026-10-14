#!/usr/bin/env python3
"""Synthetic tests for role aliasing and suffix-only historical interchange."""

import itertools
from collections import Counter

import numpy as np
import pytest
import torch

from grain_research.construction import math_native_access as A
from grain_research.construction import math_native_kv_identity as M


def test_diagonal_role_alias_has_two_distinct_explanations():
    """Natural precision cannot distinguish bkp from bvp on k=v=o."""
    signs = np.array(list(itertools.product((-1, 1), repeat=4)))
    key = (signs[:, 0] * signs[:, 1] * signs[:, 3]).reshape(2, 2, 2, 2)
    value = (signs[:, 0] * signs[:, 2] * signs[:, 3]).reshape(2, 2, 2, 2)
    assert np.array_equal(M.diagonal(key), M.diagonal(value))
    assert not np.array_equal(key, value)
    assert M.characters(key)[11] == 1
    assert M.characters(value)[13] == 1


def test_full_restriction_identity_not_only_bop():
    """Every character follows explicit substitution on the diagonal."""
    x = np.arange(16 * 3).reshape(2, 2, 2, 2, 3) ** 2
    np.testing.assert_allclose(
        M.restrict_coefficients(M.characters(x)),
        A.characters(M.diagonal(x)),
        atol=1e-12,
    )


def test_interaction_coalesces_to_common_bp_not_ownership():
    """A bkvp interaction becomes bp; it is not an exclusive owner channel."""
    s = np.array(list(itertools.product((-1, 1), repeat=4)))
    x = np.prod(s, axis=1).reshape(2, 2, 2, 2)
    beta = M.restrict_coefficients(M.characters(x))
    assert beta[5] == 1 and beta[7] == 0


@pytest.mark.parametrize("bad", [np.zeros((2, 2, 2, 1)), np.full((2, 2, 2, 2), np.nan)])
def test_incomplete_cube_is_not_filled(bad):
    """Do not supply a full mechanism claim from missing or invalid support."""
    with pytest.raises(ValueError):
        M.characters(bad)


def test_suffix_interchange_preserves_prefix_and_self_cache():
    """Match donor K/V exactly while stock body attention remains unchanged."""
    q = torch.arange(12, dtype=torch.float32).reshape(1, 1, 4, 3) / 12
    k, v = q.clone(), q.clone() * 2

    def attention(_module, query, key, value, _mask, scaling, **_kwargs):
        """Minimal native attention with the real token/output axis convention."""
        weights = torch.softmax(query @ key.transpose(-1, -2) * scaling, -1)
        return (weights @ value).transpose(1, 2), weights

    expected, _ = attention(None, q, k, v, None, 1)
    got, _ = M.suffix_interchange(
        attention, q, k, v, None, 1, 2, [1], k[0, :, [1]], v[0, :, [1]], None
    )
    assert torch.equal(expected, got)
    changed, _ = M.suffix_interchange(
        attention, q, k, v, None, 1, 2, [1], k[0, :, [1]], v[0, :, [1]] + 20, None
    )
    assert torch.equal(changed[:, :2], expected[:, :2])
    assert not torch.equal(changed[:, 2:], expected[:, 2:])


def paired_rows():
    """Create complete positional support without accessing research materials."""
    result = []
    for bank, family, b, o in itertools.product((0, 3), range(6), (-1, 1), (-1, 1)):
        result.append(
            {
                "id": len(result),
                "bank": bank,
                "family": family,
                "polarity": b,
                "acting_person": o,
                "input_ids": list(range(187)),
                "history": list(range(46, 168)),
                "current": list(range(178, 187)),
                "padding": [167],
                "history_text": f"common events{family}/{b}\nowner{o}",
            }
        )
    return result


def test_design_has_complete_independent_kv_support():
    """Off-diagonal cells remain separate; controls do not fill scientific cells."""
    from scripts import run_grain_native_kv_identity_hm132 as R

    rows = paired_rows()
    design = R.fixed_design(rows)
    assert len(design) == 100
    assert Counter(d["mode"] for d in design) == {"capture": 48, "mixed": 48, "noop": 4}
    support = {
        (
            rows[d["row"]]["bank"],
            rows[d["row"]]["family"],
            rows[d["row"]]["polarity"],
            d["k"],
            d["v"],
        )
        for d in design
        if d["mode"] != "noop"
    }
    assert support == set(
        itertools.product((0, 3), range(6), (-1, 1), (-1, 1), (-1, 1))
    )


def test_unaligned_owner_padding_is_rejected():
    """Cache swaps must not change which historical positions can be read."""
    from scripts import run_grain_native_kv_identity_hm132 as R

    rows = paired_rows()
    rows[0]["padding"] = [166, 167]
    with pytest.raises(ValueError, match="identical positional support"):
        R.fixed_design(rows)


def test_native_interface_is_restored_after_failure(monkeypatch):
    """An unsuccessful model forward must not leave a global attention override."""
    from types import SimpleNamespace

    import transformers.models.qwen2.modeling_qwen2 as qwen

    original = qwen.eager_attention_forward
    observer = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(layers=[0]),
            config=SimpleNamespace(_attn_implementation="eager"),
        )
    )

    def failed(*_args, **_kwargs):
        """Fail before any actual model execution in this synthetic test."""
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(A, "native_batch", failed)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        M.native_batch(
            observer,
            {"input_ids": [1], "history": [0]},
            {"suffixes": [[1]], "alphabet": []},
        )
    assert qwen.eager_attention_forward is original
