#!/usr/bin/env python3
"""Primitive and independent synthetic-oracle tests for the numeric construction."""

import itertools

import numpy as np
import pytest
import torch

from grain_research.construction import math_recursive_collector as C


def toy():
    """Provide two fixed layers, independent context donors and a current seed."""
    rng = np.random.default_rng(112)
    config = {
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
    }

    def random(shape):
        """Return stable float32 arrays, avoiding accidental saturated examples."""
        return rng.normal(0, 0.2, shape).astype(np.float32)

    library = {
        "seed": random((2, 8)),
        "cos": np.cos(np.arange(22).reshape(11, 2) * 0.1).astype(np.float32),
        "sin": np.sin(np.arange(22).reshape(11, 2) * 0.1).astype(np.float32),
    }
    names = (
        [("p", 0)]
        + [
            (f"e{j}_l{level}", j + 1)
            for j, level in itertools.product(range(6), range(3))
        ]
        + [(f"r_o{owner}", 7) for owner in (-1, 1)]
    )
    for name, pos in names:
        library[name + "_pos"] = np.array([pos])
        for channel in ("k", "v"):
            library[name + "_" + channel] = random((2, 1, 2, 2))
    weights = []
    for _ in range(3):
        w = {
            "input_layernorm.weight": torch.ones(8),
            "post_attention_layernorm.weight": torch.ones(8),
        }
        for name, shape in (("q", (8, 8)), ("k", (4, 8)), ("v", (4, 8))):
            w[f"self_attn.{name}_proj.weight"] = torch.from_numpy(random(shape))
            w[f"self_attn.{name}_proj.bias"] = torch.from_numpy(random((shape[0],)))
        for name, shape in (
            ("self_attn.o_proj.weight", (8, 8)),
            ("mlp.gate_proj.weight", (12, 8)),
            ("mlp.up_proj.weight", (12, 8)),
            ("mlp.down_proj.weight", (8, 12)),
        ):
            w[name] = torch.from_numpy(random(shape))
        weights.append(w)
    return library, weights, config


def oracle(library, case, weights, config, broadcast=False):
    """Independent NumPy token/head loops, no compiler attention/block helpers."""
    h = library["seed"].astype(float)
    cos, sin = library["cos"].astype(float), library["sin"].astype(float)
    context_length = len(cos) - len(h)

    def norm(x, weight):
        """Use a second RMS implementation for the reference oracle."""
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config["rms_norm_eps"])
            * weight
        )

    def rotate(x, indices):
        """Apply an explicit paired coordinate swap independent of Torch code."""
        return (
            x * cos[indices, None]
            + np.stack((-x[..., 1], x[..., 0]), axis=-1) * sin[indices, None]
        )

    for layer in range(2):
        w = {key: value.numpy().astype(float) for key, value in weights[layer].items()}
        x = norm(h, w["input_layernorm.weight"])
        q, k, v = [
            (
                (
                    x @ w[f"self_attn.{name}_proj.weight"].T
                    + w[f"self_attn.{name}_proj.bias"]
                ).reshape(2, heads, 2)
            )
            for name, heads in (("q", 4), ("k", 2), ("v", 2))
        ]
        q, k = rotate(q, np.arange(9, 11)), rotate(k, np.arange(9, 11))
        kk, vv = np.zeros((11, 2, 2)), np.zeros((11, 2, 2))
        valid = np.zeros(context_length, bool)
        names = [
            "p",
            *(f"e{j}_l{level}" for j, level in enumerate(case["levels"])),
            f"r_o{case['owner']}",
        ]
        for name in names:
            pos = library[name + "_pos"]
            source_layer = 1 if broadcast else layer
            kk[pos] = rotate(library[name + "_k"][source_layer].astype(float), pos)
            vv[pos] = library[name + "_v"][source_layer]
            valid[pos] = True
        kk[9:], vv[9:] = k, v
        update = np.zeros((2, 4, 2))
        for i, head in itertools.product(range(2), range(4)):
            support = list(np.flatnonzero(valid)) + list(range(9, 10 + i))
            scores = np.array(
                [q[i, head] @ kk[j, head // 2] / np.sqrt(2) for j in support]
            )
            p = np.exp(scores - max(scores))
            p /= sum(p)
            update[i, head] = sum(
                weight * vv[j, head // 2] for weight, j in zip(p, support, strict=True)
            )
        h += update.reshape(2, 8) @ w["self_attn.o_proj.weight"].T
        x = norm(h, w["post_attention_layernorm.weight"])
        gate = x @ w["mlp.gate_proj.weight"].T
        h += ((gate / (1 + np.exp(-gate))) * (x @ w["mlp.up_proj.weight"].T)) @ w[
            "mlp.down_proj.weight"
        ].T
    w = {key: value.numpy().astype(float) for key, value in weights[2].items()}
    x = norm(h, w["input_layernorm.weight"])
    k, v = [
        (
            (
                x @ w[f"self_attn.{name}_proj.weight"].T
                + w[f"self_attn.{name}_proj.bias"]
            ).reshape(2, 2, 2)
        )
        for name in ("k", "v")
    ]
    return rotate(k, np.arange(9, 11)), v


@pytest.mark.parametrize("variant", ("layered", "broadcast_last_history"))
def test_numeric_recurrence_against_independent_oracle(variant):
    """Padding, causality, GQA/RoPE, RMSNorm, MLP and residuals all enter the test."""
    library, weights, config = toy()
    cases = [{"levels": [0, 1, 2, 1, 1, 0], "owner": owner} for owner in (-1, 1)]
    k, v = C.compile_current(library, cases, weights, config, variant, "cpu")
    for i, case in enumerate(cases):
        kk, vv = oracle(library, case, weights, config, variant != "layered")
        np.testing.assert_allclose(k[i], kk, atol=3e-7, rtol=2e-6)
        np.testing.assert_allclose(v[i], vv, atol=3e-7, rtol=2e-6)


def test_positive_same_pair_marginals():
    """Two author laws differ only in degree3, not any first/two-way observation."""
    signs, counts = C.parity_laws()
    powers = np.array(list(itertools.product((0, 1), repeat=3)))
    moments = counts @ np.prod(signs[:, None] ** powers[None], axis=2) / 40
    np.testing.assert_array_equal(moments[0, :-1], moments[1, :-1])
    np.testing.assert_allclose(moments[:, -1], [0.4, -0.4])
    assert counts.min() == 3 and counts.sum(1).tolist() == [40, 40]
    f = 2 + signs.sum(1) + signs[:, 0] * signs[:, 1]
    scores, coefficient = C.parity_account(f[:, None])
    np.testing.assert_allclose(scores[0], scores[1])
    np.testing.assert_allclose(coefficient, 0)


def test_exact_author_third_character():
    """Explicit40-record weights reconstruct the observed three-way difference."""
    signs, counts = C.parity_laws()
    f = np.prod(signs, axis=1)[:, None] * np.arange(1, 7)[None]
    score, coef = C.parity_account(f)
    explicit = np.stack([np.repeat(f, w, axis=0).mean(0) for w in counts])
    np.testing.assert_allclose(score, explicit)
    np.testing.assert_allclose(score[0] - score[1], 0.8 * coef)


def test_no_silent_missing_or_wrong_owner():
    """Undefined event levels, owners and incomplete author supports are rejected."""
    library, _, _ = toy()
    with pytest.raises(ValueError):
        C.assemble_context(library, [{"levels": [1] * 6, "owner": 0}], 0, 9)
    with pytest.raises(ValueError):
        C.assemble_context(library, [{"levels": [3] * 6, "owner": 1}], 0, 9)
    with pytest.raises(ValueError):
        C.parity_account(np.zeros((7, 6)))


def test_history_layer_not_final_vector_sum():
    """Changing the layer address changes this fixed operation without any fit."""
    library, weights, config = toy()
    case = [{"levels": [1] * 6, "owner": 1}]
    a = C.compile_current(library, case, weights, config, "layered", "cpu")
    b = C.compile_current(
        library, case, weights, config, "broadcast_last_history", "cpu"
    )
    assert np.linalg.norm(a[0] - b[0]) > 1e-4


def test_streamed_hash(tmp_path):
    """Large-artifact hashing uses the same exact bytes as standard SHA256."""
    import hashlib

    path = tmp_path / "public_test.bin"
    path.write_bytes(b"public synthetic" * 100)
    assert C.stream_sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
