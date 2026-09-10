#!/usr/bin/env python3
"""Endpoint composition, native derivatives and source-only selection tests."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.selection_endpoint import (
    account,
    endpoint,
    endpoint_gradients,
    key_vectors,
    source_reader,
)
from grain_research.vector_mechanisms.source_query_selection import (
    evidence_inputs,
    key_actions,
    source_query,
)
from grain_research.vector_mechanisms.upstream_attention_channels import (
    decompose,
    export_weights,
)
from tests.test_grain_complete_transport import tiny_model


@pytest.mark.parametrize("length", [7, 13])
def test_full_vectors_recover_native_key_and_both_local_readers(length):
    model = tiny_model()
    rng = np.random.default_rng(length)
    x, d, readers = (
        rng.normal(size=(length, 16)),
        rng.normal(size=(4, length, 16)),
        rng.normal(size=(2, 16)),
    )
    w = export_weights(model, block=1)
    kw = cached_suffix_kwargs(model, torch.tensor(x[None], dtype=torch.float32))
    cos, sin = [a.numpy()[0] for a in kw["position_embeddings"]]
    mask = np.zeros(length)
    mask[1] = -np.inf
    evidence = evidence_inputs(
        {"x": x, "d": d, "cos": cos, "sin": sin, "mask": mask}, w, readers
    )
    q = source_query(x[-1], w)
    anchors = np.stack([q, q * 0.8, q * 1.1])
    pure = {k: v for k, v in evidence.items() if k != "g"}
    vector = key_vectors(anchors, pure, w["o"])
    native = decompose(x, d, cos, sin, mask, w)["channels"][1]
    np.testing.assert_allclose(vector[0], native, atol=2e-15)
    np.testing.assert_allclose(
        np.einsum("afh,mh->amf", vector, readers),
        key_actions(anchors, evidence),
        atol=2e-15,
    )
    with pytest.raises(ValueError, match="Pure"):
        key_vectors(anchors, pure | {"q": q}, w["o"])


def test_real_suffix_reverse_and_forward_derivatives_and_injection():
    model = tiny_model()
    state = torch.randn(1, 9, 16)
    delta = torch.randn_like(state)
    kw = cached_suffix_kwargs(model, state)
    ids, c = torch.tensor([1, 2, 3]), torch.tensor([-0.5, 0.0, 0.5])
    score, g, r, post = endpoint_gradients(model, state, kw, ids, c, block=1)
    _, jvp = torch.func.jvp(
        lambda x: endpoint(model, x, kw, ids, c, block=1), (state,), (delta,)
    )
    torch.testing.assert_close(jvp, (g * delta).sum(), rtol=1e-4, atol=1e-7)
    vec = torch.randn(16)
    full = torch.zeros_like(state)
    full[:, -1] = vec
    direct, injected = torch.func.jvp(
        lambda x: endpoint(model, x, kw, ids, c, block=1, post=True), (post,), (full,)
    )
    torch.testing.assert_close(score, direct)
    torch.testing.assert_close(injected, (r * vec).sum(), rtol=1e-4, atol=1e-7)


def test_local_projection_does_not_prove_endpoint_composition():
    u, uh = np.array([1.0, 0.0]), np.array([1.0, 2.0])
    local = np.array([1.0, 0.0])
    suffix = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert local @ u == local @ uh
    assert local @ suffix @ u != local @ suffix @ uh
    row = account(1.0, 3.0, 2.0, 5.0)
    assert row["total_error"] == row["selection_error"] + row["reader_error"]
    assert (
        row["total_error"] ** 2
        == row["selection_error"] ** 2
        + row["reader_error"] ** 2
        + 2 * row["selection_error"] * row["reader_error"]
    )
    assert row["physical_patch"] == 7 and row["compiled_swap"] == 6


def test_source_reader_excludes_held_author_and_all_new8_share_one_reader():
    bank = {
        (i, q): np.ones(3) * (i + 10 * j)
        for i in range(4)
        for j, q in enumerate(QUERIES)
    }
    v, members = source_reader(bank, "source4", 3, QUERIES[0])
    assert members == [0, 1, 2]
    np.testing.assert_array_equal(v, 1)
    for i in range(8):
        v, members = source_reader(bank, "new8", i, QUERIES[1])
        assert members == [0, 1, 2, 3]
        np.testing.assert_array_equal(v, 11.5)
