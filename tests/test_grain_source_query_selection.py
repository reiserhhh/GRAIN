#!/usr/bin/env python3
"""Source exclusions, recipient RoPE and actual-query key-channel identity."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.source_query_selection import (
    FIELDS,
    anchors,
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
def test_actual_query_recovers_existing_key_channel_in_recipient_rope(length):
    model = tiny_model()
    rng = np.random.default_rng(length)
    x = rng.normal(size=(length, 16))
    d = rng.normal(size=(4, length, 16))
    shared = rng.normal(size=(2, 16))
    weights = export_weights(model, block=1)
    kw = cached_suffix_kwargs(model, torch.tensor(x[None], dtype=torch.float32))
    cos, sin = [v.numpy()[0] for v in kw["position_embeddings"]]
    mask = np.zeros(length)
    mask[2] = -np.inf
    c = {"x": x, "d": d, "cos": cos, "sin": sin, "mask": mask}
    e = evidence_inputs(c, weights, shared)
    assert set(e) == FIELDS
    q = source_query(x[-1], weights)
    actual = decompose(x, d, cos, sin, mask, weights)["channels"][1] @ shared.T
    np.testing.assert_allclose(key_actions(q[None], e)[0], actual.T, atol=1e-15)
    bad = e | {"q": q}
    with pytest.raises(ValueError, match="Only pure"):
        key_actions(q[None], bad)


def test_source_exclusion_and_identical_source_limits():
    bank = {
        "index": np.repeat(np.arange(4), 2),
        "query": np.tile(QUERIES, 4),
        "q": np.array(
            [np.ones((4, 4)) * (i + 10 * q) for i in range(4) for q in range(2)]
        ),
    }
    a, s = anchors(bank, "source4", 3, QUERIES[0])
    assert s == [0, 1, 2]
    np.testing.assert_allclose(a[:, 0, 0], [1, 11, 6])
    b, s = anchors(bank, "new8", 0, QUERIES[1])
    assert s == [0, 1, 2, 3]
    np.testing.assert_allclose(b[:, 0, 0], [11.5, 1.5, 6.5])
    bank["q"][:] = 2
    np.testing.assert_array_equal(anchors(bank, "new8", 7, QUERIES[0])[0], 2)
    bank["index"][0] = 1
    with pytest.raises(ValueError, match="Exact paired"):
        anchors(bank, "new8", 0, QUERIES[0])
