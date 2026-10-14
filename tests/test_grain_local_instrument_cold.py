#!/usr/bin/env python3
"""Check the cold wrapper's fixed pair-forward contract without model downloads."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.score_grain_local_instrument_cold import loaded_reader


class Tokenizer:
    """Record pair alignment and return a controlled batch length."""

    def __init__(self, length=3):
        self.length = length
        self.calls = []

    def __call__(self, sentences, items, **kwargs):
        self.calls.append((sentences, items, kwargs))
        return {
            "input_ids": torch.ones((len(sentences), self.length), dtype=torch.long)
        }


class Model:
    """Return deterministic logits in the checkpoint's explicit label order."""

    config = SimpleNamespace(
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
    )

    def __call__(self, input_ids):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(
            logits=torch.tensor([0.0, 2.0, 1.0]).repeat(len(input_ids), 1)
        )


def test_loaded_reader_keeps_pairs_labels_and_batching():
    """The runtime-only repair retains the original eight-pair, no-truncation readout."""
    tokenizer = Tokenizer()
    frame = pd.DataFrame(
        {
            "audit_id": range(9),
            "sentence": [f"s{i}" for i in range(9)],
            "item_text": [f"h{i}" for i in range(9)],
        }
    )
    result = loaded_reader(tokenizer, Model(), torch)(frame)
    assert result.audit_id.tolist() == list(range(9))
    assert result.nli.tolist() == ["entailment"] * 9
    assert [len(c[0]) for c in tokenizer.calls] == [8, 1]
    assert tokenizer.calls[1][:2] == (["s8"], ["h8"])
    assert tokenizer.calls[0][2] == {
        "padding": True,
        "truncation": False,
        "return_tensors": "pt",
    }
    np.testing.assert_allclose(
        result[["p_contradiction", "p_entailment", "p_neutral"]].iloc[0],
        torch.softmax(torch.tensor([0.0, 2.0, 1.0]), -1).numpy(),
    )


def test_loaded_reader_rejects_overlong_pairs():
    """A material-length error must not be repaired by silent truncation."""
    frame = pd.DataFrame({"audit_id": [0], "sentence": ["s"], "item_text": ["h"]})
    with pytest.raises(ValueError, match="no silent truncation"):
        loaded_reader(Tokenizer(513), Model(), torch)(frame)
