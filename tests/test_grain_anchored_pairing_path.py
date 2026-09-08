#!/usr/bin/env python3
"""Actual suffix hook is local to the final head and always removed."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_grain_anchored_pairing_path as packet


def test_patch_changes_only_final_head_and_cleans_up(monkeypatch):
    projection = torch.nn.Identity()
    observer = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection))
                    for _ in range(18)
                ]
            )
        )
    )
    state = torch.arange(12.0).reshape(1, 3, 4)
    captured = []

    def replay(_observer, value, _kwargs, _alphabet):
        changed = projection(value)
        captured.append(changed.clone())
        return {"conditional_score": float(changed[0, -1].sum())}

    monkeypatch.setattr(packet, "cached_suffix_kwargs", lambda *_: {})
    monkeypatch.setattr(packet, "replay_suffix", replay)
    direction = np.array([0.1, -0.3, 0.2, 0.4])
    score, error = packet.paired_suffix(
        observer, state, {}, state[0, -1].numpy(), direction
    )
    assert error == 0
    assert torch.equal(captured[0][0, :-1], state[0, :-1])
    assert score == pytest.approx(float(state[0, -1].sum()) - direction.sum())
    assert not projection._forward_pre_hooks
    with pytest.raises(ValueError, match="differs"):
        packet.paired_suffix(observer, state, {}, np.ones(4), direction)
    assert not projection._forward_pre_hooks
