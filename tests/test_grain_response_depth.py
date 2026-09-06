"""Small synthetic causal model controls for registered-site activation patches."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from grain_research.vector_mechanisms.response_context import ResponseObserver
from scripts.run_grain_response_depth import SITES, summarize


class CausalLayer(nn.Module):
    """Position-causal mixing makes a body patch unable to change its prefix."""

    def forward(self, hidden):
        count = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[
            None, :, None
        ]
        return hidden + hidden.cumsum(dim=1) / count * 0.02


class TinyBase(nn.Module):
    """A no-cache causal stack with a positionwise final normalization."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 8)
        self.layers = nn.ModuleList(CausalLayer() for _ in range(28))
        self.norm = nn.LayerNorm(8)

    def forward(self, input_ids, attention_mask, **kwargs):
        h = self.embed_tokens(input_ids) * attention_mask[..., None]
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))


class TinyLM(nn.Module):
    """Only the interfaces used by the observer, not a language-performance test."""

    def __init__(self):
        super().__init__()
        self.model = TinyBase()
        self.lm_head = nn.Linear(8, 64)


def test_registered_patches_replay_and_respect_causality():
    torch.manual_seed(731)
    model = TinyLM()
    observer = ResponseObserver(
        model,
        SimpleNamespace(pad_token_id=0),
        blocks=(4, 10, 16, 22, 27),
        capture_sites=SITES,
    )
    body = [11, 12, 13, 14]
    actual = observer.forward([2] * 500, body)
    mismatch = observer.forward([3] * 500, body)
    assert not np.array_equal(actual["token_logp"], mismatch["token_logp"])
    for site in SITES:
        sham = observer.forward(
            [3] * 500, body, patch_site=site, patch=mismatch["trace"][site]["states"]
        )
        patched = observer.forward(
            [3] * 500, body, patch_site=site, patch=actual["trace"][site]["states"]
        )
        assert np.array_equal(sham["token_logp"], mismatch["token_logp"])
        assert patched["token_logp"][0] == mismatch["token_logp"][0]
        if site in {"block_27", "final_norm"}:
            assert np.array_equal(patched["token_logp"][1:], actual["token_logp"][1:])
    with pytest.raises(ValueError, match="not registered"):
        observer.forward([3], body, patch=np.zeros((4, 8)), patch_site="unknown")
    with pytest.raises(ValueError, match="span"):
        observer.forward([3], body, patch=np.zeros((2, 8)))
    assert all(len(m._forward_hooks) == 0 for m in observer.sites.values())


def test_default_observer_retains_rc1_sites_and_state_scope():
    o = ResponseObserver(TinyLM(), SimpleNamespace(pad_token_id=0))
    v = o.forward([1], [2, 3])
    assert set(v["trace"]) == {"input_embedding", "block_10", "block_16", "final_norm"}
    assert {k for k, value in v["trace"].items() if "states" in value} == {"block_10"}


def test_depth_summary_does_not_divide_by_individual_gap():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "participant_code": str(i),
                "site": site,
                "actual_tail_logp": gap,
                "matched_tail_logp": 0.0,
                "patch_tail_logp": 0.5 * gap,
                "self_patch_max_error": 0,
                "patch_actual_tail_max_error": abs(gap) * 0.5,
                "patch_first_token_change": 0,
            }
            for site in SITES
            for i, gap in enumerate([0, 1, -0.2])
        ]
    )
    summary = summarize(frame)
    assert summary["effects"]["block_10"]["ratio_of_group_means"] == pytest.approx(0.5)
    assert all(x["mean"] == 0 for x in summary["adjacent_layer_contrasts"].values())
