#!/usr/bin/env python3
"""CPU64 normalization companion for the unchanged sealed HM255 native reader."""

from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from .math_item_bridge import common_prefix


def native_items(model, sequences, alphabet, device, batch_size=4):
    """Preserve cached native inference; normalize copied logits on CPU, not MPS."""
    import torch

    if batch_size < 1 or len(alphabet) != 2 or len(set(alphabet)) != 2:
        raise ValueError("Fixed positive batch size and binary alphabet required")
    length = common_prefix(sequences)
    states, logits, normalizers, top = [], [], [], []
    with torch.inference_mode():
        prefix = torch.tensor([sequences[0][:length]], dtype=torch.long, device=device)
        cached = model.model(
            input_ids=prefix, use_cache=True, return_dict=True
        ).past_key_values
        for start in range(0, len(sequences), batch_size):
            suffixes = [s[length:] for s in sequences[start : start + batch_size]]
            width = max(map(len, suffixes))
            ids = torch.zeros((len(suffixes), width), dtype=torch.long, device=device)
            mask = torch.zeros(
                (len(suffixes), length + width), dtype=torch.long, device=device
            )
            mask[:, :length] = 1
            for i, suffix in enumerate(suffixes):
                ids[i, : len(suffix)] = torch.tensor(suffix, device=device)
                mask[i, length : length + len(suffix)] = 1
            cache = deepcopy(cached)
            cache.batch_repeat_interleave(len(suffixes))
            output = model.model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            indices = torch.tensor([len(s) - 1 for s in suffixes], device=device)
            h = output.last_hidden_state[
                torch.arange(len(suffixes), device=device), indices
            ]
            full = model.lm_head(h).float()
            if not torch.isfinite(full).all() or not torch.isfinite(h).all():
                raise ValueError("Nonfinite native item observation")
            states.append(h.float().cpu().numpy())
            logits.append(full[:, alphabet].cpu().numpy())
            normalizers.append(logsumexp(full.cpu().numpy().astype(float), -1))
            top.append(full.argmax(-1).cpu().numpy())
    return {
        "states": np.concatenate(states),
        "logits": np.concatenate(logits).astype(float),
        "normalizer": np.concatenate(normalizers),
        "top": np.concatenate(top),
        "prefix_tokens": np.array(length),
    }
