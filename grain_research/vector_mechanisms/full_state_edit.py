#!/usr/bin/env python3
"""Role-aligned natural edits and finite context/query state interventions."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

import numpy as np
import torch

from .direct_traits import native_readout


def align_excerpts(
    full_text,
    edited_text,
    full_spans,
    edited_spans,
    kept,
    full_encoding,
    edited_encoding,
):
    """Map body tokens by source excerpt and scaffold tokens by displayed ordinal.

    Every transported donor has the same token ID and mapped character interval.
    Retokenized scaffold/boundary tokens use -1: keep their edited states in all
    cells and account for them in rebasing. Interior body mismatches fail closed.
    """
    if len(kept) != len(edited_spans) or not kept or sorted(set(kept)) != list(kept):
        raise ValueError("Ordered distinct retained excerpts required")
    if min(kept) < 0 or max(kept) >= len(full_spans):
        raise ValueError("Retained excerpt out of range")
    segments = []
    end = 0
    for i, source in enumerate(kept):
        elo, ehi = edited_spans[i]
        # The visible Excerpt i title retains the same ordinal wording as full i.
        flo = 0 if i == 0 else full_spans[i - 1][1]
        fhi = full_spans[i][0]
        segments.append((end, elo, flo, fhi, "scaffold"))
        blo, bhi = full_spans[source]
        segments.append((elo, ehi, blo, bhi, "body"))
        end = ehi
    segments.append(
        (end, len(edited_text), full_spans[-1][1], len(full_text), "suffix")
    )
    for lo, hi, flo, fhi, _role in segments:
        if edited_text[lo:hi] != full_text[flo:fhi]:
            raise ValueError("Declared text-role segments are not exact")
    lookup = defaultdict(list)
    for i, (token, (lo, hi)) in enumerate(
        zip(full_encoding["input_ids"], full_encoding["offset_mapping"], strict=True)
    ):
        lookup[(int(lo), int(hi), int(token))].append(i)
    starts = [s[0] for s in segments]
    result, used = [], defaultdict(int)
    fresh = {"scaffold": 0, "body_boundary": 0}
    for token, (lo, hi) in zip(
        edited_encoding["input_ids"], edited_encoding["offset_mapping"], strict=True
    ):
        if hi <= lo:
            raise ValueError("Token has no alignable character interval")
        segment = segments[bisect_right(starts, lo) - 1]
        elo, ehi, flo, fhi, role = segment
        a, b = flo + lo - elo, flo + hi - elo
        key = (a, b, int(token))
        candidates = lookup.get(key, [])
        if full_text[a:b] != edited_text[lo:hi] or not candidates:
            touches_boundary = (
                lo == elo
                or hi >= ehi
                or any(
                    x < b and y > a and (x < flo or y > fhi)
                    for x, y in full_encoding["offset_mapping"]
                )
            )
            if role == "body" and not touches_boundary:
                raise ValueError(
                    f"Unaligned interior body token at edited position {len(result)}"
                )
            result.append(-1)
            fresh["body_boundary" if role == "body" else "scaffold"] += 1
            continue
        occurrence = used[key]
        if occurrence >= len(candidates):
            raise ValueError("Ambiguous repeated source-token assignment")
        result.append(candidates[occurrence])
        used[key] += 1
    indices = np.asarray(result, dtype=int)
    valid = indices >= 0
    if not np.array_equal(
        np.asarray(full_encoding["input_ids"])[indices[valid]],
        np.asarray(edited_encoding["input_ids"])[valid],
    ):
        raise ValueError("Token identity mismatch")
    if indices[-1] != len(full_encoding["input_ids"]) - 1:
        raise ValueError("Final query endpoint must map to original final position")
    return indices, {
        "tokens": len(indices),
        "exact_token_matches": int(valid.sum()),
        "fresh_tokens": fresh,
        "renumbered_headers": sum(source != i for i, source in enumerate(kept)),
        "monotone": bool(np.all(np.diff(indices[valid]) > 0)),
        "duplicate_donors": int(valid.sum()) - len(set(indices[valid].tolist())),
        "policy": "exact-body/source-scaffold-role; retokenized-boundaries-kept-edited-in-rebase",
    }


def align_states(full, edited, mapping):
    """Transport matched states; every unmatched boundary retains its edited state."""
    indices = torch.as_tensor(mapping, device=full.device)
    if full.ndim != 3 or edited.ndim != 3 or len(indices) != edited.shape[1]:
        raise ValueError("State mapping does not cover the entire edited sequence")
    if full.shape[0] != edited.shape[0] or full.shape[2] != edited.shape[2]:
        raise ValueError("State dimensions differ")
    if bool(torch.any(indices < -1)) or bool(torch.any(indices >= full.shape[1])):
        raise ValueError("Invalid donor index")
    valid = indices >= 0
    result = edited.clone()
    result[:, valid] = full[:, indices[valid]]
    return result


def grid_components(full, f00, f10, f01, f11):
    """Exact four-cell main/interaction and two-order allocations, in score units."""
    values = np.asarray([full, f00, f10, f01, f11], float)
    if not np.isfinite(values).all():
        raise ValueError("Finite endpoint scores required")
    return {
        "natural": f11 - full,
        "rebase": f00 - full,
        "context_main": f10 - f00,
        "query_main": f01 - f00,
        "interaction": f11 - f10 - f01 + f00,
        "context_allocated": ((f10 - f00) + (f11 - f01)) / 2,
        "query_allocated": ((f01 - f00) + (f11 - f10)) / 2,
    }


def mix_states(aligned, edited, context, query):
    """Select all nonfinal states and final state without mutating either donor."""
    if aligned.shape != edited.shape or aligned.ndim != 3 or aligned.shape[1] < 2:
        raise ValueError("Matched batch/token/hidden states required")
    if context not in (0, 1) or query not in (0, 1):
        raise ValueError("Binary cell coordinates required")
    h = (edited if context else aligned).clone()
    h[:, -1] = (edited if query else aligned)[:, -1]
    return h


def capture_boundary(observer, ids, alphabet, block=16):
    """Capture a same-run full score, block output and native suffix arguments."""
    model = observer.model
    layers = model.model.layers
    saved = {}

    def state_hook(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        saved["state"] = h.detach().clone()

    def arguments(_module, _args, kwargs):
        saved["kwargs"] = dict(kwargs)

    handles = [
        layers[block].register_forward_hook(state_hook),
        layers[block + 1].register_forward_pre_hook(arguments, with_kwargs=True),
    ]
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    observer.calls += 1
    try:
        with torch.no_grad():
            output = model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            readout = native_readout(
                output.logits[0, -1].float().cpu().numpy(), alphabet
            )
        if saved["kwargs"].get("past_key_values") is not None or saved["kwargs"].get(
            "use_cache"
        ):
            raise ValueError("Cached KV would invalidate full-state replacement")
        return saved["state"], saved["kwargs"], readout
    finally:
        for handle in handles:
            handle.remove()


def replay_suffix(observer, state, kwargs, alphabet, block=16):
    """Run only remaining blocks with the native mask/position/RoPE arguments."""
    model = observer.model
    if any(
        layer.self_attn.sliding_window is not None
        for layer in model.model.layers[block + 1 :]
    ):
        raise ValueError("This packet supports only the frozen full-attention suffix")
    observer.calls += 1
    with torch.no_grad():
        h = state
        for layer in model.model.layers[block + 1 :]:
            h = layer(h, **kwargs)
            h = h[0] if isinstance(h, tuple) else h
        logits = model.lm_head(model.model.norm(h[:, -1:]))[0, -1]
        return native_readout(logits.float().cpu().numpy(), alphabet)
