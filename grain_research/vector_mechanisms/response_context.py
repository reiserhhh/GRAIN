"""Frozen same-response context interventions, without personality labels."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment


def native_parts(tokenizer, response, context=None):
    """Separate native-template prefix and complete response tokens by offsets."""
    messages = []
    if context is not None:
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": response})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    start = rendered.rfind(response)
    if not response or start < 0:
        raise ValueError("Response is absent from native template")
    stop = start + len(response)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a >= start and b <= stop]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Noncontiguous or empty response token interval")
    if any(
        b > start and a < stop and (a < start or b > stop) for a, b in offsets if b > a
    ):
        raise ValueError("Token crosses response/template boundary")
    ids = list(encoded["input_ids"])
    native = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if isinstance(native, Mapping):
        native = native["input_ids"]
    if ids != list(native):
        raise ValueError("Native template and offset tokenization differ")
    return ids[: indices[0]], ids[indices[0] : indices[-1] + 1]


def select_responses(views, tokenizer):
    """Take the last complete eligible user/preceding-assistant pair per free chat."""
    cases, coverage = [], []
    for view in views:
        meta = view["metadata"]
        if meta["view_kind"] != "free_chat":
            continue
        counts, selected = Counter(), None
        turns = view["turns"]
        for index in range(len(turns) - 1, 0, -1):
            user, ai = turns[index], turns[index - 1]
            if user["role"] != "user" or ai["role"] != "assistant":
                continue
            counts["candidate_pairs"] += 1
            try:
                p, body = native_parts(tokenizer, user["text"], ai["text"])
                _, alone = native_parts(tokenizer, user["text"])
            except ValueError:
                counts["template_boundary_rejected"] += 1
                continue
            n_context = len(tokenizer.encode(ai["text"], add_special_tokens=False))
            if not 16 <= len(body) <= 256 or not 16 <= n_context <= 640 or len(p) > 768:
                counts["length_rejected"] += 1
                continue
            if body != alone:
                raise ValueError("Changed response tokens under context")
            selected = {
                "participant_code": meta["participant_code"],
                "response": user["text"],
                "context": ai["text"],
                "response_sequence": user["source_sequence"],
                "context_sequence": ai["source_sequence"],
                "response_tokens": len(body),
                "context_tokens": n_context,
                "response_sha256": hashlib.sha256(user["text"].encode()).hexdigest(),
                "context_sha256": hashlib.sha256(ai["text"].encode()).hexdigest(),
            }
            cases.append(selected)
            break
        coverage.append(
            {
                "participant_code": meta["participant_code"],
                "eligible": selected is not None,
                **counts,
            }
        )
    return cases, coverage


def context_derangements(lengths):
    """One global length-matched assignment and one fixed random derangement."""
    x = np.asarray(lengths, dtype=float)
    if len(x) < 4 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("Need at least four positive context lengths")
    cost = abs(np.log(x[:, None]) - np.log(x[None, :]))
    cost += np.random.default_rng(731).uniform(0, 1e-10, size=cost.shape)
    np.fill_diagonal(cost, np.inf)
    rows, matched = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(x))) or (matched == rows).any():
        raise ValueError("Invalid matched derangement")
    rng = np.random.default_rng(1731)
    for _ in range(10000):
        random = rng.permutation(len(x))
        if not (random == rows).any():
            return matched, random
    raise ValueError("Random derangement failed; no silent fallback")


def packed_sequence(prefix, body, pad_id, *, compact=False):
    """Right-align the valid prefix so every response has an identical fixed origin."""
    if not prefix or not 1 <= len(body) <= 256 or len(prefix) > 768:
        raise ValueError("Unsupported complete prefix/response lengths")
    start = len(prefix) if compact else 768
    left = start - len(prefix)
    right = 1024 - start - len(body)
    ids = [pad_id] * left + list(prefix) + list(body) + [pad_id] * right
    mask = [0] * left + [1] * (len(prefix) + len(body)) + [0] * right
    return ids, mask, list(range(start, start + len(body)))


def cosine(left, right):
    """Finite cosine between vector summaries; zero vectors are unsupported."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Undefined representation cosine")
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


class ResponseObserver:
    """Observe current-token states and likelihood without generating or fitting."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.calls = 0
        self.sites = {
            "input_embedding": model.model.embed_tokens,
            "block_10": model.model.layers[10],
            "block_16": model.model.layers[16],
            "final_norm": model.model.norm,
        }

    def forward(self, prefix, body, *, compact=False, patch=None):
        """Optional full-current-span block10 replacement; all target logits are causal."""
        import torch

        ids, mask, indices = packed_sequence(
            prefix, body, self.tokenizer.pad_token_id, compact=compact
        )
        device = next(self.model.parameters()).device
        ix = torch.tensor(indices, device=device)
        captures, handles = {}, []

        def hook(name):
            def apply(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if name == "block_10" and patch is not None:
                    donor = torch.as_tensor(patch, device=device, dtype=hidden.dtype)
                    if donor.shape != hidden[0, ix].shape:
                        raise ValueError("Patch does not match current-token span")
                    hidden = hidden.clone()
                    hidden[0, ix] = donor
                current = hidden[0, ix].detach().float().cpu().numpy()
                if not np.isfinite(current).all():
                    raise ValueError("Nonfinite user-token state")
                captures[name] = {"mean": current.mean(axis=0)}
                if name == "block_10":
                    captures[name]["states"] = current
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            return apply

        try:
            for name, module in self.sites.items():
                handles.append(module.register_forward_hook(hook(name)))
            with torch.inference_mode():
                hidden = self.model.model(
                    input_ids=torch.tensor([ids], device=device),
                    attention_mask=torch.tensor([mask], device=device),
                    position_ids=torch.arange(1024, device=device)[None, :],
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                logits = self.model.lm_head(hidden[0, ix - 1]).float()
                targets = torch.tensor(body, device=device)
                logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[
                    :, 0
                ]
                values = logp.cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        self.calls += 1
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite response log probability")
        return {
            "trace": captures,
            "token_logp": values,
            "mean_logp": float(values.mean()),
            "tail_logp": float(values[1:].mean()),
        }


def paired_interval(values):
    """Conditional material bootstrap only, not population or donor-assignment inference."""
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Missing paired effects")
    rng = np.random.default_rng(731)
    means = x[rng.integers(len(x), size=(1000, len(x)))].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(x.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "positive_count": int((x > 0).sum()),
        "participants": len(x),
    }
