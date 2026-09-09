#!/usr/bin/env python3
"""Portable conditional-navigation scores from four frozen Transformer blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .conditional_inputs import evidence_mask, evidence_summaries
from .conditional_models import predict_response_kernel
from .navigation_runtime import checked_path, file_hash, read
from .state_transport import pack_state


def encode_with_evidence(tokenizer, texts, spec, maximum=3056):
    """Reproduce the frozen renderer and excerpt-only offsets without truncation."""
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(t, str) or not t for t in texts)
    ):
        raise ValueError("Nonempty complete excerpts required")
    recipe, spans = spec["recipe"], []
    text = recipe["prefix"]
    for i, body in enumerate(texts, 1):
        if i > 1:
            text += recipe["separator"].replace("{index}", str(i))
        spans.append((len(text), len(text) + len(body)))
        text += body
    question_start = len(text) + recipe["question_offset_in_suffix"]
    text += recipe["suffix"]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids, offsets = encoded.ids, encoded.offsets
    if not ids or len(ids) > maximum:
        raise ValueError("Frozen capacity exceeded; no truncation")
    mask = np.asarray([b > question_start for _, b in offsets], bool)
    mask[-1] = False
    if not mask.any() or np.asarray(ids)[mask].tolist() != spec["question_tokens"]:
        raise ValueError("Query coordinates changed")
    return ids, mask, evidence_mask(offsets, spans, mask)


def conditional_scores(inputs, models, x):
    """Evaluate source-centered fixed programs; no current-batch statistics or labels."""
    x = np.asarray(x, float)
    if x.shape != (inputs["x"].shape[1],) or not np.isfinite(x).all():
        raise ValueError("Invalid early input coordinates")
    base = float(inputs["anchor"]) + float(
        (x @ inputs["reader"] + inputs["bias"]).sum()
    )
    cross = (x @ inputs["x"].T)[None]
    return {
        name: base + float(predict_response_kernel(model, cross).sum())
        for name, model in models.items()
    }


class ConditionalRuntime:
    """One shallow forward per text/query shared across preselected fixed programs."""

    def __init__(self, bundle, device="mps"):
        """Load a self-contained private bundle, not a full decoder or LM head."""
        import torch
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import Qwen2Config, Qwen2Model

        from .context_program import CroppedContext

        self.root = Path(bundle).resolve()
        self.manifest = read(self.root / "manifest.json")
        self.accessed = []
        for name, sha in self.manifest["files"].items():
            if file_hash(checked_path(self.root, name)) != sha:
                raise ValueError("Changed conditional dependency: " + name)
            self.accessed.append(name)
        cfg = Qwen2Config.from_dict(read(self.root / "config.json"))
        if cfg.num_hidden_layers != 4:
            raise ValueError("Exactly four extracted layers required")
        cfg._attn_implementation = "eager"
        torch.set_num_threads(2)
        model = Qwen2Model(cfg).float()
        model.load_state_dict(load_file(self.root / "context.safetensors"), strict=True)
        model.to(device)
        self.context = CroppedContext(model, cuts=(3,), max_tokens=3056)
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.spec = read(self.root / "queries.json")
        self.inputs, self.models, self.calls = {}, {}, 0
        for q, methods in self.manifest["selection"].items():
            with np.load(
                self.root / "compiled" / f"inputs_{q}.npz", allow_pickle=False
            ) as z:
                self.inputs[q] = dict(z)
            self.models[q] = {}
            for method in methods:
                with np.load(
                    self.root / "compiled" / f"{q}_{method}.npz", allow_pickle=False
                ) as z:
                    self.models[q][method] = dict(z)

    def score_ids(self, ids, mask, query, body=None):
        """Accept frozen input token coordinates only, never upper teacher states."""
        if query not in self.models:
            raise ValueError("Query not selected by the construction-only decision")
        mask = np.asarray(mask, bool)
        if (
            len(mask) != len(ids)
            or mask[-1]
            or np.asarray(ids)[mask].tolist() != self.spec[query]["question_tokens"]
        ):
            raise ValueError("Invalid input query coordinates")
        h = self.context.capture(ids)[3]
        self.calls += 1
        x = pack_state(h[mask], h[-1])
        if self.inputs[query]["x"].shape[1] != len(x):
            if body is None:
                raise ValueError("Augmented query requires exact excerpt mask")
            x = np.r_[x, evidence_summaries(h, body, mask).ravel()]
        return conditional_scores(self.inputs[query], self.models[query], x)

    def score(self, texts, query):
        """Map complete raw excerpts to all preselected unclipped mechanism scores."""
        if query not in self.models:
            raise ValueError("Query not selected")
        ids, mask, body = encode_with_evidence(self.tokenizer, texts, self.spec[query])
        return self.score_ids(ids, mask, query, body)

    def dependency_report(self):
        """Expose retained neural dependence and private construction-anchor scope."""
        return {
            "program": "conditional",
            "runtime_layers": 4,
            "runtime_lm_head": False,
            "full_LLM_runtime": False,
            "Transformer_free": False,
            "neural_calls": self.calls,
            "torch_imported": "torch" in sys.modules,
            "transformers_imported": "transformers" in sys.modules,
            "human_labels_accessed": False,
            "files": self.accessed,
            "private_author_derived_anchor_vectors": True,
            "public_redistribution_authorized": False,
        }
