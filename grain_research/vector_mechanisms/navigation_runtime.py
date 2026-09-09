#!/usr/bin/env python3
"""Portable frozen navigation programs; no repository/model access at runtime."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .projection_program import METHODS, pool_pairings, score_token_ids


def read(path):
    """Read UTF-8 runtime metadata without executing serialized Python objects."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    """Stream a SHA-256 digest without materializing a weight file in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(bundle, relative):
    """Reject external or symlink-escaped dependency paths."""
    root = Path(bundle).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or external runtime dependency: " + str(relative))
    return path


def encode_texts(tokenizer, texts, recipe, expected, maximum=3056):
    """Reproduce frozen prompt bytes and query slots, rejecting silent truncation."""
    if (
        not texts
        or not isinstance(texts, list)
        or any(not isinstance(t, str) or not t for t in texts)
    ):
        raise ValueError("texts must be a nonempty list of complete excerpts")
    text = recipe["prefix"] + texts[0]
    for i, body in enumerate(texts[1:], 2):
        text += recipe["separator"].replace("{index}", str(i)) + body
    question_start = len(text) + recipe["question_offset_in_suffix"]
    text += recipe["suffix"]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids
    if not ids or len(ids) > maximum:
        raise ValueError("Input exceeds frozen capacity; no automatic truncation")
    mask = np.asarray([b > question_start for _, b in encoded.offsets], bool)
    mask[-1] = False
    if not mask.any() or mask[:-1].all() or np.asarray(ids)[mask].tolist() != expected:
        raise ValueError("Input did not preserve frozen question coordinates")
    return ids, mask


class NavigationRuntime:
    """Frozen raw-text arithmetic with explicit optional four-block dependence."""

    def __init__(self, bundle, program="token", device="cpu"):
        """Load only selected-program weights; never load a teacher or its head."""
        from tokenizers import Tokenizer

        if program not in ("token", "context4", "state1", "state2"):
            raise ValueError("Unknown exported program")
        self.root, self.program = Path(bundle).resolve(), program
        self.manifest = read(self.root / "manifest.json")
        self.accessed = []
        dependencies = self.manifest["program_dependencies"][program]
        for relative in dependencies:
            path = checked_path(self.root, relative)
            if file_hash(path) != self.manifest["files"][relative]:
                raise ValueError("Changed runtime dependency: " + relative)
            self.accessed.append(relative)
        self.spec = read(self.root / "queries.json")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.charts, self.calls = {}, 0
        if program == "token":
            self.table = np.load(
                self.root / "embedding_table.npy", mmap_mode="r", allow_pickle=False
            )
            for q in self.spec:
                self.charts[q] = []
                for i in range(2):
                    with np.load(
                        self.root / "token_charts" / f"{q}_{i}.npz", allow_pickle=False
                    ) as z:
                        chart = dict(z)
                    chart["offsets"] = {
                        method: pool_pairings(
                            chart["embeddings"],
                            chart["embeddings"],
                            chart["gradients"],
                            method,
                        )
                        for method in METHODS
                    }
                    self.charts[q].append(chart)
        else:
            import torch
            from safetensors.torch import load_file
            from transformers import Qwen2Config, Qwen2Model

            from .context_program import CroppedContext

            torch.set_num_threads(2)
            cfg = Qwen2Config.from_dict(read(self.root / "config.json"))
            cfg._attn_implementation = "eager"
            model = Qwen2Model(cfg).float()
            model.load_state_dict(
                load_file(self.root / "context.safetensors"), strict=True
            )
            model.to(device)
            self.context = CroppedContext(model, cuts=(3,), max_tokens=3056)
            for q in self.spec:
                if program == "context4":
                    self.charts[q] = []
                    for i in range(2):
                        with np.load(
                            self.root / "context_charts" / f"{q}_{i}.npz",
                            allow_pickle=False,
                        ) as z:
                            chart = dict(z)
                        e = ~chart["mask"].astype(bool)
                        e[-1] = False
                        chart["offset"] = pool_pairings(
                            chart["hidden"][e],
                            chart["hidden"][e],
                            chart["gradients"][:, e],
                            "semantic_order",
                        )
                        self.charts[q].append(chart)
                else:
                    with np.load(
                        self.root / "compiled" / f"{q}.npz", allow_pickle=False
                    ) as z:
                        self.charts[q] = dict(z)

    def encode(self, texts, query):
        """Tokenize the original excerpt list with a fixed query-specific recipe."""
        if query not in self.spec:
            raise ValueError("Query absent from the exported atlas")
        q = self.spec[query]
        return encode_texts(self.tokenizer, texts, q["recipe"], q["question_tokens"])

    def score(self, texts, query):
        """Return an unclipped mechanism score, not a calibrated questionnaire value."""
        ids, mask = self.encode(texts, query)
        anchor = self.spec[query]["anchor"]
        if self.program == "token":
            values = score_token_ids(
                ids,
                mask,
                self.table,
                self.charts[query],
                anchor,
                "semantic_order",
            )
            values.pop("charts")
        else:
            from .context_program import score_context_charts
            from .state_geometry import score_compiled_projection
            from .state_transport import pack_state

            h = self.context.capture(ids)[3]
            self.calls += 1
            if self.program == "context4":
                values = score_context_charts(self.charts[query], h, mask, anchor)[
                    "mean"
                ]
            else:
                values = score_compiled_projection(
                    self.charts[query],
                    pack_state(h[mask], h[-1]),
                    degree=int(self.program[-1]),
                )
        return {"query": query, "program": self.program, "tokens": len(ids), **values}

    def dependency_report(self):
        """Expose actual imports, dependency hashes and retained-neural execution."""
        return {
            "program": self.program,
            "declared_class": self.manifest["runtime_classes"][self.program],
            "verified_files": self.accessed,
            "torch_imported": "torch" in sys.modules,
            "transformers_imported": "transformers" in sys.modules,
            "transformer_forward_calls": self.calls,
            "retained_layers": 0 if self.program == "token" else 4,
            "LM_head": False,
            "teacher_PQ_inputs": False,
            "frozen_query_set": list(self.spec),
            "output_is_calibrated_personality_probability": False,
        }
