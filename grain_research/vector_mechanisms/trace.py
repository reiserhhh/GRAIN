"""Bounded decoder residual-stream capture and full-sequence scoring.

Hooks sample only specified positions. No generated answer is used as a
feature. The final normalization site is distinct from decoder block outputs.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Patch:
    """Intervention at a decoder block output, in that model's coordinates."""

    site: str
    position: int
    donor: np.ndarray
    alpha: float = 1.0
    basis: np.ndarray | None = None


def patch_delta(
    current: np.ndarray, donor: np.ndarray, basis: np.ndarray | None = None
) -> np.ndarray:
    """Project a donor difference onto an orthonormal column basis."""
    delta = np.asarray(donor) - np.asarray(current)
    if basis is None:
        return delta
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[0] != delta.shape[-1]:
        raise ValueError("Subspace shape does not match hidden dimension")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5):
        raise ValueError("Subspace basis must have orthonormal columns")
    return basis @ (basis.T @ delta)


class DecoderObserver:
    """Observe a Hugging Face decoder-only model without changing its weights."""

    def __init__(
        self, model, tokenizer, *, max_tokens=2048, layer_count=6, pad_to_length=None
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.pad_to_length = pad_to_length
        if pad_to_length is not None and not 1 <= pad_to_length <= max_tokens:
            raise ValueError("Invalid fixed execution length")
        decoder = getattr(model, "model", None)
        if decoder is None or not hasattr(decoder, "layers"):
            raise ValueError("Unsupported decoder layout: expected model.model.layers")
        indices = sorted(
            set(
                np.linspace(
                    0,
                    len(decoder.layers) - 1,
                    min(layer_count, len(decoder.layers)),
                    dtype=int,
                ).tolist()
            )
        )
        self.sites = {"input_embedding": decoder.embed_tokens}
        self.sites.update({f"block_{i:02d}": decoder.layers[i] for i in indices})
        self.sites["final_norm"] = decoder.norm
        self.calls = 0
        self.input_tokens = 0
        self.processed_tokens = 0

    @classmethod
    def load(
        cls,
        name,
        *,
        revision=None,
        device="cpu",
        dtype="float32",
        max_tokens=2048,
        local_files_only=True,
        pad_to_length=None,
    ):
        """Load an explicitly pinned public checkpoint; never enable remote code."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformer_logging

        transformer_logging.disable_progress_bar()

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=getattr(torch, dtype),
            attn_implementation="eager",
        ).to(device)
        return cls(model, tokenizer, max_tokens=max_tokens, pad_to_length=pad_to_length)

    def prompt_ids(self, system: str, user: str) -> list[int]:
        """Tokenize the actual chat template with an unanswered assistant prefix."""
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if not ids or not all(isinstance(item, int) for item in ids):
            raise ValueError(
                "Chat template must produce one flat integer token sequence"
            )
        if len(ids) + 16 > self.max_tokens:
            raise ValueError(
                f"Prompt exceeds token budget ({len(ids)}); no silent truncation"
            )
        return list(ids)

    def candidate_ids(self, ids: list[int], answer: str) -> list[int]:
        """Append an explicit token sequence and record exactly what is scored.

        The prompt's terminal chat-control boundary is preserved. Candidate
        tokens are defined independently, rather than retokenizing that prefix.
        """
        tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        if not tokens or len(ids) + len(tokens) > self.max_tokens:
            raise ValueError("Empty answer or candidate exceeds token budget")
        return list(tokens)

    def content_position(self, system: str, user: str, content: str) -> int:
        """Locate a unique content endpoint before the question using tokenizer offsets.

        This point cannot attend to later question tokens in a causal decoder.
        It separates evidence representation from the decision suffix.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not content or text.count(content) != 1:
            raise ValueError(
                "Content anchor must occur exactly once in the rendered prompt"
            )
        begin = text.index(content)
        end = begin + len(content)
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        if list(encoded["input_ids"]) != self.prompt_ids(system, user):
            raise ValueError(
                "Offset tokenization differs from actual chat-template tokens"
            )
        eligible = [
            i
            for i, (start, stop) in enumerate(encoded["offset_mapping"])
            if stop > start >= begin and stop <= end
        ]
        if not eligible:
            raise ValueError("No token ends inside the content anchor")
        position = eligible[-1]
        # Exclude tokens merged with the following delimiter, rather than reading
        # a token that could already contain part of the question.
        return position

    @contextmanager
    def hooks(self, positions: list[int], patch: Patch | None = None):
        """Capture/patch selected absolute positions and always remove all hooks."""
        captured = {}
        handles = []
        if patch is not None and patch.site not in self.sites:
            raise ValueError(f"Unknown patch site {patch.site}")
        if patch is not None and patch.basis is not None:
            patch_delta(np.zeros_like(patch.donor), patch.donor, patch.basis)

        def hook_for(name):
            def callback(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if patch is not None and patch.site == name:
                    if not 0 <= patch.position < hidden.shape[1]:
                        raise ValueError("Patch position outside sequence")
                    hidden = hidden.clone()
                    current = hidden[0, patch.position]
                    donor = torch.as_tensor(
                        patch.donor, device=current.device, dtype=current.dtype
                    )
                    delta = donor - current
                    if patch.basis is not None:
                        basis = torch.as_tensor(
                            patch.basis, device=current.device, dtype=current.dtype
                        )
                        delta = basis @ (basis.T @ delta)
                    hidden[0, patch.position] = current + patch.alpha * delta
                captured[name] = (
                    hidden[0, positions].detach().float().cpu().numpy().copy()
                )
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            return callback

        try:
            for name, module in self.sites.items():
                handles.append(module.register_forward_hook(hook_for(name)))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _forward(self, ids):
        """Execute one deterministic full forward and account for its cost."""
        device = next(self.model.parameters()).device
        target_length = self.pad_to_length or len(ids)
        if len(ids) > target_length:
            raise ValueError("Input exceeds fixed execution length")
        padding = target_length - len(ids)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        pad_id = 0 if pad_id is None else pad_id
        tokens = torch.tensor(
            [ids + [pad_id] * padding], dtype=torch.long, device=device
        )
        mask = torch.tensor(
            [[1] * len(ids) + [0] * padding], dtype=torch.long, device=device
        )
        self.calls += 1
        self.input_tokens += len(ids)
        self.processed_tokens += target_length
        with torch.inference_mode():
            output = self.model(
                input_ids=tokens,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
        # Do not expose padding positions as candidate readout positions.
        output.logits = output.logits[:, : len(ids)]
        return output

    def capture(self, ids: list[int], *, positions=None, patch=None):
        """Capture pre-answer states and last-position logits without generation."""
        positions = positions if positions is not None else [len(ids) - 1]
        if not ids or any(p < 0 or p >= len(ids) for p in positions):
            raise ValueError("Capture position outside prompt")
        if patch is not None and not 0 <= patch.position < len(ids):
            raise ValueError("Patch position outside sequence")
        with self.hooks(positions, patch) as captured:
            output = self._forward(ids)
        logits = output.logits[0, len(ids) - 1].detach().float().cpu().numpy()
        return captured, logits

    def score(self, ids: list[int], answer: str, *, patch=None) -> float:
        """Sum teacher-forced log probabilities for ALL candidate answer tokens."""
        answer_ids = self.candidate_ids(ids, answer)
        if patch is not None and not 0 <= patch.position < len(ids):
            raise ValueError("Patch must target a pre-answer prompt position")
        with self.hooks([len(ids) - 1], patch):
            # The final answer token is a target, not an input. In particular,
            # single-token answers must use the exact same prefill as capture.
            # An unnecessary extra token can alter low-precision kernel rounding.
            output = self._forward(ids + answer_ids[:-1])
        start = len(ids) - 1
        logits = output.logits[0, start : start + len(answer_ids)].float()
        target = torch.tensor(answer_ids, device=logits.device)
        return float(
            torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).sum().item()
        )

    def margin(self, ids, *, patch=None) -> float:
        """Return the Yes/No sequence log-likelihood contrast, not a trait score."""
        return self.score(ids, "Yes", patch=patch) - self.score(ids, "No", patch=patch)

    def manifest(self):
        """Describe the observer, including numerical and tokenization choices."""
        import transformers

        config = self.model.config
        parameter = next(self.model.parameters())
        template = str(self.tokenizer.chat_template)
        return {
            "model": config._name_or_path,
            "revision": getattr(config, "_commit_hash", None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "sites": list(self.sites),
            "max_tokens": self.max_tokens,
            "pad_to_length": self.pad_to_length,
            "attention": "eager",
            "use_cache": False,
            "training": self.model.training,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "candidate_ids": {
                s: self.tokenizer.encode(s, add_special_tokens=False)
                for s in ("Yes", "No")
            },
            "candidate_definition": "append_explicit_token_sequence_to_frozen_prompt_prefix",
            "unix_time": time.time(),
        }
