#!/usr/bin/env python3
"""Cold query-conditioned programs and exact source-chart carrier accounting."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .affine_program import score_affine
from .functional_profiles import FunctionalQuery
from .navigation_runtime import checked_path, encode_texts
from .projection_program import pool_pairings, score_token_ids
from .source_span import predict_linear
from .state_transport import pack_state, unpack_state


def digest(path):
    """Hash a declared artifact without reading arbitrary source data."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    """Read metadata without executable serialization."""
    return json.loads(Path(path).read_text())


def arrays(path):
    """Load non-pickled numerical source coefficients."""
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def chart_carriers(chart, hidden, mask):
    """Split the existing B chart into C/P/Q without changing its projection."""
    h, mask = np.asarray(hidden, np.float32), np.asarray(mask, bool)
    r, g, rm = chart["hidden"], chart["gradients"], chart["mask"].astype(bool)
    if (
        h.ndim != 2
        or len(h) != len(mask)
        or mask[-1]
        or rm[-1]
        or not mask.any()
        or mask.sum() != rm.sum()
        or h.shape[1] != r.shape[1]
        or not np.isfinite(h).all()
    ):
        raise ValueError("Matched query coordinates and finite hidden states required")
    re, xe = ~rm, ~mask
    re[-1] = xe[-1] = False
    if not re.any() or not xe.any():
        raise ValueError("Common context cannot be empty")
    c = pool_pairings(r[re], h[xe], g[:, re], "semantic_order")
    c -= pool_pairings(r[re], r[re], g[:, re], "semantic_order")
    p = np.einsum("apd,pd->a", g[:, rm], h[mask] - r[rm])
    q = g[:, -1] @ (h[-1] - r[-1])
    return {"origin": np.asarray(chart["projection_origin"]), "C": c, "P": p, "Q": q}


def context_scores(charts, hidden, mask, anchor):
    """Return the unchanged full reader and source-fixed query-context ablations."""
    values = [chart_carriers(c, hidden, mask) for c in charts]
    parts = {k: float(np.mean([v[k].sum() for v in values])) for k in values[0]}
    origin = anchor + parts["origin"]
    return {
        "context_all": origin + parts["C"] + parts["P"] + parts["Q"],
        "context_C": origin + parts["C"],
        "context_PQ": origin + parts["P"] + parts["Q"],
        "context_origin": origin,
        **{"B_" + k: v for k, v in parts.items()},
    }


class QueryPrograms:
    """Use only a portable, source-frozen bundle, not current teacher states."""

    def __init__(self, bundle, mode, device="mps"):
        """Load selected dependencies and keep token mode free of neural imports."""
        from tokenizers import Tokenizer

        if mode not in ("token", "context"):
            raise ValueError("Unknown cold mode")
        self.root, self.mode, self.calls = Path(bundle).resolve(), mode, 0
        self.manifest = read(self.root / "manifest.json")
        self.verified = self.manifest["dependencies"][mode]
        for n in self.verified:
            if digest(checked_path(self.root, n)) != self.manifest["files"][n]:
                raise ValueError("Frozen dependency changed: " + n)
        self.spec = read(self.root / "queries.json")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        if mode == "token":
            self.table = np.load(self.root / "embedding_table.npy", mmap_mode="r")
            self.charts = {
                q: [arrays(self.root / f"token_{q}_{i}.npz") for i in range(2)]
                for q in self.spec
            }
        else:
            import torch
            from safetensors.torch import load_file
            from transformers import Qwen2Config, Qwen2Model

            from .context_program import CroppedContext

            torch.set_num_threads(2)
            cfg = Qwen2Config.from_dict(read(self.root / "config.json"))
            cfg._attn_implementation = "eager"
            m = Qwen2Model(cfg).float()
            m.load_state_dict(load_file(self.root / "context.safetensors"), strict=True)
            self.context = CroppedContext(m.to(device), cuts=(3,), max_tokens=1024)
            self.charts = {
                q: [arrays(self.root / f"context_{q}_{i}.npz") for i in range(2)]
                for q in self.spec
            }
            self.models = {q: arrays(self.root / f"state_{q}.npz") for q in self.spec}
            self.compiled = {
                q: arrays(self.root / f"compiled_{q}.npz") for q in self.spec
            }
            z = arrays(self.root / "reader.npz")
            self.readers = {
                q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
                for q in self.spec
            }

    def score(self, texts, query):
        """Compute from raw text; auxiliary arrays are generated here, never supplied."""
        q = self.spec[query]
        ids, mask = encode_texts(
            self.tokenizer, texts, q["recipe"], q["question_tokens"], maximum=1024
        )
        base = {"tokens": len(ids), "query": query}
        if self.mode == "token":
            return {
                **base,
                **{
                    "token_" + mode: score_token_ids(
                        ids, mask, self.table, self.charts[query], q["anchor"], mode
                    )["score"]
                    for mode in ("uniform", "semantic_order")
                },
            }, {"ids": np.asarray(ids), "mask": mask}
        h = self.context.capture(ids)[3]
        self.calls += 1
        early = pack_state(h[mask], h[-1])
        upper = predict_linear(self.models[query], early)
        reader = self.readers[query]
        p, final = unpack_state(upper, reader.p_ref.shape)
        state_value = reader.parts(p, final)["score"]
        compiled = score_affine(self.compiled[query], early)["score"]
        if abs(state_value - compiled) > 1e-8:
            raise ValueError("Reconstruction and direct composition disagree")
        source = self.models[query]["mean_x"]
        coef = self.compiled[query]["coefficients"].sum(1)
        width = h.shape[1]
        delta = (early - source) * coef
        return {
            **base,
            "A_state": state_value,
            "A_compiled": compiled,
            "A_P": float(delta[:-width].sum()),
            "A_Q": float(delta[-width:].sum()),
            "A_origin": score_affine(self.compiled[query], source)["score"],
            **context_scores(self.charts[query], h, mask, q["anchor"]),
        }, {"hidden": h, "upper_estimated": upper, "ids": np.asarray(ids), "mask": mask}


def cold_run(bundle, input_path, output, mode, device="mps", max_seconds=600):
    """Execute bounded raw inputs; prediction metadata includes actual dependencies."""
    from threadpoolctl import threadpool_limits

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start, predictions = time.monotonic(), []
    with threadpool_limits(limits=2):
        runtime = QueryPrograms(bundle, mode, device)
        for row in read(input_path):
            if set(row) != {"id", "texts", "query"}:
                raise ValueError("Cold API accepts only id/texts/query")
            if time.monotonic() - start > max_seconds:
                raise TimeoutError(
                    "Bounded cold packet expired; do not tune/restart implicitly"
                )
            values, state = runtime.score(row["texts"], row["query"])
            name = f"input_{len(predictions):03d}.npz"
            np.savez_compressed(output / name, **state)
            predictions.append({"id": row["id"], **values, "arrays": name})
    deps = {
        "mode": mode,
        "rows": len(predictions),
        "seconds": time.monotonic() - start,
        "transformer_forward_calls": runtime.calls,
        "torch_imported": "torch" in sys.modules,
        "transformers_imported": "transformers" in sys.modules,
        "retained_layers": 0 if mode == "token" else 4,
        "LM_head": False,
        "teacher_state_inputs": False,
        "verified_files": runtime.verified,
    }
    if mode == "token" and (deps["torch_imported"] or deps["transformers_imported"]):
        raise ValueError("Token-only process imported a neural runtime")
    (output / "predictions.json").write_text(
        json.dumps(predictions, indent=2, allow_nan=False)
    )
    (output / "dependencies.json").write_text(json.dumps(deps, indent=2))
    files = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "freeze.json").write_text(json.dumps(files, indent=2))
    print(json.dumps({"status": "COMPLETE", **deps}), flush=True)
