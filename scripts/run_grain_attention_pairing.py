#!/usr/bin/env python3
"""S10: fixed-head KV-pairing intervention and covariance-based query integration."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import SYSTEM
from grain_research.vector_mechanisms.query_tilt import (
    integrated_query_response,
    pairing_component,
    tilted_readout,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import cluster_r_interval, load_panel
from scripts.run_grain_multiquery_sensitivity import (
    QUERIES,
    conditional_torch,
    native_alphabet,
    query_prompt,
    token_groups,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def rotate_half(x):
    """Apply Qwen's fixed rotary half-coordinate map in NumPy."""
    if x.shape[-1] % 2:
        raise ValueError("Rotary dimension must be even")
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def score_forward(observer, ids, alphabet):
    """One unchanged decoder forward; hooks define any declared intervention."""
    tokens = torch.tensor([ids], device=next(observer.model.parameters()).device)
    with torch.no_grad():
        logits = observer.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        ).logits[0, -1]
        score = float(conditional_torch(logits, alphabet).cpu())
    observer.calls += 1
    if not np.isfinite(score):
        raise ValueError("Nonfinite attention experiment score")
    return score


def capture_head(observer, ids, alphabet, layer=17):
    """Capture position-resolved KV, scaled final query, and actual head output."""
    attention = observer.model.model.layers[layer].self_attn
    if attention.sliding_window is not None:
        raise ValueError("This fixed experiment requires unwindowed attention")
    saved, handles = {}, []

    def attention_input(_module, _args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        saved["cos"] = cos[0].detach().float().cpu().numpy().copy()
        saved["sin"] = sin[0].detach().float().cpu().numpy().copy()
        mask = kwargs.get("attention_mask")
        if mask is not None and torch.any(mask[0, 0, -1] != 0):
            raise ValueError(
                "Last query has a masked key; compact full context required"
            )

    def projection(name):
        def capture(_module, _inputs, output):
            saved[name] = output[0].detach().float().cpu().numpy().copy()

        return capture

    def head_output(_module, args):
        saved["head_output"] = args[0][0, -1].detach().float().cpu().numpy().copy()

    try:
        handles.append(
            attention.register_forward_pre_hook(attention_input, with_kwargs=True)
        )
        for name in ("q", "k", "v"):
            handles.append(
                getattr(attention, name + "_proj").register_forward_hook(
                    projection(name)
                )
            )
        handles.append(attention.o_proj.register_forward_pre_hook(head_output))
        score = score_forward(observer, ids, alphabet)
    finally:
        for handle in handles:
            handle.remove()
    dim = attention.head_dim
    q = saved["q"].reshape(len(ids), -1, dim)
    k = saved["k"].reshape(len(ids), -1, dim)
    v = saved["v"].reshape(len(ids), -1, dim)
    cos, sin = saved["cos"][:, None], saved["sin"][:, None]
    q = (q * cos + rotate_half(q) * sin)[-1] * attention.scaling
    k = k * cos + rotate_half(k) * sin
    return score, {
        "q": q,
        "k": k,
        "v": v,
        "native_head": saved["head_output"].reshape(-1, dim),
    }


def patch_head_score(observer, ids, alphabet, donor, layer=17):
    """Replace only the final-position concatenated head vector before W_O."""
    projection = observer.model.model.layers[layer].self_attn.o_proj

    def replace(_module, args):
        state = args[0].clone()
        value = torch.as_tensor(
            donor.reshape(-1), device=state.device, dtype=state.dtype
        )
        if value.shape != state[0, -1].shape:
            raise ValueError("Head donor has the wrong dimension")
        state[0, -1] = value
        return (state, *args[1:])

    handle = projection.register_forward_pre_hook(replace)
    try:
        return score_forward(observer, ids, alphabet)
    finally:
        handle.remove()


def analyze_heads(cache, groups):
    """Resolve all heads without selecting useful heads on endpoint scores."""
    queries, keys, values, native = (cache[k] for k in ("q", "k", "v", "native_head"))
    if len(queries) % keys.shape[1]:
        raise ValueError("Unsupported grouped-query attention shape")
    repetitions = len(queries) // keys.shape[1]
    rows, pairing = [], []
    content = np.concatenate(groups)
    for head, query in enumerate(queries):
        k, v = keys[:, head // repetitions], values[:, head // repetitions]
        actual, weights = tilted_readout(k, v, query)
        coupling = pairing_component(weights, v, groups)
        mean = v.mean(0)
        target = actual - mean
        projected = k @ query
        linear = (projected - projected.mean()) @ v / len(v)
        estimate16 = integrated_query_response(
            k, v, np.zeros_like(query), query, order=16
        )
        estimate32 = integrated_query_response(
            k, v, np.zeros_like(query), query, order=32
        )
        energy = float(np.sum(target**2))
        error = float(np.max(np.abs(actual - native[head])))
        if error > 1e-4:
            raise ValueError(f"Native attention reconstruction changed: {error:.6g}")
        rows.append(
            {
                "head": head,
                "reconstruction_max_error": error,
                "comment_mass": float(weights[content].sum()),
                "pairing_norm": float(np.linalg.norm(coupling)),
                "head_norm": float(np.linalg.norm(actual)),
                "linear_mse_over_zero": float(np.sum((target - linear) ** 2) / energy)
                if energy
                else None,
                "integral16_mse_over_zero": float(
                    np.sum((target - estimate16) ** 2) / energy
                )
                if energy
                else None,
                "integral32_mse_over_zero": float(
                    np.sum((target - estimate32) ** 2) / energy
                )
                if energy
                else None,
                "integral_step_gap": float(np.linalg.norm(estimate16 - estimate32)),
            }
        )
        pairing.append(coupling)
    return rows, native - np.array(pairing)


def summarize(out):
    """Report all head diagnostics and query effects, keeping author clusters intact."""
    packets = [read(p) for p in sorted((out / "private_packets").glob("panel_*.json"))]
    if len(packets) != 8:
        raise ValueError("Require the complete eight-author mechanism panel")
    effects = pd.DataFrame([r for p in packets for r in p["effects"]])
    heads = pd.DataFrame([r for p in packets for r in p["heads"]])
    effects.to_csv(out / "private_effects.csv", index=False)
    heads.to_csv(out / "private_head_diagnostics.csv", index=False)
    rows = []
    for query, g in effects.groupby("query"):
        hg = heads[heads["query"] == query]
        rows.append(
            {
                "query": query,
                "authors": len(g),
                "effect_rms": float(np.sqrt(np.mean(g.effect**2))),
                "effect_mean_abs": float(g.effect.abs().mean()),
                "self_max_error": float(g.self_error.max()),
                "reconstruction_max_error": float(hg.reconstruction_max_error.max()),
                "median_comment_mass": float(hg.comment_mass.median()),
                "median_pairing_norm": float(hg.pairing_norm.median()),
                "median_linear_mse_over_zero": float(hg.linear_mse_over_zero.median()),
                "median_integral32_mse_over_zero": float(
                    hg.integral32_mse_over_zero.median()
                ),
                "max_integral32_mse_over_zero": float(
                    hg.integral32_mse_over_zero.max()
                ),
            }
        )
    pivot = effects.pivot(index="author_id", columns="query", values="effect")
    shared = cluster_r_interval(
        pivot.neuroticism, pivot.confirmation_seeking, pivot.index, draws=500
    )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha(out / "manifest.json"),
            "effects_sha256": sha(out / "private_effects.csv"),
            "metrics": rows,
            "shared_N_confirmation": shared,
            "questionnaire_labels_read": False,
            "scope": "fixed layer17 final-query head intervention, not a full model/personality decomposition",
        },
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print({"shared_N_confirmation": shared}, flush=True)


def main():
    """Run72 fixed forwards, no gradients; preserve complete author checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private results must remain in ignored storage")
    _, records = load_panel(args.panel)
    selected = [m for m, _, _ in records if m["view"] == 0 and m["panel_index"] < 8]
    if len(selected) != 8:
        raise ValueError("Require the fixed S1 panel")
    manifest = {
        "protocol": "GRAIN_S10_ATTENTION_PAIRING_V1",
        "script_sha256": sha(__file__),
        "algebra_sha256": sha(ROOT / "grain_research/vector_mechanisms/query_tilt.py"),
        "source_sha256": sha(args.panel / "manifest.json"),
        "model_sha256": sha(MODEL / "model.safetensors"),
        "queries": list(QUERIES),
        "authors": 8,
        "layer": 17,
        "max_forwards": 72,
        "max_seconds": 900,
        "labels_read": False,
    }
    if out.exists():
        if read(out / "manifest.json") != manifest:
            raise ValueError("Frozen attention pairing experiment changed")
    else:
        out.mkdir(parents=True)
        (out / "private_packets").mkdir()
        (out / "private_kv").mkdir()
        write(out / "manifest.json", manifest)
    state = out / "run_status.json"
    prior = read(state) if state.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(value):
        row = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "complete_authors": len(
                list((out / "private_packets").glob("panel_*.json"))
            ),
            "labels_read": False,
        }
        write(state, row)
        print(row, flush=True)

    try:
        for material in selected:
            index = material["panel_index"]
            path = out / "private_packets" / f"panel_{index:02d}.json"
            if path.exists():
                if read(path)["author_id"] != material["author_id"]:
                    raise ValueError("Resumed author changed")
                continue
            if prior["seconds"] + time.monotonic() - start >= 900:
                status("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                return
            effects, head_rows = [], []
            for query in QUERIES:
                alphabet = native_alphabet(query, observer.tokenizer)
                user, spans = query_prompt(material["texts"], query)
                ids = observer.prompt_ids(SYSTEM, user)
                groups = token_groups(observer, user, spans, ids)
                origin, cache = capture_head(observer, ids, alphabet)
                rows, donor = analyze_heads(cache, groups)
                self_score = patch_head_score(
                    observer, ids, alphabet, cache["native_head"]
                )
                if abs(origin - self_score) > 1e-5:
                    raise ValueError("Native-head copy changed the score")
                intervened = patch_head_score(observer, ids, alphabet, donor)
                np.savez_compressed(
                    out / "private_kv" / f"panel_{index:02d}_{query}.npz", **cache
                )
                effects.append(
                    {
                        "author_id": material["author_id"],
                        "query": query,
                        "origin": origin,
                        "intervened": intervened,
                        "effect": intervened - origin,
                        "self_error": abs(self_score - origin),
                    }
                )
                head_rows.extend(
                    {"author_id": material["author_id"], "query": query, **r}
                    for r in rows
                )
            write(
                path,
                {
                    "author_id": material["author_id"],
                    "effects": effects,
                    "heads": head_rows,
                },
            )
            status("RUNNING")
            if prior["calls"] + observer.calls > 72:
                raise RuntimeError("Forward budget exceeded")
        summarize(out)
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
