#!/usr/bin/env python3
"""S12: collect body, query and within-comment KV-pairing source representations."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import candidates, tokenize_prompt
from grain_research.vector_mechanisms.query_tilt import (
    pairing_component,
    tilted_readout,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_attention_pairing import capture_head
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

REPRESENTATIONS = ("body_mean", "query_state", "kv_pairing")


def state_summaries(state, groups):
    """Keep equal-comment body means and the final query state in original units."""
    h = np.asarray(state, float)
    if h.ndim != 2 or not np.isfinite(h).all() or not groups:
        raise ValueError("Finite token matrix and nonempty comment groups required")
    used, means = set(), []
    for group in groups:
        ix = np.asarray(group)
        if (
            ix.ndim != 1
            or not len(ix)
            or not np.issubdtype(ix.dtype, np.integer)
            or ix.min() < 0
            or ix.max() >= len(h) - 1
            or len(set(ix)) != len(ix)
            or used.intersection(ix)
        ):
            raise ValueError("Require disjoint body positions excluding final query")
        used.update(ix)
        means.append(h[ix].mean(0))
    return np.mean(means, axis=0), h[-1].copy()


def pairing_vector(cache, groups):
    """Return all-head within-comment pairing, with native attention replay check."""
    q, k, v, native = (cache[n] for n in ("q", "k", "v", "native_head"))
    if len(q) % k.shape[1]:
        raise ValueError("Unsupported grouped-query attention")
    ratio, parts, errors = len(q) // k.shape[1], [], []
    for head, query in enumerate(q):
        actual, weights = tilted_readout(
            k[:, head // ratio], v[:, head // ratio], query
        )
        parts.append(pairing_component(weights, v[:, head // ratio], groups))
        errors.append(float(np.max(np.abs(actual - native[head]))))
    error = max(errors)
    if error > 1e-4:
        raise ValueError(f"Captured attention replay error: {error:.6g}")
    return np.concatenate(parts), error


def capture_representations(observer, ids, alphabet, groups, block=16, layer=17):
    """Capture three paired representations in one unchanged frozen forward."""
    saved = {}

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        body, query = state_summaries(h[0].detach().float().cpu().numpy(), groups)
        saved["body_mean"], saved["query_state"] = body, query

    handle = observer.model.model.layers[block].register_forward_hook(boundary)
    try:
        score, cache = capture_head(observer, ids, alphabet, layer=layer)
    finally:
        handle.remove()
    saved["kv_pairing"], error = pairing_vector(cache, groups)
    dims = {value.shape for value in saved.values()}
    if len(dims) != 1 or len(saved) != 3:
        raise ValueError("Representation dimensions do not match")
    return score, saved, error


def inputs(source):
    """Validate S9 membership, material and cached teacher scores without reading Y."""
    manifest = read(source / "manifest.json")
    summary = read(source / "summary.json")
    if (
        summary["status"] != "COMPLETE"
        or sha(source / "private_material.json") != manifest["material_sha256"]
        or sha(source / "private_components_before_outcomes.csv")
        != summary["components_sha256"]
    ):
        raise ValueError("S9 source changed or incomplete")
    material = [
        p for p in read(source / "private_material.json") if p["query"] == "neuroticism"
    ]
    if len(material) != 213 or len({p["author_id"] for p in material}) != 213:
        raise ValueError("Require full213 N authors")
    return material


def prepare(args):
    """Freeze a source-representation collection; do not overwrite prior work."""
    source, out = args.source.resolve(), args.output_dir
    material = inputs(source)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S12_QUERY_REPRESENTATION_V1",
            "source": str(source),
            "source_material_sha256": sha(source / "private_material.json"),
            "source_components_sha256": sha(
                source / "private_components_before_outcomes.csv"
            ),
            "source_packets": {
                f"author_{p['index']:03d}.json": sha(
                    source / "private_packets" / f"author_{p['index']:03d}.json"
                )
                for p in material
            },
            "code_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/run_grain_attention_pairing.py",
                    ROOT / "grain_research/vector_mechanisms/query_tilt.py",
                )
            },
            "model_sha256": sha(MODEL / "model.safetensors"),
            "authors": 213,
            "views": 639,
            "max_forwards": 639,
            "max_seconds": 2700,
            "representations": REPRESENTATIONS,
            "block": 16,
            "layer": 17,
            "observer": {"device": args.device, "dtype": "float32", "padding": None},
            "labels_read": False,
        },
    )
    print({"status": "PREPARED", "authors": 213, "views": 639}, flush=True)


def verify(out, device):
    """Bind collection and resumption to its exact material, numerical and code state."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        sha(source / "private_material.json") != m["source_material_sha256"]
        or sha(source / "private_components_before_outcomes.csv")
        != m["source_components_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
        or any(sha(ROOT / p) != digest for p, digest in m["code_hashes"].items())
        or m["observer"]["device"] != device
        or any(
            sha(source / "private_packets" / p) != digest
            for p, digest in m["source_packets"].items()
        )
    ):
        raise ValueError("Frozen collection inputs changed")
    return m, inputs(source)


def validate_packet(path, author):
    """Check an existing complete author packet and its three-view vector artifact."""
    p = read(path)
    if p["author_id"] != author["author_id"] or len(p["views"]) != 3:
        raise ValueError("Resumed author/view mismatch")
    if sha(path.with_suffix(".npz")) != p["features_sha256"]:
        raise ValueError("Saved representation artifact changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        if any(
            z[name].shape != (3, 1536) or not np.isfinite(z[name]).all()
            for name in REPRESENTATIONS
        ):
            raise ValueError("Incomplete author representation")
    return p


def run(args):
    """Collect complete authors, checkpointing resource boundaries without Y inspection."""
    out = args.output_dir
    manifest, material = verify(out, args.device)
    source = Path(manifest["source"])
    prior_path = out / "run_status.json"
    prior = read(prior_path) if prior_path.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(name):
        """Write aggregate liveness and consumed cost without author identifiers."""
        record = {
            "status": name,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "complete_authors": len(
                list((out / "private_packets").glob("author_*.json"))
            ),
            "labels_read": False,
        }
        write(out / "run_status.json", record)
        print(record, flush=True)

    try:
        alphabet = candidates("neuroticism", observer.tokenizer)
        for i, author in enumerate(material):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                validate_packet(path, author)
                continue
            if (
                i >= args.max_authors
                or prior["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            old = read(
                source / "private_packets" / f"author_{author['index']:03d}.json"
            )
            if old["author_id"] != author["author_id"]:
                raise ValueError("Source author mismatch")
            old_views = {p["view"]: p for p in old["views"]}
            rows, arrays = [], {name: [] for name in REPRESENTATIONS}
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], "neuroticism", observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Source prompt changed")
                groups = token_groups(observer, user, spans, ids)
                score, values, error = capture_representations(
                    observer, ids, alphabet, groups
                )
                replay_error = abs(score - old_views[view]["new_teacher"])
                if replay_error > 2e-5:
                    raise ValueError(f"S9 score replay changed: {replay_error:.6g}")
                for name in REPRESENTATIONS:
                    arrays[name].append(values[name])
                rows.append(
                    {
                        "view": view,
                        "score": score,
                        "s9_replay_error": replay_error,
                        "head_reconstruction_error": error,
                        "tokens": len(ids),
                        "comment_tokens": sum(map(len, groups)),
                    }
                )
            np.savez_compressed(
                path.with_suffix(".npz"),
                **{k: np.asarray(v) for k, v in arrays.items()},
            )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "views": rows,
                    "features_sha256": sha(path.with_suffix(".npz")),
                },
            )
            validate_packet(path, author)
            if (i + 1) % 8 == 0:
                status("RUNNING")
            if prior["calls"] + observer.calls > manifest["max_forwards"]:
                raise RuntimeError("Native forward budget exceeded")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    """Prepare or collect the fixed source representation packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path, default=BASE / "source_remainder_s9_r1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-authors", type=int, default=24)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE):
        raise ValueError("Keep private outputs in ignored results")
    {"prepare": prepare, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
