#!/usr/bin/env python3
"""Learn one fixed text-coordinate action and test it by native head intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_operator import (
    fit_action,
    predict_action,
)
from grain_research.vector_mechanisms.direct_traits import (
    SYSTEM,
    candidates,
    question,
    render,
)
from grain_research.vector_mechanisms.reverse_trace import equal_norm_random
from scripts.run_grain_conditional_operator import QUERIES, identity, inputs
from scripts.run_grain_persona_generation_trace import (
    append_jsonl,
    load_observer,
    read_jsonl,
    sha,
    write,
)
from scripts.run_grain_persona_reverse_trace import read_score


def context(observer, text, query):
    """Use the full article and identify the token-identical reader suffix."""
    user, _ = render([text], query)
    ids = observer.prompt_ids(SYSTEM, user)
    start = observer.content_indices(SYSTEM, user, question(query))[0]
    return ids, list(range(start, len(ids)))


def cache_path(out, rid, query):
    """Address a text/query record without filename-sensitive material IDs."""
    key = hashlib.sha256(f"{rid}|{query}".encode()).hexdigest()
    return out / "states" / f"{key}.npz"


def capture(observer, ids, pos, alphabet):
    """Capture one head and score in the same frozen-model forward."""
    dim = observer.model.config.hidden_size // observer.model.config.num_attention_heads
    box = {}

    def hook(_module, xs):
        box["head"] = xs[0][0, pos, :dim].detach().cpu().numpy().copy()

    handle = observer.model.model.layers[12].self_attn.o_proj.register_forward_pre_hook(
        hook
    )
    try:
        score = read_score(observer, ids, alphabet)
    finally:
        handle.remove()
    return box["head"], score


def intervene(observer, ids, pos, alphabet, delta):
    """Add a proposed change only to L12.H0 suffix rows, with safe hook cleanup."""
    import torch

    dim = observer.model.config.hidden_size // observer.model.config.num_attention_heads
    if delta.shape != (len(pos), dim) or not np.isfinite(delta).all():
        raise ValueError("Invalid action shape or value")

    def hook(_module, xs):
        x = xs[0].clone()
        x[0, pos, :dim] += torch.as_tensor(delta, device=x.device, dtype=x.dtype)
        return (x,)

    handle = observer.model.model.layers[12].self_attn.o_proj.register_forward_pre_hook(
        hook
    )
    try:
        return read_score(observer, ids, alphabet)
    finally:
        handle.remove()


def run(args):
    """Resume complete cache records; select nothing on application effects."""
    out = args.output
    identity(out)
    (out / "states").mkdir(exist_ok=True)
    manifest = {
        "script_sha256": sha(Path(__file__)),
        "rank": 2,
        "alpha": 1.0,
        "site": "L12.H0",
    }
    mp = out / "native_manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Native run changed")
    write(mp, manifest)
    rows, pairs = inputs()
    observer = load_observer(max_tokens=1024, layer_count=2)
    write(out / "model_manifest.json", observer.manifest())
    alphabets = {q: candidates(q, observer.tokenizer) for q in QUERIES}
    start = time.monotonic()
    cached = 0
    for row in rows:
        if row["axis"] == "none":
            continue
        for qi, q in enumerate(QUERIES):
            path = cache_path(out, row["text_id"], q)
            if path.exists():
                cached += 1
                continue
            if time.monotonic() - start > args.max_seconds:
                write(
                    out / "native_status.json",
                    {"status": "PARTIAL_CACHE", "cached": cached},
                )
                return
            ids, pos = context(observer, row["text"], q)
            head, score = capture(observer, ids, pos, alphabets[q])
            if abs(score["score"] - row["scores"][qi]) > 1e-6:
                raise ValueError("Original reader score did not replay")
            temp = path.with_suffix(".tmp.npz")
            np.savez(
                temp,
                head=head,
                suffix=np.array([ids[i] for i in pos]),
                score=score["score"],
                mass=score["mass"],
                token_count=len(ids),
            )
            temp.replace(path)
            cached += 1
            if cached % 24 == 0:
                print("cached", cached, "forwards", observer.calls, flush=True)
    dx, dh, ds = [], [], []
    shapes = None
    for p in pairs:
        dx.append(np.subtract(p["high"]["g"], p["low"]["g"]))
        states = []
        pair_shapes = []
        for q in QUERIES:
            with (
                np.load(cache_path(out, p["high"]["text_id"], q)) as h,
                np.load(cache_path(out, p["low"]["text_id"], q)) as l,
            ):
                np.testing.assert_array_equal(h["suffix"], l["suffix"])
                states.append((h["head"].astype(float) - l["head"]).ravel())
                pair_shapes.append(h["head"].shape)
        if shapes is not None and shapes != pair_shapes:
            raise ValueError("Query suffix shapes changed across pairs")
        shapes = pair_shapes
        dh.append(np.concatenate(states))
        ds.append(np.subtract(p["high"]["scores"], p["low"]["scores"]))
    dx, dh, ds = np.array(dx), np.array(dh), np.array(ds)
    source = np.array([p["split"] == "source" for p in pairs])
    model = fit_action(dx[source], dh[source], rank=2)
    direct = fit_action(dx[source], ds[source])
    predicted = predict_action(model, dx)
    score_pred = predict_action(direct, dx)
    np.savez(out / "action_model.npz", **model)
    np.savez(
        out / "action_arrays.npz",
        dx=dx,
        dh=dh,
        ds=ds,
        source=source,
        predicted=predicted,
        direct=score_pred,
    )
    write(out / "pair_order.json", [p["pair"] for p in pairs])
    records = read_jsonl(out / "native_effects.jsonl")
    done = {r["key"] for r in records}
    offset = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    for i, p in enumerate(pairs):
        if p["split"] == "source":
            continue
        for qi, q in enumerate(QUERIES):
            ids, pos = context(observer, p["low"]["text"], q)
            sl = slice(offset[qi], offset[qi + 1])
            learned = predicted[i, sl].reshape(shapes[qi])
            donor = dh[i, sl].reshape(shapes[qi])
            random = equal_norm_random(
                np.zeros_like(learned), learned, f"{p['pair']}|{q}"
            )
            with np.load(cache_path(out, p["low"]["text_id"], q)) as cached_low:
                mass_before = float(cached_low["mass"])
            for kind, delta in (
                ("compiled", learned),
                ("donor", donor),
                ("random", random),
                ("self", np.zeros_like(learned)),
            ):
                key = f"{p['pair']}|{q}|{kind}"
                if key in done:
                    continue
                if time.monotonic() - start > args.max_seconds:
                    write(
                        out / "native_status.json",
                        {"status": "PARTIAL_EFFECTS", "rows": len(records)},
                    )
                    return
                result = intervene(observer, ids, pos, alphabets[q], delta)
                low = p["low"]["scores"][qi]
                r = {
                    k: p[k]
                    for k in ("pair", "axis", "family", "situation", "seed", "split")
                }
                r.update(
                    key=key,
                    query=q,
                    kind=kind,
                    low=low,
                    high=p["high"]["scores"][qi],
                    score=result["score"],
                    effect=result["score"] - low,
                    contrast=float(ds[i, qi]),
                    direct=float(score_pred[i, qi]),
                    mass=result["mass"],
                    mass_before=mass_before,
                    action_norm=float(np.linalg.norm(delta)),
                )
                append_jsonl(out / "native_effects.jsonl", r)
                done.add(key)
                records.append(r)
        print("application", p["pair"], "rows", len(records), flush=True)
    write(
        out / "native_status.json",
        {
            "status": "COMPLETE",
            "rows": len(records),
            "cached_states": cached,
            "forwards_this_invocation": observer.calls,
            "seconds_this_invocation": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-seconds", type=int, default=1200)
    run(ap.parse_args())
