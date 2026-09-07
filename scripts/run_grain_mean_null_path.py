#!/usr/bin/env python3
"""S6: mean-preserving token-state paths through the frozen block16 decoder suffix."""

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
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import (
    cluster_r_interval,
    corr,
    load_panel,
)
from scripts.run_grain_multiquery_sensitivity import (
    QUERIES,
    conditional_torch,
    native_alphabet,
    query_prompt,
    token_groups,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def mean_null_direction(h, groups, selected):
    """Project selected token groups toward their own mean, preserving each sum."""
    if h.ndim != 2 or not selected or len(set(selected)) != len(selected):
        raise ValueError(
            "A token-state matrix and distinct selected comments are required"
        )
    direction = torch.zeros_like(h)
    seen = set()
    for comment in selected:
        positions = groups[comment]
        if not positions or seen.intersection(positions):
            raise ValueError("Nonempty disjoint token groups required")
        seen.update(positions)
        direction[positions] = h[positions].mean(0, keepdim=True) - h[positions]
    return direction


def simpson(values):
    """Integrate a directional derivative at three or five equally spaced nodes."""
    if set(values) == {0.0, 0.5, 1.0}:
        return (values[0.0] + 4 * values[0.5] + values[1.0]) / 6
    if set(values) == {0.0, 0.25, 0.5, 0.75, 1.0}:
        return (
            values[0.0]
            + 4 * values[0.25]
            + 2 * values[0.5]
            + 4 * values[0.75]
            + values[1.0]
        ) / 12
    raise ValueError("Expected complete three- or five-node Simpson rule")


def path_sample(observer, ids, alphabet, groups, edits, active, alpha):
    """Differentiate a fixed-length internal path; report both directions at origin."""
    saved = {}

    def boundary(_module, _inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        h = state.detach()
        directions = {
            name: mean_null_direction(h[0], groups, comments)
            for name, comments in edits.items()
        }
        leaf = (h + alpha * directions[active][None]).detach().requires_grad_(True)
        saved.update(leaf=leaf, directions=directions)
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    device = next(observer.model.parameters()).device
    tokens = torch.tensor([ids], device=device)
    handle = observer.model.model.layers[16].register_forward_hook(boundary)
    try:
        with torch.enable_grad():
            result = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            score = conditional_torch(result.logits[0, -1], alphabet)
            (gradient,) = torch.autograd.grad(score, saved["leaf"])
        observer.calls += 1
        gradient = gradient.detach()[0]
        directions = saved["directions"]
        slopes = {
            name: float((gradient * d).sum().cpu()) for name, d in directions.items()
        }
        mean_errors = {
            name: max(
                float(d[groups[j]].sum(0).abs().max().cpu()) / len(groups[j])
                for j in edits[name]
            )
            for name, d in directions.items()
        }
        norms = {name: float(d.norm().cpu()) for name, d in directions.items()}
        value = float(score.detach().cpu())
        if not np.isfinite([value, *slopes.values(), *norms.values()]).all():
            raise ValueError("Nonfinite score or path derivative")
        return {
            "score": value,
            "slopes": slopes,
            "mean_errors": mean_errors,
            "direction_norms": norms,
        }
    finally:
        handle.remove()


def summarize(out):
    """Report actual effects, integral completeness and matched-query relationships."""
    packets = [
        read(path) for path in sorted((out / "private_packets").glob("panel_*.json"))
    ]
    rows = [item for packet in packets for item in packet["responses"]]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_responses.csv", index=False)
    metrics = []
    for query, g in frame.groupby("query"):
        natural = g.natural_delete.to_numpy()
        metrics.append(
            {
                "query": query,
                "authors": g.author_id.nunique(),
                "paths": len(g),
                "effect_rms": float(np.sqrt(np.mean(g.effect**2))),
                "effect_mean_absolute": float(g.effect.abs().mean()),
                "local_prediction_r": corr(g.effect, g.local),
                "integral_prediction_r": corr(g.effect, g.integral),
                "local_rmse": float(np.sqrt(np.mean((g.effect - g.local) ** 2))),
                "integral_rmse": float(np.sqrt(np.mean((g.effect - g.integral) ** 2))),
                "refined_paths": int((g.nodes == 5).sum()),
                "remaining_large_gaps": int(
                    (
                        abs(g.effect - g.integral)
                        > np.maximum(0.001, 0.05 * abs(g.effect))
                    ).sum()
                ),
                "natural_delete_r": corr(g.effect, natural),
                "max_mean_preservation_error": float(g.mean_preservation_error.max()),
            }
        )
    pairs = []
    pivot = frame.pivot(index=["author_id", "edit"], columns="query", values="effect")
    centered = pivot - pivot.groupby(level="edit").transform("mean")
    for j, q in enumerate(QUERIES):
        for q2 in QUERIES[j + 1 :]:
            pairs.append(
                {
                    "left": q,
                    "right": q2,
                    **cluster_r_interval(
                        centered[q],
                        centered[q2],
                        centered.index.get_level_values("author_id"),
                        draws=500,
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "shared_effects.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "metrics": metrics,
            "paired_query_effects": pairs,
            "responses_sha256": sha(out / "private_responses.csv"),
            "manifest_sha256": sha(out / "manifest.json"),
            "questionnaire_labels_read": False,
            "scope": "internal state intervention; neither natural text deletion nor personality-label change",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


def main():
    """Acquire eight complete source-author units with bounded adaptive quadrature."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private results must remain ignored")
    source_manifest, records = load_panel(args.panel)
    selected = [
        (m, p) for m, p, _ in records if m["view"] == 0 and m["panel_index"] < 8
    ]
    if len(selected) != 8:
        raise ValueError("Require the fixed eight S1 author units")
    manifest = {
        "protocol": "GRAIN_S6_MEAN_NULL_TOKEN_PATH_V1",
        "script_sha256": sha(__file__),
        "source_manifest_sha256": sha(args.panel / "manifest.json"),
        "material_sha256": source_manifest["material_sha256"],
        "model_sha256": sha(MODEL / "model.safetensors"),
        "device": args.device,
        "dtype": "float32",
        "queries": list(QUERIES),
        "authors": 8,
        "views_per_author": 1,
        "max_calls": 216,
        "max_seconds": 1800,
        "displacement": "within-comment token state -> its own mean at block16",
        "quadrature": "Simpson 3; Simpson 5 if gap > max(.001,.05*abs(effect))",
        "questionnaire_labels_read": False,
    }
    if out.exists():
        if read(out / "manifest.json") != manifest:
            raise ValueError("Do not resume with changed code, model, or inputs")
    else:
        out.mkdir(parents=True)
        (out / "private_packets").mkdir()
        write(out / "manifest.json", manifest)
    status_path = out / "run_status.json"
    prior = read(status_path) if status_path.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(state):
        payload = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "backwards": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "complete_authors": len(
                list((out / "private_packets").glob("panel_*.json"))
            ),
        }
        write(status_path, payload)
        print(payload, flush=True)

    try:
        for material, packet in selected:
            dest = out / "private_packets" / f"panel_{material['panel_index']:02d}.json"
            if dest.exists():
                if read(dest)["author_id"] != material["author_id"]:
                    raise ValueError("Incorrect resumed author")
                continue
            if prior["seconds"] + time.monotonic() - start >= 1800:
                status("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                return
            rows = []
            edits = {name: material[name] for name in ("e1", "e2")}
            for query in QUERIES:
                alphabet = native_alphabet(query, observer.tokenizer)
                user, spans = query_prompt(material["texts"], query)
                ids = observer.prompt_ids(SYSTEM, user)
                groups = token_groups(observer, user, spans, ids)
                origin = path_sample(observer, ids, alphabet, groups, edits, "e1", 0.0)
                cached = packet["queries"][query]["readouts"]
                if abs(origin["score"] - cached["full"]["conditional_score"]) > 1e-5:
                    raise ValueError("Origin disagrees with frozen S1 teacher")
                for edit in edits:
                    nodes = {0.0: origin["slopes"][edit]}
                    samples = {}
                    for alpha in (0.5, 1.0):
                        sample = path_sample(
                            observer, ids, alphabet, groups, edits, edit, alpha
                        )
                        nodes[alpha], samples[alpha] = sample["slopes"][edit], sample
                    effect = samples[1.0]["score"] - origin["score"]
                    integral3 = simpson(nodes)
                    if abs(integral3 - effect) > max(0.001, 0.05 * abs(effect)):
                        for alpha in (0.25, 0.75):
                            sample = path_sample(
                                observer, ids, alphabet, groups, edits, edit, alpha
                            )
                            nodes[alpha] = sample["slopes"][edit]
                    rows.append(
                        {
                            "author_id": material["author_id"],
                            "panel_index": material["panel_index"],
                            "query": query,
                            "edit": edit,
                            "full_score": origin["score"],
                            "collapsed_score": samples[1.0]["score"],
                            "effect": effect,
                            "local": nodes[0.0],
                            "integral3": integral3,
                            "integral": simpson(nodes),
                            "nodes": len(nodes),
                            "mean_preservation_error": origin["mean_errors"][edit],
                            "direction_norm": origin["direction_norms"][edit],
                            "natural_delete": cached[edit]["conditional_score"]
                            - cached["full"]["conditional_score"],
                            "path_slopes": {str(k): v for k, v in nodes.items()},
                        }
                    )
            write(dest, {"author_id": material["author_id"], "responses": rows})
            status("RUNNING")
            if prior["calls"] + observer.calls > 216:
                raise RuntimeError("Frozen call budget exceeded")
            if args.device == "mps":
                torch.mps.empty_cache()
        summarize(out)
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
