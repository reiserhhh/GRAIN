#!/usr/bin/env python3
"""S7: separate token-permutation response from nonlinear pooled-state response."""

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


def permutation_positions(groups, selected, seed):
    """Sample a uniform within-comment permutation, independent of trait query."""
    rng = np.random.default_rng(seed)
    source, destination = [], []
    for j in selected:
        destination.extend(groups[j])
        source.extend(rng.permutation(groups[j]).tolist())
    if len(set(destination)) != len(destination) or set(source) != set(destination):
        raise ValueError("Invalid disjoint permutation")
    return destination, source


def apply_permutation(h, groups, selected, seed):
    """Move complete state rows without changing their unordered distribution."""
    destination, source = permutation_positions(groups, selected, seed)
    output = h.clone()
    output[destination] = h[source]
    return output


def orbit_split(original, collapsed, shuffled):
    """Exact empirical accounting; shuffled mean is a finite group-average estimate."""
    orbit = float(np.mean(shuffled))
    order = original - orbit
    dispersion = orbit - collapsed
    return {
        "orbit_mean": orbit,
        "order_component": order,
        "dispersion_component": dispersion,
        "collapse_component": original - collapsed,
        "identity_error": abs(order + dispersion - (original - collapsed)),
        "mc_score_sd": float(np.std(shuffled, ddof=1)),
    }


def score(observer, ids, alphabet, groups, selected, seed):
    """Score a fixed-length body-state permutation with all other states unchanged."""
    check = {}

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        changed = apply_permutation(h[0], groups, selected, seed)
        check["mean_error"] = max(
            [
                float(
                    (changed[groups[j]].mean(0) - h[0, groups[j]].mean(0))
                    .abs()
                    .max()
                    .cpu()
                )
                for j in selected
            ],
            default=0.0,
        )
        return (
            (changed[None], *output[1:]) if isinstance(output, tuple) else changed[None]
        )

    handle = observer.model.model.layers[16].register_forward_hook(boundary)
    try:
        tokens = torch.tensor([ids], device=next(observer.model.parameters()).device)
        with torch.no_grad():
            logits = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
            value = float(conditional_torch(logits, alphabet).cpu())
        observer.calls += 1
        if not np.isfinite(value):
            raise ValueError("Nonfinite permutation score")
        return value, check["mean_error"]
    finally:
        handle.remove()


def summarize(out):
    """Report the declared empirical split without claiming full-group convergence."""
    rows = [
        r
        for p in sorted((out / "private_packets").glob("panel_*.json"))
        for r in read(p)["rows"]
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_permutations.csv", index=False)
    splits = []
    for (author, query, edit), g in frame.groupby(["author_id", "query", "edit"]):
        splits.append(
            {
                "author_id": author,
                "query": query,
                "edit": edit,
                **orbit_split(g.original.iloc[0], g.collapsed.iloc[0], g.shuffled),
            }
        )
    table = pd.DataFrame(splits)
    table.to_csv(out / "private_components.csv", index=False)
    metrics = []
    for q, g in table.groupby("query"):
        individual = frame[frame["query"] == q]
        item = {
            "query": q,
            "authors": g.author_id.nunique(),
            "paths": len(g),
            "permutations": len(individual),
            "mean_mc_score_sd": float(g.mc_score_sd.mean()),
            "max_identity_error": float(g.identity_error.max()),
            "shuffle_effect_rms": float(
                np.sqrt(np.mean((individual.shuffled - individual.original) ** 2))
            ),
            "order_dispersion_r": corr(g.order_component, g.dispersion_component),
        }
        for name in ("order_component", "dispersion_component", "collapse_component"):
            item[name + "_rms"] = float(np.sqrt(np.mean(g[name] ** 2)))
        metrics.append(item)
    pairs = []
    for component in ("order_component", "dispersion_component"):
        pivot = table.pivot(
            index=["author_id", "edit"], columns="query", values=component
        )
        pivot = pivot - pivot.groupby(level="edit").transform("mean")
        pairs.append(
            {
                "component": component,
                **cluster_r_interval(
                    pivot.neuroticism,
                    pivot.confirmation_seeking,
                    pivot.index.get_level_values("author_id"),
                    draws=500,
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "shared_components.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "metrics": metrics,
            "shared_N_confirmation": pairs,
            "rows_sha256": sha(out / "private_permutations.csv"),
            "components_sha256": sha(out / "private_components.csv"),
            "manifest_sha256": sha(out / "manifest.json"),
            "questionnaire_labels_read": False,
            "group_average": "three fixed uniform draws, not exact enumeration",
            "max_mean_error": float(frame.mean_error.max()),
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


def main():
    """Run one bounded group-action panel, without new author selection or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--mean-null", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private outputs must remain ignored")
    _, records = load_panel(args.panel)
    previous = read(args.mean_null / "summary.json")
    if (
        previous["status"] != "COMPLETE"
        or sha(args.mean_null / "private_responses.csv") != previous["responses_sha256"]
    ):
        raise ValueError("Frozen S6 endpoints required")
    fixed = pd.read_csv(
        args.mean_null / "private_responses.csv", dtype={"author_id": str}
    ).set_index(["author_id", "query", "edit"])
    selected = [
        (m, p) for m, p, _ in records if m["view"] == 0 and m["panel_index"] < 8
    ]
    if len(selected) != 8 or not fixed.index.is_unique:
        raise ValueError("Expected eight fixed authors and unique endpoints")
    manifest = {
        "protocol": "GRAIN_S7_PERMUTATION_ORBIT_V1",
        "script_sha256": sha(__file__),
        "source_sha256": sha(args.panel / "manifest.json"),
        "s6_sha256": sha(args.mean_null / "summary.json"),
        "model_sha256": sha(MODEL / "model.safetensors"),
        "device": args.device,
        "queries": list(QUERIES),
        "authors": 8,
        "permutations": 3,
        "seed_rule": "731 + panel_index*100 + edit_index*10 + repeat",
        "max_forwards": 168,
        "max_seconds": 600,
        "questionnaire_labels_read": False,
    }
    if out.exists():
        if read(out / "manifest.json") != manifest:
            raise ValueError("Changed frozen experiment")
    else:
        out.mkdir(parents=True)
        (out / "private_packets").mkdir()
        write(out / "manifest.json", manifest)
    state_file = out / "run_status.json"
    prior = read(state_file) if state_file.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(value):
        record = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "backwards": 0,
            "complete_authors": len(
                list((out / "private_packets").glob("panel_*.json"))
            ),
        }
        write(state_file, record)
        print(record, flush=True)

    try:
        for material, _ in selected:
            path = out / "private_packets" / f"panel_{material['panel_index']:02d}.json"
            if path.exists():
                if read(path)["author_id"] != material["author_id"]:
                    raise ValueError("Resumed author mismatch")
                continue
            if prior["seconds"] + time.monotonic() - start > 600:
                status("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                return
            rows = []
            for query in QUERIES:
                alphabet = native_alphabet(query, observer.tokenizer)
                user, spans = query_prompt(material["texts"], query)
                ids = observer.prompt_ids(SYSTEM, user)
                groups = token_groups(observer, user, spans, ids)
                original, _ = score(observer, ids, alphabet, groups, [], 0)
                for j, edit in enumerate(("e1", "e2")):
                    endpoint = fixed.loc[(material["author_id"], query, edit)]
                    if abs(original - endpoint.full_score) > 1e-5:
                        raise ValueError("Origin changed since S6")
                    for repeat in range(3):
                        seed = 731 + material["panel_index"] * 100 + j * 10 + repeat
                        shuffled, error = score(
                            observer, ids, alphabet, groups, material[edit], seed
                        )
                        rows.append(
                            {
                                "author_id": material["author_id"],
                                "query": query,
                                "edit": edit,
                                "repeat": repeat,
                                "seed": seed,
                                "original": original,
                                "collapsed": float(endpoint.collapsed_score),
                                "shuffled": shuffled,
                                "mean_error": error,
                            }
                        )
            write(path, {"author_id": material["author_id"], "rows": rows})
            status("RUNNING")
            if prior["calls"] + observer.calls > 168:
                raise RuntimeError("Forward budget exceeded")
        summarize(out)
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
