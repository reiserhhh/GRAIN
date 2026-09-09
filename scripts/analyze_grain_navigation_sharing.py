#!/usr/bin/env python3
"""NAV-06: fixed-site, same-event cross-query endpoint exchange with native checks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.route_exchange import (
    endpoint_delta_patch,
    linear_cka,
    partition_routes,
)
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_upstream_routes import collector_guard, plan_pairs


def run(reference, upstream, out, seconds=1200):
    """Transport event deltas, retaining own-query, shared and source-specific controls."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver
    from grain_research.vector_mechanisms.upstream_routes import (
        capture_differentiable_prefix,
    )

    pairs, design = plan_pairs(reference)
    pairs = [p for p in pairs if p["split"] == "held_scene"]
    keyed = {(p["query"], p["scene_id"], p["axis"]): p for p in pairs}
    selections = read_json(upstream / "selected_routes.json")
    if (
        digest(upstream / "selected_routes.json")
        != read_json(upstream / "route_freeze.json")["sha256"]
    ):
        raise ValueError("Construction-frozen routes changed")
    source_paths = {
        "route_freeze": upstream / "route_freeze.json",
        "upstream_manifest": upstream / "manifest.json",
        "reference_manifest": reference / "manifest.json",
        "reader": reference / "real_reference.npz",
        "design": reference / "design.json",
    }
    codes = [
        "scripts/analyze_grain_navigation_sharing.py",
        "grain_research/vector_mechanisms/route_exchange.py",
        "grain_research/vector_mechanisms/upstream_routes.py",
    ]
    expected = {
        "protocol": "NAV06_ENDPOINT_EXCHANGE_V1",
        "source_sha256": {k: digest(v) for k, v in source_paths.items()},
        "code_sha256": {k: digest(ROOT / k) for k in codes},
        "selection": "NAV02 construction scenes only; no re-ranking",
        "coordinates": "common last input slot only; never align unequal P rows",
        "human_labels_used": False,
        "held_scene_ids": ["s09", "s10", "s11"],
        "group_scope": "synthetic event conditions only; GRAIN real-author groups remain separate",
    }
    if out.exists():
        if read_json(out / "manifest.json") != expected:
            raise ValueError("Frozen exchange protocol changed")
        if read_json(out / "run_status.json")["status"] == "COMPLETE":
            print(read_json(out / "summary.json"), flush=True)
            return
    else:
        out.mkdir(parents=True)
        (out / "patches").mkdir()
        write_json(out / "manifest.json", expected)
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    prior = read_json(out / "run_status.json")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if (
        digest(MODEL / "model.safetensors")
        != read_json(reference / "manifest.json")["model_sha256"]
    ):
        raise ValueError("Source model changed")
    fs, start, rows = load_functions(source_paths["reader"]), time.monotonic(), []
    dim = observer.model.model.layers[0].self_attn.head_dim
    cache = {}

    def state(index):
        """Read only validated numeric caches from actual NAV02 forward runs."""
        if index not in cache:
            meta, z = checked_arrays(upstream / "cache" / f"state_{index:04d}_0.json")
            if meta["index"] != index:
                raise ValueError("State identity changed")
            cache[index] = {
                "score": meta["score"],
                "sites": {k[6:]: v for k, v in z.items() if k.startswith("sites_")},
            }
        return cache[index]

    def native(index):
        return read_json(reference / "packets" / f"unit_{index:04d}.json")["native"][
            "conditional_score"
        ]

    def status(name):
        value = {
            "status": name,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", value)
        return value

    try:
        for pair in pairs:
            q, r, d = pair["query"], state(pair["origin"]), state(pair["donor"])
            target = design[pair["origin"]]
            for source_q in QUERIES:
                sp = keyed[(source_q, pair["scene_id"], pair["axis"])]
                so, sd = state(sp["origin"]), state(sp["donor"])
                groups = (
                    {"own_all": selections[q]}
                    if source_q == q
                    else partition_routes(selections[q], selections[source_q])
                )
                for group_name, group in groups.items():
                    if not group:
                        continue
                    modes = (
                        ["own_delta"] if source_q == q else ["own_delta", "cross_delta"]
                    )
                    for mode in modes:
                        file = (
                            out
                            / "patches"
                            / f"{pair['index']:02d}_{source_q}_{group_name}_{mode}.json"
                        )
                        if file.exists():
                            rows.append(read_json(file))
                            continue
                        if time.monotonic() - start >= seconds:
                            raise TimeoutError
                        origin, donor = (r, d) if mode == "own_delta" else (so, sd)
                        patches = endpoint_delta_patch(r, origin, donor, group, dim)
                        result = capture_differentiable_prefix(
                            observer,
                            target["input_ids"],
                            target["mask"],
                            fs[q],
                            patches=patches,
                            gradients=False,
                            alphabet=target["alphabet"],
                        )
                        row = {
                            "target_query": q,
                            "source_query": source_q,
                            "scene_id": pair["scene_id"],
                            "axis": pair["axis"],
                            "group": group_name,
                            "mode": mode,
                            "route_count": len(group),
                            "function_effect": result["score"] - r["score"],
                            "native_effect": result["native"]["conditional_score"]
                            - native(pair["origin"]),
                            "function_natural_delta": d["score"] - r["score"],
                            "native_natural_delta": native(pair["donor"])
                            - native(pair["origin"]),
                        }
                        write_json(file, row)
                        rows.append(row)
            print(status("EXCHANGING"), flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "shared_specific_edges.csv", index=False)
        summaries = []
        for key, g in frame.groupby(["target_query", "source_query", "group", "mode"]):
            summaries.append(
                dict(
                    zip(
                        ["target_query", "source_query", "group", "mode"],
                        key,
                        strict=True,
                    )
                )
                | {
                    "n_events": len(g),
                    "n_scenes": g.scene_id.nunique(),
                    "mean_abs_native_effect": float(abs(g.native_effect).mean()),
                    "mean_abs_natural_effect": float(
                        abs(g.native_natural_delta).mean()
                    ),
                    "native_function_effect_r": quality(
                        g.native_effect, g.function_effect
                    )["r"],
                    "native_patch_natural_r": quality(
                        g.native_natural_delta, g.native_effect
                    )["r"],
                }
            )
        pd.DataFrame(summaries).to_csv(out / "exchange_summary.csv", index=False)
        comparisons = []
        for key, g in frame[frame.target_query != frame.source_query].groupby(
            ["target_query", "source_query", "group"]
        ):
            a = g[g["mode"] == "own_delta"].set_index(["scene_id", "axis"])
            b = (
                g[g["mode"] == "cross_delta"]
                .set_index(["scene_id", "axis"])
                .loc[a.index]
            )
            comparisons.append(
                dict(zip(["target_query", "source_query", "group"], key, strict=True))
                | {
                    "n_events": len(a),
                    "n_scenes": a.reset_index().scene_id.nunique(),
                    **{
                        "native_" + k: v
                        for k, v in quality(a.native_effect, b.native_effect).items()
                    },
                    "mean_abs_own": float(abs(a.native_effect).mean()),
                    "mean_abs_cross": float(abs(b.native_effect).mean()),
                    "sign_agreement": float(
                        np.mean(np.sign(a.native_effect) == np.sign(b.native_effect))
                    ),
                }
            )
        pd.DataFrame(comparisons).to_csv(out / "cross_vs_own.csv", index=False)
        kernels = []
        for qa in QUERIES:
            for qb in QUERIES:
                aa, bb, va, vb = [], [], [], []
                for scene in ("s09", "s10", "s11"):
                    for axis in ("actor", "opportunity"):
                        pa, pb = keyed[(qa, scene, axis)], keyed[(qb, scene, axis)]
                        aa.append(native(pa["donor"]) - native(pa["origin"]))
                        bb.append(native(pb["donor"]) - native(pb["origin"]))
                        common = partition_routes(selections[qa], selections[qb])[
                            "shared"
                        ]
                        for querypair, vectors in ((pa, va), (pb, vb)):
                            r, d = state(querypair["origin"]), state(querypair["donor"])
                            terms = []
                            for c in common:
                                sl = (
                                    slice(None)
                                    if c["head"] == -1
                                    else slice(c["head"] * dim, (c["head"] + 1) * dim)
                                )
                                terms.append(
                                    d["sites"][c["site"]][-1, sl]
                                    - r["sites"][c["site"]][-1, sl]
                                )
                            vectors.append(
                                np.concatenate(terms) if terms else np.zeros(1)
                            )
                ka = {(c["site"], c["head"]) for c in selections[qa]}
                kb = {(c["site"], c["head"]) for c in selections[qb]}
                kernels.append(
                    {
                        "query_a": qa,
                        "query_b": qb,
                        "n_events": 6,
                        "n_scenes": 3,
                        "native_response_r": quality(aa, bb)["r"],
                        "endpoint_delta_cka": linear_cka(va, vb),
                        "candidate_jaccard": len(ka & kb) / len(ka | kb),
                        "shared_sites": len(ka & kb),
                    }
                )
        pd.DataFrame(kernels).to_csv(out / "response_kernel.csv", index=False)
        frame.groupby(
            ["target_query", "source_query", "group", "mode", "axis"], as_index=False
        ).agg(
            native_effect=("native_effect", "mean"),
            function_effect=("function_effect", "mean"),
            n=("scene_id", "size"),
        ).to_csv(out / "group_condition_effects.csv", index=False)
        write_json(
            out / "summary.json",
            {
                "status": "COMPLETE_LOCAL_EXCHANGE",
                "patches": len(rows),
                "independent_scenes": 3,
                "held_event_edits": 6,
                "human_group_analysis_complete": False,
                "coordinate": "endpoint-only model-identical head/MLP delta transport",
                "calls": prior["calls"] + observer.calls,
            },
        )
        print(status("COMPLETE"), flush=True)
    except TimeoutError:
        print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=OUTPUT)
    parser.add_argument(
        "--upstream", type=Path, default=BASE / "navigation_v1/NAV-02/query_sites_r1"
    )
    parser.add_argument(
        "--out", type=Path, default=BASE / "navigation_v1/NAV-06/endpoint_exchange_r1"
    )
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    with collector_guard(args.out):
        run(args.reference, args.upstream, args.out, args.seconds)
