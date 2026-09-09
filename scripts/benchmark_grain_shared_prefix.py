#!/usr/bin/env python3
"""NAV-09: exact prefix reuse, independent-process timing and numerical replay."""

from __future__ import annotations

import argparse
import resource
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
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.shared_prefix import (
    capture_cached_suffix,
    capture_uncached,
    context_groups,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "navigation_v1/NAV-09/shared_prefix_r1"


def implementation():
    """Hash the actual capture and benchmark implementations."""
    names = [
        "scripts/benchmark_grain_shared_prefix.py",
        "grain_research/vector_mechanisms/shared_prefix.py",
    ]
    return {n: digest(ROOT / n) for n in names}


def prepare(out):
    """Fix two previously exposed authors and one world before computing scores."""
    if out.exists():
        raise FileExistsError("Choose a new benchmark ID")
    panel = BASE / "navigation_v1/NAV-08/author_panel_r1"
    worlds = read_json(panel / "private_worlds.json")
    chosen = sorted(
        w["world_id"] for w in worlds if w["role"] == "evaluation" and w["view"] == 0
    )[:2]
    cases = [
        [d for d in read_json(panel / "private_design.json") if d["world_id"] == w]
        for w in chosen
    ]
    construction = read_json(OUTPUT / "design.json")
    wid = min(d["world_id"] for d in construction)
    cases.append([d for d in construction if d["world_id"] == wid])
    design = []
    for ci, rows in enumerate(cases):
        if {d["query"] for d in rows} != set(QUERIES) or len(rows) != 3:
            raise ValueError("Each case must contain exactly the three queries")
        design.extend(
            {**d, "case": ci, "index": ci * 3 + j}
            for j, d in enumerate(sorted(rows, key=lambda x: x["query"]))
        )
    groups = []
    for ci in range(3):
        for prefix, rows in context_groups([d for d in design if d["case"] == ci]):
            groups.append(
                {
                    "case": ci,
                    "prefix_tokens": len(prefix),
                    "queries": [d["query"] for d in rows],
                    "share": len(rows) > 1,
                }
            )
    out.mkdir(parents=True)
    write_json(out / "private_design.json", design)
    write_json(
        out / "manifest.json",
        {
            "protocol": "NAV09_EXACT_PREFIX_V1",
            "repeats": 3,
            "design_sha256": digest(out / "private_design.json"),
            "source_sha256": {
                str(x.relative_to(ROOT)): digest(x)
                for x in (
                    panel / "private_design.json",
                    panel / "private_worlds.json",
                    OUTPUT / "design.json",
                )
            },
            "code_sha256": implementation(),
            "model_sha256": digest(MODEL / "model.safetensors"),
            "reference_sha256": digest(OUTPUT / "synthetic_reference.npz"),
            "tolerances": {"state_max_abs": 1e-4, "score_max_abs": 1e-5},
            "minimum_speedup": 1.10,
            "groups": groups,
            "memory": "process RSS peak includes model load and warmup; MPS allocation is sampled, not peak",
            "context_policy": "exact complete pre-query context; different system prompts are never shared",
            "warmup": "one full first-case pass, excluded",
            "timing": "end-to-end inference including prefix build, cache clone, native and P/Q copies; excludes disk saves",
            "human_labels_used": False,
            "full_LLM_required": True,
        },
    )
    print({"status": "PREPARED", "groups": groups}, flush=True)


def run(out, arm):
    """Time one arm in its own process; cache never survives across cases/repeats."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = read_json(out / "manifest.json")
    if m["code_sha256"] != implementation():
        raise ValueError("Benchmark implementation changed")
    if digest(out / "private_design.json") != m["design_sha256"]:
        raise ValueError("Design changed")
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer changed")
    if digest(OUTPUT / "synthetic_reference.npz") != m["reference_sha256"]:
        raise ValueError("Function changed")
    dest = out / arm
    if dest.exists():
        raise FileExistsError(
            "A timing arm is single-process/single-run; choose new benchmark ID"
        )
    dest.mkdir()
    design = read_json(out / "private_design.json")
    fs = load_functions(OUTPUT / "synthetic_reference.npz")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    capture_uncached(observer, design[0])
    torch.mps.synchronize()
    warm_calls = observer.calls
    timings, allocs = [], []
    start = time.monotonic()
    for rep in range(m["repeats"]):
        for ci in range(3):
            rows = [d for d in design if d["case"] == ci]
            torch.mps.synchronize()
            t0, calls0 = time.monotonic(), observer.calls
            results, tokens, shared = [], 0, 0
            if arm == "uncached":
                for d in rows:
                    h, native = capture_uncached(observer, d)
                    results.append((d, h, native))
                    tokens += len(d["input_ids"])
            else:
                for prefix, group in context_groups(rows):
                    if len(group) == 1:
                        d = group[0]
                        h, native = capture_uncached(observer, d)
                        results.append((d, h, native))
                        tokens += len(d["input_ids"])
                        continue
                    device = next(observer.model.parameters()).device
                    with torch.no_grad():
                        ids = torch.tensor([prefix], device=device)
                        prefix_result = observer.model.model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            use_cache=True,
                        )
                    observer.calls += 1
                    cache = prefix_result.past_key_values
                    del prefix_result
                    allocs.append(torch.mps.current_allocated_memory())
                    tokens += len(prefix)
                    for d in group:
                        h, native = capture_cached_suffix(observer, d, prefix, cache)
                        results.append((d, h, native))
                        tokens += len(d["input_ids"]) - len(prefix)
                    shared += len(group) - 1
                    del cache
            torch.mps.synchronize()
            elapsed = time.monotonic() - t0
            timings.append(
                {
                    "repeat": rep,
                    "case": ci,
                    "seconds": elapsed,
                    "forward_calls": observer.calls - calls0,
                    "processed_tokens": tokens,
                    "avoided_context_repeats": shared,
                }
            )
            allocs.append(torch.mps.current_allocated_memory())
            for d, h, native in results:
                save_unit(
                    dest / f"r{rep}_u{d['index']:02d}.json",
                    {
                        "repeat": rep,
                        "index": d["index"],
                        "query": d["query"],
                        "case": ci,
                        "native": native,
                        "parts": fs[d["query"]].parts(h["P"], h["Q"]),
                    },
                    **h,
                )
            pd.DataFrame(timings).to_csv(dest / "latency.csv", index=False)
            write_json(
                dest / "run_status.json",
                {
                    "status": "RUNNING",
                    "groups_complete": len(timings),
                    "calls": observer.calls - warm_calls,
                },
            )
    result = {
        "status": "COMPLETE",
        "arm": arm,
        "calls": observer.calls - warm_calls,
        "wall_seconds": time.monotonic() - start,
        "inference_seconds": sum(x["seconds"] for x in timings),
        "peak_RSS_bytes_Mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_sampled_MPS_bytes": max(allocs),
        "model": str(MODEL),
        "dtype": "float32",
        "attention": "eager",
    }
    write_json(dest / "run_status.json", result)
    print(result, flush=True)


def analyze(out):
    """Compare every stored state/score and every paired full-case latency."""
    m = read_json(out / "manifest.json")
    for arm in ("uncached", "cached"):
        if read_json(out / arm / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Both timing processes must finish")
    errors = []
    for rep in range(m["repeats"]):
        for d in read_json(out / "private_design.json"):
            a, x = checked_arrays(out / "uncached" / f"r{rep}_u{d['index']:02d}.json")
            b, y = checked_arrays(out / "cached" / f"r{rep}_u{d['index']:02d}.json")
            if any(a[k] != b[k] for k in ("repeat", "index", "query", "case")):
                raise ValueError("Replay identity mismatch")
            errors.append(
                {
                    "repeat": rep,
                    "case": d["case"],
                    "query": d["query"],
                    "P_max_abs": float(np.max(np.abs(x["P"] - y["P"]))),
                    "Q_max_abs": float(np.max(np.abs(x["Q"] - y["Q"]))),
                    "native_abs": abs(
                        a["native"]["conditional_score"]
                        - b["native"]["conditional_score"]
                    ),
                    "function_abs": abs(a["parts"]["score"] - b["parts"]["score"]),
                }
            )
    e = pd.DataFrame(errors)
    t = pd.read_csv(out / "uncached/latency.csv").merge(
        pd.read_csv(out / "cached/latency.csv"),
        on=["repeat", "case"],
        suffixes=("_uncached", "_cached"),
        validate="one_to_one",
    )
    t["speedup"] = t.seconds_uncached / t.seconds_cached
    state_error = float(e[["P_max_abs", "Q_max_abs"]].max().max())
    score_error = float(e[["native_abs", "function_abs"]].max().max())
    numerical = (
        state_error <= m["tolerances"]["state_max_abs"]
        and score_error <= m["tolerances"]["score_max_abs"]
    )
    speed = float(t.speedup.median())
    e.to_csv(out / "numerical_equivalence.csv", index=False)
    t.to_csv(out / "paired_timings.csv", index=False)
    result = {
        "status": "COMPLETE_BENCHMARK",
        "numerical_pass": bool(numerical),
        "state_max_abs": state_error,
        "score_max_abs": score_error,
        "median_speedup": speed,
        "minimum_speedup": m["minimum_speedup"],
        "decision": "OPT_IN_EXACT_PREFIX_CACHE"
        if numerical and speed >= m["minimum_speedup"]
        else "RETAIN_UNCACHED_DEFAULT",
        "not_independent_runtime": True,
    }
    write_json(out / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "analyze"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--arm", choices=("uncached", "cached"), default="uncached")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.out)
    elif a.action == "analyze":
        analyze(a.out)
    else:
        with collector_guard(a.out):
            run(a.out, a.arm)
