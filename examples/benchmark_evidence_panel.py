#!/usr/bin/env python3
"""Benchmark the same GRAIN open-stream calculation before and after indexing."""

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np
from grain_core.selective_silence import compute_selective_silence_panel
from threadpoolctl import threadpool_limits


def workload(events, venues, actors, seed):
    rng = np.random.default_rng(seed)
    author = rng.integers(0, actors, events)
    condition = (author + rng.integers(0, 5, events)) % venues
    return (rng.uniform(0, 16, events), author, condition, rng.normal(size=(events, 8)))


def compare(before, after):
    assert len(before) == len(after)
    err = 0.0
    for a, b in zip(before, after, strict=True):
        assert a.keys() == b.keys()
        for k, x in a.items():
            y = b[k]
            if isinstance(x, str):
                assert x == y
                continue
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)
            delta = np.abs(np.asarray(x, dtype=float) - y)
            if np.isfinite(delta).any():
                err = max(err, float(np.max(delta[np.isfinite(delta)])))
    return err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "baseline", help="trusted local Git ref with the previous implementation"
    )
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    if args.repeat < 1:
        p.error("repeat must be positive")
    ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.baseline], text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{ref}:suica_core/selective_silence.py"], text=True
    )
    namespace = {}
    exec(compile(source, f"{ref}:selective_silence.py", "exec"), namespace)  # noqa: S102 - caller selects a trusted local baseline
    prior = namespace["compute_selective_silence_panel"]
    result = {
        "baseline": ref,
        "machine": platform.machine(),
        "platform": platform.system(),
        "threads": 1,
        "cases": {},
    }
    with threadpool_limits(limits=1):
        for n, venues, actors in ((2400, 16, 120), (24000, 128, 800)):
            data = workload(n, venues, actors, n)
            funcs = {"old": prior, "new": compute_selective_silence_panel}
            opts = {"window_edges": np.arange(17), "min_incumbent_events": 2}
            old, new = (f(*data, **opts) for f in funcs.values())
            error = compare(old, new)
            times = {k: [] for k in funcs}
            for i in range(args.repeat):
                keys = list(funcs)
                for key in keys if i % 2 == 0 else reversed(keys):
                    start = time.perf_counter()
                    funcs[key](*data, **opts)
                    times[key].append(1000 * (time.perf_counter() - start))
            median = {k: statistics.median(v) for k, v in times.items()}
            result["cases"][str(n)] = {
                "venues": venues,
                "actors": actors,
                "rows": len(new),
                "max_abs_difference": error,
                "median_ms": median,
                "samples_ms": times,
                "speedup": median["old"] / median["new"],
            }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
