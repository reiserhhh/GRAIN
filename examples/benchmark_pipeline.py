#!/usr/bin/env python3
"""Synthetic GRAIN timings. Run as a module from the checkout being measured."""

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples import handbook_study as study
from grain_core import dual_view_increment as views
from grain_core import factor_discovery as fd
from grain_core import liquid_instrument as li


def timing(f, repeat):
    f()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        times.append(1000 * (time.perf_counter() - start))
    return {"ms": times, "median_ms": statistics.median(times)}


def gpu_probe(repeat):
    """Actual source-to-readout contraction; includes transfers and synchronization."""
    import torch

    devices = [
        d
        for d, ok in (
            ("mps", torch.backends.mps.is_available()),
            ("cuda", torch.cuda.is_available()),
        )
        if ok
    ]
    result = {}
    rng = np.random.default_rng(90)
    for t, d in ((24, 8), (128, 32), (128, 64)):
        p = rng.dirichlet(np.ones(d), size=t)
        effect = rng.normal(0, 0.01, (t, t * d))
        expected = p.T @ effect
        case = {"cpu_float64": timing(lambda x=p, y=effect: x.T @ y, repeat)}
        a, b = p.astype("float32"), effect.astype("float32")
        case["cpu_float32"] = timing(lambda x=a, y=b: x.T @ y, repeat)
        for device in devices:

            def contract(dev=device, left=p, right=effect):
                x = torch.from_numpy(left).to(device=dev, dtype=torch.float32)
                y = torch.from_numpy(right).to(device=dev, dtype=torch.float32)
                return (x.T @ y).cpu().numpy()

            got = contract()
            case[device] = {
                **timing(contract, repeat),
                "max_abs_vs_cpu64": float(np.max(abs(got - expected))),
            }
        result[f"{t}x{d}"] = case
    return result


def benchmark(repeat):
    rng = np.random.default_rng(31)
    docs = study.corpus("bench", 64)
    items = li.episodes(docs, 0)
    texts = [r[2] for _, rows in items for r in rows]
    atlas = li.fit_atlas(texts, 8, "gmm_soft")
    paths = li.native_paths(atlas, items)
    early = rng.normal(size=(120, 64))
    late = early * 0.4 + rng.normal(size=early.shape)
    n, d = 96, 16
    nodes = rng.dirichlet(np.ones(d), size=n)
    effects = rng.normal(0, 0.01, (n, n, d))
    effects[np.triu_indices(n, 1)[::-1]] = 0
    obs = {
        "readout": nodes,
        "reset": nodes,
        "missing_effect": effects,
        "reverse_endpoint": nodes[-1],
        "middle_swap_endpoint": None,
        "all_missing": nodes - effects.sum(0),
    }
    y = np.sin(np.arange(64) * 0.04)
    with tempfile.TemporaryDirectory() as base:
        root = Path(base)
        study.run(root / "reference")
        target = study.corpus("new", 30)

        def workflow():
            with tempfile.TemporaryDirectory(dir=root) as sub:
                study.run(Path(sub) / "data")

        def cold():
            with tempfile.TemporaryDirectory(dir=root) as sub:
                study.score(Path(sub), root / "reference", target)

        cases = {
            "handbook_build_and_place": workflow,
            "cold_30_people": cold,
            "atlas_fit_1536_events": lambda: li.fit_atlas(texts, 8, "gmm_soft"),
            "atlas_apply_256_episodes": lambda: li.native_paths(atlas, items),
            "family_features": lambda: li.participant_features(
                items, paths, ["departure", "detour"]
            ),
            "factor_99_permutations": lambda: fd.fit_stable_crossview(
                early, late, n_permutations=99
            ),
            "trajectory_postprocess_96x16": lambda: li.trajectory(nodes, obs),
            "two_view_cutoff_64": lambda: views.assess(
                y,
                y * 0.8,
                half_widths=np.ones(64),
                rates=[0.8, 0.5, 0.25],
                budgets=[0.2, 0.4, 0.4],
                c_bound=2.0,
                calibration_id="fixture",
            ),
        }
        return {k: timing(v, repeat) for k, v in cases.items()}


def frozen_benchmark(path, transport, repeat, output, batch_docs):
    ref, doc = li.load(path), study.corpus("bench", 8)
    if ref["width"] != 0 or ref.get("cut_policy", {}).get("kind") not in (
        "fixed_or_turn",
        "whole",
        None,
    ):
        raise ValueError("benchmark expects a whole-event reference")
    paths = li.native_paths(ref["atlas"], li.episodes(doc, 0))
    if batch_docs:
        if not 2 <= batch_docs <= 16:
            raise ValueError("use 2 to 16 documents")
        docs = [
            dict(doc, episodes=list(x))
            for x in np.array_split(
                np.asarray(doc["episodes"], dtype=object), batch_docs
            )
        ]

        def serial():
            return [li.run("score", x, ref, transport) for x in docs]

        def batched():
            return li.score_batch(docs, ref, transport)

        if serial() != batched():
            raise AssertionError(
                "serial and batch outputs differ in the synthetic benchmark"
            )
        return {
            "documents": len(docs),
            "episodes": len(paths),
            "equal_outputs": True,
            "serial": timing(serial, repeat),
            "batched": timing(batched, repeat),
        }

    def run():
        answer = li.run("score", doc, ref, transport)
        return answer

    if output:
        li.save(output, run())
    return {
        "episodes": len(paths),
        "unique_numeric": len(set(map(li.digest, paths))),
        "reference": li.digest(ref),
        "score": timing(run, repeat),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--gpu-probe", action="store_true", help="optional PyTorch; no training"
    )
    parser.add_argument(
        "--liquid-reference",
        type=Path,
        help="instead time cold scoring with an existing whole-event LTC; never train",
    )
    parser.add_argument("--transport", choices=["local", "ssh"], default="local")
    parser.add_argument(
        "--score-out",
        type=Path,
        help="optional fresh output for independent parity checking",
    )
    parser.add_argument(
        "--batch-docs",
        type=int,
        default=0,
        help="also compare separate and shared frozen worker calls",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    with threadpool_limits(limits=1):
        if args.liquid_reference:
            result = frozen_benchmark(
                args.liquid_reference,
                args.transport,
                args.repeat,
                args.score_out,
                args.batch_docs,
            )
        else:
            result = {
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "timing": benchmark(args.repeat),
                "device_probe": gpu_probe(args.repeat) if args.gpu_probe else None,
            }
        print(json.dumps(result, indent=2))
