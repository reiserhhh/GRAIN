#!/usr/bin/env python3
"""F10-04 bounded derivative triage of the six PV01 finite-difference failures.

Pre-declared (manifest written before any model load): the six original cases
(pair, query, direction seed), score f(delta) = sum(softmax(candidate logits) *
values) with delta added to layer-12 o_proj input rows of head 0 at the reader
suffix (identical to PV01), computed entirely in the model dtype.
Arms: cpu float64 (analytic autograd + central FD at h=1e-2,1e-3,1e-4),
cpu float32 (central FD at the same steps), mps float32 (central FD at h=1e-2,
original setting replay). Budget: 6 + 36 + 36 + 12 = 90 forwards (<=100).
Classification rules are frozen in the manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PV01 = ROOT / "results/grain_measurement_discovery_v1/pv01/r1/gradient_checks.json"
MODEL_DIR = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
STEPS = (1e-2, 1e-3, 1e-4)
RULES = {
    "STORED_ANALYTIC_ERROR": "stored FP32-MPS analytic differs from FP64 analytic by >5% relative",
    "FORMULA_ERROR": "FP64 central FD fails to approach FP64 analytic: relative error at h=1e-3 and 1e-4 both >1e-3",
    "CURVATURE_TRUNCATION": "FP64 FD converges (rel err at h=1e-4 <=1e-3) and FP64 FD at h=1e-2 reproduces the original h=.01 error within a factor 2",
    "PRECISION_FP32": "FP64 FD converges but FP32 FD (cpu or mps) at h=1e-2 deviates from FP64 FD at h=1e-2 by more than the original tolerance",
    "UNRESOLVED": "none of the above",
}


def load(device, dtype):
    """Load the pinned local observer in the requested device/dtype (offline)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from grain_research.vector_mechanisms.trace import DecoderObserver

    observer = DecoderObserver.load(
        str(MODEL_DIR),
        device=device,
        dtype=dtype,
        max_tokens=1024,
        local_files_only=True,
    )
    observer.model.requires_grad_(False)
    return observer


def score_fn(observer, ids, positions, alphabet, counters):
    """Return f(delta, grad=False) in the model dtype with the PV01 hook site."""
    import torch

    model = observer.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dim = model.config.hidden_size // model.config.num_attention_heads
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    token_ids = torch.tensor(alphabet["token_ids"], dtype=torch.long, device=device)
    values = torch.tensor(alphabet["values"], dtype=dtype, device=device)
    module = model.model.layers[12].self_attn.o_proj

    def f(delta, grad=False):
        box = {}

        def hook(_m, args):
            x = args[0].clone()
            if grad:
                x = x.detach().requires_grad_(True)
                box["x"] = x
                return (x,)
            x[0, positions, :dim] += torch.as_tensor(delta, device=device, dtype=dtype)
            return (x,)

        handle = module.register_forward_pre_hook(hook)
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                logits = model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1]
                s = (torch.softmax(logits[token_ids], dim=0) * values).sum()
                counters["forward"] += 1
                if counters["forward"] > counters["max"]:
                    raise TimeoutError("forward budget exceeded")
                if grad:
                    s.backward()
                    g = (
                        box["x"]
                        .grad[0, positions, :dim]
                        .detach()
                        .to(torch.float64)
                        .cpu()
                        .numpy()
                    )
                    return float(s.detach().cpu().to(torch.float64)), g
                return float(s.detach().cpu().to(torch.float64))
        finally:
            handle.remove()

    return f


def cases(observer):
    """Reconstruct the six original cases exactly as PV01 did."""
    from grain_research.vector_mechanisms.direct_traits import candidates
    from scripts.run_grain_conditional_operator import inputs
    from scripts.run_grain_conditional_operator_native import context

    _, pairs = inputs()
    source_pairs = [p for p in pairs if p["split"] == "source"]
    original = json.loads(PV01.read_text())
    out = []
    for index, query in [(i, "extraversion") for i in range(3)] + [
        (i, "neuroticism") for i in range(3)
    ]:
        pair = source_pairs[index % len(source_pairs)]
        ids, positions = context(observer, pair["low"]["text"], query)
        dim = (
            observer.model.config.hidden_size
            // observer.model.config.num_attention_heads
        )
        shape = (len(positions), dim)
        rng = np.random.default_rng(20260915 + index)
        direction = rng.normal(size=int(np.prod(shape)))
        direction /= np.linalg.norm(direction)
        orig = next(
            o for o in original if o["pair"] == pair["pair"] and o["query"] == query
        )
        out.append(
            {
                "pair": pair["pair"],
                "query": query,
                "ids": ids,
                "positions": positions,
                "direction": direction.reshape(shape),
                "alphabet": candidates(query, observer.tokenizer),
                "original": orig,
            }
        )
    return out


def run_arm(device, dtype, steps, with_grad, counters):
    """One precision arm over all six cases."""
    observer = load(device, dtype)
    rows = []
    for c in cases(observer):
        f = score_fn(observer, c["ids"], c["positions"], c["alphabet"], counters)
        row = {"pair": c["pair"], "query": c["query"], "device": device, "dtype": dtype}
        if with_grad:
            base, g = f(None, grad=True)
            row["score"] = base
            row["analytic"] = float(np.sum(g * c["direction"]))
        for h in steps:
            plus, minus = f(h * c["direction"]), f(-h * c["direction"])
            row[f"fd_{h:g}"] = (plus - minus) / (2 * h)
        row["original"] = {
            k: c["original"][k]
            for k in (
                "analytic_directional_derivative",
                "fd_h_0.01",
                "fd_h_0.005",
                "abs_error_h_0.01",
                "tolerance_h_0.01",
            )
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "arm": f"{device}/{dtype}",
                    "pair": c["pair"],
                    "query": c["query"],
                    "forwards": counters["forward"],
                }
            ),
            flush=True,
        )
    del observer
    gc.collect()
    return rows


def classify(r64, r32, rmps):
    """Apply the frozen rules case by case."""
    out = []
    for a, b, m in zip(r64, r32, rmps, strict=True):
        o = a["original"]
        analytic = a["analytic"]
        rel = {
            h: abs(a[f"fd_{h:g}"] - analytic) / max(abs(analytic), 1e-12) for h in STEPS
        }
        stored = abs(o["analytic_directional_derivative"] - analytic) / max(
            abs(analytic), 1e-12
        )
        fp32_dev = max(
            abs(b["fd_0.01"] - a["fd_0.01"]), abs(m["fd_0.01"] - a["fd_0.01"])
        )
        if stored > 0.05:
            label = "STORED_ANALYTIC_ERROR"
        elif rel[1e-3] > 1e-3 and rel[1e-4] > 1e-3:
            label = "FORMULA_ERROR"
        elif (
            rel[1e-4] <= 1e-3
            and abs(a["fd_0.01"] - analytic) >= 0.5 * o["abs_error_h_0.01"]
            and abs(a["fd_0.01"] - analytic) <= 2 * o["abs_error_h_0.01"]
        ):
            label = "CURVATURE_TRUNCATION"
        elif rel[1e-4] <= 1e-3 and fp32_dev > o["tolerance_h_0.01"]:
            label = "PRECISION_FP32"
        else:
            label = "UNRESOLVED"
        out.append(
            {
                "pair": a["pair"],
                "query": a["query"],
                "fp64_analytic": analytic,
                "stored_fp32_mps_analytic": o["analytic_directional_derivative"],
                "stored_relative_error": stored,
                "fp64_fd_relative_error": {str(h): v for h, v in rel.items()},
                "fp64_fd_0.01_abs_error": abs(a["fd_0.01"] - analytic),
                "original_abs_error_0.01": o["abs_error_h_0.01"],
                "fp32_cpu_fd_0.01": b["fd_0.01"],
                "fp32_mps_fd_0.01": m["fd_0.01"],
                "original_mps_fd_0.01": o["fd_h_0.01"],
                "fp32_max_deviation_from_fp64_fd_0.01": fp32_dev,
                "classification": label,
            }
        )
    return out


def main():
    """Write manifest, run three arms sequentially, classify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "cases": [
            {
                "pair": o["pair"],
                "query": o["query"],
                "direction_seed": o["direction_seed"],
            }
            for o in json.loads(PV01.read_text())
        ],
        "arms": [
            ["cpu", "float64", list(STEPS), "analytic+fd"],
            ["cpu", "float32", list(STEPS), "fd"],
            ["mps", "float32", [1e-2], "fd replay"],
        ],
        "planned_forwards": 90,
        "max_forwards": 100,
        "rules": RULES,
        "model_dir": str(MODEL_DIR),
        "no_training": True,
        "no_api": True,
    }
    (args.output / "manifest_before_run.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    counters = {"forward": 0, "max": 100}
    started = time.monotonic()
    r64 = run_arm("cpu", "float64", STEPS, True, counters)
    (args.output / "arm_cpu_float64.json").write_text(json.dumps(r64, indent=1, default=float) + "\n")
    r32 = run_arm("cpu", "float32", STEPS, False, counters)
    (args.output / "arm_cpu_float32.json").write_text(json.dumps(r32, indent=1, default=float) + "\n")
    rmps = run_arm("mps", "float32", (1e-2,), False, counters)
    (args.output / "arm_mps_float32.json").write_text(json.dumps(rmps, indent=1, default=float) + "\n")
    table = classify(r64, r32, rmps)
    (args.output / "arms.json").write_text(
        json.dumps(
            {"cpu_float64": r64, "cpu_float32": r32, "mps_float32": rmps},
            indent=1,
            default=float,
        )
        + "\n"
    )
    (args.output / "triage.json").write_text(json.dumps(table, indent=1) + "\n")
    (args.output / "accounting.json").write_text(
        json.dumps(
            {
                "forwards": counters["forward"],
                "backwards": 6,
                "seconds": time.monotonic() - started,
            },
            indent=1,
        )
        + "\n"
    )
    for t in table:
        print(
            t["pair"],
            t["query"],
            t["classification"],
            f"stored_rel={t['stored_relative_error']:.2e}",
            {k: f"{v:.2e}" for k, v in t["fp64_fd_relative_error"].items()},
        )


if __name__ == "__main__":
    main()
