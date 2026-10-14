#!/usr/bin/env python3
"""F10-04 derivative triage r2: one repair-only rerun (supervisor amendment).

Authorization: project persona reports/claude_research_loop_20260926/
F10_R2_BOUNDED_AUTHORIZATION.md (2026-09-26 02:34 JST): at most 90 additional
local forwards and 6 backward passes; cumulative use including failed r1 is
~169 forwards. Frozen scope reused from r1: the same six cases, direction
seeds, steps, classification rules and local model (r1 manifest SHA-256
033537726e465c6f408e914accdda97493bb61ec99613ff1f3d6bf4d87799143, verified
before any model load). Repairs: MPS results are moved to CPU before float64
conversion; the cap is enforced BEFORE each attempted model call; every case
and arm is saved immediately with call accounting, including failure paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_grain_gptpro_f10_derivative_triage import RULES, STEPS, cases, load

R1 = ROOT / "results/grain_gptpro_foundation_v3/F10/derivative_triage_r1"
R1_MANIFEST_SHA = "033537726e465c6f408e914accdda97493bb61ec99613ff1f3d6bf4d87799143"
AUTH = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/F10_R2_BOUNDED_AUTHORIZATION.md")
MAX_FORWARDS = 90
MAX_BACKWARDS = 6
ARMS = (("cpu", "float64", STEPS, True), ("cpu", "float32", STEPS, False), ("mps", "float32", (1e-2,), False))


class Budget:
    """Call accounting with the cap enforced before every attempt."""

    def __init__(self, path):
        self.path = path
        self.forward_attempts = 0
        self.forward_completed = 0
        self.backward_attempts = 0
        self.failed_calls = []

    def before_forward(self, grad):
        if self.forward_attempts + 1 > MAX_FORWARDS:
            raise RuntimeError("forward cap reached before call")
        if grad and self.backward_attempts + 1 > MAX_BACKWARDS:
            raise RuntimeError("backward cap reached before call")
        self.forward_attempts += 1
        if grad:
            self.backward_attempts += 1

    def save(self, **extra):
        self.path.write_text(json.dumps({
            "r2_forward_attempts": self.forward_attempts,
            "r2_forward_completed": self.forward_completed,
            "r2_backward_attempts": self.backward_attempts,
            "r2_caps": {"forward": MAX_FORWARDS, "backward": MAX_BACKWARDS},
            "r1_estimated_forwards": 79,
            "r1_estimated_backwards": 6,
            "cumulative_estimated_forwards": 79 + self.forward_attempts,
            "failed_calls": self.failed_calls,
            **extra,
        }, indent=1) + "\n")


def score_fn(observer, ids, positions, alphabet, budget):
    """PV01 hook-site score; identical computation to r1 except cap/cast order."""
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

        budget.before_forward(grad)
        handle = module.register_forward_pre_hook(hook)
        try:
            with torch.enable_grad() if grad else torch.no_grad():
                logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True).logits[0, -1]
                s = (torch.softmax(logits[token_ids], dim=0) * values).sum()
                if grad:
                    s.backward()
                    g = box["x"].grad[0, positions, :dim].detach().cpu().to(torch.float64).numpy()
                    out = (float(s.detach().cpu().to(torch.float64)), g)
                else:
                    out = float(s.detach().cpu().to(torch.float64))
            budget.forward_completed += 1
            return out
        except Exception as exc:
            budget.failed_calls.append({"device": str(device), "dtype": str(dtype), "grad": grad, "error": repr(exc)[:300]})
            raise
        finally:
            handle.remove()

    return f


def run_arm(device, dtype, steps, with_grad, budget, out):
    """One arm; each case appended to <arm>.jsonl as soon as it is complete."""
    import gc

    observer = load(device, dtype)
    rows = []
    log = out / f"arm_{device}_{dtype}.jsonl"
    for c in cases(observer):
        f = score_fn(observer, c["ids"], c["positions"], c["alphabet"], budget)
        row = {"pair": c["pair"], "query": c["query"], "device": device, "dtype": dtype}
        if with_grad:
            base, g = f(None, grad=True)
            row["score"] = base
            row["analytic"] = float(np.sum(g * c["direction"]))
        for h in steps:
            plus, minus = f(h * c["direction"]), f(-h * c["direction"])
            row[f"fd_{h:g}"] = (plus - minus) / (2 * h)
        row["original"] = {k: c["original"][k] for k in ("analytic_directional_derivative", "fd_h_0.01", "fd_h_0.005", "abs_error_h_0.01", "tolerance_h_0.01")}
        rows.append(row)
        with log.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        budget.save(status="RUNNING", last_arm=f"{device}/{dtype}", last_case=[c["pair"], c["query"]])
        print(json.dumps({"arm": f"{device}/{dtype}", "pair": c["pair"], "query": c["query"], "forwards": budget.forward_attempts}), flush=True)
    del observer
    gc.collect()
    return rows


def classify(r64, r32, rmps):
    """Frozen r1 rules; PRECISION_FP32 needs both FP32 arms, else UNRESOLVED_MISSING_ARM."""
    out = []
    for i, a in enumerate(r64):
        o = a["original"]
        analytic = a["analytic"]
        rel = {h: abs(a[f"fd_{h:g}"] - analytic) / max(abs(analytic), 1e-12) for h in STEPS}
        stored = abs(o["analytic_directional_derivative"] - analytic) / max(abs(analytic), 1e-12)
        b = r32[i] if i < len(r32) else None
        m = rmps[i] if i < len(rmps) else None
        fp32_dev = max(abs(b["fd_0.01"] - a["fd_0.01"]), abs(m["fd_0.01"] - a["fd_0.01"])) if b and m else None
        if stored > 0.05:
            label = "STORED_ANALYTIC_ERROR"
        elif rel[1e-3] > 1e-3 and rel[1e-4] > 1e-3:
            label = "FORMULA_ERROR"
        elif rel[1e-4] <= 1e-3 and 0.5 * o["abs_error_h_0.01"] <= abs(a["fd_0.01"] - analytic) <= 2 * o["abs_error_h_0.01"]:
            label = "CURVATURE_TRUNCATION"
        elif fp32_dev is None:
            label = "UNRESOLVED_MISSING_ARM"
        elif rel[1e-4] <= 1e-3 and fp32_dev > o["tolerance_h_0.01"]:
            label = "PRECISION_FP32"
        else:
            label = "UNRESOLVED"
        out.append({
            "pair": a["pair"], "query": a["query"], "fp64_analytic": analytic,
            "stored_fp32_mps_analytic": o["analytic_directional_derivative"], "stored_relative_error": stored,
            "fp64_fd_relative_error": {str(h): v for h, v in rel.items()},
            "fp64_fd_0.01_abs_error": abs(a["fd_0.01"] - analytic), "original_abs_error_0.01": o["abs_error_h_0.01"],
            "original_tolerance_0.01": o["tolerance_h_0.01"],
            "fp32_cpu_fd_0.01": b["fd_0.01"] if b else None, "fp32_mps_fd_0.01": m["fd_0.01"] if m else None,
            "original_mps_fd_0.01": o["fd_h_0.01"], "fp32_max_deviation_from_fp64_fd_0.01": fp32_dev,
            "classification": label,
        })
    return out


def main():
    """Verify frozen scope, snapshot code, run arms with incremental saves, classify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = R1 / "manifest_before_run.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != R1_MANIFEST_SHA:
        raise SystemExit(f"r1 manifest hash mismatch {digest}")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, args.output / "r1_manifest_frozen.json")
    shutil.copy2(AUTH, args.output / "supervisor_authorization.md")
    snap = args.output / "code_snapshot"
    snap.mkdir()
    for p in (Path(__file__), ROOT / "scripts/run_grain_gptpro_f10_derivative_triage.py"):
        shutil.copy2(p, snap / p.name)
    (args.output / "amendment_before_run.json").write_text(json.dumps({
        "r1_manifest_sha256": digest,
        "authorization": str(AUTH),
        "authorization_sha256": hashlib.sha256(AUTH.read_bytes()).hexdigest(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snap.iterdir())},
        "caps": {"forward": MAX_FORWARDS, "backward": MAX_BACKWARDS},
        "planned": {"cpu/float64": "6 grad forwards (+6 backward) + 36 FD", "cpu/float32": "36 FD", "mps/float32": "12 FD"},
        "cumulative_note": "r1 ~79 forwards (failed, numbers lost) + r2 <=90 => ~169; not <=100",
        "rules": RULES,
        "unchanged": ["cases", "direction seeds", "steps", "classification thresholds", "local model"],
        "failure_policy": "preserve surviving arms; unresolved cases recorded; no third full rerun",
    }, indent=1) + "\n")
    budget = Budget(args.output / "accounting.json")
    budget.save(status="STARTED")
    started = time.monotonic()
    results, failure = {}, None
    for device, dtype, steps, grad in ARMS:
        try:
            results[f"{device}_{dtype}"] = run_arm(device, dtype, steps, grad, budget, args.output)
            (args.output / f"arm_{device}_{dtype}.json").write_text(json.dumps(results[f"{device}_{dtype}"], indent=1, default=float) + "\n")
        except Exception:  # noqa: BLE001 - any arm failure must preserve surviving arms
            failure = {"arm": f"{device}/{dtype}", "traceback": traceback.format_exc()[-2000:]}
            budget.save(status="ARM_FAILED", failure=failure)
            break
    r64 = results.get("cpu_float64", [])
    table = classify(r64, results.get("cpu_float32", []), results.get("mps_float32", [])) if len(r64) == 6 else []
    (args.output / "triage.json").write_text(json.dumps(table, indent=1) + "\n")
    budget.save(status="FAILED" if failure else "COMPLETE", failure=failure, seconds=time.monotonic() - started,
                classifications={f"{t['pair']}|{t['query']}": t["classification"] for t in table})
    for t in table:
        print(t["pair"], t["query"], t["classification"])


if __name__ == "__main__":
    main()
