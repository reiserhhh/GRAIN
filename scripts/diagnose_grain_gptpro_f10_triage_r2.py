#!/usr/bin/env python3
"""Post-hoc diagnostic of F10 derivative triage r2 (saved arms only, no model calls).

The frozen rules labelled all six cases FORMULA_ERROR because the nominal FP64
finite differences did not converge at h=1e-3/1e-4. This diagnostic does not
relabel; it records the evidence bearing on that label: analytic agreement
across precisions, FD error scaling with step size, FP64-arm FD at the original
step against the original PV01 tolerance, MPS replay of the original failing
values, and the float32 casts present in the installed Qwen2 model code.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Write diagnostic.json into the r2 directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/grain_gptpro_foundation_v3/F10/derivative_triage_r2")
    args = parser.parse_args()
    a = [json.loads(x) for x in (args.run / "arm_cpu_float64.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (args.run / "arm_cpu_float32.jsonl").read_text().splitlines()]
    m = [json.loads(x) for x in (args.run / "arm_mps_float32.jsonl").read_text().splitlines()]
    cases = []
    for x, y, z in zip(a, b, m, strict=True):
        o = x["original"]
        g = x["analytic"]
        err = {h: abs(x[f"fd_{h:g}"] - g) for h in (0.01, 0.001, 0.0001)}
        err32 = {h: abs(y[f"fd_{h:g}"] - g) for h in (0.01, 0.001, 0.0001)}
        cases.append({
            "pair": x["pair"], "query": x["query"],
            "analytic_fp64_vs_stored_fp32_mps_rel": abs(o["analytic_directional_derivative"] - g) / abs(g),
            "fp64arm_fd_abs_error": {str(h): v for h, v in err.items()},
            "fp32cpu_fd_abs_error": {str(h): v for h, v in err32.items()},
            "fp64arm_error_times_h": {str(h): v * h for h, v in err.items()},
            "fp64arm_fd_0.01_within_original_tolerance": err[0.01] <= o["tolerance_h_0.01"],
            "original_tolerance_0.01": o["tolerance_h_0.01"],
            "mps_fd_0.01": z["fd_0.01"], "original_mps_fd_0.01": o["fd_h_0.01"],
            "mps_replay_rel_diff": abs(z["fd_0.01"] - o["fd_h_0.01"]) / abs(o["fd_h_0.01"]),
            "mps_fd_0.01_abs_error_vs_fp64_analytic": abs(z["fd_0.01"] - g),
            "mps_fails_original_tolerance": abs(z["fd_0.01"] - g) > o["tolerance_h_0.01"],
        })
    from transformers.models.qwen2 import modeling_qwen2 as q

    src = inspect.getsource(q.Qwen2RMSNorm.forward)
    diag = {
        "cases": cases,
        "summary": {
            "max_analytic_rel_disagreement": max(c["analytic_fp64_vs_stored_fp32_mps_rel"] for c in cases),
            "fp64arm_fd_0.01_within_tolerance": sum(c["fp64arm_fd_0.01_within_original_tolerance"] for c in cases),
            "mps_fails_original_tolerance": sum(c["mps_fails_original_tolerance"] for c in cases),
            "max_mps_replay_rel_diff": max(c["mps_replay_rel_diff"] for c in cases),
            "median_fp64arm_error_ratio_1e-4_over_1e-2": float(np.median([c["fp64arm_fd_abs_error"]["0.0001"] / c["fp64arm_fd_abs_error"]["0.01"] for c in cases])),
            "qwen2_rmsnorm_casts_to_float32": "hidden_states.to(torch.float32)" in src,
            "transformers_version": __import__("transformers").__version__,
        },
        "interpretation_boundary": "post-hoc diagnostic; frozen classification FORMULA_ERROR is retained as the protocol outcome; no third rerun",
    }
    (args.run / "diagnostic_posthoc.json").write_text(json.dumps(diag, indent=1) + "\n")
    print(json.dumps(diag["summary"], indent=1))


if __name__ == "__main__":
    main()
