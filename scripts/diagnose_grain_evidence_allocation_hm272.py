#!/usr/bin/env python3
"""One unmodified first-batch support diagnosis, not a scientific-arm retry."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_join_encoding as E
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Inspect all-layer bank support and padding membership, without reweighting."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    parent = ROOT / "results/grain_math_response_v1/evidence_allocation_r1"
    manifest = R.check(parent)
    runtime = R.read(parent / "runtime.json")
    if runtime["state"] != "FAILED" or runtime["done_batches"]:
        raise ValueError("Only the actual first-batch support failure is diagnosed")
    out = parent / "support_diagnosis"
    out.mkdir()
    R.write(
        out / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(__file__)},
            "parent_manifest": R.digest(parent / "manifest.json"),
            "parent_failed_runtime": R.digest(parent / "runtime.json"),
            "operation": "one unchanged isolated prefix and first8 native candidates; only bank sums/pad membership exported; no modified scientific arm",
        },
    )
    case = R.read(parent / "cases.json")[0]
    bank = R.read(parent / "bank.json")[:8]
    address = case["supports"][0][0]
    suffix = case["suffix"]
    lengths = [len(suffix) + len(r["target_ids"]) - 1 for r in bank]
    width = max(lengths)
    valid = np.array([[False] * (width - n) + [True] * n for n in lengths])
    observer = J.offline_model(manifest)
    began = time.monotonic()
    cache, encoders = E.encode(
        observer, case["prefixes"][0][0], address["body_start"], address["first_end"]
    )
    original, arrays = qwen.eager_attention_forward, {}

    def hook(module, query, key, value, attention_mask, scaling, **kwargs):
        """Observe native support only; weights/output are returned unchanged."""
        output, weights = original(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
        for name, lo, hi in (
            ("first", address["body_start"], address["first_end"]),
            ("second", address["second_start"], address["prefix_length"]),
        ):
            arrays[f"{module.layer_idx}/{name}"] = (
                weights[..., lo:hi].sum(-1).detach().float().cpu().numpy()
            )
        return output, weights

    try:
        qwen.eager_attention_forward = hook
        B.observe_candidates(observer, cache, suffix, [r["target_ids"] for r in bank])
        torch.mps.synchronize()
    finally:
        qwen.eager_attention_forward = original
    elapsed = time.monotonic() - began
    rows = []
    for name, mass in arrays.items():
        zero = mass == 0
        valid_full = np.broadcast_to(valid[:, None, :], mass.shape)
        rows.append(
            {
                "bank": name,
                "zero_valid": int((zero & valid_full).sum()),
                "zero_pad": int((zero & ~valid_full).sum()),
                "mass_min_valid": float(mass[valid_full].min()),
            }
        )
    R.archive(out / "bank_masses.npz", **arrays, valid_queries=valid)
    R.write(
        out / "diagnosis.json",
        {
            "rows": rows,
            "model_seconds": elapsed,
            "zero_valid_total": sum(r["zero_valid"] for r in rows),
            "zero_pad_total": sum(r["zero_pad"] for r in rows),
            "encoder_receipts": encoders,
            "one_unmodified_batch_not_retried_scientific_arm": True,
        },
    )
    R.seal(out, "diagnosed_seal.json", ("bank_masses.npz", "diagnosis.json"))
    print(R.read(out / "diagnosis.json"), flush=True)


if __name__ == "__main__":
    main()
