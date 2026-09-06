#!/usr/bin/env python3
"""Bounded source-only token replay or pinned-Qwen history instrument qualification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import ARMS
from grain_research.vector_mechanisms.history_instrument import (
    archive_instrument,
    material_layouts,
    qualify_triplet,
    technical_cases,
    tokenizer_matches,
)
from grain_research.vector_mechanisms.ready_comparison import digest
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json

REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SITES = ("input_embedding", "block_04", "block_10", "block_16", "final_norm")


def load_observer(a, material):
    """Refuse concurrent collection and require the original CUDA/BF16 observer."""
    if a.observer_queue_dir is None:
        raise ValueError("Real qualification requires the completed observer queue")
    q = json.loads((a.observer_queue_dir / "queue_status.json").read_text())
    qm = json.loads((a.observer_queue_dir / "manifest.json").read_text())
    if (
        q["status"] != "COMPLETE"
        or len(q["completed"]) != 72
        or {(r["stage"], r["shard"]) for r in q["completed"]}
        != {(s, i) for s in ("bge_large", "e5_large", "qwen") for i in range(24)}
        or qm["material_sha256"] != material["common_material_sha256"]
        or q["labels_read"]
    ):
        raise ValueError(
            "Original full observer queue must finish; no competing weights"
        )
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 6_000_000_000:
        raise ValueError("Need .20 CUDA observer and at least6GB free GPU memory")
    torch.set_num_threads(4)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    obs = DecoderObserver.load(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision=REVISION,
        device="cuda",
        dtype="bfloat16",
        max_tokens=3072,
        pad_to_length=3072,
        local_files_only=True,
    )
    layers = obs.model.model.layers
    obs.sites = {"input_embedding": obs.model.model.embed_tokens}
    obs.sites.update({f"block_{i:02d}": layers[i] for i in (4, 10, 16)})
    obs.sites["final_norm"] = obs.model.model.norm
    if (
        obs.manifest()["revision"] != REVISION
        or material_hash(obs.tokenizer.chat_template)
        != material_hash(a.tokenizer.chat_template)
        or material["qwen_revision"] != REVISION
    ):
        raise ValueError("Actual model/template does not match registered material")
    return obs


def run(a):
    """No psychological labels; a token preflight never counts as neural qualification."""
    if not 60 <= a.max_seconds <= 1800:
        raise ValueError("Require bounded runtime")
    started = time.monotonic()
    by_author, cases, material, audit = technical_cases(
        a.views, a.material_dir, a.audit_dir
    )
    tokenizer_matches(a.qwen_tokenizer, material)
    a.tokenizer = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Keep private instrument artifacts in results")
    out.mkdir(parents=True, exist_ok=False)
    archive_instrument(ROOT, out / "source_snapshot.zip")
    fields = {
        "protocol": "VM05_HISTORY_INSTRUMENT_V1",
        "preflight_only": a.preflight_only,
        "material_identity": material["identity"],
        "audit_identity": audit["identity"],
        "reference_rule": "Original99 dictionary; strict no-near-copy target registry",
        "sample_rule": "Two Big5/four MBTI source authors; hash history_instrument_731",
        "selected_cases": cases,
        "authors": len(cases),
        "expected_layouts": len(cases) * 54,
        "planned_instrument_forwards": 0 if a.preflight_only else len(cases) * 108,
        "history_envelope": "body_and_separator",
        "qwen_revision": REVISION,
        "sites": SITES,
        "pad_to_length": 3072,
        "max_seconds": a.max_seconds,
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "code_sha256": {
            name: digest(ROOT / name)
            for name in (
                "scripts/qualify_grain_source_history.py",
                "grain_research/vector_mechanisms/source_history.py",
                "grain_research/vector_mechanisms/history_instrument.py",
                "grain_research/vector_mechanisms/trace.py",
                "grain_research/vector_mechanisms/external_material.py",
                "grain_research/vector_mechanisms/context.py",
                "grain_research/vector_mechanisms/ready_comparison.py",
            )
        },
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    checks, serialized, arrays = [], [], {}
    obs = None
    try:
        all_layouts = [material_layouts(c, by_author, a.tokenizer) for c in cases]
        if not a.preflight_only:
            obs = load_observer(a, material)
            write_json(out / "actual_observer.json", obs.manifest())
            torch.cuda.reset_peak_memory_stats()
        for ci, layouts in enumerate(all_layouts):
            for vi in range(3):
                for arm in ARMS:
                    previous = None
                    for task in QUESTIONS:
                        if time.monotonic() - started > a.max_seconds:
                            raise TimeoutError(
                                "Keep partial instrument; no qualification"
                            )
                        rows = {
                            h: layouts[vi, arm, task, h]
                            for h in ("own", "matched", "invisible")
                        }
                        for h, row in rows.items():
                            serialized.append(
                                {
                                    "case": ci,
                                    "view": vi,
                                    "arm": arm,
                                    "task": task,
                                    "history_arm": h,
                                    "layout": row,
                                }
                            )
                        if obs is None:
                            continue
                        captured, errors = qualify_triplet(obs, rows)
                        prefix_error = 0.0
                        if previous is not None:
                            prefix_error = max(
                                float(
                                    np.max(
                                        np.abs(
                                            captured[h][0][s][1:]
                                            - previous[h][0][s][1:]
                                        )
                                    )
                                )
                                for h in rows
                                for s in SITES
                            )
                            if prefix_error != 0:
                                raise ValueError(
                                    "Pre-question current changed with later task"
                                )
                        previous = captured
                        for h, (states, logits) in captured.items():
                            key = f"case{ci}_view{vi}_{arm}_{task}_{h}"
                            for s in SITES:
                                arrays[f"{key}_{s}"] = states[s]
                            arrays[f"{key}_logits"] = logits
                        checks.append(
                            {
                                "case": ci,
                                "view": vi,
                                "arm": arm,
                                "task": task,
                                "causal_prefix_error": prefix_error,
                                **errors,
                            }
                        )
            progress = {
                "status": "RUNNING",
                "authors": ci + 1,
                "planned": len(cases),
                "seconds": time.monotonic() - started,
                "forwards": obs.calls if obs else 0,
            }
            write_json(out / "run_status.json", progress)
        write_json(out / "private_layouts.json", serialized)
        if obs:
            np.savez_compressed(out / "private_instrument_states.npz", **arrays)
            pd.DataFrame(checks).to_csv(out / "instrument_checks.csv", index=False)
        summary = {
            "status": "COMPLETE",
            "parent_identity": fields["identity"],
            "authors": len(cases),
            "token_layouts_replayed": len(serialized),
            "independent_token_implementation": False,
            "real_neural_qualification_passed": obs is not None,
            "forwards": obs.calls if obs else 0,
            "all_instrument_errors_zero": all(
                v == 0 for r in checks for k, v in r.items() if k.endswith("error")
            )
            if obs
            else None,
            "seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if obs else 0,
            "labels_read": False,
            "bridge_outcomes_read": False,
            "outputs_sha256": {
                p.name: digest(p)
                for p in out.iterdir()
                if p.is_file()
                and not p.name.startswith("._")
                and p.name
                not in ("manifest.json", "run_status.json", "source_snapshot.zip")
            },
        }
        if len(serialized) != fields["expected_layouts"] or (
            obs and obs.calls != fields["planned_instrument_forwards"]
        ):
            raise ValueError("Incomplete qualification coverage")
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as e:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED",
                "exception_type": type(e).__name__,
                "error": str(e),
                "real_neural_qualification_passed": False,
                "forwards": obs.calls if obs else 0,
                "seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    """Use --preflight-only on Mac; real qualification waits for .20 collection."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "material-dir", "audit-dir", "qwen-tokenizer", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--observer-queue-dir", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--max-seconds", type=int, default=1800)
    run(p.parse_args())


if __name__ == "__main__":
    main()
