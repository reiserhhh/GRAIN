#!/usr/bin/env python3
"""VM01: bounded, resumable real-LLM semantic controls and residual traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import (
    fact_summary,
    factorial_interactions,
    probe_layers,
)
from grain_research.vector_mechanisms.controls import (
    SYSTEM,
    build_controls,
    control_content_anchor,
    control_user_message,
    material_hash,
    validate_controls,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch


def write_json(path, value):
    """Atomically checkpoint JSON with finite values only."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def instrument_checks(observer, rows):
    """Check no-hook replay, self-donor, restored state and embedding invariance."""
    records = []
    for row in [r for r in rows if r["variant"] == 0][:8]:
        ids = observer.prompt_ids(SYSTEM, control_user_message(row))
        baseline = observer._forward(ids).logits[0, -1].detach().float().cpu().numpy()
        states, logits = observer.capture(ids)
        _, repeated = observer.capture(ids)
        site = list(observer.sites)[len(observer.sites) // 2]
        patch = Patch(site, len(ids) - 1, states[site][0])
        _, sham = observer.capture(ids, patch=patch)
        _, restored = observer.capture(ids)
        alternative = list(ids)
        alternative[1] = (alternative[1] + 1) % observer.model.config.vocab_size
        changed, _ = observer.capture(alternative)
        values = {
            "no_hook_delta": float(np.max(np.abs(logits - baseline))),
            "repeat_delta": float(np.max(np.abs(logits - repeated))),
            "self_donor_delta": float(np.max(np.abs(logits - sham))),
            "restore_delta": float(np.max(np.abs(logits - restored))),
            "fixed_input_embedding_delta": float(
                np.max(np.abs(states["input_embedding"] - changed["input_embedding"]))
            ),
        }
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite replay result")
        records.append(
            {"case_id": row["case_id"], **values, "pass": max(values.values()) <= 5e-4}
        )
    return records


def run(args):
    """Resume only identical code/material/config; checkpoint every semantic case."""
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for field in ("device", "dtype", "output_dir"):
        value = getattr(args, field, None)
        if value is not None:
            config[field] = str(value)
    rows = build_controls()
    if args.limit:
        # Include all four families and at least one held-out scenario in smoke runs.
        rows = [
            r
            for r in rows
            if r["variant"] in (0, 1)
            and r["group_id"] in ("scenario_00", "scenario_08")
        ][: args.limit]
    validation = validate_controls(rows)
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "private_cache"
    cache.mkdir(exist_ok=True)
    manifest = {
        "protocol": config["protocol"],
        "config": config,
        "material": validation,
        "code_sha256": code_fingerprint(ROOT),
        "label_scope": "synthetic_semantic_facts_only",
    }
    manifest["identity"] = material_hash(manifest)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["identity"] != manifest["identity"]:
            raise ValueError("Run identity changed; choose a new output directory")
    write_json(manifest_path, manifest)
    archive_source(ROOT, output / "source_snapshot.zip")
    write_json(output / "contrast_manifest.json", rows)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    if config["device"] == "cuda":
        free, total = torch.cuda.mem_get_info()
        write_json(
            output / "device_preflight.json",
            {
                "free_bytes": free,
                "total_bytes": total,
                "name": torch.cuda.get_device_name(0),
            },
        )
        if free < 4_000_000_000:
            raise RuntimeError(
                "Insufficient free VRAM for this full-precision model; not disturbing other jobs"
            )
    started = time.monotonic()
    observer = DecoderObserver.load(
        config["model"],
        revision=config["revision"],
        device=config["device"],
        dtype=config["dtype"],
        max_tokens=config["max_tokens"],
        local_files_only=config["local_files_only"],
        pad_to_length=config.get("pad_to_length"),
    )
    write_json(output / "model_manifest.json", observer.manifest())
    checks = instrument_checks(observer, rows)
    write_json(output / "hook_replay_tests.json", checks)
    if not checks or not all(item["pass"] for item in checks):
        raise RuntimeError("INSTRUMENT_FAILED: semantic interpretation is disabled")
    print(f"INSTRUMENT_PASS {len(checks)} replay/self/restore/input tests", flush=True)
    completed = []
    partial_reason = None
    for row in rows:
        path = cache / f"{row['case_id']}.json"
        feature_path = cache / f"{row['case_id']}.npz"
        if path.is_file() and feature_path.is_file():
            completed.append(json.loads(path.read_text()))
            continue
        if (
            time.monotonic() - started >= config["maximum_seconds"]
            or observer.calls + 3 > config["maximum_calls"]
        ):
            partial_reason = "runtime_or_call_budget"
            break
        if (
            sum(p.stat().st_size for p in cache.iterdir() if p.is_file())
            >= config["maximum_cache_bytes"]
        ):
            partial_reason = "cache_budget"
            break
        ids = observer.prompt_ids(SYSTEM, control_user_message(row))
        before = time.monotonic()
        content_position = observer.content_position(
            SYSTEM, control_user_message(row), control_content_anchor(row)
        )
        states, _ = observer.capture(ids, positions=[content_position, len(ids) - 1])
        margin = observer.margin(ids)
        if not np.isfinite(margin) or not all(
            np.isfinite(v).all() for v in states.values()
        ):
            raise ValueError("Non-finite model state or output")
        measurement = {
            **row,
            "margin": margin,
            "prompt_tokens": len(ids),
            "prompt_token_sha256": material_hash(ids),
            "last_token_id": ids[-1],
            "content_position": content_position,
            "content_token_id": ids[content_position],
            "seconds": time.monotonic() - before,
        }
        np.savez_compressed(
            feature_path,
            **{
                f"{key}__{view}": value[index]
                for key, value in states.items()
                for index, view in enumerate(("content", "decision"))
            },
        )
        write_json(path, measurement)
        completed.append(measurement)
        if len(completed) % 8 == 0:
            print(
                f"PROGRESS {len(completed)}/{len(rows)} calls={observer.calls} seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    frame = pd.DataFrame(completed)
    if frame.empty:
        raise RuntimeError("No completed semantic cases")
    frame.drop(columns=["passage", "question"]).to_csv(
        output / "fact_predictions.csv", index=False
    )
    summary = pd.DataFrame(fact_summary(frame))
    summary.to_csv(output / "capability_by_family.csv", index=False)
    interactions = factorial_interactions(frame)
    interactions.to_csv(output / "evidence_interactions.csv", index=False)
    features = {}
    for row in completed:
        with np.load(cache / f"{row['case_id']}.npz") as states:
            for key in states.files:
                features.setdefault(key, []).append(states[key])
    features = {key: np.stack(value) for key, value in features.items()}
    probes = probe_layers(frame, features)
    probes.to_csv(output / "layer_observability.csv", index=False)
    state = {
        "status": "COMPLETE" if len(completed) == len(rows) else "PARTIAL",
        "completed": len(completed),
        "planned": len(rows),
        "partial_reason": partial_reason,
        "calls_this_invocation": observer.calls,
        "input_tokens_this_invocation": observer.input_tokens,
        "processed_tokens_this_invocation": observer.processed_tokens,
        "seconds_this_invocation": time.monotonic() - started,
        "instrument_pass": True,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "meaning": "model semantic instrument; no human personality evidence",
    }
    if config["device"] == "cuda":
        state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "run_status.json", state)
    lines = [
        "# GRAIN VM01 Semantic Instrument Report",
        "",
        f"Status: {state['status']}",
        "",
        "These labels are explicit semantic facts, NOT personality scores.",
        "Probes read pre-answer states. Test stories are absent from discovery.",
        "Only four held-out scenarios: intervals are descriptive and limited.",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "Input-embedding suffix is a negative control, not a content embedding baseline.",
        "Factorial interaction is output-scale nonadditivity, not personality or PID synergy.",
        "A decoded fact does not establish that the model uses that particular subspace.",
        "This run has no cross-model, answer-code, or human generalization conclusion.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


def main():
    """Expose bounded smoke/full runs and persist failure traces for diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_vector_mechanisms_v1.json"
    )
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        cfg = json.loads(args.config.read_text())
        output = args.output_dir or Path(cfg["output_dir"])
        if not output.is_absolute():
            output = ROOT / output
        output.mkdir(parents=True, exist_ok=True)
        (output / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
