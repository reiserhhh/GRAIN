#!/usr/bin/env python3
"""Bounded invented semantics/order panel using the already-qualified direct observer."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_material import (
    archive_direct,
    load_registration,
    verify_tokenizer,
)
from grain_research.vector_mechanisms.direct_semantics import panel, prepare, summarize
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.history_collection import observation_signature
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.collect_grain_direct_traits import CODE, qualified_parent
from scripts.qualify_grain_source_history import load_observer
from scripts.run_grain_vector_mechanism_controls import write_json


def require_idle_history(folder):
    """Fail closed while the known GPU history owner is active; never interrupt it."""
    if folder is None:
        raise ValueError("Real semantic panel requires history queue status")
    s = json.loads((folder / "queue_status.json").read_text(encoding="utf-8"))
    if (
        s["status"] not in ("COMPLETE", "PAUSED_BUDGET_BETWEEN_SHARDS")
        or (folder / "queue.lock").exists()
    ):
        raise ValueError("History GPU queue is not idle")
    import psutil

    for key in ("pid", "child_pid"):
        if s.get(key) and psutil.pid_exists(int(s[key])):
            raise ValueError("History owner or child is still alive")
    return {k: s.get(k) for k in ("status", "parent_identity", "pid", "child_pid")}


def run(a):
    """Freeze the whole invented panel before forwards; do not read any outcome values."""
    if not 60 <= a.max_seconds <= 1800:
        raise ValueError("Semantic panel requires a60-1800s bound")
    start = time.monotonic()
    _, whole, material = load_registration(a.views, a.material_dir)
    verify_tokenizer(a.tokenizer_dir, material)
    a.tokenizer = AutoTokenizer.from_pretrained(
        str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    code = {name: digest(ROOT / name) for name in CODE}
    qm, qs = qualified_parent(a.qualification_dir, material, code, whole)
    idle = None if a.preflight_only else require_idle_history(a.history_queue_dir)
    cases = panel()
    prompts = [prepare(c, a.tokenizer) for c in cases]
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Keep observations under results")
    out.mkdir(parents=True, exist_ok=False)
    archive_direct(ROOT, out / "source_snapshot.zip")
    extra = (
        "grain_research/vector_mechanisms/direct_semantics.py",
        "scripts/qualify_grain_direct_traits_semantics.py",
    )
    manifest = {
        "protocol": "VM05_DIRECT_SEMANTICS_V1",
        "preflight_only": a.preflight_only,
        "panel_sha256": material_hash(cases),
        "material_identity": material["identity"],
        "qualification_identity": qm["identity"],
        "qualified_model_signature": qs["model_signature"],
        "code_sha256": code | {name: digest(ROOT / name) for name in extra},
        "source_snapshot_sha256": digest(out / "source_snapshot.zip"),
        "max_seconds": a.max_seconds,
        "idle_history": idle,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    rows, obs = [], None
    try:
        if not a.preflight_only:
            # Recheck ownership immediately before loading weights.
            require_idle_history(a.history_queue_dir)
            obs = load_observer(a, material)
            if observation_signature(obs.manifest()) != qs["model_signature"]:
                raise ValueError("Qualified direct observer changed")
            write_json(out / "actual_observer.json", obs.manifest())
        for c, (ids, alphabet) in zip(cases, prompts, strict=True):
            if time.monotonic() - start > a.max_seconds:
                raise TimeoutError("Preserve partial semantic observations")
            row = c | {
                "prompt_tokens": len(ids),
                "prompt_sha256": material_hash(ids),
                "alphabet": alphabet,
            }
            if obs:
                _, logits = obs.capture(ids)
                _, again = obs.capture(ids)
                if not np.array_equal(logits, again):
                    raise ValueError("Native semantic replay changed")
                row.update(native_readout(logits, alphabet))
            rows.append(row)
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "cases": len(rows),
                    "forwards": obs.calls if obs else 0,
                },
            )
        write_json(out / "observations.json", rows)
        if obs:
            contrasts, tasks = summarize(rows)
            write_json(out / "semantic_contrasts.json", contrasts)
            write_json(out / "task_sensitivity.json", tasks)
        summary = {
            "status": "COMPLETE",
            "parent_identity": manifest["identity"],
            "cases": len(rows),
            "real_neural_collection": obs is not None,
            "forwards": obs.calls if obs else 0,
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "bridge_outcomes_read": False,
            "scope": "Invented necessary capability contrasts; not human validity or equivalence",
            "outputs_sha256": {
                p.name: digest(p)
                for p in out.iterdir()
                if p.name
                in (
                    "observations.json",
                    "actual_observer.json",
                    "semantic_contrasts.json",
                    "task_sensitivity.json",
                )
            },
        }
        write_json(out / "summary.json", summary)
        write_json(out / "run_status.json", summary)
        print(json.dumps(summary))
    except Exception as exc:
        write_json(out / "partial_observations.json", rows)
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "cases": len(rows),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        # DecoderObserver owns no close() API; this bounded CLI exits after run.
        del obs


def main():
    """Run token-only now; real panel waits for the history GPU owner's terminal boundary."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "material-dir",
        "tokenizer-dir",
        "qualification-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("observer-queue-dir", "history-queue-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--max-seconds", type=int, default=900)
    run(p.parse_args())


if __name__ == "__main__":
    main()
