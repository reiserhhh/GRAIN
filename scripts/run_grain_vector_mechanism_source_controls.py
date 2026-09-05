#!/usr/bin/env python3
"""Bounded, serial, predeclared source controls while the CUDA observer queue runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof


def cases():
    """Do every frozen axis/arm, never select only the previously strongest axis."""
    items = []
    for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont"):
        for arm in ("raw", "cue_mask", "placebo_mask"):
            for mode, seeds in [
                ("retain", [731]),
                ("purge", [731]),
                ("random_count", range(731, 736)),
            ]:
                for seed in seeds:
                    items.append(
                        {
                            "task": task,
                            "arm": arm,
                            "mode": mode,
                            "seed": seed,
                            "feature": "grain_intermediate",
                            "exclude_reference": True,
                        }
                    )
    for arm in ("raw", "cue_mask", "placebo_mask"):
        for mode, seeds in [("purge", [731]), ("random_count", range(731, 736))]:
            for seed in seeds:
                items.append(
                    {
                        "task": "big5",
                        "arm": arm,
                        "mode": mode,
                        "seed": seed,
                        "feature": "word_char",
                        "exclude_reference": False,
                    }
                )
    for item in items:
        item["id"] = f"{item['task']}_{item['arm']}_{item['mode']}_{item['seed']}"
    return items


def scoring_commitment():
    """Bind score-defining sources, allowing only unrelated analysis/docs to evolve."""
    names = (
        "scripts/run_grain_vector_mechanism_external_links.py",
        "grain_research/vector_mechanisms/external_heads.py",
        "grain_research/vector_mechanisms/external_controls.py",
        "grain_research/vector_mechanisms/external_material.py",
        "scripts/run_grain_vector_mechanism_external_features.py",
        "suica_core/v7_psychometric.py",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }


def main():
    """Check each completed OOF result, halt on errors, and preserve every output."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--feature-cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if not 600 <= a.max_seconds <= 28800:
        raise ValueError("Source-control batch must have a bounded budget")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = {
        "protocol": "VM05_SOURCE_CONTROLS_V1",
        "cases": cases(),
        "scoring_code": scoring_commitment(),
        "views_sha256": hashlib.sha256(a.views.read_bytes()).hexdigest(),
        "preflight_manifest_sha256": hashlib.sha256(
            (a.preflight_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "bridge_outcomes_used": False,
    }
    fields["identity"] = material_hash(fields)
    path = out / "batch_manifest.json"
    if path.exists():
        if not a.resume_completed or json.loads(path.read_text()) != fields:
            raise ValueError("Existing batch requires explicit identical resume")
    elif any(out.iterdir()):
        raise ValueError("Use new batch directory")
    else:
        write_json(path, fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    (out / "job_logs").mkdir(exist_ok=True)
    start, completed = time.monotonic(), []
    status = {
        "status": "RUNNING",
        "planned": len(fields["cases"]),
        "parent_identity": fields["identity"],
    }
    try:
        for case in fields["cases"]:
            child = out / case["id"]
            if scoring_commitment() != fields["scoring_code"]:
                raise ValueError("Scoring implementation changed during batch")
            if not child.exists():
                remaining = a.max_seconds - (time.monotonic() - start)
                if remaining < 600:
                    write_json(
                        out / "batch_status.json",
                        status | {"status": "PAUSED_BUDGET", "completed": completed},
                    )
                    return
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_grain_vector_mechanism_external_links.py"),
                ]
                for flag in (
                    "views",
                    "preflight_dir",
                    "parent_dir",
                    "prepared_root",
                    "feature_cache_dir",
                ):
                    command += ["--" + flag.replace("_", "-"), str(getattr(a, flag))]
                command += [
                    "--task",
                    case["task"],
                    "--feature",
                    case["feature"],
                    "--arm",
                    case["arm"],
                    "--thread-control",
                    case["mode"],
                    "--drop-seed",
                    str(case["seed"]),
                    "--output-dir",
                    str(child),
                    "--max-seconds",
                    str(min(3600, int(remaining) - 60)),
                ]
                if case["exclude_reference"]:
                    command += ["--exclude-reference"]
                status.update(
                    current=case["id"],
                    completed=completed,
                    seconds=time.monotonic() - start,
                )
                write_json(out / "batch_status.json", status)
                with (out / "job_logs" / (case["id"] + ".log")).open(
                    "w", encoding="utf-8"
                ) as handle:
                    result = subprocess.run(
                        command,
                        check=False,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Source-control child failed: {case['id']}; inspect scoped log"
                    )
            verified = load_oof(child)
            m, s = verified["manifest"], verified["summary"]
            if (
                m["task"] != case["task"]
                or m["arm"] != case["arm"]
                or m["thread_control"] != case["mode"]
                or m["drop_seed"] != case["seed"]
                or m["exclude_reference"] != case["exclude_reference"]
            ):
                raise ValueError("Completed case metadata differs")
            completed.append(
                {"case": case["id"], "score": s["score"], "authors": s["authors"]}
            )
            write_json(
                out / "batch_status.json",
                status | {"completed": completed, "seconds": time.monotonic() - start},
            )
            print(
                json.dumps(
                    completed[-1]
                    | {"completed": len(completed), "planned": len(fields["cases"])}
                ),
                flush=True,
            )
        write_json(
            out / "batch_status.json",
            status
            | {
                "status": "COMPLETE",
                "completed": completed,
                "seconds": time.monotonic() - start,
            },
        )
    except Exception as error:
        write_json(
            out / "batch_status.json",
            status
            | {"status": "FAILED_STOP", "completed": completed, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
