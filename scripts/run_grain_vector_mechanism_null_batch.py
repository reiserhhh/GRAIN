#!/usr/bin/env python3
"""Serial, bounded complete-null family for every retained source arm and axis."""

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
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_source_controls import legacy_path


def cases(control_dir, legacy_dir):
    """Use all three arms and all four MBTI axes; do not select favorable outcomes."""
    result = [
        {
            "id": f"{task}_{arm}_noref",
            "task": task,
            "arm": arm,
            "parent": str(control_dir / f"{task}_{arm}_retain_731"),
        }
        for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
        for arm in ("raw", "cue_mask", "placebo_mask")
    ]
    return result + [
        {
            "id": f"big5_{arm}_sparse",
            "task": "big5",
            "arm": arm,
            "parent": str(legacy_path(legacy_dir, "big5", arm)),
        }
        for arm in ("raw", "cue_mask", "placebo_mask")
    ]


def main():
    """Resume only verified indexed draws with the same code/material commitments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--legacy-dir", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--feature-cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not 600 <= a.max_seconds <= 28800:
        raise ValueError("Invalid bounded batch budget")
    names = (
        "scripts/run_grain_vector_mechanism_external_null.py",
        "scripts/run_grain_vector_mechanism_external_links.py",
        "scripts/run_grain_vector_mechanism_external_features.py",
        "grain_research/vector_mechanisms/external_null.py",
        "grain_research/vector_mechanisms/external_heads.py",
        "grain_research/vector_mechanisms/external_controls.py",
        "grain_research/vector_mechanisms/external_material.py",
        "suica_core/v7_psychometric.py",
    )

    def code():
        return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}

    fields = {
        "protocol": "VM05_ALL_RETAINED_SOURCE_NULLS_V1",
        "cases": cases(a.control_dir, a.legacy_dir),
        "code": code(),
        "views_sha256": hashlib.sha256(a.views.read_bytes()).hexdigest(),
        "preflight_sha256": hashlib.sha256(
            (a.preflight_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "bridge_labels_read": False,
    }
    fields["identity"] = material_hash(fields)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "batch_manifest.json"
    if path.exists():
        if not a.resume or json.loads(path.read_text()) != fields:
            raise ValueError("Existing null batch requires explicit identical resume")
    elif any(out.iterdir()):
        raise ValueError("Use a new output directory")
    else:
        write_json(path, fields)
    (out / "job_logs").mkdir(exist_ok=True)
    start, completed = time.monotonic(), []
    current = None
    try:
        for case in fields["cases"]:
            current = case["id"]
            child = out / current
            if code() != fields["code"]:
                raise ValueError("Null scoring code changed")
            remaining = int(a.max_seconds - (time.monotonic() - start))
            if remaining < 600 or not (Path(case["parent"]) / "summary.json").exists():
                write_json(
                    out / "batch_status.json",
                    {
                        "status": "PAUSED_BUDGET"
                        if remaining < 600
                        else "WAITING_SOURCE_PARENT",
                        "current": current,
                        "completed": completed,
                        "planned": len(fields["cases"]),
                    },
                )
                return
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_grain_vector_mechanism_external_null.py"),
                "--parent-run",
                case["parent"],
                "--output-dir",
                str(child),
                "--max-seconds",
                str(min(3600, remaining - 60)),
            ]
            for key in ("views", "preflight_dir", "feature_cache_dir"):
                command += ["--" + key.replace("_", "-"), str(getattr(a, key))]
            if child.exists():
                command += ["--resume"]
            write_json(
                out / "batch_status.json",
                {
                    "status": "RUNNING",
                    "current": current,
                    "completed": completed,
                    "planned": len(fields["cases"]),
                },
            )
            with (out / "job_logs" / f"{current}.log").open(
                "a", encoding="utf-8"
            ) as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=False,
                )
            if result.returncode:
                raise ValueError(f"Null child failed: {current}")
            status = json.loads((child / "run_status.json").read_text())
            if status["status"] != "COMPLETE":
                write_json(
                    out / "batch_status.json",
                    {
                        "status": status["status"],
                        "current": current,
                        "completed": completed,
                        "planned": len(fields["cases"]),
                    },
                )
                return
            completed.append(
                {
                    "case": current,
                    "observed": status["observed"],
                    "p_upper": status["p_upper"],
                    "draws": status["draws"],
                }
            )
            print(json.dumps(completed[-1] | {"completed": len(completed)}), flush=True)
        write_json(
            out / "batch_status.json",
            {
                "status": "COMPLETE",
                "completed": completed,
                "planned": len(fields["cases"]),
                "seconds": time.monotonic() - start,
            },
        )
    except Exception as error:
        write_json(
            out / "batch_status.json",
            {
                "status": "FAILED_STOP",
                "current": current,
                "completed": completed,
                "error": str(error),
                "planned": len(fields["cases"]),
            },
        )
        raise


if __name__ == "__main__":
    main()
