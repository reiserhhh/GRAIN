#!/usr/bin/env python3
"""OS-level sealed replay (macOS sandbox-exec) for a fixed subset of public bundles.

The in-process audit hook only observes Python-level opens. This check runs the
same cold scorer under an OS sandbox that denies reading file contents under the
results/data trees of both repositories (except the declared bundle and request
file) and denies network, then compares outputs byte-for-byte with the
hook-sealed replay and records an OS-level refused truth read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD = Path("/Volumes/mobile3/projects/project persona")
RUN = ROOT / "results/grain_gptpro_foundation_v3/F16/serialize_full_r1"
SEEDS = (20260915, 20260916, 20260917)
PICK = (
    "numeric_t1_{s}_conditional_conditional_mixture_T1",
    "numeric_dynamic_{s}_conditional_conditional_mixture_T3",
    "text_t1_{s}_mixture_T1",
    "text_dynamic_{s}_mixture_T3",
)
DENIED = [
    ROOT / "results",
    ROOT / "data",
    COORD / "results",
    COORD / "data_sets",
    COORD / "analysis",
    COORD / "reports",
]
TRUTH = (
    ROOT
    / "results/grain_gptpro_foundation_v3/F16/dynamic_transport_full_r1/20260915/application_truth.npz"
)


def profile(allowed):
    """SBPL profile: deny content reads of data trees, allow declared files, deny network."""
    rules = ["(version 1)", "(allow default)", "(deny network*)"]
    rules += [f'(deny file-read-data (subpath "{p}"))' for p in DENIED if p.exists()]
    rules += [f'(allow file-read-data (literal "{p}"))' for p in allowed]
    return "".join(rules)


def run_one(name, output):
    """Score one bundle under the OS sandbox; compare with the hook-sealed output."""
    index = {e["name"]: e for e in json.loads((RUN / "bundle_index.json").read_text())}
    entry = index[name]
    bundle = RUN / "bundles" / f"{name}.json"
    requests = RUN / "cold" / f"{name}.requests.jsonl"
    out = output / f"{name}.scores.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    command = [
        "sandbox-exec",
        "-p",
        profile([bundle.resolve(), requests.resolve()]),
        sys.executable,
        "-m",
        "grain_research.gptpro_foundation_v3.f16_serialize",
        "score",
        "--bundle",
        str(bundle),
        "--sha256",
        entry["sha256"],
        "--input",
        str(requests),
        "--output",
        str(out),
    ]
    done = subprocess.run(
        command, env=env, capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    reference = (RUN / "cold" / f"{name}.scores.jsonl").read_bytes()
    return {
        "name": name,
        "returncode": done.returncode,
        "stderr_tail": done.stderr[-500:],
        "byte_identical_to_hook_sealed": out.exists() and out.read_bytes() == reference,
    }


def main():
    """Run the OS-sealed subset and a raw OS-level truth-read probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = [p.format(s=s) for s in SEEDS for p in PICK]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda n: run_one(n, args.output), names))
    probe = subprocess.run(
        [
            "sandbox-exec",
            "-p",
            profile([]),
            sys.executable,
            "-c",
            f"open({str(TRUTH)!r},'rb').read(1)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    summary = {
        "runs": len(results),
        "all_returncode_zero": all(r["returncode"] == 0 for r in results),
        "all_byte_identical": all(r["byte_identical_to_hook_sealed"] for r in results),
        "os_truth_probe_refused": probe.returncode != 0
        and "Operation not permitted" in probe.stderr,
        "os_truth_probe_stderr_tail": probe.stderr[-300:],
        "denied_subpaths": [str(p) for p in DENIED if p.exists()],
        "scope": "OS sandbox denies file-content reads under the listed data trees and all network; code/venv/stdlib remain readable. Subset: 4 routes x 3 seeds mixture bundles.",
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
