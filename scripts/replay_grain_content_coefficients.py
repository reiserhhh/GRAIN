#!/usr/bin/env python3
"""Batch score numeric context/GRAIN packets while denying undeclared file reads."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.content_coefficients import predict_coefficients


def replay(jobs_path, output_path, forbidden):
    """Allow only declared fixed models/current inputs after module loading."""
    jobs = json.loads(jobs_path.read_text())
    reads = {str(Path(j[k]).resolve()) for j in jobs for k in ("model", "inputs")}
    output = str(output_path.resolve())
    denied = []

    def guard(event, args):
        if event != "open":
            return
        path, _mode, flags = args
        if isinstance(path, int):
            raise PermissionError("Undeclared descriptor open")
        path = os.path.realpath(os.fsdecode(path))
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if (writing and path != output) or (not writing and path not in reads):
            denied.append(path)
            raise PermissionError("Undeclared file denied")

    buffer = io.BytesIO()
    np.savez(buffer, probe=np.zeros(1))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as z:
        _ = z["probe"]
    with threadpool_limits(limits=4):
        sys.addaudithook(guard)
        for path in forbidden:
            try:
                with open(path, "rb"):
                    pass
            except PermissionError:
                continue
            raise AssertionError("Undeclared target accessible")
        predicted = {}
        for j in jobs:
            with np.load(j["model"], allow_pickle=False) as z:
                model = dict(z)
            with np.load(j["inputs"], allow_pickle=False) as z:
                if set(z.files) != {"hidden", "grain", "positions"}:
                    raise ValueError("Unexpected inference input key")
                h, g, pos = z["hidden"], z["grain"], z["positions"]
                if pos.shape != (len(h),) or np.any(np.diff(pos) <= 0):
                    raise ValueError("Selected body order changed")
                predicted[j["key"]] = predict_coefficients(h, g, model)
        np.savez_compressed(output_path, **predicted)
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "denial_probes": len(forbidden),
                "extra_denials": len(denied) - len(forbidden),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forbid", type=Path, action="append", default=[])
    a = p.parse_args()
    replay(a.jobs, a.output, a.forbid)
