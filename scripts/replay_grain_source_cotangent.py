#!/usr/bin/env python3
"""Restricted numeric-only attention reader; no current-target packet access."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.source_cotangent import predict_attention


def replay(jobs_path, forbidden, max_seconds=120):
    """Whitelist exact input/model/projection files and declared output packets."""
    jobs = json.loads(jobs_path.read_text())
    reads = {
        str(Path(j[k]).resolve()) for j in jobs for k in ("model", "inputs", "weights")
    }
    writes = {str(Path(j["output"]).resolve()) for j in jobs}
    if (
        len(writes) != len(jobs)
        or reads & writes
        or any(Path(p).exists() for p in writes)
    ):
        raise ValueError("Unique fresh prediction outputs required")
    denied = []

    def guard(event, args):
        if event != "open":
            return
        path, _mode, flags = args
        if isinstance(path, int):
            raise PermissionError("Undeclared descriptor open")
        path = os.path.realpath(os.fsdecode(path))
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if (writing and path not in writes) or (not writing and path not in reads):
            denied.append(path)
            raise PermissionError("Undeclared file denied")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, probe=np.zeros(1))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as z:
        _ = z["probe"]
    with threadpool_limits(limits=2):
        sys.addaudithook(guard)
        for path in forbidden:
            try:
                with open(path, "rb"):
                    pass
            except PermissionError:
                continue
            raise AssertionError("Forbidden target accessible")
        start = time.monotonic()
        for j in jobs:
            if time.monotonic() - start >= max_seconds:
                raise TimeoutError("Bounded numerical reader budget reached")
            data = []
            for name in ("inputs", "model", "weights"):
                with np.load(j[name], allow_pickle=False) as z:
                    data.append(dict(z))
            predicted = predict_attention(*data)
            np.savez_compressed(j["output"], **predicted)
        elapsed = time.monotonic() - start
        if elapsed > max_seconds:
            raise TimeoutError("Completed last unit beyond numerical budget")
    print(
        json.dumps(
            {
                "units": len(jobs),
                "programs_per_unit": 2,
                "denial_probes": len(forbidden),
                "extra_denials": len(denied) - len(forbidden),
                "seconds": elapsed,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--forbid", type=Path, action="append", default=[])
    args = p.parse_args()
    replay(args.jobs, args.forbid)
