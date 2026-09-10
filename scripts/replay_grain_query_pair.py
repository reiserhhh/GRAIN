#!/usr/bin/env python3
"""Reproduce complete numerical scoring with all undeclared file reads denied."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_pair import ARMS, BLOCKS
from grain_research.vector_mechanisms.query_pair import predict_query_pair


def replay(model_path, input_path, output_path, forbidden):
    """Read only frozen assets and current numerical inputs after installing guard."""
    reads = {str(p.resolve()) for p in (model_path, input_path)}
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
    sys.addaudithook(guard)
    for path in forbidden:
        try:
            with open(path, "rb"):
                pass
        except PermissionError:
            continue
        raise AssertionError("Source file was accessible")
    with np.load(model_path, allow_pickle=False) as z:
        model = dict(z)
    with np.load(input_path, allow_pickle=False) as z:
        blocks = {k: z[k] for k in BLOCKS}
        phi = {k: z[k] for k in ARMS}
    np.savez(output_path, **predict_query_pair(blocks, phi, model))
    print(
        {
            "source_denial_checks": len(forbidden),
            "extra_denials": len(denied) - len(forbidden),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forbid", type=Path, action="append", default=[])
    a = p.parse_args()
    replay(a.model, a.inputs, a.output, a.forbid)
