#!/usr/bin/env python3
"""Cold numeric routing: no texts, evaluation roles, logits or model import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_relational_cells as M


def score(program_path, observations_path, output_path):
    """Apply sealed programs and co-addressed covectors to numeric observations."""
    with np.load(program_path, allow_pickle=False) as loaded:
        programs = {k: loaded[k] for k in loaded.files}
    with np.load(observations_path, allow_pickle=False) as loaded:
        observations = {k: loaded[k] for k in loaded.files}
    states, values = observations["states"], observations["values"]
    if states.ndim != 3 or states.shape[:2] != values.shape[:2]:
        raise ValueError("Six co-addressed record states per world required")
    out = {"ids": observations["ids"]}
    for arm in ("real", "permuted", "position"):
        if arm == "position":
            coordinates = np.broadcast_to(programs["position"], (*states.shape[:2], 2))
        else:
            program = {
                k: programs[f"{arm}_{k}"]
                for k in ("center", "scale", "coefficients", "intercept")
            }
            coordinates = M.predict(
                program, states.reshape(-1, states.shape[-1])
            ).reshape(*states.shape[:2], 2)
        roles = M.assign(coordinates.reshape(-1, 2)).reshape(*states.shape[:2], 2)
        readings, means, counts = [], [], []
        for role, carrier in zip(roles, values, strict=True):
            reading = M.target_values(carrier, programs["directions"])
            mean, count = M.cell_ledger(role, reading)
            readings.append(reading)
            means.append(mean)
            counts.append(count)
        out[f"{arm}_coordinates"] = coordinates
        out[f"{arm}_roles"] = roles
        out[f"{arm}_means"] = np.stack(means)
        out[f"{arm}_counts"] = np.stack(counts)
    out["readings"] = np.stack(readings)
    with Path(output_path).open("xb") as handle:
        np.savez_compressed(handle, **out)


def main():
    """Run the label-free scorer from an explicit three-path interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    score(args.program, args.observations, args.output)


if __name__ == "__main__":
    main()
