#!/usr/bin/env python3
"""Endpoint-equivalent masked-cache gauge test, not quadrature refinement."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H


def representative(low, high, gate_low, gate_high):
    """Replace only zero-mass endpoint states with their visible partner values."""
    l, h = low.clone(), high.clone()
    l_inactive = (gate_low == 0) & (gate_high > 0)
    h_inactive = (gate_high == 0) & (gate_low > 0)
    l = l.where(~l_inactive[:, None, :, None, None], high)
    h = h.where(~h_inactive[:, None, :, None, None], low)
    return l, h


def prepare(parent):
    """Freeze the result-after mathematical gauge comparison before running it."""
    out = parent / "support_gauge_r1"
    out.mkdir(exist_ok=False)
    plan = {
        "timing": "Result-after companion;triggered by primary quadrature error,not preregistered",
        "question": "Do endpoint-invisible padded K/V choices alter the linear interpolated focus response while all endpoint observations remain unchanged?",
        "parent": str(parent.resolve()),
        "pairs": 144,
        "pair_batch": 8,
        "max_numeric_seconds": 120,
        "midpoint": 0.5,
        "new_model_calls": 0,
        "fits": 0,
        "intervention": "For low-masked/high-visible slot set low K/V to high;reverse for high-masked/low-visible. Keep every positive-mass endpoint,mask,text,query,GRAIN mark and weight unchanged",
        "parent_compatibility_or_integration_failure_rescued": False,
        "code": {
            str(Path(__file__).relative_to(ROOT)): H.sha(__file__),
            "grain_research/construction/math_marked_response.py": H.sha(
                ROOT / "grain_research/construction/math_marked_response.py"
            ),
        },
        "sources": {
            str((parent / name).relative_to(ROOT)): H.sha(parent / name)
            for name in (
                "00/manifest.json",
                "00/pairs.json",
                "02/summary.json",
                "02/moments.npz",
            )
        },
    }
    H.write(out / "plan.json", plan)
    write_json(
        out / "status.json",
        {"state": "PREPARED", "completed": 0, "pending": None, "seconds": 0.0},
    )


def run(parent):
    """Compare both representatives at endpoints and one fixed interior point."""
    import torch

    out = parent / "support_gauge_r1"
    plan = H.load(out / "plan.json")
    H.verify_hashes(ROOT, plan["code"] | plan["sources"])
    manifest = H.load(parent / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    pairs = H.load(parent / "00/pairs.json")
    spec = manifest["input_descriptor"]
    state = H.load(out / "status.json")
    if state["state"] != "PREPARED":
        raise ValueError("Do not retry a started companion")
    torch.set_num_threads(2)
    device = manifest["spec"]["device"]
    tensor = lambda x: torch.as_tensor(
        np.asarray(x), dtype=torch.float32, device=device
    )
    weights = C.load_weights(spec["weights"], spec["blocks"], device)
    with np.load(spec["coordinates"], allow_pickle=False) as z:
        seeds, cos, sin = (tensor(z[name]) for name in ("seeds", "cos", "sin"))
    b = spec["reader"]
    query = tensor(
        np.stack(
            [
                R.rotate(
                    np.asarray(b["raw_queries"])[q],
                    spec["endpoint"] + b["gaps"][q],
                    b["frequencies"],
                    b["scaling"],
                )
                for q in range(6)
            ]
        )
    )
    directions = tensor(b["directions"])
    valid = np.asarray(spec["valid"])
    with torch.no_grad():
        for start in range(0, len(pairs), plan["pair_batch"]):
            if state["seconds"] >= plan["max_numeric_seconds"]:
                state["state"] = "BUDGET_STOP"
                write_json(out / "status.json", state)
                raise TimeoutError("Fixed gauge budget;no renewal")
            packet = pairs[start : start + plan["pair_batch"]]
            state.update(state="RUNNING", pending=[start, start + len(packet)])
            write_json(out / "status.json", state)
            begun = time.perf_counter()
            buffers = {name: [[], []] for name in ("k", "v")}
            for p in packet:
                for side, name in enumerate(("low", "high")):
                    path = Path(spec["contexts"][p[name]])
                    if C.stream_sha(path) != spec["hashes"][str(path.resolve())]:
                        raise ValueError("Changed source state")
                    with np.load(path, allow_pickle=False) as z:
                        for key in buffers:
                            buffers[key][side].append(z[key])
            k, v = [
                [tensor(np.stack(values)) for values in buffers[key]]
                for key in ("k", "v")
            ]
            gate = [
                tensor(valid[[p[name] for p in packet]]) for name in ("low", "high")
            ]
            kk = representative(*k, *gate)
            vv = representative(*v, *gate)
            points = [(kk[s], vv[s], gate[s]) for s in range(2)]
            points += [
                ((k[0] + k[1]) / 2, (v[0] + v[1]) / 2, (gate[0] + gate[1]) / 2),
                ((kk[0] + kk[1]) / 2, (vv[0] + vv[1]) / 2, (gate[0] + gate[1]) / 2),
            ]
            readings = np.stack(
                [
                    M.read(
                        *M.construct(*point, seeds, cos, sin, weights, spec["config"]),
                        query,
                        directions,
                    )
                    .cpu()
                    .numpy()
                    for point in points
                ]
            )
            with (out / f"batch_{start:04d}.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    readings=readings,
                    ids=np.array([[p["low"], p["high"]] for p in packet]),
                )
            state.update(
                completed=start + len(packet),
                pending=None,
                seconds=state["seconds"] + time.perf_counter() - begun,
            )
            write_json(out / "status.json", state)
            print(
                "HM120 gauge",
                state["completed"],
                round(state["seconds"], 3),
                flush=True,
            )
    state["state"] = (
        "COMPLETE" if state["seconds"] <= plan["max_numeric_seconds"] else "BUDGET_STOP"
    )
    write_json(out / "status.json", state)
    H.write(out / "seal.json", H.seals(out, out))


def analyze(parent):
    """Read all paired gauge contrasts; never infer a personality effect from them."""
    out = parent / "support_gauge_r1"
    state = H.load(out / "status.json")
    H.verify_hashes(out, H.load(out / "seal.json"))
    readings, ids = [], []
    for path in sorted(out.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            readings.append(z["readings"])
            ids.append(z["ids"])
    readings, ids = np.concatenate(readings, axis=1), np.concatenate(ids)
    with np.load(parent / "02/moments.npz", allow_pickle=False) as z:
        original = z["endpoints"]
    endpoint_error = float(np.max(abs(readings[:2] - original)))
    difference = readings[3] - readings[2]
    focus = difference[:, 0] - difference[:, 1]
    spec = H.load(parent / "00/manifest.json")["input_descriptor"]
    valid = np.asarray(spec["valid"])
    changed = np.any(valid[ids[:, 0]] != valid[ids[:, 1]], axis=1)
    result = {
        "status": "FINITE_MASKED_CACHE_GAUGE_OBSERVED_NOT_GLOBAL"
        if state["state"] == "COMPLETE"
        else "PARTIAL",
        "pairs": len(ids),
        "changed_availability_pairs": int(changed.sum()),
        "numeric_seconds": state["seconds"],
        "endpoint_max_difference": endpoint_error,
        "midpoint_reading_max_difference": float(abs(difference).max()),
        "midpoint_focus_difference_RMS": np.sqrt(
            np.mean(focus**2, axis=(0, 2))
        ).tolist(),
        "midpoint_focus_difference_RMS_changed": np.sqrt(
            np.mean(focus[changed] ** 2, axis=(0, 2))
        ).tolist(),
        "midpoint_focus_max_difference_unchanged": float(abs(focus[~changed]).max()),
        "original_HM119_compatibility": "FAILED_RETAINED",
        "primary_quadrature": "FAILED_FOR_RELIABLE_LOCALIZATION_RETAINED",
        "fits_or_new_model_calls": [0, 0],
        "scope": "Numerical path nonuniqueness under endpoint-invisible K/V representatives;not human trait/state change",
    }
    H.write(out / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage](
        args.parent.resolve()
    )
