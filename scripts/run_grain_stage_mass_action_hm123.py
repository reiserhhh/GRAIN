#!/usr/bin/env python3
"""Run one fixed early/late GRAIN intervention battery, without accuracy grids."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_grain_mass_action as G
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.construction import math_stage_mass_action as T
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H

CODE = (
    "grain_research/construction/math_stage_mass_action.py",
    "scripts/run_grain_stage_mass_action_hm123.py",
    "scripts/readback_grain_stage_mass_action_hm123.py",
    "tests/test_grain_stage_mass_action.py",
    "configs/grain_stage_mass_action_hm123.json",
    "docs/GRAIN_STAGE_MASS_ACTION_PLAN_HM123.md",
)


def save(path, **arrays):
    """Never overwrite a started scientific result."""
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def check(out):
    """Authenticate the admitted finite instrument and source results."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Freeze one stage cut and both schedules before their responses exist."""
    if out.exists():
        raise FileExistsError("Retain previous and failed experiments")
    spec = H.load(ROOT / "configs/grain_stage_mass_action_hm123.json")
    src = ROOT / spec["source"]
    old = H.load(src / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(src, old["inputs"])
    for folder in ("01", "02"):
        H.verify_hashes(src, H.load(src / folder / "seal.json"))
    desc = old["input_descriptor"]
    if spec["cut"] != desc["blocks"] // 2 or desc["blocks"] != 17:
        raise ValueError("Frozen17-block history and mechanical midpoint required")
    with np.load(src / "00/interventions.npz", allow_pickle=False) as z:
        early, late = T.mass_pairs(z["arms"])
        scale = z["read_scale"]
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    save(out / "00/schedules.npz", early=early, late=late, read_scale=scale)
    H.write(out / "00/pairs.json", H.load(src / "00/pairs.json"))
    sources = old["code"] | old["source"]
    sources |= {
        str((src / n).relative_to(ROOT)): H.sha(src / n)
        for n in (
            "00/manifest.json",
            "00/interventions.npz",
            "02/responses.npz",
            "02/summary.json",
            "02/readback.json",
            "02/seal.json",
        )
    }
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": sources,
            "inputs": H.seals(out, out / "00"),
            "input_descriptor": desc,
            "source_failure_retained": True,
            "historically_exposed_material": True,
            "cut_selected_from_outcomes": False,
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed_units": 0,
            "pending": None,
            "seconds": 0.0,
            "attempts": 0,
        },
    )
    print("HM123 PREPARED", {"units": 288, "arms": 49, "cut": spec["cut"]}, flush=True)


def run(out):
    """Execute finite schedules once, with safe bounded checkpointing."""
    import torch

    m = check(out)
    spec, inputs = m["spec"], m["input_descriptor"]
    path = out / "01/status.json"
    state = H.load(path)
    if state["state"] != "PREPARED" or state["pending"] is not None:
        raise ValueError("Do not restart a started packet")
    if "transformers" in sys.modules:
        raise RuntimeError("No full model or generated-answer calls")
    torch.set_num_threads(2)
    if C.stream_sha(inputs["weights"]) != inputs["hashes"][inputs["weights"]]:
        raise ValueError("Immutable pretrained maps changed")
    tensor = lambda a: torch.as_tensor(
        np.asarray(a), dtype=torch.float32, device=spec["device"]
    )
    weights = C.load_weights(inputs["weights"], inputs["blocks"], spec["device"])
    with np.load(inputs["coordinates"], allow_pickle=False) as z:
        seeds, cos, sin = [tensor(z[k]) for k in ("seeds", "cos", "sin")]
    reader = inputs["reader"]
    queries = tensor(
        np.stack(
            [
                R.rotate(
                    np.asarray(reader["raw_queries"])[q],
                    inputs["endpoint"] + reader["gaps"][q],
                    reader["frequencies"],
                    reader["scaling"],
                )
                for q in range(6)
            ]
        )
    )
    directions = tensor(reader["directions"])
    with np.load(out / "00/schedules.npz", allow_pickle=False) as z:
        early, late = z["early"], z["late"]
    length = early.shape[-1]
    for start in range(0, spec["units"], spec["unit_batch"]):
        if state["seconds"] >= spec["max_numeric_seconds"]:
            state["state"] = "BUDGET_STOP"
            write_json(path, state)
            break
        stop = min(start + spec["unit_batch"], spec["units"])
        state.update(
            state="RUNNING", pending=[start, stop], attempts=state["attempts"] + 1
        )
        write_json(path, state)
        begun = time.perf_counter()
        try:
            buffers = {"k": [], "v": []}
            for unit in range(start, stop):
                file = Path(inputs["contexts"][unit])
                if C.stream_sha(file) != inputs["hashes"][str(file.resolve())]:
                    raise ValueError("Historical input changed")
                with np.load(file, allow_pickle=False) as z:
                    for name, buffer in buffers.items():
                        buffer.append(z[name])
            with torch.no_grad():
                k, v = [
                    tensor(np.stack(buffers[n])).repeat_interleave(49, 0)
                    for n in ("k", "v")
                ]
                output = T.construct(
                    k,
                    v,
                    tensor(early[start:stop].reshape(-1, length)),
                    tensor(late[start:stop].reshape(-1, length)),
                    np.arange(length),
                    length,
                    seeds,
                    cos,
                    sin,
                    weights,
                    inputs["config"],
                    spec["cut"],
                )
                readings = (
                    M.read(*output, queries, directions)
                    .cpu()
                    .numpy()
                    .reshape(stop - start, 49, 2, 2, 6)
                )
            if not np.isfinite(readings).all():
                raise FloatingPointError("Nonfinite scientific run retained")
            seconds = time.perf_counter() - begun
            save(
                out / f"01/units/units_{start:04d}.npz",
                readings=readings,
                ids=np.arange(start, stop),
            )
            H.write(
                out / f"01/units/units_{start:04d}.json",
                {"units": stop - start, "seconds": seconds},
            )
            state.update(
                completed_units=stop, pending=None, seconds=state["seconds"] + seconds
            )
            write_json(path, state)
            print("HM123", stop, round(state["seconds"], 3), flush=True)
            del k, v, output, readings
        except Exception:
            state.update(
                state="FAILED", seconds=state["seconds"] + time.perf_counter() - begun
            )
            write_json(path, state)
            raise
    if state["completed_units"] == spec["units"]:
        state["state"] = (
            "COMPLETE"
            if state["seconds"] <= spec["max_numeric_seconds"]
            else "BUDGET_STOP"
        )
        write_json(path, state)
    H.write(out / "01/seal.json", H.seals(out, out / "01/units"))


def gram(x):
    """Keep complete signed response configurations of all six input cues."""
    a = np.asarray(x, float).reshape(6, -1)
    norms = np.linalg.norm(a, axis=1)
    if (norms <= 1e-12).any():
        raise ValueError("Unavailable response direction cannot be called zero")
    a = a / norms[:, None]
    return a @ a.T


def analyze(out):
    """Keep nonadditive effects and both stage comparisons, including failures."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    state, spec = H.load(out / "01/status.json"), m["spec"]
    files = sorted((out / "01/units").glob("units_*.npz"))
    with_readings = []
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            with_readings.append(z["readings"])
    if not with_readings or state["completed_units"] != 288:
        H.write(
            out / "02/partial.json", {"status": "PARTIAL_NOT_PROMOTED", "state": state}
        )
        return
    readings = np.concatenate(with_readings).astype(float)
    early, late = T.route_readings(readings)
    with np.load(ROOT / spec["source"] / "02/responses.npz", allow_pickle=False) as z:
        full = z["readings"].astype(float)
    with np.load(out / "00/schedules.npz", allow_pickle=False) as z:
        scale = z["read_scale"]
    base = np.repeat(readings[:, :1], 25, 1)
    account = T.finite_account(base, early, late, full)
    pairs = H.load(out / "00/pairs.json")
    fields = {n: G.response_field(x, pairs) for n, x in account.items()}
    rows = []
    for name in ("early", "late", "total", "interaction"):
        rows.extend(
            {"stage": name, **r}
            for r in G.selectivity(fields[name]["actor_odd"], scale)
        )
    primary = []
    for wording in range(2):
        values = {
            n: fields[n]["actor_odd"][2, wording, :, :, :, 0, 1] / scale[1]
            for n in fields
        }
        rms = {n: float(np.sqrt(np.mean(x**2))) for n, x in values.items()}
        total_sq = float(np.sum(values["total"] ** 2))
        primary.append(
            {
                "wording": wording,
                "RMS_source_scaled": rms,
                "interaction_fraction": None
                if rms["total"] <= 1e-12
                else rms["interaction"] / rms["total"],
                "early_late_cosine": T.cosine(values["early"], values["late"]),
                "signed_total_alignments": {
                    n: None
                    if total_sq <= 1e-24
                    else float(np.sum(values[n] * values["total"]) / total_sq)
                    for n in ("early", "late", "interaction")
                },
            }
        )
    selectivity = []
    for name in ("early", "late", "total"):
        for wording in range(2):
            means = [
                float(
                    np.mean(
                        [
                            r["selectivity"]
                            for r in rows
                            if r["stage"] == name
                            and (r["variant"], r["wording"], r["mode"], r["alignment"])
                            == (2, wording, 1, align)
                        ]
                    )
                )
                for align in (0, 1)
            ]
            selectivity.append(
                {
                    "stage": name,
                    "wording": wording,
                    "correct": means[0],
                    "wrong": means[1],
                    "gain": means[0] - means[1],
                }
            )
    couplings = {}
    for v, align, mode in np.ndindex(3, 2, 2):
        for name in ("early", "late", "total", "interaction"):
            a = fields[name]["actor_odd"][v, :, :, :, :, align, mode] / scale[mode]
            couplings[f"{name}_{v}_{align}_{mode}"] = gram(a.transpose(3, 0, 1, 2, 4))
        e, l = [
            fields[n]["actor_odd"][v, :, :, :, :, align, mode] / scale[mode]
            for n in ("early", "late")
        ]
        couplings[f"early_late_cosines_{v}_{align}_{mode}"] = np.array(
            [T.cosine(e[:, :, :, q], l[:, :, :, q]) for q in range(6)], float
        )
    identity = float(
        abs(
            account["total"]
            - account["early"]
            - account["late"]
            - account["interaction"]
        ).max()
    )
    baseline_error = float(abs(readings[:, 0] - full[:, 0]).max())
    save(
        out / "02/response_fields.npz",
        readings=readings,
        full_readings=full,
        read_scale=scale,
        **{
            f"{name}_{component}": x
            for name, field in fields.items()
            for component, x in field.items()
        },
    )
    save(out / "02/couplings.npz", **couplings)
    H.table(out / "02/selectivity.csv", rows)
    compatible = baseline_error <= spec["baseline_tolerance"]
    complete = compatible and state["state"] == "COMPLETE"
    H.write(
        out / "02/summary.json",
        {
            "status": "FINITE_STAGE_ACTION_COMPLETE_NOT_GLOBAL",
            "units": 288,
            "numeric_seconds": state["seconds"],
            "cut": spec["cut"],
            "baseline_max_error": baseline_error,
            "numeric_compatible": compatible,
            "account_max_error": identity,
            "primary": primary,
            "selectivity": selectivity,
            "nonadditive_composition_trigger": complete
            and all(
                r["interaction_fraction"] is not None
                and r["interaction_fraction"] > spec["substantial_interaction_fraction"]
                for r in primary
            ),
            "hidden_selective_stages": [
                n
                for n in ("early", "late")
                if complete
                and all(
                    r["correct"] > 0 and r["gain"] > 0
                    for r in selectivity
                    if r["stage"] == n
                )
            ],
            "scientific_scope": "Defined instrument schedules,not unique natural paths or psychological mediation",
            "original_HM122_failure_retained": True,
            "original_scope_complete": False,
        },
    )
    print(H.load(out / "02/summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[a.command](a.out.resolve())
