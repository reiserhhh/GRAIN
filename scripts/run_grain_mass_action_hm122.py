#!/usr/bin/env python3
"""Execute one fixed finite GRAIN-to-evidence action experiment, without grids."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_action_expression as A
from grain_research.construction import math_grain_mass_action as G
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.construction import math_supported_attention_safe as S
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H

CODE = (
    "grain_research/construction/math_grain_mass_action.py",
    "scripts/run_grain_mass_action_hm122.py",
    "scripts/readback_grain_mass_action_hm122.py",
    "tests/test_grain_mass_action.py",
    "configs/grain_mass_action_hm122.json",
    "docs/GRAIN_MASS_ACTION_PLAN_HM122.md",
)


def save(path, **arrays):
    """Preserve all failed and completed files; do not overwrite a run."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def check(out):
    """Verify immutable source, observation law and intervention masses."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def reference(src, pairs):
    """Address saved finite endpoints, not full-model generated answers."""
    with np.load(src / "02/moments.npz", allow_pickle=False) as z:
        e = z["endpoints"].astype(float)
    r = np.zeros((288, 2, 2, 6))
    for p, row in enumerate(pairs):
        r[row["low"]], r[row["high"]] = e[0, p], e[1, p]
    return r


def prepare(out):
    """Freeze source-only mark axes and every mass action before any new reading."""
    if out.exists():
        raise FileExistsError("Preserve old and partial experiments")
    spec = H.load(ROOT / "configs/grain_mass_action_hm122.json")
    src, marks_src = (ROOT / spec[k] for k in ("source", "marks_source"))
    old = H.load(src / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(src, old["inputs"])
    H.verify_hashes(src, H.load(src / "01/seal.json"))
    H.verify_hashes(src, H.load(src / "02/seal.json"))
    pairs = H.load(src / "00/pairs.json")
    ids = [p[k] for p in pairs for k in ("low", "high")]
    if sorted(ids) != list(range(288)):
        raise ValueError("Every history appears exactly once in the paired design")
    inputs = old["input_descriptor"]
    mass = np.asarray(inputs["valid"], float)
    with np.load(marks_src / "00/marks.npz", allow_pickle=False) as z:
        marks, visible = z["g"].astype(float), z["visible"].astype(bool)
    if not np.array_equal(mass * visible, visible.astype(float)):
        raise ValueError("Equal physical token masses required for multiset control")
    geometry = G.source_geometry(marks, mass, visible, pairs)
    score = G.scores(marks, mass, visible, geometry["axes"], spec["score_clip"])
    arms = G.intervention_masses(mass, visible, score, spec["epsilon"])
    base = reference(src, pairs)
    fields = G.response_field(np.repeat(base[:, None], 25, axis=1), pairs)
    read_scale = np.sqrt(np.mean(fields["baseline_actor_odd"][0] ** 2, axis=(0, 1, 2)))
    read_available = read_scale > 1e-10
    read_scale = np.where(read_available, read_scale, 1)
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    H.write(out / "00/pairs.json", pairs)
    save(
        out / "00/interventions.npz",
        marks=marks,
        visible=visible,
        mass=mass,
        scores=score,
        arms=arms,
        read_scale=read_scale,
        read_available=read_available,
        **geometry,
    )
    sources = dict(old["code"] | old["source"])
    sources |= {
        str((src / n).relative_to(ROOT)): H.sha(src / n)
        for n in (
            "00/manifest.json",
            "00/pairs.json",
            "01/seal.json",
            "02/seal.json",
            "02/moments.npz",
            "02/summary.json",
        )
    }
    sources[str((marks_src / "00/marks.npz").relative_to(ROOT))] = H.sha(
        marks_src / "00/marks.npz"
    )
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "source": sources,
            "inputs": H.seals(out, out / "00"),
            "input_descriptor": inputs,
            "historically_exposed_application_material": True,
            "source_axes_use_application_answers_or_gradients": False,
            "arms_per_history": 25,
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
    print(
        {
            "prepared": 288,
            "source_ids": len(geometry["source_ids"]),
            "rank": int(geometry["rank"]),
            "axes_available": geometry["available"].tolist(),
        },
        flush=True,
    )


def run(out):
    """Evaluate all fixed finite mass arms through unchanged pretrained recurrence."""
    import torch

    m = check(out)
    spec, inputs = m["spec"], m["input_descriptor"]
    path, state = out / "01/status.json", H.load(out / "01/status.json")
    if state["state"] != "PREPARED" or state["pending"] is not None:
        raise ValueError("Never restart a started packet")
    if "transformers" in sys.modules:
        raise RuntimeError("No full model or answer invocation")
    torch.set_num_threads(2)
    if C.stream_sha(inputs["weights"]) != inputs["hashes"][inputs["weights"]]:
        raise ValueError("Frozen pretrained maps changed")
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
    with np.load(out / "00/interventions.npz", allow_pickle=False) as z:
        arms = z["arms"]
    length = arms.shape[-1]
    positions = np.arange(length)
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
                    raise ValueError("History carrier changed")
                with np.load(file, allow_pickle=False) as z:
                    for key, values in buffers.items():
                        values.append(z[key])
            with torch.no_grad():
                k, v = [
                    tensor(np.stack(buffers[key])).repeat_interleave(25, 0)
                    for key in ("k", "v")
                ]
                mass = tensor(arms[start:stop].reshape(-1, length))
                reading = M.read(
                    *S.construct(
                        k,
                        v,
                        mass,
                        positions,
                        length,
                        seeds,
                        cos,
                        sin,
                        weights,
                        inputs["config"],
                    ),
                    queries,
                    directions,
                )
                reading = reading.cpu().numpy().reshape(stop - start, 25, 2, 2, 6)
            if not np.isfinite(reading).all():
                raise FloatingPointError("Preserve nonfinite run; no numerical rescue")
            seconds = time.perf_counter() - begun
            save(
                out / f"01/units/units_{start:04d}.npz",
                readings=reading,
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
            print("HM122", stop, round(state["seconds"], 3), flush=True)
            del k, v, mass, reading
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


def analyze(out):
    """Report targeted and broad actions separately, never select a winning arm."""
    m = check(out)
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    state, spec = H.load(out / "01/status.json"), m["spec"]
    files = sorted((out / "01/units").glob("units_*.npz"))
    readings = np.concatenate(
        [np.load(f, allow_pickle=False)["readings"] for f in files]
    ).astype(float)
    if len(readings) != 288:
        H.write(
            out / "02/partial.json",
            {"status": "PARTIAL_NOT_PROMOTED", "units": len(readings), "state": state},
        )
        return
    pairs = H.load(out / "00/pairs.json")
    with np.load(out / "00/interventions.npz", allow_pickle=False) as z:
        scale, available, cue_available = (
            z["read_scale"],
            z["read_available"],
            z["available"],
        )
    fields = G.response_field(readings, pairs)
    rows = G.selectivity(fields["actor_odd"], scale)
    primary = []
    for wording in range(2):
        means = [
            float(
                np.mean(
                    [
                        r["selectivity"]
                        for r in rows
                        if (r["variant"], r["wording"], r["mode"], r["alignment"])
                        == (2, wording, 1, align)
                    ]
                )
            )
            for align in (0, 1)
        ]
        primary.append(
            {
                "wording": wording,
                "correct_selectivity": means[0],
                "wrong_selectivity": means[1],
                "gain": means[0] - means[1],
            }
        )
    comparisons = []
    for component in ("actor_odd", "actor_even"):
        field = fields[component] / scale
        for alignment, mode, variant, wording in np.ndindex(2, 2, 2, 2):
            v = variant + 1
            source = field[0, 0, :, :, :, alignment, mode].reshape(6, 2, -1)
            target = field[v, wording, :, :, :, alignment, mode].reshape(6, 2, -1)
            comparisons.append(
                {
                    "component": component,
                    "alignment": alignment,
                    "mode": mode,
                    "variant": v,
                    "wording": wording,
                    **A.direction_match(source, target),
                }
            )
    ref = reference(ROOT / spec["source"], pairs)
    error = float(abs(readings[:, 0] - ref).max())
    compatible = error <= spec["baseline_tolerance"]
    resolved = (
        compatible
        and state["state"] == "COMPLETE"
        and bool(available.all() and cue_available.all())
    )
    success = resolved and all(
        r["correct_selectivity"] > 0 and r["gain"] > 0 for r in primary
    )
    summary = {
        "status": "FINITE_MASS_ACTION_COMPLETE_NOT_GLOBAL",
        "units": 288,
        "numeric_seconds": state["seconds"],
        "baseline_max_error": error,
        "numeric_compatible": compatible,
        "source_cue_available": cue_available.tolist(),
        "read_scale_available": available.tolist(),
        "primary": primary,
        "positive_targeted_GRAIN_action": success,
        "comparisons": comparisons,
        "current_state_arithmetic_evaluations": 288 * 25 * 2,
        "reverse_gradients_full_captures_answers_API_training": [0, 0, 0, 0, 0],
        "original_scope_complete": False,
        "scope": spec["scope"],
    }
    save(out / "02/responses.npz", readings=readings, read_scale=scale, **fields)
    H.table(out / "02/selectivity.csv", rows)
    H.write(out / "02/summary.json", summary)
    print({k: v for k, v in summary.items() if k != "comparisons"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage](args.out.resolve())
