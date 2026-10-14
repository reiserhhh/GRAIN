#!/usr/bin/env python3
"""Execute a fixed supported-measure response packet, not an accuracy search."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_action_expression as A
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_marked_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.construction import math_supported_measure as S
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_support_gauge_hm120 as G

CODE = (
    "grain_research/construction/math_supported_measure.py",
    "grain_research/construction/math_marked_response.py",
    "grain_research/construction/math_recursive_collector.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/construction/math_action_expression.py",
    "scripts/run_grain_supported_measure_hm121.py",
    "scripts/run_grain_support_gauge_hm120.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/construction/math_cold_evidence.py",
    "tests/test_grain_supported_measure.py",
    "tests/test_grain_marked_response_constructor.py",
    "configs/grain_supported_measure_hm121.json",
    "docs/GRAIN_SUPPORTED_MEASURE_PLAN_HM121.md",
)


def save(path, **arrays):
    """Create one result without replacing a failed or completed artifact."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def prepare(out):
    """Freeze inherited public histories, marks and the new path definition."""
    if out.exists():
        raise FileExistsError("Preserve previous experiment")
    spec = H.load(ROOT / "configs/grain_supported_measure_hm121.json")
    src = ROOT / spec["source"]
    old = H.load(src / "00/manifest.json")
    H.verify_hashes(ROOT, old["code"] | old["source"])
    H.verify_hashes(src, old["inputs"])
    H.verify_hashes(src, H.load(src / "01/seal.json"))
    H.verify_hashes(src, H.load(src / "02/seal.json"))
    pairs = H.load(src / "00/pairs.json")
    if len(pairs) != spec["pairs"]:
        raise ValueError("All144 addressed pairs required")
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    H.write(out / "00/pairs.json", pairs)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "source": {
                str((src / name).relative_to(ROOT)): H.sha(src / name)
                for name in (
                    "00/manifest.json",
                    "00/marks.npz",
                    "00/pairs.json",
                    "02/moments.npz",
                    "02/summary.json",
                )
            },
            "inputs": H.seals(out, out / "00"),
            "input_descriptor": old["input_descriptor"],
            "original_scope_complete": False,
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed_pairs": 0,
            "pending": None,
            "seconds": 0.0,
            "attempts": 0,
        },
    )
    print("HM121 prepared", len(pairs), flush=True)


def check(out):
    """Authenticate immutable computation and addressed scientific input."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    return m


def run(out):
    """Evaluate invariant mixtures and fixed mass derivatives with safe receipts."""
    import torch

    m = check(out)
    spec, inputs = m["spec"], m["input_descriptor"]
    pairs = H.load(out / "00/pairs.json")
    path = out / "01/status.json"
    state = H.load(path)
    if state["state"] != "PREPARED" or state["pending"] is not None:
        raise ValueError("Never retry a started packet")
    if "transformers" in sys.modules:
        raise RuntimeError("No full neural model instantiation")
    torch.set_num_threads(2)
    if C.stream_sha(inputs["weights"]) != inputs["hashes"][inputs["weights"]]:
        raise ValueError("Frozen weights changed")
    tensor = lambda x: torch.as_tensor(
        np.asarray(x), dtype=torch.float32, device=spec["device"]
    )
    weights = C.load_weights(inputs["weights"], inputs["blocks"], spec["device"])
    with np.load(inputs["coordinates"], allow_pickle=False) as z:
        seeds, cos, sin = (tensor(z[name]) for name in ("seeds", "cos", "sin"))
    bundle = inputs["reader"]
    queries = tensor(
        np.stack(
            [
                R.rotate(
                    np.asarray(bundle["raw_queries"])[q],
                    inputs["endpoint"] + bundle["gaps"][q],
                    bundle["frequencies"],
                    bundle["scaling"],
                )
                for q in range(6)
            ]
        )
    )
    directions = tensor(bundle["directions"])
    valid = np.asarray(inputs["valid"], float)
    length = valid.shape[1]
    positions = np.tile(np.arange(length), 2)
    nodes, node_weights = M.quadrature(spec["quadrature_nodes"])
    for start in range(0, len(pairs), spec["pair_batch"]):
        if state["seconds"] >= spec["max_numeric_seconds"]:
            state["state"] = "BUDGET_STOP"
            write_json(path, state)
            raise TimeoutError("No finite budget renewal")
        packet = pairs[start : start + spec["pair_batch"]]
        state.update(
            state="RUNNING",
            pending=[start, start + len(packet)],
            attempts=state["attempts"] + 1,
        )
        write_json(path, state)
        begun = time.perf_counter()
        try:
            buffers = {name: [[], []] for name in ("k", "v")}
            for pair in packet:
                for side, name in enumerate(("low", "high")):
                    file = Path(inputs["contexts"][pair[name]])
                    if C.stream_sha(file) != inputs["hashes"][str(file.resolve())]:
                        raise ValueError("Frozen history changed")
                    with np.load(file, allow_pickle=False) as z:
                        for key in buffers:
                            buffers[key][side].append(z[key])
            k, v = [[tensor(np.stack(a)) for a in buffers[key]] for key in ("k", "v")]
            masses = [
                tensor(valid[[p[name] for p in packet]]) for name in ("low", "high")
            ]
            atom_k, atom_v, origin, delta = S.pack(*k, *v, *masses)
            kk, vv = G.representative(*k, *masses), G.representative(*v, *masses)
            gauge_k, gauge_v, gauge_origin, gauge_delta = S.pack(*kk, *vv, *masses)
            pack_error = max(
                float(abs(a - b).max().cpu())
                for a, b in zip(
                    (atom_k, atom_v, origin, delta),
                    (gauge_k, gauge_v, gauge_origin, gauge_delta),
                    strict=True,
                )
            )

            def read(ak, av, mass):
                """Observe both fixed focuses through the same explicit constructor."""
                return M.read(
                    *S.construct(
                        ak,
                        av,
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

            with torch.no_grad():
                endpoint = [
                    read(atom_k, atom_v, origin + t * delta).cpu().numpy()
                    for t in (0.0, 1.0)
                ]
                midpoint = [
                    read(ak, av, ao + spec["gauge_midpoint"] * ad).cpu().numpy()
                    for ak, av, ao, ad in (
                        (atom_k, atom_v, origin, delta),
                        (gauge_k, gauge_v, gauge_origin, gauge_delta),
                    )
                ]
            action = np.zeros((len(packet), 2, 6, 2 * length))
            for t, w in zip(nodes, node_weights, strict=True):
                mass = (origin + float(t) * delta).detach().requires_grad_(True)
                reading = read(atom_k, atom_v, mass)
                focus = reading[:, 0] - reading[:, 1]
                for mode in range(2):
                    for query in range(6):
                        grad = torch.autograd.grad(
                            focus[:, mode, query].sum(),
                            mass,
                            retain_graph=not (mode == 1 and query == 5),
                        )[0]
                        action[:, mode, query] += (
                            w * (grad * delta).detach().cpu().numpy()
                        )
                del mass, reading, focus, grad
            seconds = time.perf_counter() - begun
            save(
                out / f"01/units/pairs_{start:04d}.npz",
                action=action,
                endpoints=np.stack(endpoint),
                midpoint=np.stack(midpoint),
                pack_error=np.array(pack_error),
                ids=np.array([[p["low"], p["high"]] for p in packet]),
            )
            H.write(
                out / f"01/units/pairs_{start:04d}.json",
                {"pairs": len(packet), "seconds": seconds},
            )
            state.update(
                completed_pairs=start + len(packet),
                pending=None,
                seconds=state["seconds"] + seconds,
            )
            write_json(path, state)
            print(
                "HM121",
                state["completed_pairs"],
                round(state["seconds"], 3),
                flush=True,
            )
        except Exception:
            state.update(
                state="FAILED", seconds=state["seconds"] + time.perf_counter() - begun
            )
            write_json(path, state)
            raise
    state["state"] = (
        "COMPLETE" if state["seconds"] <= spec["max_numeric_seconds"] else "BUDGET_STOP"
    )
    write_json(path, state)
    H.write(out / "01/seal.json", H.seals(out, out / "01/units"))


def analyze(out):
    """Keep every family/control comparison and the failed-path branch."""
    m = check(out)
    spec, state = m["spec"], H.load(out / "01/status.json")
    src = ROOT / spec["source"]
    files = sorted((out / "01/units").glob("pairs_*.npz"))
    if not files:
        raise ValueError("No completed numerical work")
    buffers = {
        name: [] for name in ("action", "endpoints", "midpoint", "ids", "pack_error")
    }
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            for key, values in buffers.items():
                values.append(z[key])
    a, e, midpoint, ids = [
        np.concatenate(buffers[k], axis=1 if k in ("endpoints", "midpoint") else 0)
        for k in ("action", "endpoints", "midpoint", "ids")
    ]
    pack_error = max(float(x) for x in buffers["pack_error"])
    with np.load(src / "00/marks.npz", allow_pickle=False) as z:
        marks = np.concatenate((z["g"][ids[:, 0]], z["g"][ids[:, 1]]), axis=1)
        visible = np.concatenate(
            (z["visible"][ids[:, 0]], z["visible"][ids[:, 1]]), axis=1
        )
    length = marks.shape[1] // 2
    moments = S.moments(a, marks, visible, np.repeat([0, 1], length))
    truth = (e[1, :, 0] - e[1, :, 1]) - (e[0, :, 0] - e[0, :, 1])
    residual = a.sum(-1) - truth
    with np.load(src / "02/moments.npz", allow_pickle=False) as z:
        reference = z["endpoints"][:, : len(ids)]
    endpoint_error = float(abs(e - reference).max())
    gauge_error = float(abs(midpoint[0] - midpoint[1]).max())
    error_fraction = float(np.linalg.norm(residual) / np.linalg.norm(truth))
    localization = (
        state["state"] == "COMPLETE"
        and error_fraction <= spec["localization_completeness_RMS_fraction_limit"]
        and endpoint_error <= spec["endpoint_rounding_tolerance"]
        and gauge_error <= spec["gauge_reading_tolerance"]
        and pack_error == 0
    )
    maps = {
        name: moments[name]
        for name in ("mass", "first", "connected", "misaligned_connected")
    }
    comparisons = []
    pairs = H.load(out / "00/pairs.json")
    if len(a) == len(pairs):
        for name, values in maps.items():
            field = np.zeros((3, 2, 6, 2, 2, *values.shape[1:]))
            for p, value in zip(pairs, values, strict=True):
                variant, wording, family, condition, owner = p["key"]
                field[
                    variant, wording, family, (condition + 1) // 2, (owner + 1) // 2
                ] = value
            own = field[:, :, :, :, 1]
            odd = (own - field[:, :, :, :, 0]) / 2
            for component, x in (("own", own), ("actor_odd", odd)):
                for mode in range(2):
                    for variant, wording in ((1, 0), (1, 1), (2, 0), (2, 1)):
                        comparisons.append(
                            {
                                "map": name,
                                "component": component,
                                "mode": mode,
                                "variant": variant,
                                "wording": wording,
                                **A.direction_match(
                                    x[0, 0, :, :, mode].reshape(6, 2, -1),
                                    x[variant, wording, :, :, mode].reshape(6, 2, -1),
                                ),
                            }
                        )

    def selected(name, wording):
        """Return only the two prespecified primary comparisons, not best cases."""
        return next(
            x["matched_mean_cosine"]
            for x in comparisons
            if (x["map"], x["component"], x["mode"], x["variant"], x["wording"])
            == (name, "actor_odd", 1, 2, wording)
        )

    gains = (
        [selected("connected", w) - selected("misaligned_connected", w) for w in (0, 1)]
        if comparisons
        else []
    )
    summary = {
        "status": "COMPLETE_SUPPORTED_MEASURE_NOT_GLOBAL"
        if state["state"] == "COMPLETE"
        else "PARTIAL",
        "pairs": len(a),
        "numeric_seconds": state["seconds"],
        "pack_invariance_max_error": pack_error,
        "midpoint_gauge_reading_max_error": gauge_error,
        "endpoint_reference_max_error": endpoint_error,
        "completeness_RMS": float(np.sqrt(np.mean(residual**2))),
        "response_RMS": float(np.sqrt(np.mean(truth.astype(float) ** 2))),
        "completeness_RMS_fraction": error_fraction,
        "localization_resolved": localization,
        "primary_mark_gain_both_wordings": gains,
        "positive_GRAIN_localization": bool(localization and gains and min(gains) > 0),
        "comparisons": comparisons,
        "current_state_arithmetic_evaluations": len(a) * 16,
        "reverse_gradient_calls": len(a) * 12,
        "fits_answers_full_history_captures_or_API": [0, 0, 0, 0],
        "HM119_compatibility": "FAILED_RETAINED",
        "HM120_localization": "FAILED_RETAINED",
        "original_scope_complete": False,
        "scope": spec["scope"],
    }
    save(
        out / "02/moments.npz",
        action=a,
        endpoints=e,
        midpoint=midpoint,
        ids=ids,
        marks=marks,
        visible=visible,
        truth=truth,
        residual=residual,
        **moments,
    )
    H.write(out / "02/summary.json", summary)
    if comparisons:
        H.table(
            out / "02/comparisons.csv",
            [
                {
                    key: row.get(key)
                    for key in (
                        "map",
                        "component",
                        "mode",
                        "variant",
                        "wording",
                        "status",
                        "matched_mean_cosine",
                        "permutation_mean",
                        "matching_rank_with_ties",
                    )
                }
                for row in comparisons
            ],
        )
    print({k: v for k, v in summary.items() if k != "comparisons"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage](
        args.output.resolve()
    )
