#!/usr/bin/env python3
"""GFP02: actual native suffix interventions along independent factor alignments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_path_routes import SEED, split_direction
from grain_research.vector_mechanisms.full_state_edit import (
    capture_boundary,
    replay_suffix,
)
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_conditional_edits import compatible_observers, load_panel
from scripts.run_grain_factor_path_routes import LEVEL, PLAN, SOURCE
from scripts.run_grain_real_author_path import MODEL


def prepare(source, out):
    """Bind all native units and fixed alignment bytes before native calls."""
    if read(source / "status.json")["status"] != "CPU_COMPLETE":
        raise ValueError("Complete and freeze source factor directions first")
    if out.exists():
        raise ValueError("Use a fresh output directory")
    _, _, native_manifest, hashes = load_panel()
    dm = read(source / "direction_freeze.json")
    if sha(source / "private_directions.npz") != dm["sha256"]:
        raise ValueError("Source directions changed")
    rows = [
        d
        for d in read(LEVEL / "private_design.json")
        if d["query"] in ("TF_cont", "neuroticism")
    ]
    source_authors = {d["author_id"] for d in read(SOURCE / "private_design.json")}
    if source_authors & {d["author_id"] for d in rows}:
        raise ValueError("Intervention evaluation authors entered source fit")
    tokens = read(SOURCE / "manifest.json")
    if tokens["model_sha256"] != native_manifest["model_sha256"]:
        raise ValueError("Source and native model checkpoints differ")
    # P must denote the same literal query token sequence, not just equal shapes.
    for d in read(SOURCE / "private_design.json"):
        if d["query"] not in ("TF_cont", "neuroticism"):
            continue
        for v in d["views"]:
            actual = list(
                np.asarray(v["input_ids"])[np.asarray(v["question_mask"], bool)]
            )
            if actual != native_manifest["question_tokens"][d["query"]]:
                raise ValueError("Source/native query tokens differ")
    for p in (
        source / "manifest.json",
        source / "private_directions.npz",
        source / "direction_freeze.json",
        PLAN,
        Path(__file__),
    ):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    out.mkdir(parents=True)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", rows)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FACTOR_PATH_NATIVE_INTERVENTION_V1",
            "source": str(source.relative_to(ROOT)),
            "units": len(rows),
            "max_calls": len(rows) * 17,
            "max_seconds_per_invocation": 1800,
            "design_sha256": sha(out / "private_design.json"),
            "inputs_sha256": hashes,
            "model_sha256": native_manifest["model_sha256"],
            "amplitude": 0.25,
            "human_labels_used": False,
        },
    )
    write(
        out / "status.json",
        {"status": "PREPARED", "completed_units": 0, "calls": 0, "seconds": 0},
    )


def run(out):
    """Keep full context fixed; patch only literal query P and disjoint Q."""
    m, previous = read(out / "manifest.json"), read(out / "status.json")
    if previous["status"] == "COMPLETE":
        return
    for p, digest in m["inputs_sha256"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen native input changed: " + p)
    if (
        sha(out / "private_design.json") != m["design_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Native checkpoint/design changed")
    source = ROOT / m["source"]
    direction_meta = read(source / "direction_freeze.json")
    with np.load(source / "private_directions.npz", allow_pickle=False) as z:
        directions = {k: z[k].copy() for k in z}
    functions_path = ROOT / read(source / "manifest.json")["functions"]
    if (
        sha(functions_path)
        != read(source / "manifest.json")["input_sha256"][
            str(functions_path.relative_to(ROOT))
        ]
    ):
        raise ValueError("Query functions changed")
    with np.load(functions_path, allow_pickle=False) as z:
        functions = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in ("TF_cont", "neuroticism")
        }
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if not compatible_observers(observer.manifest(), read(LEVEL / "observer.json")):
        raise ValueError("Native observer implementation changed")
    write(out / "observer.json", observer.manifest())
    start, complete = time.monotonic(), 0

    def status(value):
        result = {
            "status": value,
            "completed_units": complete,
            "planned_units": m["units"],
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "human_labels_used": False,
        }
        write(out / "status.json", result)
        print(result, flush=True)

    try:
        for d in read(out / "private_design.json"):
            q = d["query"]
            stem = f"unit_{d['index']:02d}_{q}"
            dest = out / "private_packets" / f"{stem}.json"
            if dest.exists():
                retained = read(dest)
                if (
                    retained["author_id"] != d["author_id"]
                    or len(retained["effects"]) != 8
                ):
                    raise ValueError("Incomplete retained intervention unit")
                complete += 1
                continue
            if (
                time.monotonic() - start > m["max_seconds_per_invocation"]
                or previous["calls"] + observer.calls + 17 > m["max_calls"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            state, kwargs, native = capture_boundary(
                observer, d["full_ids"], d["alphabet"]
            )
            with np.load(
                LEVEL / "private_packets" / f"{stem}.npz", allow_pickle=False
            ) as z:
                state_error = float(
                    np.max(np.abs(state[0].detach().cpu().numpy() - z["full_state"]))
                )
            if (
                state_error > 2e-5
                or abs(native["conditional_score"] - d["source_full"]) > 2e-6
            ):
                raise ValueError("Native cached full state does not replay")
            mask = np.asarray(d["full_mask"], bool)
            if mask[-1]:
                raise ValueError("Query and final state must be disjoint")
            p, qv = (
                state[0, mask].detach().cpu().numpy(),
                state[0, -1].detach().cpu().numpy(),
            )
            info = next(r for r in direction_meta["queries"] if r["query"] == q)
            if list(p.shape) != info["p_shape"] or list(qv.shape) != info["q_shape"]:
                raise ValueError("Source/native state dimensions differ")
            effects = []
            for arm in ("real", "null"):
                for j, factor in enumerate(direction_meta["factors"]):
                    dp, dq = split_direction(
                        directions[f"{q}__{arm}"][j] * m["amplitude"], p.shape, qv.shape
                    )
                    values, affine = {}, {}
                    for sign in (-1, 1):
                        changed = state.clone()
                        changed[:, mask] = torch.as_tensor(
                            p + sign * dp, dtype=state.dtype, device=state.device
                        )
                        changed[:, -1] = torch.as_tensor(
                            qv + sign * dq, dtype=state.dtype, device=state.device
                        )
                        values[str(sign)] = replay_suffix(
                            observer, changed, kwargs, d["alphabet"]
                        )["conditional_score"]
                        affine[str(sign)] = functions[q].parts(
                            p + sign * dp, qv + sign * dq
                        )["score"]
                    effects.append(
                        {
                            "factor": factor,
                            "arm": arm,
                            "plus": values["1"],
                            "minus": values["-1"],
                            "effect": (values["1"] - values["-1"]) / 2,
                            "curvature": (values["1"] + values["-1"]) / 2
                            - native["conditional_score"],
                            "affine_effect": (affine["1"] - affine["-1"]) / 2,
                            "direction_norm": float(
                                np.linalg.norm(np.r_[dp.ravel(), dq.ravel()])
                            ),
                        }
                    )
            write(
                dest,
                {
                    "author_id": d["author_id"],
                    "index": d["index"],
                    "query": q,
                    "native": native["conditional_score"],
                    "state_error": state_error,
                    "effects": effects,
                },
            )
            complete += 1
            if complete % 4 == 0:
                status("RUNNING")
        status("COMPLETE")
    except Exception:
        status("FAILED")
        raise


def analyze(out):
    """Summarize paired author response and distinguish shuffled control effects."""
    if read(out / "status.json")["status"] != "COMPLETE":
        raise ValueError("Do not analyze a selected partial panel")
    rows = []
    for d in read(out / "private_design.json"):
        packet = read(
            out / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
        )
        rows.extend(
            {"author_id": d["author_id"], "query": d["query"], **e}
            for e in packet["effects"]
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_effects.csv", index=False)
    output = []
    for (q, factor), group in frame.groupby(["query", "factor"]):
        real = group[group.arm == "real"].sort_values("author_id")
        null = group[group.arm == "null"].sort_values("author_id")
        if list(real.author_id) != list(null.author_id) or len(real) != 12:
            raise ValueError("Incomplete paired twelve-author effects")
        a, b = real.effect.to_numpy(), null.effect.to_numpy()
        rng = np.random.default_rng(SEED)
        ix = rng.integers(0, len(a), (2000, len(a)))
        means, delta = a[ix].mean(1), (np.abs(a[ix]) - np.abs(b[ix])).mean(1)
        output.append(
            {
                "query": q,
                "factor": factor,
                "authors": len(a),
                "real_mean_effect": float(a.mean()),
                "real_lower": float(np.quantile(means, 0.025)),
                "real_upper": float(np.quantile(means, 0.975)),
                "real_positive_fraction": float(np.mean(a > 0)),
                "real_mean_abs": float(np.mean(np.abs(a))),
                "null_mean_effect": float(b.mean()),
                "null_mean_abs": float(np.mean(np.abs(b))),
                "abs_gain": float(np.mean(np.abs(a) - np.abs(b))),
                "abs_gain_lower": float(np.quantile(delta, 0.025)),
                "abs_gain_upper": float(np.quantile(delta, 0.975)),
                "affine_effect": float(real.affine_effect.mean()),
                "affine_native_sign_match": float(
                    np.mean(np.sign(a) == np.sign(real.affine_effect))
                ),
                "max_abs_curvature": float(np.abs(real.curvature).max()),
                "direction_norm": float(real.direction_norm.mean()),
            }
        )
    pd.DataFrame(output).to_csv(out / "native_factor_effect_summary.csv", index=False)
    write(
        out / "analysis_status.json",
        {
            "status": "COMPLETE_GFP02",
            "units": 24,
            "directions": 8,
            "signed_suffix_calls": 384,
            "new_native_calls": read(out / "status.json")["calls"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "prepare", "run", "analyze"), default="all"
    )
    a = parser.parse_args()
    source, output = a.source_dir.resolve(), a.output_dir.resolve()
    if not source.is_relative_to(BASE.resolve()) or not output.is_relative_to(
        BASE.resolve()
    ):
        parser.error("Use private ignored mechanism directories")
    if a.stage in ("all", "prepare"):
        prepare(source, output)
    if a.stage in ("all", "run"):
        run(output)
    if a.stage in ("all", "analyze"):
        analyze(output)
