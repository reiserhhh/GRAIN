#!/usr/bin/env python3
"""M24: actual suffix effects of query mean, configuration and final state edits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_coordinates import FAMILIES
from grain_research.vector_mechanisms.coordinate_cube import (
    anchored_effects,
    query_cube,
)
from grain_research.vector_mechanisms.full_state_edit import (
    capture_boundary,
    replay_suffix,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_conditional_edits import (
    LEVEL,
    compatible_observers,
    load_panel,
    paired_metrics,
)
from scripts.run_grain_real_author_path import MODEL

PLAN = ROOT / "docs/GRAIN_QUERY_COORDINATE_INTERVENTION_M24.md"


def prepare(out, readers):
    """Freeze complete same-author cube membership and readers before new calls."""
    _, _, lm, hashes = load_panel()
    if read(readers / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M23 readers incomplete")
    design = [
        d
        for d in read(LEVEL / "private_design.json")
        if d["query"] in ("TF_cont", "neuroticism")
    ]
    authors = {d["author_id"] for d in design}
    for query in ("TF_cont", "neuroticism"):
        for family in FAMILIES:
            for fold in range(5):
                p = readers / "private_models" / f"{query}_{family}_{fold}.joblib"
                a = joblib.load(p)
                if (
                    set(a["train_authors"]) & authors
                    or a["question_tokens"] != lm["question_tokens"][query]
                ):
                    raise ValueError("Reader membership or template mismatch")
                hashes[str(p.relative_to(ROOT))] = sha(p)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COORDINATE_CUBE_M24_V1",
            "plan_sha256": sha(PLAN),
            "script_sha256": sha(__file__),
            "cube_module_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/coordinate_cube.py"
            ),
            "input_sha256": sha(out / "private_design.json"),
            "model_sha256": lm["model_sha256"],
            "units": 24,
            "max_calls": 192,
            "max_seconds": 900,
            "readout_dir": str(readers.relative_to(ROOT)),
            "private_input_hashes": hashes,
            "human_labels_read": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )


def run(out):
    """Replay full states and run seven cached-displacement suffixes per unit."""
    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    for path, digest in m["private_input_hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen model or material changed")
    if sha(__file__) != m["script_sha256"] or sha(PLAN) != m["plan_sha256"]:
        raise ValueError("Frozen implementation changed")
    if (
        sha(ROOT / "grain_research/vector_mechanisms/coordinate_cube.py")
        != m["cube_module_sha256"]
    ):
        raise ValueError("Frozen cube construction changed")
    if (
        sha(out / "private_design.json") != m["input_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Design or observer checkpoint changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if not compatible_observers(observer.manifest(), read(LEVEL / "observer.json")):
        raise ValueError("Observer configuration changed")
    write(out / "observer.json", observer.manifest())
    start, complete = time.monotonic(), 0

    def status(state):
        """Report actual costs, including completed prefixes and retained failed work."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": complete,
            "planned_units": m["units"],
            "human_labels_read": False,
            "gradients": 0,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for d in read(out / "private_design.json"):
            stem = f"unit_{d['index']:02d}_{d['query']}"
            dest, old = (
                out / "private_packets" / f"{stem}.json",
                LEVEL / "private_packets" / f"{stem}.npz",
            )
            if dest.exists():
                if read(dest)["input_array_sha256"] != sha(old):
                    raise ValueError("Retained unit changed")
                complete += 1
                continue
            if (
                prior["calls"] + observer.calls + 8 > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            with np.load(old, allow_pickle=False) as z:
                cube = query_cube(
                    z["full_P"], z["full_Q"], z["edited_P"], z["edited_Q"]
                )
                state_expected = z["full_state"]
            state, kwargs, native = capture_boundary(
                observer, d["full_ids"], d["alphabet"]
            )
            state_error = float(
                np.max(np.abs(state[0].detach().cpu().numpy() - state_expected))
            )
            if (
                state_error > 2e-5
                or abs(native["conditional_score"] - d["source_full"]) > 2e-6
            ):
                raise ValueError("Full-state replay mismatch")
            values = {"000": native["conditional_score"]}
            mask = np.asarray(d["full_mask"], bool)
            if mask[-1]:
                raise ValueError("P and Q regions must be disjoint")
            for name, (p, q) in cube.items():
                if name == "000":
                    continue
                changed = state.clone()
                changed[:, mask] = torch.as_tensor(
                    p, dtype=state.dtype, device=state.device
                )
                changed[:, -1] = torch.as_tensor(
                    q, dtype=state.dtype, device=state.device
                )
                values[name] = replay_suffix(observer, changed, kwargs, d["alphabet"])[
                    "conditional_score"
                ]
            predicted = {}
            for family in FAMILIES:
                row = {}
                for fold in range(5):
                    artifact = joblib.load(
                        ROOT
                        / m["readout_dir"]
                        / "private_models"
                        / f"{d['query']}_{family}_{fold}.joblib"
                    )
                    r = artifact["reader"]
                    for name, (p, q) in cube.items():
                        row.setdefault(name, []).append(
                            float(r.predict(p[None], q[None])[0])
                        )
                predicted[family] = {name: float(np.mean(v)) for name, v in row.items()}
            write(
                dest,
                {
                    "index": d["index"],
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "values": values,
                    "effects": anchored_effects(values),
                    "reader_values": predicted,
                    "natural_delta": d["source_edit"] - d["source_full"],
                    "state_error": state_error,
                    "input_array_sha256": sha(old),
                    "calls": 8,
                },
            )
            complete += 1
            if complete % 4 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def analyze(out):
    """Keep conditional block effects separate from complete natural-text changes."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Cube run incomplete")
    rows = []
    for path in sorted((out / "private_packets").glob("unit_*.json")):
        d = read(path)
        for model, values in {"native": d["values"], **d["reader_values"]}.items():
            effects = anchored_effects(values)
            rows.append(
                {
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "model": model,
                    "conditional_delta": values["111"] - values["000"],
                    "natural_delta": d["natural_delta"],
                    **effects,
                }
            )
    frame = pd.DataFrame(rows)
    metrics, components = [], []
    for query, g in frame.groupby("query"):
        teacher = g[g.model == "native"].set_index("author_id")
        for model, part in g.groupby("model"):
            part = part.set_index("author_id").loc[teacher.index]
            for effect in ("conditional_delta", "mean", "configuration", "final"):
                metrics.append(
                    {
                        "query": query,
                        "model": model,
                        "effect": effect,
                        **paired_metrics(
                            teacher[effect].to_numpy(), part[effect].to_numpy()
                        ),
                    }
                )
        for effect in (
            "mean",
            "configuration",
            "final",
            "mean_configuration",
            "mean_final",
            "configuration_final",
            "triple",
        ):
            components.append(
                {
                    "query": query,
                    "effect": effect,
                    **paired_metrics(
                        teacher.conditional_delta.to_numpy(), teacher[effect].to_numpy()
                    ),
                }
            )
        components.append(
            {
                "query": query,
                "effect": "conditional_to_natural",
                **paired_metrics(
                    teacher.natural_delta.to_numpy(),
                    teacher.conditional_delta.to_numpy(),
                ),
            }
        )
    frame.to_csv(out / "private_effects.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "reader_effect_metrics.csv", index=False)
    pd.DataFrame(components).to_csv(out / "native_component_metrics.csv", index=False)
    interaction = ["mean_configuration", "mean_final", "configuration_final", "triple"]
    max_affine = float(
        np.max(np.abs(frame.loc[frame.model != "native", interaction].to_numpy()))
    )
    error = float(
        np.max(
            np.abs(
                frame.conditional_delta
                - frame[["mean", "configuration", "final", *interaction]].sum(axis=1)
            )
        )
    )
    write(
        out / "summary.json",
        {
            **read(out / "run_status.json"),
            "max_affine_interaction": max_affine,
            "accounting_error": error,
            "authors": 12,
            "queries": 2,
        },
    )
    print(pd.DataFrame(components).to_string(index=False))


def main():
    """Prepare, acquire or analyze the finite intervention packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--readout-dir", type=Path, default=BASE / "conditional_coordinates_m23_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use a private result directory")
    if args.stage == "prepare":
        prepare(out, args.readout_dir.resolve())
    elif args.stage == "run":
        run(out)
    else:
        analyze(out)


if __name__ == "__main__":
    main()
