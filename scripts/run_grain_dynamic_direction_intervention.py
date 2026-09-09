#!/usr/bin/env python3
"""One source-selected dynamic direction, 12 held authors, 60 native calls."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_path_routes import split_direction
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
)
from scripts.run_grain_dynamic_path_channels import PLAN, PRIOR, SEED, SOURCE
from scripts.run_grain_real_author_path import MODEL


def prepare(out):
    """Freeze transition directions and exact query-token/checkpoint identities."""
    if out.exists():
        raise ValueError("Do not overwrite the intervention packet")
    _, _, manifest, hashes = load_panel()
    source_meta = read(SOURCE / "manifest.json")
    if source_meta["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("Source/native model identity mismatch")
    grains = pd.read_csv(PRIOR / "private_factor_views.csv")
    allowed = set(grains.author_id)
    design = sorted(
        [
            d
            for d in read(SOURCE / "private_design.json")
            if d["query"] == "TF_cont" and d["author_id"] in allowed
        ],
        key=lambda d: d["author_id"],
    )
    ids = [d["author_id"] for d in design]
    native = [d for d in read(LEVEL / "private_design.json") if d["query"] == "TF_cont"]
    if (
        len(ids) != 63
        or len(native) != 12
        or set(ids) & {d["author_id"] for d in native}
    ):
        raise ValueError("Source/native authors are not the expected disjoint panels")
    x = np.diff(
        grains[grains.author_id.isin(ids)]
        .pivot(index="author_id", columns="view", values="whole:F02")
        .loc[ids]
        .to_numpy(),
        axis=1,
    )
    x -= x.mean(0)
    sd = x.std()
    x /= sd
    states = []
    for d in design:
        for v in d["views"]:
            if (
                list(np.asarray(v["input_ids"])[np.asarray(v["question_mask"], bool)])
                != manifest["question_tokens"]["TF_cont"]
            ):
                raise ValueError("Literal source/native query-token mismatch")
        p = SOURCE / f"private_packets/unit_{d['index']:04d}.npz"
        if sha(p) != read(p.with_suffix(".json"))["array_sha256"]:
            raise ValueError("Source state packet changed")
        hashes[str(p.relative_to(ROOT))] = sha(p)
        with np.load(p) as z:
            pshape, qshape = z["P"].shape[1:], z["Q"].shape[1:]
            states.append(np.diff(np.c_[z["P"].reshape(3, -1), z["Q"]], axis=0))
    y = np.asarray(states, float)
    y -= y.mean(0)
    denominator = np.sum(x * x) + 100
    real = np.einsum("nt,ntd->d", x, y) / denominator
    shuffled = x[np.random.default_rng(SEED).permutation(len(x))]
    null = np.einsum("nt,ntd->d", shuffled, y) / denominator
    if min(np.linalg.norm(real), np.linalg.norm(null)) < 1e-12:
        raise ValueError("Degenerate direction")
    null *= np.linalg.norm(real) / np.linalg.norm(null)
    out.mkdir(parents=True)
    np.savez_compressed(out / "private_directions.npz", real=real, null=null)
    write(out / "private_design.json", native)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_DYNAMIC_DIRECTION_NATIVE_V1",
            "query": "TF_cont",
            "factor": "whole:F02",
            "source_authors": 63,
            "source_transitions": 126,
            "evaluation_authors": 12,
            "alpha": 100,
            "amplitude": 0.25,
            "max_calls": 60,
            "max_seconds": 600,
            "change_scale": float(sd),
            "direction_norm": float(np.linalg.norm(real)),
            "P_shape": pshape,
            "Q_shape": qshape,
            "human_label_fit": False,
            "selection": "exploratory source co-change; native authors reused from earlier work",
            "source_sha256": hashes,
            "direction_sha256": sha(out / "private_directions.npz"),
            "design_sha256": sha(out / "private_design.json"),
            "model_sha256": manifest["model_sha256"],
            "plan_sha256": sha(PLAN),
            "code_sha256": sha(Path(__file__)),
        },
    )


def run(out):
    """Patch only query and final states; retain local native model and all text."""
    m = read(out / "manifest.json")
    for name, digest in m["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen source changed")
    if (
        sha(MODEL / "model.safetensors") != m["model_sha256"]
        or sha(out / "private_directions.npz") != m["direction_sha256"]
        or sha(out / "private_design.json") != m["design_sha256"]
    ):
        raise ValueError("Native inputs changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if not compatible_observers(observer.manifest(), read(LEVEL / "observer.json")):
        raise ValueError("Native observer differs")
    write(out / "observer.json", observer.manifest())
    with np.load(out / "private_directions.npz") as z:
        directions = {k: z[k].copy() for k in z.files}
    start, rows, errors = time.monotonic(), [], []
    for d in read(out / "private_design.json"):
        if (
            time.monotonic() - start > m["max_seconds"]
            or observer.calls + 5 > m["max_calls"]
        ):
            write(
                out / "status.json",
                {"status": "PAUSED_RESOURCE_BOUNDARY", "calls": observer.calls},
            )
            return
        state, kwargs, native = capture_boundary(observer, d["full_ids"], d["alphabet"])
        stem = f"unit_{d['index']:02d}_TF_cont"
        with np.load(LEVEL / f"private_packets/{stem}.npz") as z:
            error = float(
                np.max(np.abs(state[0].detach().cpu().numpy() - z["full_state"]))
            )
        if error > 2e-5 or abs(native["conditional_score"] - d["source_full"]) > 2e-6:
            raise ValueError("Native state/score does not replay")
        errors.append(error)
        mask = np.asarray(d["full_mask"], bool)
        if mask[-1]:
            raise ValueError("P and Q overlap")
        p, q = (
            state[0, mask].detach().cpu().numpy(),
            state[0, -1].detach().cpu().numpy(),
        )
        for arm, direction in directions.items():
            dp, dq = split_direction(direction * m["amplitude"], p.shape, q.shape)
            scores = []
            for sign in (-1, 1):
                changed = state.clone()
                changed[:, mask] = torch.as_tensor(
                    p + sign * dp, dtype=state.dtype, device=state.device
                )
                changed[:, -1] = torch.as_tensor(
                    q + sign * dq, dtype=state.dtype, device=state.device
                )
                scores.append(
                    replay_suffix(observer, changed, kwargs, d["alphabet"])[
                        "conditional_score"
                    ]
                )
            rows.append(
                {
                    "author_id": d["author_id"],
                    "arm": arm,
                    "minus": scores[0],
                    "plus": scores[1],
                    "effect": (scores[1] - scores[0]) / 2,
                    "native": native["conditional_score"],
                }
            )
        pd.DataFrame(rows).to_csv(out / "private_effects.csv", index=False)
        print(
            f"Dynamic intervention {len(rows) // 2}/12, calls={observer.calls}",
            flush=True,
        )
    frame = pd.DataFrame(rows).pivot(index="author_id", columns="arm", values="effect")
    delta = frame.real.abs().to_numpy() - frame["null"].abs().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = delta[rng.integers(0, len(delta), (4999, len(delta)))].mean(1)
    write(
        out / "summary.json",
        {
            "authors": len(frame),
            "mean_signed_real": float(frame.real.mean()),
            "mean_abs_real": float(frame.real.abs().mean()),
            "mean_abs_null": float(frame["null"].abs().mean()),
            "abs_ratio": float(frame.real.abs().mean() / frame["null"].abs().mean()),
            "paired_abs_gain": float(delta.mean()),
            "gain_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "real_positive": int((frame.real > 0).sum()),
            "real_negative": int((frame.real < 0).sum()),
            "max_state_replay_error": max(errors),
            "null_directions": 1,
        },
    )
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    output = BASE / "dynamic_path_channels_v1/native_direction"
    with threadpool_limits(limits=1):
        prepare(output)
    run(output)
