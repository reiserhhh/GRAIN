#!/usr/bin/env python3
"""NEXT33: construct one four-source complete conditional pair function."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_pair import ARMS, BLOCKS
from grain_research.vector_mechanisms.joint_readout import centered_logits
from grain_research.vector_mechanisms.kernel_action import training_geometry
from grain_research.vector_mechanisms.query_pair import (
    fit_query_pair,
    predict_query_pair,
)
from scripts.run_grain_complete_pair import contexts, endpoints
from scripts.run_grain_conditional_energy import PARENT, features, source_inputs
from scripts.run_grain_joint_carrier_readout import OLD, PROJECTION, check, verify
from scripts.run_grain_query_baseline import verified as verified_baseline
from scripts.run_grain_real_author_path import BASE, read, sha, write

PLAN = ROOT / "docs/GRAIN_QUERY_DIRECTED_PAIR_ASSEMBLY_PLAN_V1.md"
BASELINE = BASE / "logic_query_baseline_v1"


def membership(authors, design):
    """Align source roles by both author and index, never inherited CV fold labels."""
    roles = {(str(r["author_id"]), r["index"]): r["role"] for r in design}
    keys = [(str(r["author_id"]), r["index"]) for r in authors]
    if len(keys) != 24 or len(set(keys)) != 24 or set(keys) != set(roles):
        raise ValueError("Exactly the same 24 baseline authors required")
    if set(roles.values()) != {"reference", "evaluation"}:
        raise ValueError("Unexpected source role")
    train = np.array([roles[k] == "reference" for k in keys])
    if train.sum() != 4:
        raise ValueError("Four fixed sources required")
    return train


def source_geometry(authors, packets, train):
    """Freeze within-comment deviation scales using four source authors/three views."""
    means, variance, keep = [], [], []
    for r, yes in zip(authors, train, strict=True):
        for v in range(3):
            z = packets[r["index"]]
            means.append(z[f"c_{v}"].mean(0))
            variance.append((z[f"d_{v}"] ** 2).mean(0))
            keep.append(yes)
    return training_geometry(means, variance, np.asarray(keep))


def construct(name):
    """Freeze five numerical fits and endpoint-denied predictions before evaluation."""
    verify(PARENT)
    verified_baseline(BASELINE)
    bf = read(BASELINE / "numerical/prediction_freeze.json")
    check(BASELINE / "source_freeze.json", bf["source_freeze_sha256"])
    for name0, digest in bf["files"].items():
        check(BASELINE / "numerical" / name0, digest)
    out = BASE / name
    out.mkdir(exist_ok=False)
    code = [PLAN, Path(__file__), ROOT / "scripts/replay_grain_query_pair.py"]
    code += [
        ROOT / f"grain_research/vector_mechanisms/{p}.py"
        for p in ("query_pair", "complete_pair", "joint_readout", "kernel_action")
    ]
    code += [
        ROOT / f"scripts/{p}.py"
        for p in ("run_grain_complete_pair", "run_grain_conditional_energy")
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_DIRECTED_PAIR_V1",
            "new_neural_calls": 0,
            "human_labels_read": False,
            "penalty": 0.01,
            "baseline_freeze_sha256": sha(
                BASELINE / "numerical/prediction_freeze.json"
            ),
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
        },
    )
    design = read(BASELINE / "private_design.json")
    pairs, packets, eta = source_inputs()
    authors = pairs[::2]
    train = membership(authors, design)
    blocks, signature = contexts(authors)
    # Reconstruct the old random draw without fitting on any author.
    queries = sorted({r["query"] for r in read(OLD / "private_design.json")})
    rng = np.random.default_rng(20260910 + queries.index("neuroticism"))
    w = rng.normal(size=(1536, 32)) / np.sqrt(1536)
    eta0 = rng.normal(size=32)
    with np.load(PROJECTION / "private_directions.npz") as z:
        if not np.array_equal(z["neuroticism__W"], w) or not np.array_equal(eta, eta0):
            raise ValueError("Original data-independent random projection changed")
    center, scale = source_geometry(authors, packets, train)
    phi = {}
    for arm in ARMS:
        a, b = features(
            pairs, packets, eta, {"center": center, "geometry_scale": scale, "arm": arm}
        )
        if not np.array_equal(a[::2], a[1::2]):
            raise ValueError("Comment0 repeated features differ")
        phi[arm] = np.stack([a[::2], b[::2], b[1::2]], 1)
    sources = [r for r, yes in zip(authors, train, strict=True) if yes]
    z0, increments = endpoints(sources)
    mixed, values = [], None
    for r in sources:
        p = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if values is not None and values != p["alphabet"]["values"]:
            raise ValueError("Source alphabet changed")
        values = p["alphabet"]["values"]
        s = p["scores"]
        mixed.append(
            [
                centered_logits(
                    np.array(s[n]["candidate_logits"])
                    - s[f"s{i}"]["candidate_logits"]
                    - np.array(s[f"s{j}"]["candidate_logits"])
                    + s["zero"]["candidate_logits"]
                )
                for n, i, j in (("adjacent", 0, 1), ("distant", 0, 11))
            ]
        )
    with np.load(BASELINE / "numerical/private_model.npz") as z:
        affine = {k: z[k] for k in ("intercept", "jm", "jr")}
    model = fit_query_pair(
        {k: v[train] for k, v in blocks.items()},
        {k: v[train] for k, v in phi.items()},
        z0,
        increments,
        np.array(mixed),
        affine,
        values,
    )
    inputs = {k: v[~train] for k, v in blocks.items()} | {
        k: v[~train] for k, v in phi.items()
    }
    predicted = predict_query_pair(
        {k: inputs[k] for k in BLOCKS}, {k: inputs[k] for k in ARMS}, model
    )
    paths = {
        key: out / f"private_{key}.npz"
        for key in ("model", "inputs", "predictions", "replay")
    }
    np.savez_compressed(paths["model"], **model)
    np.savez_compressed(paths["inputs"], **inputs)
    np.savez(paths["predictions"], **predicted)
    np.savez(out / "private_geometry.npz", center=center, scale=scale, eta=eta, w=w)
    forbidden = [
        OLD
        / "private_packets"
        / f"author_{authors[np.flatnonzero(~train)[0]]['index']:03d}.json",
        PARENT / "private_joint_predictions.json",
        paths["predictions"],
    ]
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_grain_query_pair.py"),
        "--model",
        str(paths["model"]),
        "--inputs",
        str(paths["inputs"]),
        "--output",
        str(paths["replay"]),
    ]
    for path in forbidden:
        command += ["--forbid", str(path)]
    replay = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=120
    )
    with np.load(paths["replay"]) as z:
        replay_error = max(float(np.max(abs(v - z[k]))) for k, v in predicted.items())
    held = [r for r, yes in zip(authors, ~train, strict=True) if yes]
    old_held = [r for r in design if r["role"] == "evaluation"]
    reorder = [
        next(i for i, s in enumerate(old_held) if s["index"] == r["index"])
        for r in held
    ]
    with np.load(BASELINE / "numerical/private_predictions.npz") as z:
        base_replay = float(
            np.max(abs(predicted["affine_base"] - z["logits"][reorder]))
        )
        kernel_replay = float(
            np.max(abs(predicted["base"] - z["kernel_logits"][reorder]))
        )
    if replay_error != 0 or max(base_replay, kernel_replay) > 1e-12:
        raise ValueError("Frozen baseline or independent runtime replay changed")
    write(
        out / "private_membership.json",
        {
            "sources": sources,
            "evaluation": held,
            "source_denial": replay.stdout.strip(),
        },
    )
    write(
        out / "construction_summary.json",
        {
            "source_authors": 4,
            "evaluation_authors": 20,
            "pairs_per_author": 2,
            "source_geometry_views": 12,
            "fits": 5,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "source_denied_replay_max": replay_error,
            "baseline_replay_max": base_replay,
            "kernel_replay_max": kernel_replay,
            "denial_probes": 3,
            "context_signature": signature,
            "input_dimensions": {k: v.shape[-1] for k, v in inputs.items()},
        },
    )
    files = list(paths.values()) + [
        out / k
        for k in (
            "private_geometry.npz",
            "private_membership.json",
            "construction_summary.json",
        )
    ]
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "files": {p.name: sha(p) for p in files},
        },
    )
    print(
        {
            "status": "FROZEN",
            "fits": 5,
            "held_authors": 20,
            "runtime_replay_max": replay_error,
        }
    )


def verified(out):
    """Reject changed code or numerical assets before evaluation."""
    m, f = read(out / "manifest.json"), read(out / "prediction_freeze.json")
    check(out / "manifest.json", f["manifest_sha256"])
    for p, digest in m["code_hashes"].items():
        check(ROOT / p, digest)
    for p, digest in f["files"].items():
        check(out / p, digest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_query_pair_v1")
    construct(p.parse_args().name)
