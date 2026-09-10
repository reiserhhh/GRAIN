#!/usr/bin/env python3
"""NEXT26: freeze complete pair predictions before endpoint evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.comment_carrier import partition
from grain_research.vector_mechanisms.complete_pair import (
    ARMS,
    BLOCKS,
    fit_complete,
    predict_complete,
)
from grain_research.vector_mechanisms.joint_readout import centered_logits
from scripts.run_grain_conditional_energy import PARENT, features, source_inputs
from scripts.run_grain_joint_carrier_readout import OLD, STATES, check, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write

PLAN = ROOT / "docs/GRAIN_COMPLETE_PAIR_INTERFACE_PLAN_V1.md"


def contexts(authors):
    """Extract all means, retained states and layout without reading source text/Y."""
    designs = {r["index"]: r for r in read(OLD / "private_design.json")}
    blocks = {k: [] for k in BLOCKS}
    signature = None
    for row in authors:
        view = designs[row["index"]]["views"][0]
        g, gaps, _, rest = partition(view["groups"], view["p_mask"])
        with np.load(STATES / f"author_{row['index']:03d}.npz") as z:
            h = z["view_0"].astype(float)
        sig = (len(g), tuple(map(len, gaps)), sum(view["p_mask"]), h.shape[1])
        if signature is not None and sig != signature:
            raise ValueError("Context role grid changed; no silent interpolation")
        signature = sig
        blocks["M"].append(np.stack([h[x].mean(0) for x in g]).ravel())
        blocks["R"].append(h[rest].ravel())
        layout = [a for x in g for a in (x[0], x[-1] + 1)]
        layout += list(rest) + [
            np.flatnonzero(view["p_mask"])[0],
            sum(view["p_mask"]),
            len(h),
        ]
        blocks["L"].append(layout)
    return {k: np.array(v) for k, v in blocks.items()}, signature


def endpoints(authors):
    """Read model teacher endpoints for source-fitting, not questionnaire labels."""
    z0, inc = [], []
    for r in authors:
        packet = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if (
            str(packet["author_id"]) != r["author_id"]
            or packet["query"] != "neuroticism"
        ):
            raise ValueError("Endpoint author/query mismatch")
        s = packet["scores"]
        zero = centered_logits(s["zero"]["candidate_logits"])
        z0.append(zero)
        inc.append(
            [centered_logits(s[f"s{i}"]["candidate_logits"]) - zero for i in (0, 1, 11)]
        )
    return np.array(z0), np.array(inc)


def source_partition(authors, fold):
    """Select source authors before fitting, requiring complete disjoint folds."""
    if len({r["author_id"] for r in authors}) != len(authors):
        raise ValueError("Author rows must be unique")
    train = np.array([r["fold"] != fold for r in authors])
    if train.sum() != 18 or (~train).sum() != 6:
        raise ValueError("Inherited 18/6 author split required")
    return train


def construct(name):
    """Build exactly 12 small source-only heads and replay with endpoint files denied."""
    verify(PARENT)
    for path, digest in read(PARENT / "joint_freeze.json")["files"].items():
        check(PARENT / path, digest)
    out = BASE / name
    out.mkdir(exist_ok=False)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "scripts/replay_grain_complete_pair.py",
        ROOT / "grain_research/vector_mechanisms/complete_pair.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPLETE_CONDITIONAL_PAIR_V1",
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "old_freeze_sha256": sha(PARENT / "joint_freeze.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "new_neural_calls": 0,
            "human_labels_read": False,
            "penalty": 0.01,
            "target": "fixed-PQ conditional N pair, not full text or questionnaire",
        },
    )
    pairs, packets, eta = source_inputs()
    authors = pairs[::2]
    blocks, signature = contexts(authors)
    old_models = {
        (r["fold"], r["arm"]): r for r in read(PARENT / "private_joint_models.json")
    }
    feature_replay, runtime_replay = 0.0, 0.0
    files, audit = [], []
    for fold in range(4):
        train = source_partition(authors, fold)
        # Native targets are loaded only for this fold's source-author membership.
        z0, increments = endpoints(
            [r for r, yes in zip(authors, train, strict=True) if yes]
        )
        first_source = authors[np.flatnonzero(train)[0]]
        values = read(
            OLD / "private_packets" / f"author_{first_source['index']:03d}.json"
        )["alphabet"]["values"]
        phi = {}
        for arm in ARMS:
            old = old_models[fold, arm]
            a, b = features(pairs, packets, eta, old)
            if not np.array_equal(a[::2], a[1::2]):
                raise ValueError("Repeated comment0 features differ")
            phi[arm] = np.stack([a[::2], b[::2], b[1::2]], 1)
            pair_train = np.repeat(train, 2)
            feature_replay = max(
                feature_replay,
                float(np.max(abs(a[pair_train] - old["left"]))),
                float(np.max(abs(b[pair_train] - old["right"]))),
            )
        model = fit_complete(
            {k: b[train] for k, b in blocks.items()},
            {a: p[train] for a, p in phi.items()},
            z0,
            increments,
            {a: old_models[fold, a] for a in ARMS},
            values=values,
        )
        inputs = {k: b[~train] for k, b in blocks.items()} | {
            a: p[~train] for a, p in phi.items()
        }
        predicted = predict_complete(
            {k: inputs[k] for k in BLOCKS}, {a: inputs[a] for a in ARMS}, model
        )
        paths = {
            key: out / f"private_fold{fold}_{key}.npz"
            for key in ("model", "inputs", "predictions", "replay")
        }
        np.savez_compressed(paths["model"], **model)
        np.savez_compressed(paths["inputs"], **inputs)
        np.savez(paths["predictions"], **predicted)
        forbidden = [
            OLD
            / "private_packets"
            / f"author_{authors[np.flatnonzero(~train)[0]]['index']:03d}.json",
            PARENT / "private_joint_predictions.json",
            paths["predictions"],
        ]
        command = [
            sys.executable,
            str(ROOT / "scripts/replay_grain_complete_pair.py"),
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
            command, check=True, capture_output=True, text=True, timeout=120
        )
        with np.load(paths["replay"]) as z:
            for k, v in predicted.items():
                runtime_replay = max(runtime_replay, float(np.max(abs(v - z[k]))))
        audit.append(
            {
                "fold": fold,
                "source": [
                    r["author_id"] for r, yes in zip(authors, train, strict=True) if yes
                ],
                "held": [
                    r["author_id"]
                    for r, yes in zip(authors, ~train, strict=True)
                    if yes
                ],
                "source_denial": replay.stdout.strip(),
                "constant_blocks": [k for k in BLOCKS if bool(model[f"{k}_constant"])],
            }
        )
        files += list(paths.values())
        print(
            {
                "fold": fold,
                "status": "FROZEN_AND_SOURCE_DENIED_REPLAYED",
                "fit_authors": 18,
                "held_authors": 6,
            },
            flush=True,
        )
    if feature_replay > 1e-14 or runtime_replay != 0:
        raise ValueError("Input lineage or numerical runtime replay changed")
    write(out / "private_membership.json", {"authors": authors, "folds": audit})
    write(
        out / "construction_summary.json",
        {
            "authors": 24,
            "pairs": 48,
            "arms": 2,
            "fits": 12,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "context_signature": signature,
            "block_dimensions": {k: v.shape[1] for k, v in blocks.items()},
            "old_feature_replay_max": feature_replay,
            "source_denied_replay_max": runtime_replay,
            "denial_probes": 12,
            "constant_blocks_by_fold": [r["constant_blocks"] for r in audit],
        },
    )
    files += [out / "private_membership.json", out / "construction_summary.json"]
    write(
        out / "prediction_freeze.json",
        {
            "files": {p.name: sha(p) for p in files},
            "manifest_sha256": sha(out / "manifest.json"),
        },
    )


def verified(out):
    """Check construction code and all frozen numerical outputs before analysis."""
    m, f = read(out / "manifest.json"), read(out / "prediction_freeze.json")
    check(out / "manifest.json", f["manifest_sha256"])
    for p, digest in m["code_hashes"].items():
        check(ROOT / p, digest)
    for p, digest in f["files"].items():
        check(out / p, digest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_complete_pair_v1")
    construct(p.parse_args().name)
