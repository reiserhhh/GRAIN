#!/usr/bin/env python3
"""Alternate incidence/count audit of SAME-evidence directed trait measurements."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_trait_path_hm172 as R


def alternate_parts(array):
    """Solve incidence least squares, separately from the closed-form scorer."""
    f = np.asarray(array, float)
    edges = list(itertools.combinations(range(8), 2))
    incidence = np.zeros((8, len(edges)))
    for k, (i, j) in enumerate(edges):
        incidence[i, k], incidence[j, k] = 1, -1
    symmetric, odd = (f + f.swapaxes(0, 1)) / 2, (f - f.swapaxes(0, 1)) / 2
    edge_values = np.asarray([odd[i, j] for i, j in edges])
    potential = np.linalg.lstsq(incidence.T, edge_values, rcond=None)[0]
    gradient = potential[:, None] - potential[None, :]
    return symmetric, gradient, odd - gradient


def alternative_bag(path):
    """Enumerate each ordered internal identity pair, not matrix count products."""
    b = np.zeros((8, 8))
    internal = path[1:-1]
    for i, a in enumerate(internal):
        b[path[0], a] += 1
        b[a, path[-1]] += 1
        for j, other in enumerate(internal):
            if i != j:
                b[a, other] += 1
    return b / (len(internal) * (len(path) - 1))


def main():
    """Replay source laws, all cold components and all fixed-end null values."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=R.DEFAULT)
    a = p.parse_args()
    manifest = R.check(a.out, R.STAGES)
    z = R.observations(a.out)
    _, parent = R.parent(manifest["config"])
    labels = R.P.labels(parent["config"])
    src = z["source_train"]
    node = R.R.read(R.ROOT / manifest["config"]["parent"] / "source_law.json")
    binary = (labels[src] > node["label_cutoffs"]).astype(int)
    perm = R.R.read(a.out / "source_diagnostics.json")[
        "same_HM171_source_label_permutation"
    ]
    error, count = 0.0, 0

    def compare(x, y):
        nonlocal error, count
        x, y = np.asarray(x, float), np.asarray(y, float)
        error = max(error, float(np.max(np.abs(x - y))))
        count += x.size

    bags = np.asarray(
        [[alternative_bag(row) for row in author] for author in z["paths"]]
    )
    compare(bags, z["bag"])
    for name in manifest["config"]["readers"]:
        b = R.R.read(a.out / f"{name}.json")
        mass = (z["bag"] if name == "bag_endpoint_source" else z["joint"])[src].mean(1)
        y = binary[perm] if name == "permuted_source" else binary
        ref = sum(mass) / len(src)
        joint = sum(mass[u, :, :, None] * y[u] for u in range(len(src))) / len(src)
        compare(ref, b["reference"])
        compare(joint / ref[:, :, None], b["field"])
        compare(y.mean(0), b["prior"])
        fs, fg, fc = alternate_parts(b["field"])
        saved = R.score_arrays(a.out, name)
        for u in range(624):
            for w in range(2):
                pth = z["paths"][u, w]
                empirical = np.zeros((8, 8))
                for i, j in itertools.pairwise(pth):
                    empirical[i, j] += 1 / 17
                compare(empirical, z["joint"][u, w])
                bg, delta = bags[u, w] - ref, empirical - bags[u, w]
                ms, mg, mc = alternate_parts(empirical - ref)
                bs, bgrad, bc = alternate_parts(bg)
                ds, dg, dc = alternate_parts(delta)
                c = lambda x, f: np.sum(x[:, :, None] * f, axis=(0, 1))
                values = {
                    "relative": c(empirical - ref, np.asarray(b["field"])),
                    "bag_expected": c(bg, np.asarray(b["field"])),
                    "reorder": c(delta, np.asarray(b["field"])),
                    "traffic": c(ms, fs),
                    "endpoint": c(mg, fg),
                    "cycle": c(mc, fc),
                    "bag_traffic": c(bs, fs),
                    "bag_endpoint": c(bgrad, fg),
                    "bag_cycle": c(bc, fc),
                    "reorder_traffic": c(ds, fs),
                    "reorder_endpoint": c(dg, fg),
                    "reorder_cycle": c(dc, fc),
                    "old_relative": np.mean(
                        np.asarray(b["node_field"])[pth[:-1]], axis=0
                    )
                    - b["node_prior"],
                }
                for key, value in values.items():
                    compare(value, saved[key][u, w])
    law = R.R.read(a.out / "ordered_source.json")
    fc = alternate_parts(law["field"])[2]
    expected_cycle = np.sum(bags[..., None] * fc, axis=(2, 3))
    expected_full = np.sum(bags[..., None] * law["field"], axis=(2, 3))
    rng = np.random.default_rng(manifest["config"]["seed"])
    with np.load(a.out / "null_scores.npz", allow_pickle=False) as f:
        null = f["scores"]
    for k in range(manifest["config"]["permutations"]):
        order = np.argsort(rng.random(z["paths"][:, :, 1:-1].shape), axis=-1)
        inner = np.take_along_axis(z["paths"][:, :, 1:-1], order, axis=-1)
        pth = np.concatenate(
            (z["paths"][:, :, :1], inner, z["paths"][:, :, -1:]), axis=-1
        )
        compare(
            fc[pth[:, :, :-1], pth[:, :, 1:]].mean(2) - expected_cycle, null[k, ..., 0]
        )
        compare(
            np.asarray(law["field"])[pth[:, :, :-1], pth[:, :, 1:]].mean(2)
            - expected_full,
            null[k, ..., 1],
        )
    R.check(a.out, R.STAGES)
    print(
        json.dumps(
            {
                "status": "PASS" if error < 2e-13 else "FAIL",
                "alternate_scalar_values": count,
                "max_error": error,
                "same_agent_not_blind": True,
                "new_neural_forwards": 0,
                "all_old_and_new_seals_unchanged": True,
            }
        )
    )
    if error >= 2e-13:
        raise ValueError("Alternate measurement mismatch;retain failure")


if __name__ == "__main__":
    main()
