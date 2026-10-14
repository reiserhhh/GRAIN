#!/usr/bin/env python3
"""Readonly direct-count audit of source author laws and every cold score."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Recompute with separate author loops, cell counts and full finite sums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    out = args.out
    R.check(out, ("compiled_seal.json", "cold_seal.json", "analysis_seal.json"))
    z, law = R.prepared(out), R.read(out / "source_law.json")
    vectors, directions, venue, timestamps, _, roles = R.K.load_numeric()
    raw = np.einsum("utd,dk->utk", vectors, directions, optimize=False)
    idx = np.array([sorted(range(36), key=lambda j: (row[j], j)) for row in timestamps])
    projected = np.array([raw[a, idx[a]] for a in range(624)])
    max_error = float(np.max(np.abs(projected - z["coordinates"])))
    ordered_venue = np.array([venue[a, idx[a]] for a in range(624)])
    old = R.read(ROOT / R.BUNDLE)
    condition = np.array(
        [[int(v in old["context_rule"]["top"]) for v in row] for row in ordered_venue]
    )
    if not np.array_equal(condition, z["conditions"]):
        raise ValueError("Recomputed source condition address disagrees")
    if not np.array_equal(idx, z["original_units"]):
        raise ValueError("Chronology differs")
    pairs = []
    for a in range(624):
        for window, start in enumerate((0, 18)):
            for t in range(start, start + 17):
                if timestamps[a, idx[a, t + 1]] > timestamps[a, idx[a, t]]:
                    pairs.append((a, window, t, t + 1))
    if not np.array_equal(np.array(pairs), z["pairs"]):
        raise ValueError("Independent event pairing disagrees")
    source = set(roles["source_train"])
    source_pairs = [p for p in pairs if p[0] in source]
    source_cuts = np.median(
        [projected[a, t, [0, 5]] for a, _, t, _ in source_pairs], axis=0
    )
    max_error = max(max_error, float(np.max(np.abs(source_cuts - law["thresholds"]))))
    # Use the sealed partition to avoid a floating boundary recoding; verify raw
    # source units independently above before discrete addresses are compared.
    cut = np.array(law["thresholds"])
    state = (z["coordinates"][:, :, 0] > cut[0]).astype(int) + 2 * (
        z["coordinates"][:, :, 5] > cut[1]
    ).astype(int)
    outcome = {}
    for a, w, t, nxt in pairs:
        outcome[(a, w, t)] = np.array(
            [
                ordered_venue[a, t] == ordered_venue[a, nxt],
                z["coordinates"][a, nxt, 0] > cut[0],
                z["coordinates"][a, nxt, 5] > cut[1],
            ],
            float,
        )
    count = Counter(p[0] for p in source_pairs)
    mass, pos, supp = np.zeros((2, 4)), np.zeros((2, 4, 3)), np.zeros((2, 4), int)
    for a, w, t, _ in source_pairs:
        c, s = condition[a, t], state[a, t]
        weight = 1.0 / len(source) / count[a]
        mass[c, s] += weight
        pos[c, s] += weight * outcome[(a, w, t)]
        supp[c, s] += 1
    psi = pos / mass[:, :, None]
    mu = np.array([pos[c].sum(0) / mass[c].sum() for c in range(2)])
    ref = pos.sum(axis=(0, 1))
    if not np.array_equal(supp, law["support"]):
        raise ValueError("Source support mismatch")
    for real, saved in (
        (mass, law["mass"]),
        (pos, law["positive_mass"]),
        (psi, law["probabilities"]),
        (mu, law["condition_reference"]),
        (ref, law["reference"]),
    ):
        max_error = max(max_error, float(np.max(np.abs(real - saved))))
    cold = [
        json.loads(line)
        for line in (out / "cold_scores.jsonl").read_text().splitlines()
    ]
    inputs = [
        json.loads(line)
        for line in (out / "cold_inputs.jsonl").read_text().splitlines()
    ]
    observed = np.zeros((624, 2, 3))
    early = np.zeros((624, 3))
    score_checks = 0
    for i, (rec, got) in enumerate(zip(inputs, cold, strict=True)):
        a, window = divmod(i, 2)
        events = [p for p in pairs if p[:2] == (a, window)]
        if len(rec["events"]) != len(events) or rec["author_key"] != got["author_key"]:
            raise ValueError("Cold event/window identity disagreement")
        freq = Counter(
            (int(condition[a, t]), int(state[a, t])) for _, _, t, _ in events
        )
        pm = np.zeros((2, 4))
        for cell, n in freq.items():
            pm[cell] = n / len(events)
        means = np.sum(pm[:, :, None] * psi, axis=(0, 1))
        choice = np.sum((pm.sum(1) - mass.sum(1))[:, None] * mu, axis=0)
        within = np.sum(pm[:, :, None] * (psi - mu[:, None]), axis=(0, 1))
        for name, expected in (
            ("mean", means),
            ("relative", means - ref),
            ("choice", choice),
            ("within", within),
        ):
            max_error = max(max_error, float(np.max(np.abs(expected - got[name]))))
            score_checks += 3
        max_error = max(max_error, float(np.max(np.abs(means - ref - choice - within))))
        observed[a, window] = np.mean(
            [outcome[(a, window, t)] for _, _, t, _ in events], axis=0
        )
        if window == 0:
            early[a] = means - ref
    with np.load(out / "author_accounts.npz") as f:
        max_error = max(max_error, float(np.max(np.abs(observed - f["observed"]))))
        strata = f["strata"]
    permutation_checks = 0
    with np.load(out / "permutation_accounts.npz") as f:
        for role in ("source_train", "source_validation", "held"):
            ids = z[role]
            groups = strata[ids]
            permutations = f[role + "_indices"]
            x, y = early[ids].copy(), observed[ids, 1].copy()
            for g in sorted(set(groups)):
                ix = np.flatnonzero(groups == g)
                x[ix] -= x[ix].mean(0)
                y[ix] -= y[ix].mean(0)
            actual = np.sum(x * y, axis=0) / len(ids)
            max_error = max(
                max_error, float(np.max(np.abs(actual - f[role + "_observed"])))
            )
            for i, p in enumerate(permutations):
                if sorted(p) != list(range(len(ids))) or not np.array_equal(
                    groups, groups[p]
                ):
                    raise ValueError(
                        "Permutation broke complete author profile/stratum"
                    )
                stat = np.sum(x * y[p], axis=0) / len(ids)
                max_error = max(
                    max_error, float(np.max(np.abs(stat - f[role + "_statistics"][i])))
                )
                permutation_checks += 3
    if max_error > 1e-12:
        raise ValueError(
            "Readonly alternate construction audit failed: " + str(max_error)
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "max_error": max_error,
                "source_pair_counts": len(source_pairs),
                "cold_score_values": score_checks,
                "permutation_values": permutation_checks,
                "new_neural_forwards": 0,
                "same_agent_not_blind": True,
                "old_and_new_seals_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
