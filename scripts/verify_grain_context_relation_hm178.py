#!/usr/bin/env python3
"""Alternate actual-event enumeration audit;not an independent blind experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_context_relation_hm178 as H
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Enumerate pairs and R contacts directly,without using scoring functions."""
    out = H.DEFAULT
    spec = R.check(out, ("scored_seal.json", "analysis_seal.json"))["config"]
    z = R.prepared(out)
    b = R.read(out / "source_bundle.json")
    fields = np.asarray(b["pair_fields"], float)
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        scored, pairs = a["mass"], a["pairs"]
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        actions, native = a["actions"], a["native_log_odds_actions"]
    direct = np.zeros_like(scored)
    dc = np.zeros(pairs.shape, int)
    contacts = np.zeros_like(actions)
    enumerated = 0
    for arm, u, v in np.ndindex(2, 24, 2):
        s, c = set(z["selected"][u, v]), set(z["control"][u, v])
        rest = set(range(12)) - s - c
        groups = (set(range(12)), set(range(12)) - s, set(range(12)) - c, s, c)
        for k, g in enumerate(groups):
            for i in sorted(g):
                for j in sorted(g):
                    if i < 6 <= j:
                        a, d = z["panel_cells"][arm, u, v, [i, j]]
                        direct[arm, u, v, k] += fields[v, :, :, a, d] / 36
                        dc[arm, u, v, k] += 1
                        enumerated += 1
        for r in rest:
            for k, g in ((1, s), (-1, c)):
                for i in g:
                    if (r < 6) == (i < 6):
                        continue
                    left, right = (r, i) if r < 6 else (i, r)
                    a, d = z["panel_cells"][arm, u, v, [left, right]]
                    contacts[arm, u, v] += k * fields[v, :, :, a, d] / 36
    if not np.array_equal(np.isnan(direct), np.isnan(scored)) or not np.array_equal(
        dc, pairs
    ):
        raise ValueError("Scored support/opportunity differs from actual enumeration")
    finite = np.isfinite(direct)
    err = float(np.max(np.abs(direct[finite] - scored[finite])))
    known = np.isfinite(contacts) & np.isfinite(actions)
    contact_err = float(np.max(np.abs(contacts[known] - actions[known])))
    table = np.asarray(b["profile_tables"])
    margin_error = 0.0
    profiles = np.asarray(b["profiles"])
    source_law_error = 0.0
    for v, z0, z1 in np.ndindex(2, 2, 2):
        sl = slice(4 * z0, 4 * z0 + 4)
        sr = slice(4 * z1, 4 * z1 + 4)
        f, h = table[v, 2, :, sl, sr], table[v, 1, :, sl, sr]
        for axis in (0, 1, 2):
            margin_error = max(
                margin_error, float(np.max(np.abs(f.sum(axis) - h.sum(axis))))
            )
    p, y = z["source_mass"], z["source_targets"]
    observed_pairs = 0
    for v, c in np.ndindex(2, len(profiles)):
        take = np.all(y == profiles[c], axis=1)
        matrix = np.zeros((8, 8))
        for u in np.flatnonzero(take):
            for a, d in np.ndindex(8, 8):
                matrix[a, d] += p[u, v, 0, a] * p[u, v, 1, d] / len(p)
            observed_pairs += 36
        source_law_error = max(
            source_law_error, float(np.max(np.abs(matrix - table[v, 2, c])))
        )
    par = ROOT / spec["native_parent"]
    with np.load(par / "composition_accounts.npz", allow_pickle=False) as a:
        logits = a["native_logits"]
    nd = np.empty_like(native)
    for u, v, q in np.ndindex(24, 2, 9):
        x = logits[u, v, :, q + 1] - logits[u, v, :, 0]
        nd[u, v, q] = (x[2] - x[1]) - (x[3] - x[4])
    native_error = float(np.max(np.abs(nd - native)))
    if (
        max(err, contact_err, source_law_error, native_error) > 2e-12
        or margin_error > 1.1e-10
    ):
        raise ValueError("Finite scientific account failed")
    receipt = {
        "enumerated_panel_pairs": enumerated,
        "source_pairs": observed_pairs,
        "cold_values": int(scored.size),
        "source_law_error": source_law_error,
        "cold_value_error": err,
        "contact_identity_error": contact_err,
        "source_margin_error": margin_error,
        "native_nine_axis_error": native_error,
        "same_agent_not_blind": True,
        "new_neural_calls": 0,
    }
    R.write(out / "alternate_audit.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
