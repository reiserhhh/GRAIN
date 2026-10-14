#!/usr/bin/env python3
"""Replay frozen SAME-event accounts once using direct sums and source counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_grain_event_trait_action_hm173 as H
from scripts import run_grain_trait_anchor_hm171 as T


def main():
    """Perform read-only algebraic verification, not another scientific trial."""
    out = H.DEFAULT
    m = H.check(out, ("compiled_seal.json", "analysis_seal.json"))
    z = H.load(out)
    if set(z["source_indices"]) & set(z["application_indices"]):
        raise ValueError("Application authors leaked into source meaning")
    laws = [H.R.read(out / name) for name in ("source_law.json", "permuted_law.json")]
    y = T.labels(m["old_source_config"])[z["source_indices"]]
    binary = (y > np.median(y, axis=0)).astype(float)
    with np.load(out / "source_permutation.npz", allow_pickle=False) as p:
        shuffled = binary[p["order"]]
    source_error = 0.0
    for b, labels in zip(laws, (binary, shuffled), strict=True):
        ref = np.zeros(8)
        joint = np.zeros((8, 5))
        for mass, targets in zip(z["source_mass"], labels, strict=True):
            for cell in range(8):
                ref[cell] += mass[cell] / 386
                joint[cell] += mass[cell] * targets / 386
        source_error = max(
            source_error,
            float(np.max(np.abs(joint / ref[:, None] - b["probabilities"]))),
        )
    with np.load(out / "action_accounts.npz", allow_pickle=False) as p:
        expected = {k: p[k] for k in p.files}
    variant_error = effect_error = normalization_error = 0.0
    checks = 0
    for arm in range(2):
        for u in range(24):
            for v in (0, 1):
                x, c = z["coordinates"][u, arm, v], z["conditions"][u, v]
                cut = laws[0]["coordinate_cutoffs"]
                cell = (
                    4 * c
                    + (x[:, 0] > cut[0]).astype(int)
                    + 2 * (x[:, 5] > cut[1]).astype(int)
                )
                fields = np.column_stack(
                    [
                        *(
                            np.asarray(b["probabilities"])[cell] - b["prior"]
                            for b in laws
                        ),
                        z["signed_poles"][u, v],
                    ]
                )
                s, control = z["selected"][u, v], z["control"][u, v]
                ids = np.arange(12)
                groups = [
                    ids,
                    ids[~np.isin(ids, s)],
                    ids[~np.isin(ids, control)],
                    s,
                    control,
                ]
                variants = np.column_stack(
                    [np.sum(fields[ix], axis=0) / len(ix) for ix in groups]
                )
                ds, dc = (
                    variants[:, 1] - variants[:, 0],
                    variants[:, 2] - variants[:, 0],
                )
                contrast = (
                    np.sum(fields[control], axis=0) - np.sum(fields[s], axis=0)
                ) / 8
                effects = np.column_stack(
                    [
                        contrast,
                        contrast,
                        contrast,
                        ds,
                        dc,
                        variants[:, 3] - variants[:, 4],
                    ]
                )
                variant_error = max(
                    variant_error,
                    float(np.max(np.abs(variants - expected["variants"][arm, u, v]))),
                )
                effect_error = max(
                    effect_error,
                    float(
                        np.max(np.abs(effects - expected["mathematical"][arm, u, v]))
                    ),
                )
                normalization_error = max(
                    normalization_error,
                    float(np.max(np.abs(ds - 0.5 * (variants[:, 0] - variants[:, 3])))),
                )
                checks += variants.size + effects.size
    native = H.native(z, m["config"])
    native_error = float(np.max(np.abs(native - expected["native_readouts"])))
    worst = max(
        source_error, variant_error, effect_error, normalization_error, native_error
    )
    if worst > 2e-12:
        raise ValueError("Direct source/event account disagreement")
    receipt = {
        "source_probability_error": source_error,
        "direct_variant_error": variant_error,
        "direct_effect_error": effect_error,
        "normalization_error": normalization_error,
        "historical_native_error": native_error,
        "direct_event_scalars": checks,
        "source_authors": 386,
        "application_authors": 24,
        "disjoint_source_labels": True,
        "new_scientific_trials": 0,
        "independent_blind_audit": False,
    }
    H.R.write(out / "audit_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
