#!/usr/bin/env python3
"""Post-outcome HM155 descriptions from saved results; no fitting or new forwards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import recover_grain_qualified_target_field_hm155 as C
from scripts import run_grain_qualified_target_field_hm155 as R


def describe(out=R.DEFAULT):
    """Keep all corners; inspect causal descendants without attributing effects."""
    R.check(out)
    for section in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.read(out / section).items():
            if R.H.digest(out / name) != checksum:
                raise ValueError("Frozen output changed: " + name)
    rows = R.read(out / "rows.json")
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    natural = {r["id"]: R.arrays(out / "captures" / f"{r['id']:03d}.npz") for r in rows}
    carrier_rows = []
    for r in rows:
        for j, axis in enumerate(R.M.AXES):
            donor = indexed[
                r["bank"],
                r["condition"],
                r["order"],
                -r["e"] if j == 0 else r["e"],
                -r["n"] if j == 1 else r["n"],
            ]
            own, other = r["scopes"][axis], r["scopes"][R.M.AXES[1 - j]]
            before = list(range(min(own)))
            tail = list(
                range(max(max(s) for s in r["scopes"].values()) + 1, len(r["history"]))
            )
            differences = [
                natural[donor["id"]]["history_" + name].astype(float)
                - natural[r["id"]]["history_" + name].astype(float)
                for name in ("key", "value")
            ]

            def energy(slots, arrays=differences):
                return sum(float(np.sum(d[:, :, slots, :] ** 2)) for d in arrays)

            carrier_rows.append(
                {
                    "row": r["id"],
                    "axis": axis,
                    "order": r["order"],
                    "before_max": max(
                        float(np.max(abs(d[:, :, before, :]))) for d in differences
                    ),
                    "other_energy": energy(other),
                    "tail_energy": energy(tail),
                    "own_energy": energy(own),
                    "edit_first": bool(min(own) < min(other)),
                }
            )
    chars = C.table(out / "characters.csv")
    coeff = {
        (r["bank"], r["condition"], r["order"], r["axis"], r["family"], r["mask"]): r[
            "logodds"
        ]
        for r in chars
        if r["mode"] == "natural"
    }
    free_required = [
        coeff[b, 0, o, R.M.AXES[0], f, 5] - coeff[b, 1, o, R.M.AXES[0], f, 5]
        for b in range(2)
        for o in range(2)
        for f in R.M.FAMILIES
    ]
    actions = C.table(out / "actions.csv")
    by_order = []
    for mode in ("e_local", "n_local"):
        for order in range(2):
            group = [r for r in actions if r["mode"] == mode and r["order"] == order]
            by_order.append(
                {
                    "mode": mode,
                    "order": order,
                    "signed_projection_range": [
                        min(r["signed_projection"] for r in group),
                        max(r["signed_projection"] for r in group),
                    ],
                    "relative_remainder_range": [
                        min(r["relative_remainder"] for r in group),
                        max(r["relative_remainder"] for r in group),
                    ],
                }
            )
    result = {
        "status": "POST_OUTCOME_DESCRIPTION_NOT_NEW_CONFIRMATORY_TEST",
        "canonical_summary": "serialization_recovery_r1/summary.json",
        "new_forwards": 0,
        "fitted_models": 0,
        "carrier_pairs": len(carrier_rows),
        "maximum_preceding_KV_change": max(r["before_max"] for r in carrier_rows),
        "later_other_block_changed": sum(
            bool(r["other_energy"] > 0 and r["edit_first"]) for r in carrier_rows
        ),
        "earlier_other_block_changed": sum(
            bool(r["other_energy"] > 0 and not r["edit_first"]) for r in carrier_rows
        ),
        "record_ending_changed": sum(bool(r["tail_energy"] > 0) for r in carrier_rows),
        "carriers": carrier_rows,
        "E_free_minus_required": {
            "positive": sum(bool(v > 0) for v in free_required),
            "count": len(free_required),
            "mean": float(np.mean(free_required)),
            "range": [min(free_required), max(free_required)],
        },
        "natural_character_means": {
            axis: {
                str(mask): float(
                    np.mean(
                        [v for k, v in coeff.items() if k[3] == axis and k[5] == mask]
                    )
                )
                for mask in (5, 6, 7)
            }
            for axis in R.M.AXES
        },
        "local_by_order": by_order,
        "composition_relative_remainder_range": [
            min(r["relative_remainder"] for r in actions if r["mode"] == "both_local"),
            max(r["relative_remainder"] for r in actions if r["mode"] == "both_local"),
        ],
        "interpretation": "KV changes establish candidate descendants, not which carrier mediates a reader effect. No target/prompt/weight selection.",
    }
    destination = out / "post_outcome_diagnostics_r1"
    if destination.exists():
        raise RuntimeError("Do not overwrite post-outcome diagnostics")
    destination.mkdir()
    R.H.write_json(destination / "summary.json", result)
    R.H.write_json(
        destination / "provenance.json",
        {
            "capture_seal_sha256": R.H.digest(out / "capture_seal.json"),
            "output_seal_sha256": R.H.digest(out / "output_seal.json"),
            "script_sha256": R.H.digest(Path(__file__)),
            "summary_sha256": R.H.digest(destination / "summary.json"),
            "prospective_status": False,
            "original_scope_complete": False,
        },
    )
    print({k: v for k, v in result.items() if k not in ("carriers", "local_by_order")})


if __name__ == "__main__":
    describe()
