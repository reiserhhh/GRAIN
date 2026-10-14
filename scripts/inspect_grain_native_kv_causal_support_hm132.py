#!/usr/bin/env python3
"""Read-only post-design audit of where historical ownership can alter K/V."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_native_kv_identity_hm132 as R


def main():
    """Inspect every paired cache; do not fit, select a layer, or run the model."""
    out = ROOT / R.H.load(R.SPEC)["out"]
    R.check(out)
    status = R.H.load(out / "status.json")
    if status["state"] != "COLLECTED" or status["completed"] != 100:
        raise ValueError("Do not inspect partial pair support as complete")
    R.H.verify_hashes(out, R.H.load(out / "01/raw_seal.json"))
    rows = R.H.load(out / "00/rows.json")
    lookup = R.validate_rows(rows)
    captures = {
        d["row"]: d["batch"]
        for d in R.H.load(out / "00/design.json")
        if d["mode"] == "capture"
    }
    records, maximum_before = [], 0.0
    for bank, family, b in itertools.product(R.BANKS, range(6), (-1, 1)):
        left, right = (lookup[bank, family, b, o] for o in (-1, 1))
        changed_input = [
            i
            for i, (a, c) in enumerate(
                zip(left["input_ids"], right["input_ids"], strict=True)
            )
            if a != c
        ]
        if not changed_input or not set(changed_input) <= set(left["history"]):
            raise ValueError("Need a nonempty addressed historical ownership edit")
        first = min(changed_input)
        prior = [j for j, i in enumerate(left["history"]) if i < first]
        active = [j for j, i in enumerate(left["history"]) if i not in left["padding"]]
        l, r = (R.committed(out, captures[row["id"]]) for row in (left, right))
        for name in ("history_key", "history_value"):
            delta = abs(r[name].astype(float) - l[name].astype(float))
            for layer in range(28):
                earlier = float(delta[layer, :, prior].max()) if prior else 0.0
                maximum_before = max(maximum_before, earlier)
                positions = delta[layer].max(axis=(0, 2))
                affected = [left["history"][j] for j in active if positions[j] != 0]
                records.append(
                    {
                        "bank": bank,
                        "family": family,
                        "polarity": b,
                        "primitive": name,
                        "block": layer,
                        "first_input_edit": first,
                        "edited_input_positions": ";".join(map(str, changed_input)),
                        "max_change_before_first_edit": earlier,
                        "active_affected_positions": ";".join(map(str, affected)),
                        "active_affected_count": len(affected),
                        "active_max_change": float(positions[active].max()),
                    }
                )
    if maximum_before > 1e-6:
        raise ValueError("An owner edit changed a causally prior cached state")
    folder = out / "03_causal_support"
    folder.mkdir()
    R.H.table(folder / "cache_support.csv", records)
    R.H.write(
        folder / "summary.json",
        {
            "diagnostic": "POST_DESIGN_NO_NEW_MODEL_CALL",
            "pairs": 24,
            "rows": len(records),
            "max_change_before_first_edit": maximum_before,
            "code_sha256": R.H.sha(Path(__file__)),
            "raw_seal_sha256": R.H.sha(out / "01/raw_seal.json"),
            "interpretation": "Causal prefix computations cannot retroactively bind earlier behavior-token states;ownership-dependent cached differences start within the final declaration,not a psychological validity result",
        },
    )
    R.H.write(folder / "output_seal.json", R.H.seals(out, folder))
    print(
        "CAUSAL SUPPORT HM132",
        {"pairs": 24, "rows": len(records), "max_before_edit": maximum_before},
        flush=True,
    )


if __name__ == "__main__":
    main()
