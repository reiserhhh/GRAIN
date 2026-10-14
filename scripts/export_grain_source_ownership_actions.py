#!/usr/bin/env python3
"""Export the declared HM76 application actions without new calls or fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grain_source_ownership_operator import (
    COLD,
    OUT,
    SOURCE,
    M,
    digest,
    files,
    load_groups,
    models,
    read,
    seal,
    write,
)


def main():
    """Materialize single-input actions; paired metadata only indexes the export."""
    out = OUT / "04_action_export"
    if out.exists():
        raise FileExistsError("Action companion exists; do not overwrite")
    verified = read(OUT / "verification.json")
    if verified["status"] != "VERIFIED_FINITE_HM76_NOT_GLOBAL":
        raise ValueError("Primary finite packet not verified")
    for name, sha in read(OUT / "03/output_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Primary output changed")
    fixed = models()
    groups, used = load_groups(SOURCE, lambda r: r["family"] == 1)
    for root in (COLD, OUT):
        part, inputs = load_groups(root, lambda _r: True)
        if set(part) & set(groups):
            raise ValueError("Duplicate application objects")
        groups.update(part)
        used.update(inputs)
    out.mkdir()
    write(
        out / "protocol.json",
        {
            "status": "RESULT_AFTER_DECLARED_ASSET_EXPORT_NOT_NEW_SELECTION",
            "script_sha256": digest(Path(__file__)),
            "primary_seal_sha256": digest(OUT / "03/output_seal.json"),
            "input_seal": used,
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    index, maximum, pairs = [], 0.0, 0
    for (stage, method), model in fixed.items():
        items = sorted((k, v) for k, v in groups.items() if k[1] == stage)
        plus, minus = [], []
        keys = []
        for key, value in items:
            actions = []
            for sign in (1, -1):
                h = value[sign]
                action = M.operate(model, h) - h
                alternate = (
                    -2 * model["left"] @ (model["right"] @ (h - model["origin"]))
                )
                maximum = max(maximum, float(np.max(abs(action - alternate))))
                actions.append(action)
            plus.append(actions[0])
            minus.append(actions[1])
            keys.append(list(key))
        arrays = out / f"{stage}__{method}.npz"
        np.savez_compressed(
            arrays, plus_delta=np.array(plus), minus_delta=np.array(minus)
        )
        write(
            arrays.with_suffix(".json"), {"keys": keys, "array_sha256": digest(arrays)}
        )
        index.append({"stage": stage, "method": method, "application_pairs": len(keys)})
        pairs += len(keys)
    if maximum > 1e-10 or pairs != 6048:
        raise ValueError("Declared action export does not reproduce primary operations")
    write(
        out / "summary.json",
        {
            "status": "EXPORTED_FIXED_HM76_ACTIONS",
            "application_pairs_across_methods": pairs,
            "single_input_actions": 2 * pairs,
            "max_alternate_error": maximum,
            "index": index,
            "new_model_calls": 0,
            "new_fits": 0,
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    write(out / "seal.json", seal(OUT, files(out)))
    print({"exported_pairs": pairs, "max_alternate_error": maximum}, flush=True)


if __name__ == "__main__":
    main()
