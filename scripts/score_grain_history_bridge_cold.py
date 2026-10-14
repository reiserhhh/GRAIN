#!/usr/bin/env python3
"""Account for history changes using two numeric inputs, without a model."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_bridge as M


def main():
    """Audit data IO after imports; preserve all fixed primitive accounts."""
    if len(sys.argv) != 5:
        raise ValueError("Expected bundle,export,output,audit")
    bundle, export, output, audit = [Path(p).resolve() for p in sys.argv[1:]]
    if output.exists() or audit.exists():
        raise FileExistsError("Preserve historical cold results")
    fixture = BytesIO()
    np.savez_compressed(fixture, fixture=np.zeros(1))
    fixture.seek(0)
    with np.load(fixture, allow_pickle=False) as data:
        data["fixture"]
    # Finish lazy numerical initialization before the declared data IO boundary.
    M.integral_accounts(
        (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1))),
        (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1))),
    )
    allowed, opened = {str(p) for p in (bundle, export, output, audit)}, set()

    def guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        p = str(Path(args[0]).resolve())
        if p not in allowed:
            raise PermissionError("Undeclared cold IO: " + p)
        opened.add(p)

    sys.addaudithook(guard)
    with np.load(bundle, allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(export, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    native = np.zeros((2, 2, 18, 18, 3, 6))
    corners = np.zeros(native.shape + (8,))
    integral = np.zeros(native.shape + (3,))
    references = {}
    for n, (corpus, frame, i, j) in enumerate(data["addresses"]):
        if i == j:
            references[(corpus, frame, j)] = n
    errors = []
    for n, (corpus, frame, i, j) in enumerate(data["addresses"]):
        baseline = references[(corpus, frame, j)]
        p, q = data["offsets"][n : n + 2], data["offsets"][baseline : baseline + 2]
        if not np.array_equal(
            data["token_ids"][p[0] : p[1]], data["token_ids"][q[0] : q[1]]
        ):
            raise ValueError("Current token support changed")
        low, high = M.primitives(source, data, baseline), M.primitives(source, data, n)
        address = (corpus, frame, i, j)
        native[address] = M.score(*high)
        corners[address] = M.cube(low, high)
        integral[address] = M.integral_accounts(low, high, int(source["nodes"]))
        errors.append(
            float(
                np.max(
                    np.abs(
                        integral[address].sum(-1)
                        - corners[address][..., 7]
                        + corners[address][..., 0]
                    )
                )
            )
        )
    accounts = M.finite_accounts(corners)
    np.savez_compressed(
        output, native=native, corners=corners, integral=integral, **accounts
    )
    opened.add(str(audit))
    audit.write_text(
        json.dumps(
            {
                "opened_paths": sorted(opened),
                "allowed_paths": sorted(allowed),
                "scope": "after imports;explicit data IO,not OS sandbox",
                "native_target_table_or_model_read": False,
                "new_neural_calls": 0,
                "predictor_fit": False,
                "actual_captures": len(data["addresses"]),
                "complete_token_matches": len(errors),
                "max_integral_completeness_error": max(errors),
            },
            indent=2,
        )
        + "\n"
    )
    print("COLD HM100", len(errors), "captures;path error", max(errors), flush=True)


if __name__ == "__main__":
    main()
