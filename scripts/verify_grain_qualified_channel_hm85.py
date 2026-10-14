#!/usr/bin/env python3
"""Recompute HM85 capacity certificates and all raw score/action arithmetic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_author_contrast_hm82 import load, sha, write
from scripts.run_grain_qualified_channel_hm85 import OUT


def main():
    """Verify certificates without re-solving LPs or changing coefficients."""
    frozen = load(OUT / "00/freeze.json")
    for name, digest in (frozen["code"] | frozen["inputs"]).items():
        assert sha(ROOT / name) == digest, name
    for name, digest in frozen["code"].items():
        assert sha(OUT / "00/code" / name) == digest, name
    for name, digest in frozen["assets"].items():
        assert sha(OUT / name) == digest, name
    for name, digest in load(OUT / "01/seal.json")["files"].items():
        assert sha(OUT / name) == digest, name
    for name, digest in load(OUT / "01/input_bindings.json").items():
        assert sha(ROOT / name) == digest, name
    bundle = load(OUT / "00/bundle.json")
    scores = load(OUT / "01/scores.json")
    errors, values, statuses = [], 0, {}
    with (
        np.load(OUT / "00/source.npz") as source,
        np.load(OUT / "01/contextual_observations.npz") as observed,
    ):
        for key, fn in bundle["functions"].items():
            statuses[fn["status"]] = statuses.get(fn["status"], 0) + 1
            if fn["w"] is None:
                assert scores[key] is None
                continue
            w = fn["w"]
            a, n = source[key + "_own"], source[key + "_foreign"]
            own = [sum(float(v) * c for v, c in zip(row, w, strict=True)) for row in a]
            foreign = [
                sum(float(v) * c for v, c in zip(row, w, strict=True)) for row in n
            ]
            lam, mp, mn = fn["lambda"], fn["mu_positive"], fn["mu_negative"]
            for j in range(len(w)):
                left = sum(float(row[j]) * c for row, c in zip(a, lam, strict=True))
                right = sum(
                    float(row[j]) * (p - m) for row, p, m in zip(n, mp, mn, strict=True)
                )
                errors.append(abs(left - right))
            bound = fn["foreign_action_bound"]
            assert min(own) >= 1 - 1e-7
            assert max(map(abs, foreign)) <= bound + 1e-7
            assert min(lam + mp + mn) >= -1e-9
            assert sum(mp) + sum(mn) <= 1 + 1e-7
            errors.append(abs(sum(lam) - bound))
            b = observed[fn["space"]]
            direct = [
                sum(float(v) * c for v, c in zip(row, w, strict=True)) for row in b
            ]
            errors.extend(abs(x - y) for x, y in zip(direct, scores[key], strict=True))
            values += len(own) + len(foreign) + len(direct) + len(w)
    import csv

    with (OUT / "01/actions.csv").open() as stream:
        for row in csv.DictReader(stream):
            v = scores[row["function"]]
            if v is not None:
                errors.append(
                    abs(
                        (v[int(row["high"])] - v[int(row["low"])])
                        - float(row["action"])
                    )
                )
            values += 1
    assert max(errors) < 1e-7
    summary = {
        "status": "VERIFIED_FINITE_HM85_NOT_GLOBAL",
        "source_statuses": statuses,
        "scalar_values_checked": values,
        "maximum_absolute_error": max(errors),
        "new_calls": 0,
        "same_agent_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    write(OUT / "verification.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
