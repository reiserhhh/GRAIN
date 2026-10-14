#!/usr/bin/env python3
"""Replay HM10A two-sample uncertainty via multinomial counts, not row selection."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from run_grain_math_response import ROOT, sha, write_json

PRIMARY = ROOT / "results/grain_math_response_v1/joint_law_r1"
DIAGNOSTIC = ROOT / "results/grain_math_response_v1/joint_law_diagnostic_r1"
DEST = ROOT / "results/grain_math_response_v1/joint_law_diagnostic_verification_r1"


def read_csv(path):
    """Read an archived result table without executing production code."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    """Recalculate all declared source/application bootstrap arms algebraically."""
    if (DEST / "verification.json").exists():
        raise ValueError("diagnostic verification frozen")
    hashes = 0
    for root, relative in (
        (PRIMARY, "04/artifact_hashes.json"),
        (DIAGNOSTIC, "artifact_hashes.json"),
    ):
        for name, digest in json.loads((root / relative).read_text()).items():
            if sha(root / name) != digest:
                raise ValueError("changed frozen artifact")
            hashes += 1
    tables = read_csv(PRIMARY / "03/four_corner_tables.csv")
    groups = defaultdict(list)
    for row in tables:
        groups[row["reader"], row["metric"]].append(row)
    expected = {
        (r["reader"], r["metric"], r["arm"]): r
        for r in read_csv(DIAGNOSTIC / "source_application_uncertainty.csv")
    }
    maximum, checks = 0.0, 0
    for (reader, metric), rows in groups.items():
        keys = sorted(
            {(int(r["frame"]), r["actor"], int(r["incidence"])) for r in rows}
        )
        panels = []
        for role in ("source", "application"):
            families = sorted({int(r["family"]) for r in rows if r["role"] == role})
            by = {
                (
                    int(r["family"]),
                    (int(r["frame"]), r["actor"], int(r["incidence"])),
                ): float(r["interaction"])
                for r in rows
                if r["role"] == role
            }
            panels.append(np.array([[by[f, key] for key in keys] for f in families]))
        src, app = panels
        drawn_source = np.random.default_rng(10101).integers(
            0, len(src), (2000, len(src))
        )
        drawn_application = np.random.default_rng(10102).integers(
            0, len(app), (2000, len(app))
        )
        ws = np.array(
            [np.bincount(row, minlength=len(src)) / len(src) for row in drawn_source]
        )
        wa = np.array(
            [
                np.bincount(row, minlength=len(app)) / len(app)
                for row in drawn_application
            ]
        )
        theta = ws @ src
        first, second = wa @ app, wa @ (app**2)
        shifted = [
            keys.index(((f + 1) % 4, actor, 1 - incidence))
            for f, actor, incidence in keys
        ]
        for arm in (
            "source_conditional",
            "source_global",
            "source_shifted_key_control",
        ):
            th = (
                theta
                if arm == "source_conditional"
                else theta.mean(1, keepdims=True)
                if arm == "source_global"
                else theta[:, shifted]
            )
            mse = (second - 2 * th * first + th**2).mean(1)
            improvement = (2 * th * first - th**2).mean(1)
            ci = np.quantile(improvement, [0.025, 0.975])
            rmse_ci = np.quantile(np.sqrt(np.maximum(mse, 0)), [0.025, 0.975])
            archived = expected[reader, metric, arm]
            difference = max(
                float(
                    np.max(
                        np.abs(
                            ci
                            - json.loads(archived["source_and_application_family_ci"])
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            rmse_ci
                            - json.loads(archived["source_and_application_rmse_ci"])
                        )
                    )
                ),
            )
            maximum = max(maximum, difference)
            checks += 4
            if difference > 1e-9:
                raise ValueError("two-sample bootstrap differs")
    result = {
        "status": "PASS",
        "packet": "HM10A-source-uncertainty",
        "interval_endpoints_checked": checks,
        "hashes_checked": hashes,
        "max_formula_error": maximum,
        "formula": "multinomial family counts and E[e^2]-2theta E[e]+theta^2",
        "new_observer_calls": 0,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
    }
    write_json(DEST / "verification.json", result)
    write_json(
        DEST / "artifact_hashes.json",
        {
            str(f.relative_to(DEST)): sha(f)
            for f in sorted(DEST.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
