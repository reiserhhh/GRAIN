#!/usr/bin/env python3
"""Finite mathematical witnesses plus an existing native-endpoint reanalysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.logic_levels import (
    covariance_accounting,
    endpoint_accounting,
    level_associations,
    residual_pullback_example,
)
from scripts.run_grain_logic_levels import BASE, PLAN, sha, write


def execute(out):
    """Write witnesses and aggregate endpoint statistics without neural calls."""
    if out.exists():
        raise ValueError("Use a new result directory")
    source = BASE / "dynamic_path_channels_v1/native_direction/private_effects.csv"
    # The literal arm name "null" is not a missing-data marker.
    frame = pd.read_csv(source, keep_default_na=False)
    if len(frame) != 24 or frame.duplicated(["author_id", "arm"]).any():
        raise ValueError("Expected paired 12-author real/null endpoints")
    if set(frame.arm) != {"real", "null"}:
        raise ValueError("Unknown intervention arms")
    out.mkdir(parents=True)
    rows = []
    for arm, data in frame.groupby("arm", sort=True):
        if len(data) != 12 or not np.allclose(
            (data.plus - data.minus) / 2, data.effect, atol=1e-14
        ):
            raise ValueError("Native effect definition changed")
        rows.extend(
            {"arm": arm, **r}
            for r in endpoint_accounting(data.minus, data.native, data.plus)
        )
    a, b = (
        frame[frame.arm.eq(arm)].set_index("author_id").sort_index()
        for arm in ("real", "null")
    )
    if not a.index.equals(b.index) or not np.allclose(
        a.native, b.native, rtol=0, atol=1e-14
    ):
        raise ValueError("Endpoint arms are not the same native authors")
    pd.DataFrame(rows).to_csv(out / "native_endpoint_accounting.csv", index=False)
    logits = np.linspace(-2.0, 1.0, 12)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    u, t = np.arange(9.0) - 4, np.array([-1.0, 0, 1])
    x, y = (u[:, None] + t[None, :])[:, :, None], u[:, None] - 2 * t[None, :]
    links = level_associations(x, y, u)
    account = covariance_accounting(x, y)
    n = np.array([10, 1000, 1000000])
    witnesses = {
        "opposite_levels": {
            "between_r": float(links["between"][0]),
            "within_r": float(links["within"][0]),
            "covariance_closure_error": float(account["closure_error"][0]),
        },
        "constant_logit_shift": endpoint_accounting(
            sigmoid(logits - 0.2), sigmoid(logits), sigmoid(logits + 0.2)
        ),
        "residual_pullback": residual_pullback_example(),
        "perfect_axis_with_common_patch": {
            "definition": "H_u=x_u; F(H)=H; stipulated criterion Y_u=x_u; patch adds the same 0.2 to every H",
            "level_criterion_r": float(np.corrcoef(logits, logits)[0, 1]),
            "logit_response": endpoint_accounting(
                sigmoid(logits - 0.2), sigmoid(logits), sigmoid(logits + 0.2)
            )[1],
            "implication": "A completely common unit response is compatible with a perfect between-author measurement axis; it neither proves nor refutes criterion relevance",
        },
        "iid_noise_limit": {
            "n": n.tolist(),
            "event_residual_variance": [1.0, 1.0, 1.0],
            "mean_estimator_variance": (1 / n).tolist(),
            "assumption": "iid unit-variance noise; author number or event averaging precision does not delete realized event residual",
        },
        "status": "finite_constructed_examples_not_human_data",
    }
    write(out / "mathematical_witnesses.json", witnesses)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_LOGIC_WITNESSES_V1",
            "new_model_calls": 0,
            "plan_sha256": sha(PLAN),
            "script_sha256": sha(__file__),
            "module_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/logic_levels.py"
            ),
            "source_sha256": sha(source),
        },
    )
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "native_authors": 12,
            "arms": 2,
            "coordinates": 2,
            "finite_witnesses": 5,
        },
    )
    print("LOG04/05 complete: five witnesses, 12 cached native authors; no model calls")


def main():
    """Run the bounded witness packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "logic_closure_v1/witnesses_r2"
    )
    args = parser.parse_args()
    execute(args.output)


if __name__ == "__main__":
    main()
