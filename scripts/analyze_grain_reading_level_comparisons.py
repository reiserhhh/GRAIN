#!/usr/bin/env python3
"""Paired fixed-output contrasts for the two primary reading-level queries."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_path_routes import paired_delta
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def main():
    """Compare paired authors, not independently bootstrapped coefficients."""
    source = BASE / "logic_reading_levels_v1"
    out = source / "comparisons_v1"
    if read(source / "status.json")["status"] != "COMPLETE":
        raise ValueError("Complete source run required")
    out.mkdir(exist_ok=False)
    oof = pd.read_csv(source / "private_source_oof.csv")
    criteria = pd.read_csv(
        BASE / "all_factor_source_m29_v1/private_external_links.csv"
    ).set_index(["author_id", "query"])
    native = pd.read_csv(BASE / "logic_natural_transport_v1/private_predictions.csv")
    changes = pd.read_csv(source / "private_held_changes.csv")
    rows = []
    for q in ("TF_cont", "neuroticism"):
        for target in ("c", "r"):
            pooled = (
                oof[(oof["query"] == q) & (oof.arm == "pooled")]
                .groupby("author_id")[["actual_" + target, "predicted_" + target]]
                .mean()
                .sort_index()
            )
            between = (
                oof[(oof["query"] == q) & (oof.arm == "between")]
                .groupby("author_id")[["actual_" + target, "predicted_" + target]]
                .mean()
                .loc[pooled.index]
            )
            for endpoint, y in (
                ("teacher", pooled["actual_" + target]),
                ("criterion", criteria.xs(q, level="query").loc[pooled.index].y_true),
            ):
                rows.append(
                    {
                        "scope": "source_author_mean",
                        "query": q,
                        "target": target,
                        "endpoint": endpoint,
                        "candidate": "between",
                        "reference": "pooled",
                        **paired_delta(
                            y.to_numpy(),
                            between["predicted_" + target].to_numpy(),
                            pooled["predicted_" + target].to_numpy(),
                            binary=q == "TF_cont" and endpoint == "criterion",
                            draws=1999,
                        ),
                    }
                )
            for edit in ("e1", "e2"):
                table = (
                    changes[(changes["query"] == q) & (changes.edit == edit)]
                    .pivot(
                        index="author_id",
                        columns="arm",
                        values="predicted_" + target + "_delta",
                    )
                    .sort_index()
                )
                y = (
                    native[(native["query"] == q) & (native.edit == edit)]
                    .set_index("author_id")
                    .loc[table.index]
                    .native_delta
                )
                for candidate in ("between", "within"):
                    rows.append(
                        {
                            "scope": "held_" + edit,
                            "query": q,
                            "target": target,
                            "endpoint": "native_delta",
                            "candidate": candidate,
                            "reference": "pooled",
                            **paired_delta(
                                y.to_numpy(),
                                table[candidate].to_numpy(),
                                table.pooled.to_numpy(),
                                draws=1999,
                            ),
                        }
                    )
    pd.DataFrame(rows).to_csv(out / "paired_comparisons.csv", index=False)
    write(
        out / "manifest.json",
        {
            "script_sha256": sha(__file__),
            "prediction_freeze_sha256": sha(source / "prediction_freeze.json"),
            "rows": len(rows),
            "new_fits": 0,
            "draws": 1999,
        },
    )
    print(pd.DataFrame(rows).query("target=='r'").to_string(index=False))


if __name__ == "__main__":
    main()
