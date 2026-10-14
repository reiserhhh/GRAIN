#!/usr/bin/env python3
"""Post-outcome description of whole-exchange compatibility, no new forwards."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_exchange_support import exchange_parts
from scripts import run_grain_endpoint_action_hm161 as E

CODE = (
    "grain_research/construction/math_exchange_support.py",
    "tests/test_grain_exchange_support.py",
    "scripts/describe_grain_endpoint_exchange_hm161.py",
)


def describe(out=E.DEFAULT):
    """Keep every endpoint and distinguish actor-orbit actions from partial edits."""
    E.check(out)
    E.G.authenticate_results(out)
    destination = out / "post_outcome_exchange_support_r1"
    destination.mkdir(exist_ok=False)
    E.C.P.write_json(
        destination / "manifest.json",
        {
            "packet": "HM161_POST_OUTCOME_EXCHANGE_DESCRIPTION",
            "not_preregistered_or_blind": True,
            "new_neural_forwards": 0,
            "hashes": {p: E.C.H.digest(ROOT / p) for p in CODE},
            "input_characters_sha256": E.C.H.digest(out / "characters.csv"),
            "scope": "Known Z2 exchange identity applied to existing endpoint actions, not a new theorem or psychological test",
        },
    )
    with (out / "characters.csv").open() as f:
        rows = list(csv.DictReader(f))
    fields = {}
    for row in rows:
        key = (
            int(row["condition"]),
            int(row["order"]),
            row["axis"],
            row["target"],
            row["family"],
            row["measure"],
            row["component"],
        )
        fields.setdefault(key, np.full(8, np.nan))[int(row["mask"])] = float(
            row["coefficient"]
        )
    descriptions = []
    natural_even_max = 0.0
    for axis, target, family, measure, component in itertools.product(
        ("e", "n"), E.M.AXES, E.M.FAMILIES, E.M.MEASURES, ("source", "natural")
    ):
        field = np.stack(
            [
                fields[c, o, axis, target, family, measure, component]
                for c, o in itertools.product(range(2), repeat=2)
            ]
        )
        result = exchange_parts(field, axis)
        if component == "natural":
            natural_even_max = max(natural_even_max, float(np.max(abs(result["even"]))))
        descriptions.append(
            {
                "axis": axis,
                "target": target,
                "family": family,
                "measure": measure,
                "component": component,
                **{k: v for k, v in result.items() if k not in ("even", "odd")},
            }
        )
    if natural_even_max > 1e-10:
        raise ValueError(
            "Actual complete natural exchanges violate their algebraic identity"
        )
    E.write_csv(destination / "exchange_parts.csv", descriptions)
    summary = {
        "status": "POST_OUTCOME_EXCHANGE_SUPPORT_DESCRIBED_NOT_NEW_EXPERIMENT",
        "max_natural_even_coefficient": natural_even_max,
        "all_endpoints": descriptions,
        "matched_restricted_policy_source": [
            d
            for d in descriptions
            if d["component"] == "source"
            and d["measure"] == "restricted_yes"
            and ((d["axis"] == "e") == (d["target"] == "extraversion"))
        ],
        "new_neural_forwards": 0,
        "original_scope_complete": False,
    }
    E.C.P.write_json(destination / "summary.json", summary)
    E.C.P.write_json(
        destination / "output_seal.json",
        {
            name: E.C.H.digest(destination / name)
            for name in ("summary.json", "exchange_parts.csv")
        },
    )
    print(
        {
            "natural_even_max": natural_even_max,
            "matched": summary["matched_restricted_policy_source"],
        },
        flush=True,
    )


if __name__ == "__main__":
    describe()
