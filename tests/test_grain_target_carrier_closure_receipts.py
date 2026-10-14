#!/usr/bin/env python3
"""Post-outcome endpoint receipt checks with no model or private data access."""

import csv
import itertools
import math

import numpy as np
import pytest

from scripts import run_grain_target_carrier_closure_hm156 as R


def test_actual_qualified_effects_from_raw_endpoints():
    out = R.DEFAULT
    description = out / "post_outcome_description_r1/qualified_actions.csv"
    if not description.exists():
        pytest.skip("Actual finite receipt unavailable; synthetic module tests remain")
    R.check(out)
    manifest = R.R.read(out / "manifest.json")
    parent = R.ROOT / manifest["parent"]
    rows = R.R.read(out / "rows.json")
    indexed = {
        (r["bank"], r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows
    }
    paths = {
        (v["mode"], v["row"]): out / "captures" / f"{v['batch']:03d}.npz"
        for v in R.R.read(out / "design.json")
    }
    paths.update(
        {
            ("natural", v["row"]): parent / "captures" / f"{v['batch']:03d}.npz"
            for v in R.R.read(parent / "design.json")
            if v["mode"] == "natural"
        }
    )
    logits = {}
    for key, path in paths.items():
        with np.load(path, allow_pickle=False) as a:
            logits[key] = a["selected_logits"].astype(float)
    with description.open() as f:
        actual = list(csv.DictReader(f))
    assert len(actual) == 768
    for record in actual:
        group = R.R.GROUPS[int(record["group"])]
        mode, driver = record["mode"], record["actor_driver"]
        axis, family = (
            R.R.M.AXES.index(record["axis"]),
            R.R.M.FAMILIES.index(record["family"]),
        )
        terms = {"base": [], "patched": [], "flip": []}
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row = indexed[*group, e, n]
            changed_e = -e if mode.startswith(("e_", "both_")) else e
            changed_n = -n if mode.startswith(("n_", "both_")) else n
            donor = indexed[*group, changed_e, changed_n]
            qi, sign = (
                6 * axis + 2 * family + (p + 1) // 2,
                (e if driver == "e" else n) * p,
            )
            for name, key in (
                ("base", ("natural", row)),
                ("patched", (mode, row)),
                ("flip", ("natural", donor)),
            ):
                terms[name].append(sign * (logits[key][qi, 0] - logits[key][qi, 1]))
        base, patched, flip = (
            math.fsum(terms[name]) / 8 for name in ("base", "patched", "flip")
        )
        assert float(record["effect_coefficient"]) == pytest.approx(
            patched - base, abs=1e-12
        )
        assert float(record["natural_flip_effect_coefficient"]) == pytest.approx(
            flip - base, abs=1e-12
        )


def test_actual_composition_scope_is_ending_only():
    path = R.DEFAULT / "post_outcome_description_r1/summary.json"
    if not path.exists():
        pytest.skip("Actual receipt unavailable")
    summary = R.R.read(path)
    assert summary["post_outcome_not_prospective"] is True
    assert summary["new_model_forwards"] == 0
    assert "weak ending carrier only" in summary["note"]
