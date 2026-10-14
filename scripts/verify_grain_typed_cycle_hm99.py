#!/usr/bin/env python3
"""Verify typed currents through conditional averaging, not Helmert bases."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_observer_parity_hm96 import seal, write
from scripts.run_grain_typed_cycle_hm99 import check, settings, stage


def alternate(f, b, l):
    """Use node conditional expectation and a literal partial-index swap."""
    hb, hl = np.eye(b) - 1 / b, np.eye(l) - 1 / l
    jb, jl = np.full((b, b), 1 / b), np.full((l, l), 1 / l)
    pb, pl, pi = np.kron(hb, jl), np.kron(jb, hl), np.kron(hb, hl)
    odd = (f - f.swapaxes(0, 1)) / 2
    base = lambda x, y: np.stack(
        [x @ odd[..., q] @ y for q in range(f.shape[2])], axis=-1
    )
    inside = base(pi, pi).reshape(b, l, b, l, -1)
    swapped = inside.transpose(2, 1, 0, 3, 4)
    return {
        "category_cycle": base(pb, pb),
        "level_cycle": base(pl, pl),
        "category_level": base(pb, pl) + base(pl, pb),
        "category_interaction": base(pb, pi) + base(pi, pb),
        "level_interaction": base(pl, pi) + base(pi, pl),
        "interaction_category_skew": ((inside - swapped) / 2).reshape(f.shape),
        "interaction_level_skew": ((inside + swapped) / 2).reshape(f.shape),
    }


def run():
    """Recompute all actual parts, law contrasts and energy accounts."""
    spec, _, out = settings()
    check(out, include_application=True)
    for n in (1, 2, 3):
        stage(out, n)
    with np.load(out / "01/compiled.npz", allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(out / "03/accounts.npz", allow_pickle=False) as z:
        app = {k: z[k].copy() for k in z.files}
    with np.load(out / "02/predictions.npz", allow_pickle=False) as z:
        cold = {k: z[k].copy() for k in z.files}
    errors, count = [], 0

    def same(x, y):
        nonlocal count
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("Alternate shape mismatch")
        errors.append(float(np.max(np.abs(x - y))))
        count += x.size

    difference = source["plus"] - source["minus"]
    same(source["plus"].sum(1), source["minus"].sum(1))
    same(
        source["plus"] + source["plus"].transpose(0, 2, 1),
        source["minus"] + source["minus"].transpose(0, 2, 1),
    )
    if min(source["plus"].min(), source["minus"].min()) <= 0:
        raise ValueError("Nonpositive author law")
    for c in range(2):
        for kind, kernels in (
            ("source", source["source_kernel"]),
            ("application", app["application_kernel"]),
        ):
            parts = alternate(kernels[c], spec["categories"], spec["levels"])
            expected_energy = sum(np.sum(x * x, axis=(0, 1)) for x in parts.values())
            same(expected_energy, np.sum(sum(parts.values()) ** 2, axis=(0, 1)))
            for name, x in parts.items():
                stored = (
                    source[f"part_{c}_{name}"]
                    if kind == "source"
                    else app[f"application_part_{c}_{name}"]
                )
                same(x, stored)
                actual = np.stack(
                    [
                        np.array([np.sum(d * x[..., q]) for q in range(18)])
                        for d in difference
                    ]
                )
                if kind == "source":
                    same(actual, cold[f"prediction_{c}_{name}"])
                    for a in range(2):
                        same(actual, app[f"predicted_{a}_{c}_{name}"])
                else:
                    for s in range(2):
                        same(actual, app[f"actual_{c}_{s}_{name}"])
    with (out / "03/sector_metrics.csv").open() as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != 504:
        raise ValueError("Incomplete sector/frame/target metrics")
    for row in metric_rows:
        s, a = (
            spec["contexts"].index(row["source_frame"]),
            spec["contexts"].index(row["application_frame"]),
        )
        q = spec["fields"].index(row["field"]) * 6 + spec["targets"].index(
            row["target"]
        )
        x = source[f"part_{s}_{row['sector']}"][..., q]
        y = app[f"application_part_{a}_{row['sector']}"][..., q]
        energies = (np.sum(x * x), np.sum(y * y), np.sum(x * y), np.sum((x - y) ** 2))
        for key, value in zip(
            ("source_energy", "application_energy", "cross", "defect_energy"),
            energies,
            strict=True,
        ):
            same(float(row[key]), value)
        cosine = (
            np.nan
            if min(energies[:2]) == 0
            else energies[2] / np.sqrt(energies[0] * energies[1])
        )
        if np.isnan(cosine):
            if not np.isnan(float(row["cosine"])):
                raise ValueError("Undefined cosine filled in")
        else:
            same(float(row["cosine"]), cosine)
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Alternate formula mismatch")
    summary = {
        "status": "VERIFIED_FINITE_HM99_NOT_GLOBAL",
        "scalar_checks": count,
        "max_error": max(errors),
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
        "psychological_validity_claim": False,
    }
    dest = out / "04"
    dest.mkdir()
    write(dest / "summary.json", summary)
    write(out / "04_seal.json", seal(out, dest))
    write(out / "verification.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
