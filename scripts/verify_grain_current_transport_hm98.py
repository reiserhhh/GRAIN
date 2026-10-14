#!/usr/bin/env python3
"""Alternate finite HM98 recomputation; no neural rerun or fitted parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_current_transport_hm98 import check, settings, stage_seal
from scripts.run_grain_hierarchical_query_hm88 import unit
from scripts.run_grain_observer_parity_hm96 import load, seal, verify, write


def run():
    """Independently verify probability maps, scalar readouts and stop traces."""
    spec, out = settings()
    check(out)
    for number in (2, 3, 4, 5):
        stage_seal(out, number)
    verify(
        out,
        {
            n: h
            for n, h in load(out / "01/raw_seal.json").items()
            if not Path(n).name.startswith("._")
        },
    )
    rows = load(out / "00/rows.json")
    for r in rows:
        unit(out, r["unit"])
    source = np.load(out / "03/compiled_bundle.npz", allow_pickle=False)
    application = np.load(out / "02/application_coordinates.npz", allow_pickle=False)
    data = np.load(out / "05/measured_transport.npz", allow_pickle=False)
    errors, checks = [], 0

    def compare(x, y):
        nonlocal checks
        a, b = np.asarray(x), np.asarray(y)
        if a.shape != b.shape:
            raise ValueError("Alternate shape mismatch")
        errors.append(float(np.max(np.abs(a - b))) if a.size else 0)
        checks += a.size

    for context in range(2):
        for bank in spec["banks"]:
            x, y = (
                source[f"points_{context}_{bank}"],
                application[f"points_{context}_{bank}"],
            )
            distances = np.array(
                [[np.sqrt(np.sum((yy - xx) ** 2)) for xx in x] for yy in y]
            )
            alternate = np.zeros((18, 18))
            for i, d in enumerate(distances):
                matched = np.where(d <= spec["node_match_atol"])[0]
                if len(matched):
                    alternate[i, matched[0]] = 1
                else:
                    inverse = 1 / d**2
                    alternate[i] = inverse / inverse.sum()
            compare(alternate, data[f"map_{context}_{bank}"])
        for arm in (*spec["banks"], "uniform_null", "known_correspondence"):
            t = data[f"map_{context}_{arm}"]
            f = source["kernel"][context]
            alternate = np.stack([t @ f[..., q] @ t.T for q in range(18)], axis=-1)
            compare(alternate, data[f"kernel_{context}_{arm}"])
            for a, d in enumerate(data["current"]):
                difference = data["plus"][a] - data["minus"][a]
                transported = t.T @ difference @ t
                compare(transported.sum(1), np.zeros(18))
                direct = np.array(
                    [np.sum(difference * alternate[..., q]) for q in range(18)]
                )
                dual = np.array([np.sum(transported * f[..., q]) for q in range(18)])
                compare(direct, dual)
                compare(direct, data[f"contrasts_{context}_{arm}"][a])
                native = np.array(
                    [
                        np.sum(difference * data["native_kernel"][context, ..., q])
                        for q in range(18)
                    ]
                )
                compare(native, data[f"native_contrasts_{context}"][a])
                p, m = t.T @ data["plus"][a] @ t, t.T @ data["minus"][a] @ t
                compare(p.sum(1), m.sum(1))
                compare(p + p.T, m + m.T)
                if min(p.min(), m.min()) < 0:
                    raise ValueError("Alternate negative law")
    audit = load(out / "04/audit.json")
    if (
        set(audit["opened_paths"]) != set(audit["allowed_paths"])
        or audit["application_target_tables_queries_truth_or_text"]
    ):
        raise ValueError("Cold file audit failed")
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Alternate arithmetic mismatch")
    dest = out / "06"
    dest.mkdir()
    summary = {
        "status": "VERIFIED_FINITE_HM98_NOT_GLOBAL",
        "scalar_checks": checks,
        "alternate_max_error": max(errors),
        "actual_trace_checks": len(rows),
        "cold_file_access": "PASS after-import explicit IO audit",
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
        "psychological_validity_claim": False,
    }
    write(dest / "summary.json", summary)
    write(out / "06_seal.json", seal(out, dest))
    write(out / "verification.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
