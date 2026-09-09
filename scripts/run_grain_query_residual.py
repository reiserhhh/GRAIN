#!/usr/bin/env python3
"""Score source-referenced query-readable residual parts on twelve held authors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_profiles import (
    FixedAuthorCoordinates,
    FunctionalQuery,
)
from grain_research.vector_mechanisms.natural_transport import (
    fit_operator,
    paired_summary,
    signed_energy_allocation,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_conditional_edits import load_panel
from scripts.run_grain_natural_transport import FACTORS, KINDS, QUERIES, SOURCE

PARTS = ("mean", "configuration", "final")
PLAN = ROOT / "docs/GRAIN_LOGIC_QUERY_RESIDUAL_PLAN_V1.md"


def run(source, out):
    """Derive reference positions and actual changes with no held-author fitting."""
    if read(source / "status.json")["status"] != "COMPLETE":
        raise ValueError("Complete fixed transport packet required")
    prior = read(source / "manifest.json")
    for rel, expected in prior["hashes"].items():
        p = ROOT / rel
        if rel.startswith(("docs/", "scripts/", "grain_research/")):
            continue
        if sha(p) != expected:
            raise ValueError("Transport input data changed")
    factors = pd.read_csv(
        BASE / "factor_path_routes_v1/source_r2/private_factor_views.csv"
    )
    paths = pd.read_csv(BASE / "dynamic_path_channels_v1/private_path_views.csv")
    held = pd.read_csv(source / "private_factor_scores.csv").set_index(
        ["author_id", "edit"]
    )
    total_maps = read(source / "source_operators.json")
    native, states, _, native_hashes = load_panel()
    sm = read(SOURCE / "manifest.json")
    functions = Path(sm["functions"]) / "nine_query_functions.npz"
    with np.load(functions, allow_pickle=False) as z:
        fun = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    maps, refs, source_rows = {}, {}, []
    coef_errors, intercept_errors = [], []
    for q in QUERIES:
        joined = factors.merge(
            paths[paths["query"] == q], on=["author_id", "view"], validate="one_to_one"
        ).sort_values(["author_id", "view"])
        x = joined[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        y = joined[list(PARTS)].to_numpy().reshape(-1, 3, 3)
        for kind in KINDS:
            maps[q, kind] = [fit_operator(x, y[:, :, j], kind) for j in range(3)]
            coefs = np.asarray([m["coef"] for m in maps[q, kind]])
            intercepts = np.asarray([m["intercept"] for m in maps[q, kind]])
            total = total_maps[f"{q}::{kind}"]
            coef_errors.append(float(np.max(np.abs(coefs.sum(0) - total["coef"]))))
            intercept_errors.append(
                float(
                    abs(
                        intercepts.sum()
                        + (fun[q].anchor if kind == "between" else 0)
                        - total["intercept"]
                    )
                )
            )
            if kind == "between":
                r = y.mean(1) - (x.mean(1) @ coefs.T + intercepts)
                refs[q] = FixedAuthorCoordinates().fit(r)
                for uid, a in zip(joined.author_id.unique(), r, strict=True):
                    source_rows.append(
                        dict(author_id=uid, query=q, **dict(zip(PARTS, a, strict=True)))
                    )
    if max(coef_errors + intercept_errors) > 1e-10:
        raise ValueError("Part operators do not sum to frozen total")
    level_rows, delta_rows = [], []
    for n, row in native.iterrows():
        uid, q = row.author_id, row["query"]
        pfull = fun[q].parts(states[n]["full_P"], states[n]["full_Q"])
        pedit = fun[q].parts(states[n]["edited_P"], states[n]["edited_Q"])
        values = np.array([pfull[c] for c in PARTS])
        delta = np.array([pedit[c] - pfull[c] for c in PARTS])
        g = held.loc[(uid, "full"), FACTORS].to_numpy(float)
        dg = held.loc[(uid, "e1"), FACTORS].to_numpy(float) - g
        for kind in KINDS:
            coefs = np.asarray([m["coef"] for m in maps[q, kind]])
            pred = dg @ coefs.T
            r = delta - pred
            native_delta = row.teacher_edit - row.teacher_full
            item = {
                "author_id": uid,
                "query": q,
                "kind": kind,
                "native_delta": native_delta,
                "functional_delta": float(delta.sum()),
                "grain": float(pred.sum()),
                "query_remainder": float(r.sum()),
                "reader_gap": float(native_delta - delta.sum()),
            }
            item.update(
                {"residual_" + c: float(v) for c, v in zip(PARTS, r, strict=True)}
            )
            item.update(
                {"functional_" + c: float(v) for c, v in zip(PARTS, delta, strict=True)}
            )
            delta_rows.append(item)
            if kind == "between":
                residual = values - (
                    g @ coefs.T + [m["intercept"] for m in maps[q, kind]]
                )
                ref = refs[q].transform(residual[None, :])[0]
                level_rows.append(
                    dict(
                        author_id=uid,
                        query=q,
                        **{
                            "residual_" + c: float(v)
                            for c, v in zip(PARTS, residual, strict=True)
                        },
                        **{"z_" + c: float(v) for c, v in zip(PARTS, ref, strict=True)},
                    )
                )
    delta, levels = pd.DataFrame(delta_rows), pd.DataFrame(level_rows)
    metrics, allocations, cross = [], [], []
    for (q, kind), frame in delta.groupby(["query", "kind"]):
        names = ["grain", *["residual_" + c for c in PARTS], "reader_gap"]
        ch = frame[names].to_numpy()
        shares = signed_energy_allocation(frame.native_delta, ch)
        if np.max(np.abs(ch.sum(1) - frame.native_delta)) > 1e-10:
            raise ValueError("Response parts do not reconstruct native change")
        for name, value in zip(names, shares, strict=True):
            allocations.append(
                {
                    "query": q,
                    "kind": kind,
                    "channel": name,
                    "signed_energy_allocation": float(value),
                    "rms": float(np.sqrt(np.mean(frame[name] ** 2))),
                }
            )
        for arm in (
            "query_remainder",
            *["residual_" + c for c in PARTS],
            *["functional_" + c for c in PARTS],
        ):
            metrics.append(
                dict(
                    query=q,
                    kind=kind,
                    arm=arm,
                    **paired_summary(frame.native_delta, frame[arm]),
                )
            )
        for target_q in QUERIES:
            other = (
                delta[(delta["query"] == target_q) & (delta.kind == kind)]
                .set_index("author_id")
                .loc[frame.author_id]
            )
            for arm in ("query_remainder", *["residual_" + c for c in PARTS]):
                cross.append(
                    dict(
                        source_query=q,
                        target_query=target_q,
                        kind=kind,
                        arm=arm,
                        **paired_summary(
                            other.native_delta, frame[arm], calibrated=False
                        ),
                    )
                )
    delta.to_csv(out / "private_residual_changes.csv", index=False)
    levels.to_csv(out / "private_residual_positions.csv", index=False)
    pd.DataFrame(source_rows).to_csv(out / "private_reference_authors.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(allocations).to_csv(out / "signed_energy_allocation.csv", index=False)
    pd.DataFrame(cross).to_csv(out / "cross_query.csv", index=False)
    write(
        out / "residual_reference.json",
        {
            q: {
                "center": r.center.tolist(),
                "scale": r.scale.tolist(),
                "active": r.active.tolist(),
                "source_authors": 63 if q == "TF_cont" else 213,
                "parts": list(PARTS),
            }
            for q, r in refs.items()
        },
    )
    write(
        out / "part_operators.json",
        {f"{q}::{kind}": m for (q, kind), m in maps.items()},
    )
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/natural_transport.py",
        source / "manifest.json",
        source / "source_operators.json",
        source / "private_factor_scores.csv",
        functions,
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_LOGIC_QUERY_RESIDUAL_V1",
            "native_hashes": native_hashes,
            "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "source": str(source.relative_to(ROOT)),
            "max_coefficient_sum_error": max(coef_errors),
            "max_intercept_sum_error": max(intercept_errors),
            "new_neural_calls": 0,
            "human_labels_read": False,
            "reference_authors": 276,
            "held_authors": 12,
            "source_type": "frozen query-readable states minus source-fitted GRAIN component",
        },
    )
    write(out / "status.json", {"status": "COMPLETE"})
    print(
        pd.DataFrame(metrics)
        .query("kind == 'transition'")[
            ["query", "arm", "r", "r_lower", "r_upper", "mse_over_zero"]
        ]
        .to_string(index=False),
        flush=True,
    )
    print(
        pd.DataFrame(allocations).query("kind == 'transition'").to_string(index=False),
        flush=True,
    )


def main():
    """Run a fixed follow-up without changing the original transport experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=BASE / "logic_natural_transport_v1")
    p.add_argument("--output-dir", type=Path, default=BASE / "logic_query_residual_v1")
    args = p.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not source.is_relative_to(BASE) or not out.is_relative_to(BASE):
        raise ValueError("Private result paths required")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "status.json", {"status": "RUNNING"})
    try:
        with threadpool_limits(limits=2):
            run(source, out)
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()
