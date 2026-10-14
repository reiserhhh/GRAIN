#!/usr/bin/env python3
"""Compile source queries, export body primitives, then analyze history bridges."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_channel_boundary_transport as B
from grain_research.construction import math_history_bridge as M
from grain_research.construction import math_typed_cycle as T
from grain_research.construction.math_cycle_capacity import cycles
from scripts.run_grain_current_transport_hm98 import check as predecessor_check
from scripts.run_grain_hierarchical_query_hm88 import unit
from scripts.run_grain_observer_parity_hm96 import load, seal, sha, table, verify, write

SPEC = ROOT / "configs/grain_history_bridge_hm100.json"
CODE = (
    "configs/grain_history_bridge_hm100.json",
    "docs/GRAIN_HISTORY_BRIDGE_PLAN_HM100.md",
    "grain_research/construction/math_history_bridge.py",
    "scripts/run_grain_history_bridge_hm100.py",
    "scripts/score_grain_history_bridge_cold.py",
    "scripts/verify_grain_history_bridge_hm100.py",
    "tests/test_grain_math_history_bridge.py",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_body_relation_graph.py",
    "grain_research/construction/math_cold_evidence.py",
    "grain_research/construction/math_typed_cycle.py",
    "grain_research/construction/math_cycle_capacity.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_observer_parity_hm96.py",
)


def settings():
    """Return a finite fixed protocol, without optimizer parameters."""
    spec = load(SPEC)
    return spec, ROOT / spec["output"]


def check(out):
    """Authenticate declared code and frozen ancestors."""
    m = load(out / "00/manifest.json")
    verify(ROOT, m["code"] | m["source"])
    return m


def stage(out, number):
    """Check every substantive sealed stage artifact."""
    verify(out, load(out / f"{number:02d}_seal.json"))


def prepare(spec, out):
    """Freeze follow-up protocol before reading primitive account results."""
    if out.exists():
        raise FileExistsError("Preserve completed and failed packets")
    assets = {}
    for label, expected in (
        ("source", "HM97"),
        ("application", "HM98"),
        ("typed", "HM99"),
    ):
        base = ROOT / spec[label]
        if label == "application":
            predecessor_check(base)
        if (
            load(base / "verification.json")["status"]
            != f"VERIFIED_FINITE_{expected}_NOT_GLOBAL"
        ):
            raise ValueError("Verified actual predecessor required")
        names = ["00/manifest.json", "verification.json"]
        if label != "typed":
            names += ["00/bundle.json", "00/rows.json", "01/raw_seal.json"]
            verify(base, load(base / "01/raw_seal.json"))
            names += (
                ["03/measured_capacity.npz"]
                if label == "source"
                else ["05/measured_transport.npz"]
            )
        else:
            names += ["01/compiled.npz"]
        assets.update({str((base / n).relative_to(ROOT)): sha(base / n) for n in names})
    (out / "00").mkdir(parents=True)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": assets,
            "aggregate_results_already_known": True,
            "original_scope_complete": False,
            "new_neural_calls": 0,
            "fits": 0,
        },
    )
    print("PREPARED HM100", flush=True)


def compile_source(spec, out):
    """Seal previously source-derived q and covectors before body export."""
    check(out)
    source = load(ROOT / spec["source"] / "00/bundle.json")
    app = load(ROOT / spec["application"] / "00/bundle.json")
    if source != app or source["targets"] != spec["targets"]:
        raise ValueError("Identical inherited source observer required")
    dest = out / "01"
    dest.mkdir()
    np.savez_compressed(
        dest / "compiled.npz",
        **{
            k: np.asarray(source[k])
            for k in ("raw_queries", "directions", "frequencies", "scaling", "gaps")
        },
        nodes=np.asarray(spec["quadrature_nodes"]),
    )
    write(
        dest / "summary.json",
        {"source_only": True, "application_primitives_read": False},
    )
    write(out / "01_seal.json", seal(out, dest))
    print("COMPILED source observer", flush=True)


def export(spec, out):
    """Export unqueried current-span K/V and direct body boundary probabilities."""
    check(out)
    stage(out, 1)
    dest = out / "02"
    dest.mkdir()
    keys, values, gates, ids, endpoints, addresses, offsets = (
        [],
        [],
        [],
        [],
        [],
        [],
        [0],
    )
    for corpus, label in enumerate(("source", "application")):
        base = ROOT / spec[label]
        rows = load(base / "00/rows.json")
        rows = [r for r in rows if r["context"] in spec["source_contexts"]]
        if len(rows) != 648:
            raise ValueError("Two complete context supports required")
        for row in rows:
            raw = unit(base, row["unit"])
            ix, group = row["body_indices"], np.asarray(row["action_groups"][1], int)
            positions = np.asarray(ix)[group]
            transition, _ = B.channel_graphs(raw["q_body"], raw["k"][ix])
            g = B.channel_laws(transition, row["root_labels"], direct=True)[
                :, group, 0
            ].T
            keys.append(raw["k"][positions])
            values.append(raw["v"][positions])
            gates.append(g)
            ids.extend(row["input_ids"][p] for p in positions)
            endpoints.append(int(max(positions)))
            addresses.append((corpus, row["context"], row["previous"], row["current"]))
            offsets.append(offsets[-1] + len(group))
    if len(set(addresses)) != 1296:
        raise ValueError("Duplicate or missing capture addresses")
    np.savez_compressed(
        dest / "primitives.npz",
        keys=np.concatenate(keys),
        values=np.concatenate(values),
        gates=np.concatenate(gates),
        token_ids=np.asarray(ids),
        offsets=np.asarray(offsets),
        endpoints=np.asarray(endpoints),
        addresses=np.asarray(addresses),
    )
    write(
        dest / "summary.json",
        {
            "captures": len(addresses),
            "current_tokens": len(ids),
            "target_queries_applied": False,
            "new_encoder_calls": 0,
        },
    )
    write(out / "02_seal.json", seal(out, dest))
    print("EXPORTED", len(addresses), "actual body primitives", flush=True)


def cold(out):
    """Call the audited two-input numerical consumer, without target kernels."""
    check(out)
    stage(out, 1)
    stage(out, 2)
    dest = out / "03"
    dest.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_history_bridge_cold.py"),
            str(out / "01/compiled.npz"),
            str(out / "02/primitives.npz"),
            str(dest / "accounts.npz"),
            str(dest / "audit.json"),
        ],
        check=True,
    )
    write(out / "03_seal.json", seal(out, dest))


def evaluate(spec, out):
    """Replay native outcomes and keep all primitive, sector and author accounts."""
    start = time.monotonic()
    check(out)
    for n in (1, 2, 3):
        stage(out, n)
    dest = out / "04"
    dest.mkdir()
    with np.load(out / "03/accounts.npz", allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    with np.load(
        ROOT / spec["source"] / "03/measured_capacity.npz", allow_pickle=False
    ) as z:
        source = z["target_kernel"][:2].copy()
    with np.load(
        ROOT / spec["application"] / "05/measured_transport.npz", allow_pickle=False
    ) as z:
        app = z["native_kernel"].copy()
    expected = np.stack((source, app))
    native = a["native"].reshape(2, 2, 18, 18, 18)
    replay = float(np.max(np.abs(native - expected)))
    errors = [replay, float(np.max(np.abs(a["corners"][..., 7] - a["native"])))]
    bases = T.sectors(spec["categories"], spec["levels"])
    _, plus, minus = T.positive_laws(bases)
    difference = plus - minus
    arrays, metrics, sector_rows, author_rows, order_rows = {}, [], [], [], []
    for corpus in range(2):
        for frame in range(2):
            f = native[corpus, frame]
            total = cycles(f)[2]
            baseline = a["corners"][corpus, frame, ..., 0].reshape(18, 18, 18)
            errors.append(float(np.max(np.abs(cycles(baseline)[2]))))
            arrays[f"total_{corpus}_{frame}"] = total
            for account in ("integral", "shapley"):
                components = a[account][corpus, frame].reshape(18, 18, 18, 3)
                errors.append(float(np.max(np.abs(components.sum(-1) - f + baseline))))
                pieces = [cycles(components[..., k])[2] for k in range(3)]
                errors.append(float(np.max(np.abs(sum(pieces) - total))))
                for factor, piece in zip(spec["factors"], pieces, strict=True):
                    arrays[f"{account}_{factor}_{corpus}_{frame}"] = piece
                    stats = M.current_statistics(piece, total)
                    _, parts = T.project(piece, bases)
                    _, total_parts = T.project(total, bases)
                    errors.append(float(np.max(np.abs(sum(parts.values()) - piece))))
                    for q in range(18):
                        header = {
                            "corpus": ("source", "application")[corpus],
                            "frame": frame,
                            "account": account,
                            "factor": factor,
                            "field": spec["fields"][q // 6],
                            "target": spec["targets"][q % 6],
                        }
                        metrics.append(
                            {**header, **{k: float(v[q]) for k, v in stats.items()}}
                        )
                        actual = np.einsum("aij,ij->a", difference, piece[..., q])
                        for u, value in enumerate(actual):
                            author_rows.append(
                                {**header, "author_pair": u, "contrast": float(value)}
                            )
                    for name, part in parts.items():
                        arrays[f"{account}_{factor}_{corpus}_{frame}_{name}"] = part
                        statistics = M.current_statistics(part, total_parts[name])
                        for q in range(18):
                            sector_rows.append(
                                {
                                    "corpus": ("source", "application")[corpus],
                                    "frame": frame,
                                    "account": account,
                                    "factor": factor,
                                    "sector": name,
                                    "field": spec["fields"][q // 6],
                                    "target": spec["targets"][q % 6],
                                    **{k: float(v[q]) for k, v in statistics.items()},
                                }
                            )
            for order in range(6):
                for k, factor in enumerate(spec["factors"]):
                    piece = cycles(
                        a["orders"][corpus, frame, ..., order, k].reshape(18, 18, 18)
                    )[2]
                    stats = M.current_statistics(piece, total)
                    for q in range(18):
                        order_rows.append(
                            {
                                "corpus": ("source", "application")[corpus],
                                "frame": frame,
                                "order": order,
                                "factor": factor,
                                "field": spec["fields"][q // 6],
                                "target": spec["targets"][q % 6],
                                **{key: float(v[q]) for key, v in stats.items()},
                            }
                        )
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError(
            "Native/path/sector completeness mismatch: " + str(max(errors))
        )
    np.savez_compressed(dest / "cycles.npz", **arrays)
    table(dest / "component_metrics.csv", metrics)
    table(dest / "sector_metrics.csv", sector_rows)
    table(dest / "author_contrasts.csv", author_rows)
    table(dest / "order_metrics.csv", order_rows)
    elapsed = time.monotonic() - start
    if elapsed > spec["max_analysis_seconds"]:
        raise TimeoutError("Finite analysis exceeded declared budget")
    write(
        dest / "summary.json",
        {
            "status": "PENDING_ALTERNATE_HM100",
            "native_replay_error": replay,
            "max_account_error": max(errors),
            "component_rows": len(metrics),
            "sector_rows": len(sector_rows),
            "author_rows": len(author_rows),
            "order_rows": len(order_rows),
            "analysis_seconds": elapsed,
            "new_neural_calls": 0,
            "predictor_fit": False,
            "original_scope_complete": False,
        },
    )
    write(out / "04_seal.json", seal(out, dest))
    print(
        "ANALYZED HM100;native replay", replay, "account error", max(errors), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compile", "export", "cold", "evaluate")
    )
    args = parser.parse_args()
    spec, out = settings()
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare(spec, out)
        elif args.stage == "compile":
            compile_source(spec, out)
        elif args.stage == "export":
            export(spec, out)
        elif args.stage == "cold":
            cold(out)
        else:
            evaluate(spec, out)
