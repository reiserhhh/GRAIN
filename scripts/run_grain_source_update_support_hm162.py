#!/usr/bin/env python3
"""Separate source component/support omissions from application-context defects."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_exchange_support as X
from grain_research.construction import math_source_update_support as M
from scripts import run_grain_endpoint_action_hm161 as E

R, C, Q = E.R, E.C, E.Q
DEFAULT = ROOT / "results/grain_math_response_v1/source_update_support_r1"
CONFIG = "configs/grain_source_update_support_hm162.json"
CODE = (
    CONFIG,
    "grain_research/construction/math_source_update_support.py",
    "scripts/run_grain_source_update_support_hm162.py",
    "scripts/verify_grain_source_update_support_hm162.py",
    "tests/test_grain_source_update_support.py",
    "docs/GRAIN_SOURCE_UPDATE_SUPPORT_PLAN_HM162.md",
    "grain_research/construction/math_exchange_support.py",
    "grain_research/construction/math_target_carrier_closure.py",
)


def check(out):
    """Authenticate fixed source and endpoint dependencies and new arm inputs."""
    m = R.read(out / "manifest.json")
    E.check(ROOT / m["parent"])
    for name, checksum in m["hashes"].items():
        if C.H.digest(ROOT / name) != checksum:
            raise ValueError("Sealed dependency changed: " + name)
    for name, checksum in m["inputs"].items():
        if C.H.digest(out / name) != checksum:
            raise ValueError("Fixed new design changed: " + name)
    return m


def prepare(out):
    """Register the missing three arms without observing new neural responses."""
    config = R.read(ROOT / CONFIG)
    parent = ROOT / config["parent"]
    old = E.check(parent)
    E.G.authenticate_results(parent)
    rows, queries = (R.read(parent / f"{name}.json") for name in ("rows", "queries"))
    if len(rows) != 16 or len(queries["definitions"]) != 16:
        raise ValueError("Frozen complete HM161 support required")
    design = [
        {"batch": i, "row": r, "mode": "self"}
        for i, r in enumerate(config["self_rows"])
    ]
    for row, axis in config["replay_controls"]:
        design.append(
            {"batch": len(design), "row": row, "mode": axis + "_relation_replay"}
        )
    for axis, operation, row in itertools.product(
        ("e", "n"), config["new_operations"], rows
    ):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": axis + "_" + operation}
        )
    if len(design) != config["batches"]:
        raise ValueError("No arm deletion, sample expansion or design rescue")
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        C.P.write_json(out / f"{name}.json", value)
    dependencies = list(CODE)
    for name in (
        "manifest.json",
        "capture_seal.json",
        "output_seal.json",
        "alternate_audit.json",
        "design.json",
    ):
        dependencies.append(str((parent / name).relative_to(ROOT)))
    source_root = ROOT / old["parent"]
    dependencies.append(str((source_root / "source_operations.npz").relative_to(ROOT)))
    C.P.write_json(
        out / "manifest.json",
        {
            **config,
            "hashes": {name: C.H.digest(ROOT / name) for name in dependencies},
            "inputs": {
                name: C.H.digest(out / name)
                for name in ("rows.json", "queries.json", "design.json")
            },
            "source_operations": str(
                (source_root / "source_operations.npz").relative_to(ROOT)
            ),
            "model_weights_sha256": old["model_weights_sha256"],
            "maximum_tokens": old["maximum_tokens"],
        },
    )
    R.status(out, "PREPARED", batches_done=0, pending=None)
    print(
        {"prepared_batches": len(design), "new_inquiries": len(design) * 16}, flush=True
    )


def natural_captures(parent):
    """Read authenticated actual application memories without counterfactual fitting."""
    E.G.authenticate_results(parent)
    return {
        d["row"]: R.arrays(parent / "captures" / f"{d['batch']:03d}.npz")
        for d in R.read(parent / "design.json")
        if d["mode"] == "natural"
    }


def arguments(row, mode, base, source):
    """Generate source-only K/V donors and their registered time support."""
    if mode == "self":
        return (base["history_key"], base["history_value"]), list(
            range(len(row["history"]))
        )
    axis, operation = mode.split("_", 1)
    if operation == "relation_replay":
        operation = "relation_other"
    donors, support = [], None
    for channel in ("key", "value"):
        name = "history_" + channel
        donor, slots = M.transported_memory(
            base[name],
            source[f"{row['condition']}_{row['order']}_{name}"],
            row,
            axis,
            operation,
        )
        donors.append(donor)
        if support is not None and slots != support:
            raise ValueError("Key and value must share registered support")
        support = slots
    return tuple(donors), support


def collect(out):
    """Execute the100 registered batches once, retaining failure checkpoints."""
    start, m = time.monotonic(), check(out)
    if R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("Started scientific collections cannot be replayed")
    parent = ROOT / m["parent"]
    old, source = natural_captures(parent), R.arrays(ROOT / m["source_operations"])
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    replay = {}
    for d in R.read(parent / "design.json"):
        if d["mode"].endswith("_relation"):
            replay[d["row"], d["mode"][0]] = d
    done, pending = 0, None
    try:
        R.status(out, "LOADING", pid=os.getpid(), batches_done=0, pending=None)
        observer = R.J.offline_model(m)
        C.P.write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > m["collection_seconds_cap"]:
                raise TimeoutError("Finite600s collection cap, no automatic renewal")
            pending, row = entry, rows[entry["row"]]
            R.status(out, "RUNNING", pid=os.getpid(), batches_done=done, pending=entry)
            base = old[row["id"]]
            donors, slots = arguments(row, entry["mode"], base, source)
            result = R.native_batch(observer, row, queries, donors)
            prefix_error = float(
                np.max(abs(result["prefix_states"] - base["prefix_states"]))
            )
            if prefix_error > 1e-4 or float(result["cache_spread"]) > 1e-4:
                raise ValueError("Suffix operation altered raw historical computation")
            result["prefix_error"] = np.asarray(prefix_error)
            if entry["mode"] == "self" or entry["mode"].endswith("_replay"):
                reference = base
                if entry["mode"] != "self":
                    ref = replay[row["id"], entry["mode"][0]]
                    reference = R.arrays(
                        parent / "captures" / f"{ref['batch']:03d}.npz"
                    )
                error = float(
                    np.max(
                        abs(result["selected_logits"] - reference["selected_logits"])
                    )
                )
                if error > 1e-4:
                    raise ValueError("Self/source-relation replay control failed")
                result["control_error"] = np.asarray(error)
            result["donor_key"], result["donor_value"] = donors
            result["patched_slots"] = np.asarray(slots)
            np.savez_compressed(
                out / "captures" / f"{entry['batch']:03d}.npz", **result
            )
            done += 1
            R.status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=None,
                seconds=time.monotonic() - start,
            )
            print(
                {"batch": done, "total": len(design), "mode": entry["mode"]}, flush=True
            )
        C.P.write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): C.H.digest(p)
                for p in sorted((out / "captures").glob("*.npz"))
            },
        )
        R.status(
            out,
            "COLLECTED",
            pid=os.getpid(),
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=done * 16,
        )
    except Exception as exc:
        R.status(
            out,
            "FAILED",
            pid=os.getpid(),
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=repr(exc),
        )
        raise


def observations(out):
    """Merge frozen natural/relation observations and newly collected arms."""
    m = check(out)
    parent = ROOT / m["parent"]
    E.G.authenticate_results(parent)
    for name, checksum in R.read(out / "capture_seal.json").items():
        if C.H.digest(out / name) != checksum:
            raise ValueError("New actual capture changed")
    records, scores = [], {}
    for root in (parent, out):
        for d in R.read(root / "design.json"):
            mode = d["mode"]
            if root == parent and mode == "self":
                continue
            normalized = (
                mode + "_other"
                if root == parent and mode.endswith("_relation")
                else mode
            )
            a = R.arrays(root / "captures" / f"{d['batch']:03d}.npz")
            value = E.M.endpoint_scores(a["selected_logits"], a["normalizer"])
            scores[d["row"], normalized] = value
            metadata = {
                key: a[key] for key in ("control_error", "prefix_error") if key in a
            }
            records.append(
                (
                    d,
                    normalized,
                    "inherited" if root == parent else "new",
                    metadata,
                    value,
                )
            )
    return records, scores


def analyze(out):
    """Report complete endpoint actions and the non-deleted construction defect."""
    start, m = time.monotonic(), check(out)
    if (
        R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("Complete unanalysed collection required")
    rows, queries = (R.read(out / f"{name}.json") for name in ("rows", "queries"))
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    records, scores = observations(out)
    endpoints, controls = [], []
    for d, normalized, origin, a, values in records:
        if origin == "new" and "control_error" in a:
            controls.append({**d, "error": float(a["control_error"])})
        for i, q in enumerate(queries["definitions"]):
            endpoints.append(
                {
                    **d,
                    "mode": normalized,
                    "origin": origin,
                    **q,
                    **{k: rows[d["row"]][k] for k in ("condition", "order", "e", "n")},
                    **{name: float(value[i]) for name, value in values.items()},
                }
            )
    table, compatibility, fields = [], [], {}
    for c, o, axis, target, family, measure in itertools.product(
        range(2), range(2), ("e", "n"), E.M.AXES, E.M.FAMILIES, E.M.MEASURES
    ):
        if time.monotonic() - start > m["analysis_seconds_cap"]:
            raise TimeoutError("Finite analysis cap, no renewal")
        cubes = {name: np.zeros((2, 2, 2)) for name in (*M.OPERATIONS, "natural")}
        for ei, ni, pi in itertools.product(range(2), repeat=3):
            e, n, person = 2 * ei - 1, 2 * ni - 1, 2 * pi - 1
            rid = indexed[c, o, e, n]
            partner = E.M.natural_partner(rows[rid], axis, indexed)
            qi = queries["definitions"].index(
                {"axis": target, "family": family, "person": person}
            )
            baseline = scores[rid, "natural"][measure][qi]
            cubes["natural"][ei, ni, pi] = (
                scores[partner, "natural"][measure][qi] - baseline
            )
            for operation in M.OPERATIONS:
                cubes[operation][ei, ni, pi] = (
                    scores[rid, axis + "_" + operation][measure][qi] - baseline
                )
        cubes["factorial_interaction"] = (
            cubes["full_closure"]
            - cubes["full_other"]
            - cubes["relation_closure"]
            + cubes["relation_other"]
        )
        for operation, cube in cubes.items():
            coefficients = Q.characters(cube)
            fields[c, o, axis, target, family, measure, operation] = coefficients
            for mask, value in enumerate(coefficients):
                table.append(
                    {
                        "condition": c,
                        "order": o,
                        "axis": axis,
                        "target": target,
                        "family": family,
                        "measure": measure,
                        "operation": operation,
                        "mask": mask,
                        "coefficient": float(value),
                    }
                )
            parts = X.exchange_parts(coefficients, axis)
            compatibility.append(
                {
                    "condition": c,
                    "order": o,
                    "axis": axis,
                    "target": target,
                    "family": family,
                    "measure": measure,
                    "operation": operation,
                    "qualified_coefficient": float(
                        coefficients[5 if target == "extraversion" else 6]
                    ),
                    **{
                        key: value
                        for key, value in parts.items()
                        if key not in ("even", "odd")
                    },
                }
            )
    transport = []
    for axis, target, family, measure, operation in itertools.product(
        ("e", "n"), E.M.AXES, E.M.FAMILIES, E.M.MEASURES, M.OPERATIONS
    ):
        mask = 5 if target == "extraversion" else 6
        values = [
            fields[c, o, axis, target, family, measure, operation]
            for c, o in itertools.product(range(2), repeat=2)
        ]
        native = [
            fields[c, o, axis, target, family, measure, "natural"]
            for c, o in itertools.product(range(2), repeat=2)
        ]
        parts = X.exchange_parts(np.stack(values), axis)
        transport.append(
            {
                "axis": axis,
                "target": target,
                "family": family,
                "measure": measure,
                "operation": operation,
                **E.A.action_description(
                    np.array(values)[:, mask], np.array(native)[:, mask]
                ),
                "even_energy_fraction": parts["even_energy_fraction"],
                "even_coefficient_rms": parts["even_coefficient_rms"],
            }
        )
    natural = natural_captures(ROOT / m["parent"])
    source = R.arrays(ROOT / m["source_operations"])
    defects = []
    for row, axis, operation, channel in itertools.product(
        rows, ("e", "n"), M.OPERATIONS, ("key", "value")
    ):
        if time.monotonic() - start > m["analysis_seconds_cap"]:
            raise TimeoutError("Finite defect analysis cap, no renewal")
        partner = E.M.natural_partner(row, axis, indexed)
        name = "history_" + channel
        terms = M.defect_terms(
            natural[row["id"]][name],
            natural[partner][name],
            source[f"{row['condition']}_{row['order']}_{name}"],
            row,
            axis,
            operation,
        )
        defects.append(
            {
                "row": row["id"],
                "condition": row["condition"],
                "order": row["order"],
                "e": row["e"],
                "n": row["n"],
                "axis": axis,
                "operation": operation,
                "channel": channel,
                **M.energy_account(terms),
            }
        )
    summary = {
        "packet": "HM162",
        "status": "FINITE_SOURCE_UPDATE_SUPPORT_COMPLETE_NOT_GLOBAL",
        "collection": R.read(out / "collection_status.json"),
        "actual_new_batches": 100,
        "inherited_observation_batches": 48,
        "source_answer_fitting": False,
        "controls": controls,
        "transport": transport,
        "max_defect_account_error": max(r["account_max_error"] for r in defects),
        "new_neural_fits_API_private": [0, 0, 0],
        "analysis_seconds": time.monotonic() - start,
        "original_scope_complete": False,
        "interpretation": "Source-complete operation tested,not presumed coherent application-natural update. All context/omission/cross terms retained; model choices are not observed human behavior.",
    }
    for name, values in (
        ("endpoints.csv", endpoints),
        ("characters.csv", table),
        ("compatibility.csv", compatibility),
        ("transport.csv", transport),
        ("memory_defects.csv", defects),
    ):
        E.write_csv(out / name, values)
    C.P.write_json(out / "summary.json", summary)
    C.P.write_json(
        out / "output_seal.json",
        {
            p.name: C.H.digest(p)
            for p in out.iterdir()
            if p.suffix == ".csv" or p.name == "summary.json"
        },
    )
    print(
        {
            "status": summary["status"],
            "matching_prospective": [
                r
                for r in transport
                if r["measure"] == "restricted_yes"
                and r["family"] == "prospective"
                and ((r["axis"] == "e") == (r["target"] == "extraversion"))
            ],
            "max_defect_account_error": summary["max_defect_account_error"],
        },
        flush=True,
    )


def main():
    """Keep preparation, finite execution and sealed analysis separate."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "collect", "analyze"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
        args.output.resolve()
    )


if __name__ == "__main__":
    main()
