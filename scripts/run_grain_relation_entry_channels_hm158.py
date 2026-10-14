#!/usr/bin/env python3
"""Discriminate K/V entry channels of the source-only relational operation."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relation_entry_channels as M
from scripts import run_grain_carrier_relation_followup_hm157 as F

R = F.R
DEFAULT = ROOT / "results/grain_math_response_v1/relation_entry_channels_r1"
CODE = (
    "grain_research/construction/math_relation_entry_channels.py",
    "scripts/run_grain_relation_entry_channels_hm158.py",
    "scripts/verify_grain_relation_entry_channels_hm158.py",
    "tests/test_grain_relation_entry_channels.py",
    "docs/GRAIN_RELATION_ENTRY_CHANNEL_PLAN_HM158.md",
)


def authenticate_results(root):
    """Check both actual captures and numerical outputs before reusing them."""
    for seal in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.R.read(root / seal).items():
            if R.H.digest(root / name) != checksum:
                raise ValueError("Inherited observation changed: " + str(root / name))


def check(out):
    """Authenticate scientific code, source operation and inherited captures."""
    m = R.R.read(out / "manifest.json")
    R.check(ROOT / m["parent"])
    F.check(ROOT / m["followup"])
    for path, checksum in m["hashes"].items():
        if R.H.digest(ROOT / path) != checksum:
            raise ValueError("Sealed scientific dependency changed: " + path)
    if R.H.digest(out / "design.json") != m["design_sha256"]:
        raise ValueError("Sealed design changed")
    return m


def prepare(out):
    """Freeze the missing entry-channel design without new neural observations."""
    old = R.check(R.DEFAULT)
    F.check(F.DEFAULT)
    for root in (R.DEFAULT, F.DEFAULT):
        authenticate_results(root)
    design = [
        {"batch": 0, "row": 0, "mode": "self"},
        {"batch": 1, "row": 15, "mode": "self"},
        {"batch": 2, "row": 0, "mode": "e_joint_control"},
        {"batch": 3, "row": 15, "mode": "n_joint_control"},
    ]
    for axis, channel, row in itertools.product(
        ("e", "n"), ("key", "value"), range(16)
    ):
        design.append({"batch": len(design), "row": row, "mode": axis + "_" + channel})
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    R.P.write_json(out / "design.json", design)
    hashes = {name: R.H.digest(ROOT / name) for name in CODE}
    for root in (R.DEFAULT, F.DEFAULT):
        for name in ("manifest.json", "capture_seal.json", "output_seal.json"):
            path = str((root / name).relative_to(ROOT))
            hashes[path] = R.H.digest(ROOT / path)
    for name in ("rows.json", "queries.json", "source_operations.npz"):
        path = str((R.DEFAULT / name).relative_to(ROOT))
        hashes[path] = R.H.digest(ROOT / path)
    R.P.write_json(
        out / "manifest.json",
        {
            "packet": "HM158",
            "parent": str(R.DEFAULT.relative_to(ROOT)),
            "followup": str(F.DEFAULT.relative_to(ROOT)),
            "hashes": hashes,
            "design_sha256": R.H.digest(out / "design.json"),
            "model_weights_sha256": old["model_weights_sha256"],
            "batches": len(design),
            "inquiry_sequences": len(design) * 12,
            "collection_seconds_cap": 300,
            "analysis_seconds_cap": 180,
            "verification_seconds_cap": 120,
            "original_scope_complete": False,
            "scope": "Post-outcome finite K/V entry discriminator, no predictor fitting; inherited new-expression bank is not blind new material",
        },
    )
    R.R.status(out, "PREPARED", batches_done=0, pending=None)
    print({"prepared": len(design), "inquiries": len(design) * 12})


def inherited_captures(parent, followup):
    """Load only actual natural/native and joint-relation observations."""
    captured = {}
    for root in (parent, followup):
        for item in R.R.read(root / "design.json"):
            if item["mode"] == "natural" or item["mode"].endswith(
                ("_native", "_relation")
            ):
                captured[item["row"], item["mode"]] = R.R.arrays(
                    root / "captures" / f"{item['batch']:03d}.npz"
                )
    return captured


def arguments(row, mode, base, source):
    """Construct a single-entry or control edit on the fixed other carrier."""
    if mode == "self":
        return (base["history_key"], base["history_value"]), list(
            range(len(row["history"]))
        )
    axis = mode[0]
    slots = row["scopes"]["neuroticism" if axis == "e" else "extraversion"]
    donors = []
    for channel in ("key", "value"):
        name = "history_" + channel
        if mode.endswith((channel, "joint_control")):
            delta = M.relation_increment(
                source[f"{row['condition']}_{row['order']}_{name}"], row["e"], row["n"]
            )
            donors.append(R.M.apply_increment(base[name], delta, slots))
        else:
            donors.append(base[name])
    return tuple(donors), slots


def collect(out):
    """Run one finite missing-arm collection, preserving errors without retries."""
    start = time.monotonic()
    m = check(out)
    if R.R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("Started collections cannot be replayed automatically")
    parent, followup = ROOT / m["parent"], ROOT / m["followup"]
    old = inherited_captures(parent, followup)
    rows, queries = (R.R.read(parent / f"{name}.json") for name in ("rows", "queries"))
    source, design = (
        R.R.arrays(parent / "source_operations.npz"),
        R.R.read(out / "design.json"),
    )
    done, pending = 0, None
    try:
        R.R.status(out, "LOADING", pid=os.getpid(), batches_done=0)
        observer = R.R.J.offline_model(m)
        R.P.write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > m["collection_seconds_cap"]:
                raise TimeoutError("Finite300second collection cap; no renewal")
            pending, row = entry, rows[entry["row"]]
            R.R.status(
                out, "RUNNING", pid=os.getpid(), batches_done=done, pending=pending
            )
            base = old[row["id"], "natural"]
            donors, slots = arguments(row, entry["mode"], base, source)
            result = R.R.native_batch(observer, row, queries, donors)
            if not np.array_equal(result["prefix_states"], base["prefix_states"]):
                raise ValueError("Suffix entry edit changed raw prefix computation")
            if float(result["cache_spread"]) != 0:
                raise ValueError("Historical states changed across queries")
            mode = entry["mode"]
            if mode == "self" or mode.endswith("joint_control"):
                reference = (
                    base if mode == "self" else old[row["id"], mode[0] + "_relation"]
                )
                error = float(
                    np.max(
                        abs(result["selected_logits"] - reference["selected_logits"])
                    )
                )
                if error > 1e-4:
                    raise ValueError("Explicit self/joint replay control failed")
                result["control_error"] = np.asarray(error)
            result["donor_key"], result["donor_value"] = donors
            result["patched_slots"] = np.asarray(slots)
            np.savez_compressed(
                out / "captures" / f"{entry['batch']:03d}.npz", **result
            )
            done += 1
            R.R.status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=entry,
                seconds=time.monotonic() - start,
            )
            print(
                {
                    "batch": done,
                    "total": len(design),
                    "seconds": round(time.monotonic() - start, 3),
                },
                flush=True,
            )
        R.P.write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): R.H.digest(p)
                for p in sorted((out / "captures").glob("*.npz"))
            },
        )
        R.R.status(
            out,
            "COLLECTED",
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=done * 12,
        )
    except Exception as exc:
        R.R.status(
            out,
            "FAILED",
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=repr(exc),
        )
        raise


def analyze(out):
    """Report complete entry effects, qualified actions and upstream nulls."""
    start, m = time.monotonic(), check(out)
    if (
        R.R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("Analysis requires complete unanalysed collection")
    parent, followup = ROOT / m["parent"], ROOT / m["followup"]
    captured = inherited_captures(parent, followup)
    rows = R.R.read(parent / "rows.json")
    for entry in R.R.read(out / "design.json"):
        captured[entry["row"], entry["mode"]] = R.R.arrays(
            out / "captures" / f"{entry['batch']:03d}.npz"
        )
    table, upstream, identity = [], 0.0, 0.0
    for c, order, axis in itertools.product(range(2), range(2), ("e", "n")):
        if time.monotonic() - start > m["analysis_seconds_cap"]:
            raise TimeoutError("Finite analysis cap; no automatic renewal")
        group = [r for r in rows if (r["condition"], r["order"]) == (c, order)]
        fields = {}
        for mode in (
            "natural",
            axis + "_key",
            axis + "_value",
            axis + "_relation",
            axis + "_native",
        ):
            fields[mode] = np.stack(
                [
                    (
                        captured[r["id"], mode]["selected_logits"].astype(float)[:, 0]
                        - captured[r["id"], mode]["selected_logits"].astype(float)[:, 1]
                    )
                    .reshape(2, 3, 2)
                    .transpose(2, 0, 1)
                    for r in group
                ]
            ).reshape(2, 2, 2, 2, 3)
        components = M.entry_components(
            fields["natural"],
            fields[axis + "_key"],
            fields[axis + "_value"],
            fields[axis + "_relation"],
        )
        identity = max(
            identity,
            float(
                np.max(
                    abs(
                        components["joint_entry"]
                        - components["key_entry"]
                        - components["value_entry"]
                        - components["recursive_interaction"]
                    )
                )
            ),
        )
        components["native"] = fields[axis + "_native"] - fields["natural"]
        active = (axis == "e" and order == 0) or (axis == "n" and order == 1)
        for name, values in components.items():
            if not active and name != "native":
                upstream = max(upstream, float(np.max(abs(values))))
            coefficients = R.M.driver_coefficients(values)
            for mask, target, family in itertools.product(range(8), range(2), range(3)):
                table.append(
                    {
                        "condition": c,
                        "order": order,
                        "axis": axis,
                        "active_later": active,
                        "component": name,
                        "mask": mask,
                        "target": R.Q.AXES[target],
                        "family": R.Q.FAMILIES[family],
                        "coefficient": float(coefficients[mask, target, family]),
                    }
                )
    with (out / "components.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    key = lambda r: (
        r["condition"],
        r["order"],
        r["axis"],
        r["mask"],
        r["target"],
        r["family"],
    )
    active = [r for r in table if r["active_later"]]
    joint = {
        key(r): r["coefficient"] for r in active if r["component"] == "joint_entry"
    }
    native = {key(r): r["coefficient"] for r in active if r["component"] == "native"}
    summary = {
        "status": "FINITE_RELATION_ENTRY_MECHANISM_COMPLETE_NOT_GLOBAL",
        "qualified": {},
        "all_characters": {},
        "upstream_max_abs": upstream,
        "max_component_identity_error": identity,
        "original_scope_complete": False,
    }
    for component in (
        "key_entry",
        "value_entry",
        "recursive_interaction",
        "joint_entry",
    ):
        for qualified, section in ((True, "qualified"), (False, "all_characters")):
            records = [
                r
                for r in active
                if r["component"] == component
                and (not qualified or r["mask"] == (5 if r["axis"] == "e" else 6))
            ]
            x = np.array([r["coefficient"] for r in records])
            y = np.array([joint[key(r)] for r in records])
            z = np.array([native[key(r)] for r in records])
            summary[section][component] = {
                "joint_reference": M.action_description(x, y),
                "native_reference": M.action_description(x, z),
            }
    R.P.write_json(out / "summary.json", summary)
    R.P.write_json(
        out / "output_seal.json",
        {p: R.H.digest(out / p) for p in ("summary.json", "components.csv")},
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.phase](
        args.output
    )
