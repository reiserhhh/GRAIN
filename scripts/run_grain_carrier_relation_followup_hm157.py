#!/usr/bin/env python3
"""Post-outcome finite relation-only discriminator, not a precision rescue."""

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
from scripts import run_grain_carrier_edit_transport_hm157 as R

DEFAULT = R.DEFAULT / "relation_only_followup_r1"
CODE = (
    "scripts/run_grain_carrier_relation_followup_hm157.py",
    "docs/GRAIN_CARRIER_RELATION_FOLLOWUP_HM157.md",
)


def check(out):
    """Authenticate this follow-up and its unchanged sealed primary inputs."""
    m = R.R.read(out / "manifest.json")
    R.check(ROOT / m["parent"])
    for name, checksum in m["hashes"].items():
        if R.H.digest(ROOT / name) != checksum:
            raise ValueError("Follow-up sealed dependency changed: " + name)
    if R.H.digest(out / "design.json") != m["design_sha256"]:
        raise ValueError("Follow-up support changed")
    return m


def prepare(out):
    """Seal the missing relation-only arm after acknowledging primary outcomes."""
    parent = R.DEFAULT
    primary = R.check(parent)
    for seal in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.R.read(parent / seal).items():
            if R.H.digest(parent / name) != checksum:
                raise ValueError("Primary observation changed")
    design = [
        {"batch": 0, "row": 0, "mode": "self"},
        {"batch": 1, "row": 15, "mode": "self"},
    ]
    for axis, row in itertools.product(("e", "n"), range(16)):
        design.append({"batch": len(design), "row": row, "mode": axis + "_relation"})
    out.mkdir(parents=True, exist_ok=False)
    (out / "captures").mkdir()
    R.P.write_json(out / "design.json", design)
    hashes = {p: R.H.digest(ROOT / p) for p in CODE}
    for name in (
        "manifest.json",
        "capture_seal.json",
        "output_seal.json",
        "rows.json",
        "queries.json",
        "source_operations.npz",
    ):
        path = str((parent / name).relative_to(ROOT))
        hashes[path] = R.H.digest(ROOT / path)
    R.P.write_json(
        out / "manifest.json",
        {
            "packet": "HM157_POST_OUTCOME_RELATION_ONLY",
            "parent": str(parent.relative_to(ROOT)),
            "hashes": hashes,
            "design_sha256": R.H.digest(out / "design.json"),
            "model_weights_sha256": primary["model_weights_sha256"],
            "batches": 34,
            "collection_seconds_cap": 150,
            "original_scope_complete": False,
            "reason": "Primary full versus marginal does not separate relation-alone action from reader interaction. Add only missing arm;no operator/query/weight change",
        },
    )
    R.R.status(out, "PREPARED", batches_done=0)
    print({"prepared": 34, "post_outcome": True})


def collect(out):
    """Execute just the missing arm once with its own finite cap."""
    m = check(out)
    if R.R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("No automatic follow-up replay")
    parent, start = ROOT / m["parent"], time.monotonic()
    rows, queries = (R.R.read(parent / f"{n}.json") for n in ("rows", "queries"))
    design = R.R.read(out / "design.json")
    natural = {
        e["row"]: R.R.arrays(parent / "captures" / f"{e['batch']:03d}.npz")
        for e in R.R.read(parent / "design.json")
        if e["mode"] == "natural"
    }
    source, done, pending = R.R.arrays(parent / "source_operations.npz"), 0, None
    try:
        R.R.status(out, "LOADING", pid=os.getpid(), batches_done=0)
        observer = R.R.J.offline_model(m)
        for entry in design:
            if time.monotonic() - start > m["collection_seconds_cap"]:
                raise TimeoutError("Finite150second follow-up cap;no renewal")
            pending, row = entry, rows[entry["row"]]
            base, donors = natural[row["id"]], []
            axis = 0 if entry["mode"].startswith("e_") else 1
            slots = (
                list(range(len(row["history"])))
                if entry["mode"] == "self"
                else row["scopes"]["neuroticism" if axis == 0 else "extraversion"]
            )
            for name in ("history_key", "history_value"):
                if entry["mode"] == "self":
                    donors.append(base[name])
                else:
                    cube = source[f"{row['condition']}_{row['order']}_{name}"]
                    delta = R.M.edit_increment(
                        cube, row["e"], row["n"], axis
                    ) - R.M.edit_increment(
                        cube, row["e"], row["n"], axis, marginal=True
                    )
                    donors.append(R.M.apply_increment(base[name], delta, slots))
            result = R.R.native_batch(observer, row, queries, tuple(donors))
            if not np.array_equal(result["prefix_states"], base["prefix_states"]):
                raise ValueError("Follow-up changed prefix")
            if (
                entry["mode"] == "self"
                and np.max(abs(result["selected_logits"] - base["selected_logits"]))
                > 1e-4
            ):
                raise ValueError("Follow-up self contrast failed")
            result["donor_key"] = donors[0]
            result["donor_value"] = donors[1]
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
                    "total": 34,
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
    """Decompose qualified action into marginal, relation and reader interaction."""
    m = check(out)
    if (
        R.R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("Complete one-shot follow-up analysis only")
    parent = ROOT / m["parent"]
    rows = R.R.read(parent / "rows.json")
    observations = {}
    for root in (parent, out):
        for entry in R.R.read(root / "design.json"):
            a = R.R.arrays(root / "captures" / f"{entry['batch']:03d}.npz")[
                "selected_logits"
            ].astype(float)
            observations[entry["row"], entry["mode"]] = a[:, 0] - a[:, 1]
    table = []
    max_identity = 0.0
    for c, o, axis in itertools.product(range(2), range(2), ("e", "n")):
        group = [r for r in rows if (r["condition"], r["order"]) == (c, o)]
        fields = {
            mode: np.stack(
                [
                    observations[r["id"], mode].reshape(2, 3, 2).transpose(2, 0, 1)
                    for r in group
                ]
            ).reshape(2, 2, 2, 2, 3)
            for mode in (
                "natural",
                axis + "_source_full",
                axis + "_source_marginal",
                axis + "_relation",
                axis + "_native",
            )
        }
        base = fields["natural"]
        marginal = fields[axis + "_source_marginal"] - base
        relation = fields[axis + "_relation"] - base
        full = fields[axis + "_source_full"] - base
        interaction = full - marginal - relation
        max_identity = max(
            max_identity, float(np.max(abs(full - (marginal + relation + interaction))))
        )
        for component, value in (
            ("marginal", marginal),
            ("relation", relation),
            ("interaction", interaction),
            ("full", full),
            ("native", fields[axis + "_native"] - base),
        ):
            coefficients = R.M.driver_coefficients(value)
            for mask, target, family in itertools.product(range(8), range(2), range(3)):
                table.append(
                    {
                        "condition": c,
                        "order": o,
                        "axis": axis,
                        "active_later": (axis == "e" and o == 0)
                        or (axis == "n" and o == 1),
                        "component": component,
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
    active = [
        r
        for r in table
        if r["active_later"] and r["mask"] == (5 if r["axis"] == "e" else 6)
    ]
    reference = {
        (r["condition"], r["axis"], r["target"], r["family"]): r["coefficient"]
        for r in active
        if r["component"] == "native"
    }
    full = {
        (r["condition"], r["axis"], r["target"], r["family"]): r["coefficient"]
        for r in active
        if r["component"] == "full"
    }
    summary = {
        "status": "POST_OUTCOME_FINITE_RELATION_DISCRIMINATOR_COMPLETE_NOT_GLOBAL",
        "components": {},
        "max_identity_error": max_identity,
        "original_scope_complete": False,
    }
    for component in ("marginal", "relation", "interaction", "full"):
        records = [r for r in active if r["component"] == component]
        x = np.array([r["coefficient"] for r in records])
        y = np.array(
            [
                reference[r["condition"], r["axis"], r["target"], r["family"]]
                for r in records
            ]
        )
        z = np.array(
            [full[r["condition"], r["axis"], r["target"], r["family"]] for r in records]
        )
        summary["components"][component] = {
            "qualified_count": len(records),
            "same_native_direction": int(np.sum(x * y > 0)),
            "qualified_rms": float(np.sqrt(np.mean(x * x))),
            "signed_projection_on_full": float(x @ z / (z @ z)),
            "relative_difference_from_full": float(
                np.linalg.norm(x - z) / np.linalg.norm(z)
            ),
        }
    R.P.write_json(out / "summary.json", summary)
    R.P.write_json(
        out / "output_seal.json",
        {p: R.H.digest(out / p) for p in ("summary.json", "components.csv")},
    )
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "collect", "analyze"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[a.phase](a.output)
