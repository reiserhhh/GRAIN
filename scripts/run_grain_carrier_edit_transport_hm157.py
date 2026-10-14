#!/usr/bin/env python3
"""Test whether source actor-edit operations act on unobserved expressions."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_carrier_edit_transport as M
from grain_research.construction import math_qualified_target_field as Q
from scripts import run_grain_target_carrier_closure_hm156 as P

R, H = P.R, P.H
DEFAULT = ROOT / "results/grain_math_response_v1/carrier_edit_transport_r1"
MODES = tuple(
    f"{axis}_{mode}"
    for axis in ("e", "n")
    for mode in ("native", "source_full", "source_marginal", "source_reverse")
)
CODE = (
    "grain_research/construction/math_carrier_edit_transport.py",
    "scripts/run_grain_carrier_edit_transport_hm157.py",
    "tests/test_grain_carrier_edit_transport.py",
    "docs/GRAIN_CARRIER_EDIT_TRANSPORT_PLAN_HM157.md",
)


def check(out):
    """Authenticate prospective scientific dependencies and input support."""
    manifest = R.read(out / "manifest.json")
    for path, checksum in manifest["hashes"].items():
        if H.digest(ROOT / path) != checksum:
            raise ValueError("Sealed dependency changed: " + path)
    for path, checksum in manifest["inputs"].items():
        if H.digest(out / path) != checksum:
            raise ValueError("Prepared input changed: " + path)
    return manifest


def prepare(out):
    """Seal source operations and all new-text interventions before forwards."""
    from transformers import AutoTokenizer

    old = P.check(P.DEFAULT)
    for seal in ("capture_seal.json", "output_seal.json"):
        for path, checksum in R.read(P.DEFAULT / seal).items():
            if H.digest(P.DEFAULT / path) != checksum:
                raise ValueError("HM156 result changed: " + path)
    parent = ROOT / old["parent"]
    P.authenticate_parent(parent)
    tokenizer = AutoTokenizer.from_pretrained(str(R.J.MODEL), local_files_only=True)
    originals = [r for r in R.read(parent / "rows.json") if r["bank"] == 0]
    source = P.parent_natural(parent)
    prefix = tokenizer.encode(
        "<|im_start|>system\n" + Q.SYSTEM + "<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    rows = []
    for original in originals:
        row = {
            k: v
            for k, v in original.items()
            if k not in ("input_ids", "intervals", "body", "role")
        }
        row["body"] = M.new_body(original["body"])
        encoded = tokenizer(
            row["body"], add_special_tokens=False, return_offsets_mapping=True
        )
        row["input_ids"] = prefix + encoded["input_ids"]
        if len(row["input_ids"]) != len(original["input_ids"]):
            raise ValueError("No native-width repair or truncation")
        for axis, (old_start, old_stop) in original["intervals"].items():
            section = original["body"][old_start:old_stop]
            for before, after in M.REPLACEMENTS:
                section = section.replace(before, after)
            start = row["body"].index(section)
            stop = start + len(section)
            actual_scope = [
                j
                for j, (a, b) in enumerate(encoded["offset_mapping"])
                if b > a >= start and b <= stop
            ]
            if actual_scope != row["scopes"][axis]:
                raise ValueError("New expression changed declared carrier addresses")
        row["role"] = "new_application_expression_never_observed"
        rows.append(row)
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    for row in rows:
        for e, n in itertools.product((-1, 1), repeat=2):
            other = indexed[row["condition"], row["order"], e, n]
            if Counter(row["input_ids"]) != Counter(other["input_ids"]):
                raise ValueError("Actor edits changed complete token bag")
    out.mkdir(parents=True, exist_ok=False)
    for path in ("captures", "arguments"):
        (out / path).mkdir()
    queries = R.read(parent / "queries.json")
    design = [
        {"batch": i, "row": r["id"], "mode": "natural"} for i, r in enumerate(rows)
    ]
    for row_id in (0, 3, 12, 15):
        design.append({"batch": len(design), "row": row_id, "mode": "self"})
    for mode in MODES:
        for row in rows:
            design.append({"batch": len(design), "row": row["id"], "mode": mode})
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        P.write_json(out / f"{name}.json", value)
    operations = {}
    for c, o in itertools.product(range(2), repeat=2):
        group = [r for r in originals if (r["condition"], r["order"]) == (c, o)]
        for name in ("history_key", "history_value"):
            cube = np.stack([source[r["id"]][name] for r in group]).reshape(
                2, 2, *source[group[0]["id"]][name].shape
            )
            operations[f"{c}_{o}_{name}"] = cube
    np.savez_compressed(out / "source_operations.npz", **operations)
    hashes = dict(old["hashes"])
    hashes.update({p: H.digest(ROOT / p) for p in CODE})
    for name in ("manifest.json", "capture_seal.json", "output_seal.json"):
        path = str((P.DEFAULT / name).relative_to(ROOT))
        hashes[path] = H.digest(ROOT / path)
    P.write_json(
        out / "manifest.json",
        {
            "packet": "HM157",
            "hashes": hashes,
            "inputs": {
                p: H.digest(out / p)
                for p in (
                    "rows.json",
                    "queries.json",
                    "design.json",
                    "source_operations.npz",
                )
            },
            "model_weights_sha256": old["model_weights_sha256"],
            "batches": len(design),
            "inquiry_sequences": len(design) * 12,
            "collection_seconds_cap": 900,
            "analysis_seconds_cap": 180,
            "original_scope_complete": False,
            "source": str(parent.relative_to(ROOT)),
            "operation": "source0 actor-flip increments on unchanged other-clause rows;full,other-role-marginal,reverse;no answer fitting",
            "new_application": "16 structurally width-matched reception/interview records;no outputs observed before seal",
        },
    )
    R.status(out, "PREPARED", batches_done=0, pending=None)
    print(
        {"prepared": 16, "batches": len(design), "inquiries": len(design) * 12},
        flush=True,
    )


def arguments(row, mode, natural, indexed, operations):
    """Compile source-only or positive-control native increments on fixed rows."""
    base = natural[row["id"]]
    if mode == "self":
        return (base["history_key"], base["history_value"]), list(
            range(len(row["history"]))
        )
    axis = 0 if mode.startswith("e_") else 1
    slots = P.M.carrier_scopes(row)["e_other" if axis == 0 else "n_other"]
    donor = indexed[
        row["condition"],
        row["order"],
        -row["e"] if axis == 0 else row["e"],
        -row["n"] if axis == 1 else row["n"],
    ]
    result = []
    for name in ("history_key", "history_value"):
        if mode.endswith("native"):
            increment = natural[donor["id"]][name].astype(float) - base[name].astype(
                float
            )
        else:
            cube = operations[f"{row['condition']}_{row['order']}_{name}"]
            increment = M.edit_increment(
                cube, row["e"], row["n"], axis, marginal=mode.endswith("marginal")
            )
            if mode.endswith("reverse"):
                increment = -increment
        result.append(M.apply_increment(base[name], increment, slots))
    return tuple(result), slots


def collect(out):
    """Execute one bounded collection, preserving partials and refusing replay."""
    manifest = check(out)
    if R.read(out / "collection_status.json")["status"] != "PREPARED":
        raise RuntimeError("No automatic replay of started collection")
    start, done, pending = time.monotonic(), 0, None
    rows, queries, design = (
        R.read(out / f"{p}.json") for p in ("rows", "queries", "design")
    )
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    operations, natural = R.arrays(out / "source_operations.npz"), {}
    try:
        R.status(out, "LOADING", pid=os.getpid(), batches_done=0, pending=None)
        observer = R.J.offline_model(manifest)
        P.write_json(out / "model_manifest.json", observer.manifest())
        for entry in design:
            if time.monotonic() - start > manifest["collection_seconds_cap"]:
                raise TimeoutError("Finite900second cap; no renewal")
            pending, row = entry, rows[entry["row"]]
            R.status(
                out,
                "RUNNING",
                pid=os.getpid(),
                batches_done=done,
                pending=pending,
                seconds=time.monotonic() - start,
            )
            if entry["mode"] == "natural":
                donors, slots = None, []
            else:
                donors, slots = arguments(
                    row, entry["mode"], natural, indexed, operations
                )
            result = R.native_batch(observer, row, queries, donors)
            if float(result["cache_spread"]) > 1e-4:
                raise ValueError("Historical memory differs across queries")
            if donors is None:
                natural[row["id"]] = result
            else:
                base = natural[row["id"]]
                if np.max(abs(result["prefix_states"] - base["prefix_states"])) > 1e-4:
                    raise ValueError("Suffix-only edit altered body")
                if (
                    entry["mode"] == "self"
                    and np.max(abs(result["selected_logits"] - base["selected_logits"]))
                    > 1e-4
                ):
                    raise ValueError("Self donor changed reader")
                np.savez_compressed(
                    out / "arguments" / f"{entry['batch']:03d}.npz",
                    history_key=donors[0],
                    history_value=donors[1],
                    slots=np.asarray(slots),
                )
            np.savez_compressed(
                out / "captures" / f"{entry['batch']:03d}.npz", **result
            )
            done += 1
            print(
                {
                    "batch": done,
                    "total": len(design),
                    "mode": entry["mode"],
                    "seconds": round(time.monotonic() - start, 3),
                },
                flush=True,
            )
        P.write_json(
            out / "capture_seal.json",
            {
                str(p.relative_to(out)): H.digest(p)
                for folder in ("captures", "arguments")
                for p in sorted((out / folder).glob("*.npz"))
            },
        )
        R.status(
            out,
            "COLLECTED",
            batches_done=done,
            pending=None,
            seconds=time.monotonic() - start,
            model_batches=observer.calls,
            inquiry_sequences=done * 12,
        )
    except Exception as exc:
        R.status(
            out,
            "FAILED",
            batches_done=done,
            pending=pending,
            seconds=time.monotonic() - start,
            error=repr(exc),
        )
        raise


def analyze(out):
    """Retain full character effects and all target/wording/case comparisons."""
    check(out)
    if (
        R.read(out / "collection_status.json")["status"] != "COLLECTED"
        or (out / "summary.json").exists()
    ):
        raise RuntimeError("Only complete one-shot analysis accepted")
    for path, checksum in R.read(out / "capture_seal.json").items():
        if H.digest(out / path) != checksum:
            raise ValueError("Captured argument changed")
    rows, design = (R.read(out / f"{p}.json") for p in ("rows", "design"))
    endpoints = {}
    for entry in design:
        a = R.arrays(out / "captures" / f"{entry['batch']:03d}.npz")[
            "selected_logits"
        ].astype(float)
        endpoints[entry["row"], entry["mode"]] = a[:, 0] - a[:, 1]
    accounts, characters = [], []
    for c, o in itertools.product(range(2), repeat=2):
        group = [r for r in rows if (r["condition"], r["order"]) == (c, o)]
        for mode in ("natural", *MODES):
            cube = np.stack(
                [
                    endpoints[r["id"], mode].reshape(2, 3, 2).transpose(2, 0, 1)
                    for r in group
                ]
            ).reshape(2, 2, 2, 2, 3)
            coeff = M.driver_coefficients(cube)
            for mask, target, family in itertools.product(range(8), range(2), range(3)):
                characters.append(
                    {
                        "condition": c,
                        "order": o,
                        "mode": mode,
                        "mask": mask,
                        "target": Q.AXES[target],
                        "family": Q.FAMILIES[family],
                        "coefficient": float(coeff[mask, target, family]),
                    }
                )
        for axis, label in enumerate(("e", "n")):
            active = (axis == 0 and o == 0) or (axis == 1 and o == 1)
            for variant in ("source_full", "source_marginal", "source_reverse"):
                effect = np.stack(
                    [
                        endpoints[r["id"], f"{label}_{variant}"]
                        - endpoints[r["id"], "natural"]
                        for r in group
                    ]
                )
                reference = np.stack(
                    [
                        endpoints[r["id"], f"{label}_native"]
                        - endpoints[r["id"], "natural"]
                        for r in group
                    ]
                )
                for target, family in itertools.product(range(2), range(3)):
                    x = effect.reshape(2, 2, 2, 3, 2)[:, :, target, family]
                    y = reference.reshape(2, 2, 2, 3, 2)[:, :, target, family]
                    beta_x = M.driver_coefficients(x)[5 if axis == 0 else 6]
                    beta_y = M.driver_coefficients(y)[5 if axis == 0 else 6]
                    account = Q.action_account(x, y)
                    accounts.append(
                        {
                            "condition": c,
                            "order": o,
                            "axis": label,
                            "active_later": active,
                            "variant": variant,
                            "target": Q.AXES[target],
                            "family": Q.FAMILIES[family],
                            "qualified_effect": float(beta_x),
                            "native_qualified_effect": float(beta_y),
                            "same_direction": bool(beta_x * beta_y > 0),
                            **account,
                        }
                    )
    import csv

    for name, table in (("actions.csv", accounts), ("characters.csv", characters)):
        with (out / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    active = [r for r in accounts if r["active_later"]]
    summary = {
        "status": "FINITE_MECHANISM_COMPLETE_NOT_GLOBAL",
        "variants": {},
        "original_scope_complete": False,
    }
    for mode in ("source_full", "source_marginal", "source_reverse"):
        records = [r for r in active if r["variant"] == mode]
        projections = [
            r["signed_projection"]
            for r in records
            if r["signed_projection"] is not None
        ]
        summary["variants"][mode] = {
            "active_qualified_comparisons": len(records),
            "same_direction": sum(r["same_direction"] for r in records),
            "signed_projection_range": [min(projections), max(projections)],
            "mean_signed_projection": float(np.mean(projections)),
            "mean_action_rms": float(np.mean([r["rms"] for r in records])),
        }
    summary["upstream_max_rms"] = max(
        r["rms"] for r in accounts if not r["active_later"]
    )
    P.write_json(out / "summary.json", summary)
    P.write_json(
        out / "output_seal.json",
        {
            name: H.digest(out / name)
            for name in ("actions.csv", "characters.csv", "summary.json")
        },
    )
    print(summary, flush=True)


def main():
    """Run one explicit phase; no automatic search, replay or continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.phase](
        args.output
    )


if __name__ == "__main__":
    main()
