#!/usr/bin/env python3
"""Zero-forward terminal audit, keeping HM107 primary incompleteness explicit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_joint_value as M
from grain_research.construction import math_native_terminal_audit as A
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_native_joint_value_hm107_r2 as R

CODE = (
    "grain_research/construction/math_native_terminal_audit.py",
    "scripts/audit_grain_native_joint_terminal_hm107.py",
    "tests/test_grain_math_native_terminal_audit.py",
    "docs/GRAIN_NATIVE_TERMINAL_AUDIT_HM107.md",
)
UNITS = ("score", "log_odds", "mass", "entropy")


def read_records(capture):
    """Read only status-confirmed batches and independently validate each execution."""
    manifest = R.check(capture)
    spec, status = manifest["spec"], H.load(capture / "status.json")
    if status["state"] not in ("CAPTURED", "FAILED", "BUDGET_STOP"):
        raise ValueError("Do not inspect native outcomes in a running collector")
    batches = H.load(capture / "00/batches.json")
    groups = H.load(capture / "00/groups.json")
    queries = H.load(capture / "00/queries.json")
    n = status["completed_batches"]
    if not 0 <= n <= len(batches) or status["completed"] != 6 * n:
        raise ValueError("Completed batches and endpoints disagree")
    if not (capture / "00/calibration_seal.json").exists():
        raise ValueError("Terminal native calibration missing; no intervention audit")
    with np.load(capture / "00/edits.npz", allow_pickle=False) as z:
        edits = {k: z[k] for k in z.files}
    records, maximum = (
        {},
        {
            "units": 0.0,
            "prefix": 0.0,
            "edit": 0.0,
            "metadata": 0.0,
            "author": 0.0,
            "noop": 0.0,
        },
    )
    for batch in batches[:n]:
        if batch["batch"] >= n or len(batch["units"]) != 6:
            raise ValueError("Declared completed batch is malformed")
        row = groups[batch["group"]]["rows"][batch["corner"]]
        lengths = [len(row["input_ids"]) + len(s) for s in queries["suffixes"]]
        for item in batch["units"]:
            path = R.unit_path(capture, item)
            meta = H.load(path.with_suffix(".json"))
            if (
                any(meta[k] != v for k, v in item.items())
                or meta["batch"] != batch["batch"]
            ):
                raise ValueError("Actual native address does not match the design")
            if meta["sha256"] != H.sha(path.with_suffix(".npz")):
                raise ValueError("Raw native unit changed")
            native = M.native_units(
                meta["candidate_logits"], meta["full_vocab_log_normalizer"]
            )
            z = np.asarray(meta["candidate_logits"], float)
            w = np.exp(z - z.max())
            w /= w.sum()
            second = np.array(
                [
                    sum(w[i] * i / 9 for i in range(10)),
                    np.log(
                        np.exp(z[5:] - z.max()).sum() / np.exp(z[:5] - z.max()).sum()
                    ),
                    sum(np.exp(z - meta["full_vocab_log_normalizer"])),
                    -sum(w[i] * np.log(w[i]) for i in range(10)),
                ]
            )
            stored = np.array(
                [
                    meta["conditional_score"],
                    native[1],
                    meta["candidate_mass"],
                    meta["conditional_entropy"],
                ]
            )
            maximum["units"] = max(
                maximum["units"],
                float(np.max(abs(native - second))),
                float(np.max(abs(native - stored))),
            )
            tag = f"g{item['group']}__"
            arm, corner = item["arm"], item["corner"]
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as raw:
                real, length = lengths[item["target"]], max(lengths)
                support = np.tri(length, dtype=bool)
                support[:, row["padding"]] = False
                support[:, real:] = False
                for name, array in (
                    ("support_hash", support[:real, :real]),
                    ("execution_support_hash", support),
                ):
                    expected = np.frombuffer(
                        hashlib.sha256(array.tobytes()).digest(), dtype=np.uint8
                    )
                    if not np.array_equal(raw[name], expected):
                        raise ValueError("Actual native causal support changed")
                if (
                    int(raw["endpoint"]) != real - 1
                    or int(raw["execution_length"]) != length
                    or not np.array_equal(raw["masked_blocks"], np.arange(28))
                ):
                    raise ValueError("Actual native endpoint or block masks changed")
                prefix = max(
                    float(np.max(abs(raw[k] - edits[tag + ref][corner])))
                    for k, ref in (("k", "old_k"), ("before", "old_v"))
                )
                delta = (
                    edits[tag + ("joint" if arm == "joint_removed" else "random")][
                        corner
                    ]
                    if arm in ("joint_removed", "random")
                    else np.zeros_like(raw["before"])
                )
                error = float(np.max(abs(raw["after"] - raw["before"] - delta)))
                maximum["prefix"] = max(maximum["prefix"], prefix)
                maximum["edit"] = max(maximum["edit"], error)
                maximum["metadata"] = max(
                    maximum["metadata"],
                    abs(error - meta["edit_error"]),
                    abs(prefix - meta["prefix_error"]),
                )
            key = item["group"], arm, corner, item["target"]
            if key in records:
                raise ValueError("Duplicate native evidence address")
            records[key] = native
    for (g, arm, c, q), value in records.items():
        if arm == "noop":
            maximum["noop"] = max(
                maximum["noop"],
                float(np.max(abs(value - records[g, "baseline", c, q]))),
            )
    gates = {
        "units": spec["arithmetic_tolerance"],
        "metadata": spec["arithmetic_tolerance"],
        "prefix": spec["prefix_tolerance"],
        "edit": spec["edit_tolerance"],
        "noop": spec["noop_tolerance"],
    }
    if any(maximum[k] > v for k, v in gates.items()):
        raise ValueError(f"Unchanged execution/arithmetic gates failed:{maximum}")
    return manifest, status, groups, records, edits, maximum


def tables(capture):
    """Keep all available characters and restrict effects to complete matched groups."""
    manifest, status, groups, records, edits, maximum = read_records(capture)
    spec = manifest["spec"]
    ids = [g["group"] for g in groups]
    base_ids = A.complete_groups(records, ids, 6, ("baseline",))
    matched = A.complete_groups(records, ids, 6, spec["arms"])
    chars, local, rows, effects, coverage = {}, {}, [], [], []
    for group in groups:
        g, coordinates = group["group"], group["coordinates"]
        base = dict(
            group=g,
            **dict(
                zip(("bank", "bridge", "carrier", "action"), coordinates, strict=True)
            ),
        )
        complete = {
            arm: g in A.complete_groups(records, (g,), 6, (arm,))
            for arm in spec["arms"]
        }
        coverage.append(
            dict(
                **base,
                **complete,
                matched=g in matched,
                available_units=sum(k[0] == g for k in records),
            )
        )
        if g not in base_ids:
            continue
        tag = f"g{g}__"
        k, v, q, d = [edits[tag + name] for name in ("old_k", "old_v", "q", "d")]
        for arm in spec["arms"]:
            if not complete[arm]:
                continue
            field = np.stack(
                [[records[g, arm, c, target] for target in range(6)] for c in range(4)]
            )
            chars[g, arm] = A.character_table(field)
            delta = (
                edits[tag + ("joint" if arm == "joint_removed" else "random")]
                if arm != "baseline"
                else 0
            )
            p, cv = J.primitives(k, v + delta, q, d)
            local[g, arm] = A.character_table((p * cv).sum(axis=(1, 3)))
            plus, minus = (1 + 0.5 * J.BASIS[3]) / 4, (1 - 0.5 * J.BASIS[3]) / 4
            maximum["author"] = max(
                maximum["author"],
                float(
                    np.max(
                        abs(
                            sum((plus[c] - minus[c]) * field[c] for c in range(4))
                            - chars[g, arm][3]
                        )
                    )
                ),
            )
            for j, target in itertools.product(range(4), range(6)):
                rows.append(
                    dict(
                        **base,
                        arm=arm,
                        character=J.CHARACTERS[j],
                        target=spec["targets"][target],
                        local=float(local[g, arm][j, target]),
                        **dict(zip(UNITS, chars[g, arm][j, target], strict=True)),
                    )
                )
        if g in matched:
            for j, target in itertools.product(range(4), range(6)):
                effects.append(
                    dict(
                        **base,
                        character=J.CHARACTERS[j],
                        target=spec["targets"][target],
                        **{
                            f"{unit}_{label}": float(
                                chars[g, "baseline"][j, target, u]
                                - chars[g, arm][j, target, u]
                            )
                            for arm, label in (
                                ("joint_removed", "removed"),
                                ("random", "random"),
                            )
                            for u, unit in enumerate(UNITS)
                        },
                    )
                )
    sectors, contexts = [], {}
    for group in groups:
        contexts.setdefault(tuple(group["coordinates"][:3]), {})[
            group["coordinates"][3]
        ] = group["group"]
    for context, action_groups in contexts.items():
        ordered = [action_groups[a] for a in range(6)]
        if not all(g in base_ids for g in ordered):
            continue
        paired = all(g in matched for g in ordered)
        for j, u in itertools.product(range(4), range(2)):
            native = A.matrix_sectors(
                np.stack([chars[g, "baseline"][j, :, u] for g in ordered])
            )
            observer = A.matrix_sectors(
                np.stack([local[g, "baseline"][j] for g in ordered])
            )
            removed = (
                A.matrix_sectors(
                    np.stack(
                        [
                            chars[g, "baseline"][j, :, u]
                            - chars[g, "joint_removed"][j, :, u]
                            for g in ordered
                        ]
                    )
                )
                if paired
                else None
            )
            random = (
                A.matrix_sectors(
                    np.stack(
                        [
                            chars[g, "baseline"][j, :, u] - chars[g, "random"][j, :, u]
                            for g in ordered
                        ]
                    )
                )
                if paired
                else None
            )
            for name, value in native.items():
                row = {
                    "bank": context[0],
                    "bridge": context[1],
                    "carrier": context[2],
                    "character": J.CHARACTERS[j],
                    "unit": UNITS[u],
                    "sector": name,
                    "matched": paired,
                    "native_rms": A.describe(value)["rms"],
                    "local_rms": A.describe(observer[name])["rms"],
                    "native_local_cosine": A.cosine(value, observer[name]),
                }
                if paired:
                    s = A.effect_summary(value, removed[name], random[name])
                    row.update(
                        removed_rms=s["removed_effect"]["rms"],
                        random_rms=s["random_effect"]["rms"],
                        removed_alignment=s["removed_alignment"],
                        random_alignment=s["random_alignment"],
                        remaining_native_rms=s["remaining_native"]["rms"],
                    )
                sectors.append(row)
    banks = {}
    for bank in range(3):
        baseline = [g for g in base_ids if groups[g]["coordinates"][0] == bank]
        causal = [g for g in matched if groups[g]["coordinates"][0] == bank]
        summary = {"baseline_groups": len(baseline), "matched_groups": len(causal)}
        for u, unit in enumerate(UNITS[:2]):
            stats = {}
            if baseline:
                native = np.stack([chars[g, "baseline"][3, :, u] for g in baseline])
                old = np.stack([local[g, "baseline"][3] for g in baseline])
                stats.update(
                    baseline_native=A.describe(native),
                    baseline_local=A.describe(old),
                    baseline_native_local_cosine=A.cosine(native, old),
                )
            if causal:
                native = np.stack([chars[g, "baseline"][3, :, u] for g in causal])
                removed = np.stack(
                    [
                        chars[g, "baseline"][3, :, u]
                        - chars[g, "joint_removed"][3, :, u]
                        for g in causal
                    ]
                )
                random = np.stack(
                    [
                        chars[g, "baseline"][3, :, u] - chars[g, "random"][3, :, u]
                        for g in causal
                    ]
                )
                stats["matched_effects"] = A.effect_summary(native, removed, random)
            summary[unit] = stats
        banks[str(bank)] = summary
    return {
        "primary_state": status["state"],
        "primary_complete": status["state"] == "CAPTURED"
        and status["completed"] == 5208,
        "primary_protocol_acceptance": False,
        "status": "TERMINAL_CAPTURE_AUDIT_NOT_PRIMARY_ACCEPTANCE",
        "classification": "ZERO_FORWARD_COMPLETE_CASE_DIAGNOSTIC_AFTER_RUNTIME_PROJECTION",
        "actual_status": status,
        "banks": banks,
        "complete_baseline_groups": base_ids,
        "complete_matched_groups": matched,
        "completed_unit_count": len(records),
        "noop_unit_count": sum(k[1] == "noop" for k in records),
        "checks": maximum,
        "native_not_human_traits": True,
        "new_forwards": 0,
        "predictor_fits": 0,
        "original_scope_complete": False,
    }, (rows, effects, coverage, sectors)


def execute(capture, output, verify=False):
    """Create a separate sealed audit or replay its calculations without model calls."""
    if verify:
        freeze = H.load(output / "input_seal.json")
        H.verify_hashes(ROOT, freeze["code"])
        H.verify_hashes(capture, freeze["capture"])
        H.verify_hashes(output, H.load(output / "output_seal.json"))
    elif output.exists():
        raise FileExistsError("Never overwrite terminal audit history")
    summary, rows = tables(capture)
    names = ("characters", "effects", "coverage", "sectors")
    if verify:
        if summary != H.load(output / "summary.json"):
            raise ValueError("Terminal audit does not replay exactly")
        for name, values in zip(names, rows, strict=True):
            with (output / f"{name}.csv").open() as stream:
                actual = list(csv.DictReader(stream))
            columns = sorted({k for r in values for k in r})
            expected = [
                {k: "" if r.get(k) is None else str(r.get(k, "")) for k in columns}
                for r in values
            ]
            if actual != expected:
                raise ValueError(f"Terminal {name} table changed")
        print(
            "HM107 terminal audit REPLAYED;primary acceptance remains false", flush=True
        )
        return
    output.mkdir(parents=True)
    inputs = {
        str(p.relative_to(capture)): H.sha(p)
        for p in capture.rglob("*")
        if p.is_file() and p.suffix in (".json", ".npz")
    }
    H.write(
        output / "input_seal.json",
        {"code": {n: H.sha(ROOT / n) for n in CODE}, "capture": inputs},
    )
    for name, values in zip(names, rows, strict=True):
        columns = sorted({k for row in values for k in row}) or [
            "no_complete_observations"
        ]
        with (output / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    H.write(output / "summary.json", summary)
    H.write(
        output / "output_seal.json",
        {f"{n}.csv": H.sha(output / f"{n}.csv") for n in names}
        | {"summary.json": H.sha(output / "summary.json")},
    )
    print(
        "HM107 terminal audit",
        {
            k: summary[k]
            for k in (
                "primary_state",
                "completed_unit_count",
                "noop_unit_count",
                "banks",
                "checks",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("analyze", "verify"))
    p.add_argument(
        "--capture",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_joint_value_r2",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_joint_terminal_audit_r1",
    )
    args = p.parse_args()
    execute(args.capture, args.output, args.phase == "verify")
