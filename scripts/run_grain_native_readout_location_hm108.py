#!/usr/bin/env python3
"""Locate HM107 native interactions from sealed logits with zero model calls."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_reading as J
from grain_research.construction import math_native_readout_location as M
from grain_research.construction import math_native_terminal_audit as A
from scripts import audit_grain_native_joint_terminal_hm107 as T
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_native_joint_value_hm107_r2 as R

SPEC = ROOT / "configs/grain_native_readout_location_hm108.json"
CODE = (
    "configs/grain_native_readout_location_hm108.json",
    "docs/GRAIN_NATIVE_READOUT_LOCATION_PLAN_HM108.md",
    "grain_research/construction/math_native_readout_location.py",
    "scripts/run_grain_native_readout_location_hm108.py",
    "tests/test_grain_math_native_readout_location.py",
    "grain_research/construction/math_joint_reading.py",
    "grain_research/construction/math_native_joint.py",
    "grain_research/construction/math_native_terminal_audit.py",
    "grain_research/construction/math_readout_curvature.py",
)
TABLES = (
    "characters",
    "effects",
    "author_scores",
    "coverage",
    "sectors",
    "target_accounts",
)


def check(out):
    """Check this packet and inherited terminal code, calibration and outcomes."""
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    capture, audit = (
        ROOT / manifest["spec"]["capture"],
        ROOT / manifest["spec"]["terminal_audit"],
    )
    H.verify_hashes(capture, manifest["capture"])
    H.verify_hashes(audit, manifest["audit"])
    freeze = H.load(audit / "input_seal.json")
    H.verify_hashes(ROOT, freeze["code"])
    H.verify_hashes(capture, freeze["capture"])
    R.check(capture)
    return manifest


def prepare(out):
    """Freeze the no-forward protocol without replacing any prior result root."""
    if out.exists():
        raise FileExistsError("Never overwrite a previous experiment")
    spec = H.load(SPEC)
    capture, audit = ROOT / spec["capture"], ROOT / spec["terminal_audit"]
    freeze = H.load(audit / "input_seal.json")
    H.verify_hashes(ROOT, freeze["code"])
    H.verify_hashes(capture, freeze["capture"])
    H.verify_hashes(audit, H.load(audit / "output_seal.json"))
    state = H.load(capture / "status.json")
    if state["state"] != "BUDGET_STOP" or state["pending"] is not None:
        raise ValueError("Use the sealed terminal capture, not a live collector")
    if any(
        spec[k] != 0
        for k in ("new_forwards", "predictor_fits", "api_calls", "private_reads")
    ):
        raise ValueError("This protocol has no model or fitting budget")
    out.mkdir(parents=True)
    H.write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {n: H.sha(ROOT / n) for n in CODE},
            "capture": freeze["capture"],
            "audit": H.seals(audit, audit),
            "state": state,
        },
    )
    print("HM108 PREPARED: zero-forward symmetric native account", flush=True)


def write_table(path, rows):
    """Write all columns, leaving absent scientific quantities absent."""
    columns = sorted({k for row in rows for k in row})
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compute(out):
    """Account for every complete actual arm, all units, characters and authors."""
    manifest = check(out)
    spec = manifest["spec"]
    capture = ROOT / spec["capture"]
    old, status, groups, records, _, old_checks = T.read_records(capture)
    if old["spec"]["targets"] != spec["targets"] or list(M.UNITS) != spec["units"]:
        raise ValueError("Frozen target/alphabet meaning changed")
    logits = {}
    batches = H.load(capture / "00/batches.json")
    for batch in batches[: status["completed_batches"]]:
        for item in batch["units"]:
            key = item["group"], item["arm"], item["corner"], item["target"]
            logits[key] = H.load(R.unit_path(capture, item).with_suffix(".json"))[
                "candidate_logits"
            ]
    if set(logits) != set(records):
        raise ValueError("Raw logits and authenticated actual endpoints differ")
    ids = [g["group"] for g in groups]
    matched = A.complete_groups(records, ids, 6, spec["arms"])
    baseline = A.complete_groups(records, ids, 6, ("baseline",))
    checks = {
        "arithmetic": 0.0,
        "quadrature": 0.0,
        "author": 0.0,
        "native_replay": 0.0,
        "noop": 0.0,
    }
    arrays, fields, rows, effects, authors, coverage = {}, {}, [], [], [], []
    for group in groups:
        g = group["group"]
        base = dict(
            group=g,
            **dict(
                zip(
                    ("bank", "bridge", "carrier", "action"),
                    group["coordinates"],
                    strict=True,
                )
            ),
        )
        arms = [
            arm
            for arm in (*spec["arms"], "noop")
            if g in A.complete_groups(records, (g,), 6, (arm,))
        ]
        coverage.append(
            dict(
                **base,
                **{arm: arm in arms for arm in (*spec["arms"], "noop")},
                matched=g in matched,
                confirmed_units=sum(k[0] == g for k in records),
            )
        )
        for arm, target in itertools.product(arms, range(6)):
            raw = np.array([logits[g, arm, c, target] for c in range(4)])
            field = M.decompose(raw)
            integral = M.integral_account(raw, spec["quadrature_order"])
            fields[g, arm, target] = field
            arrays[f"g{g}__{arm}__q{target}__logits"] = raw
            checks["arithmetic"] = max(checks["arithmetic"], field["error"])
            checks["quadrature"] = max(checks["quadrature"], integral["error"])
            replay = np.array([records[g, arm, c, target][:2] for c in range(4)])
            checks["native_replay"] = max(
                checks["native_replay"],
                float(np.max(abs(field["corners"]["native"] - replay))),
            )
            if arm == "noop":
                for part in M.PARTS:
                    checks["noop"] = max(
                        checks["noop"],
                        float(
                            np.max(
                                abs(
                                    field["characters"][part]
                                    - fields[g, "baseline", target]["characters"][part]
                                )
                            )
                        ),
                    )
            for j, u in itertools.product(range(4), range(2)):
                rows.append(
                    dict(
                        **base,
                        arm=arm,
                        target=spec["targets"][target],
                        character=J.CHARACTERS[j],
                        unit=M.UNITS[u],
                        **{
                            part: float(field["characters"][part][j, u])
                            for part in M.PARTS
                        },
                        centered_logit_character_norm=float(
                            np.linalg.norm(field["natural"][j])
                        ),
                        joint_field_segment_integral=float(integral["lift"][j, u]),
                        additive_joint_rectangle_integral=float(
                            integral["rectangle"][u]
                        )
                        if j == 3
                        else None,
                    )
                )
            for law, p in spec["author_laws"].items():
                scores, error = M.author_account(field, p)
                checks["author"] = max(checks["author"], error)
                for u in range(2):
                    authors.append(
                        dict(
                            **base,
                            arm=arm,
                            target=spec["targets"][target],
                            law=law,
                            unit=M.UNITS[u],
                            **{part: float(scores[part][u]) for part in M.PARTS},
                        )
                    )

    def account(selected, arm, j, u, target=None):
        """Pool only the specified matched units, not missing arms or targets."""
        values = []
        targets = range(6) if target is None else (target,)
        for part in M.PARTS:
            values.append(
                np.array(
                    [
                        fields[g, "baseline", q]["characters"][part][j, u]
                        if arm == "baseline"
                        else fields[g, "baseline", q]["characters"][part][j, u]
                        - fields[g, arm, q]["characters"][part][j, u]
                        for g in selected
                        for q in targets
                    ]
                )
            )
        return M.signed_account(*values)

    for g, arm, q, j, u in itertools.product(
        matched, spec["arms"][1:], range(6), range(4), range(2)
    ):
        base = next(r for r in coverage if r["group"] == g)
        effects.append(
            dict(
                **{
                    k: base[k] for k in ("group", "bank", "bridge", "carrier", "action")
                },
                arm=arm,
                target=spec["targets"][q],
                character=J.CHARACTERS[j],
                unit=M.UNITS[u],
                **{
                    part: float(
                        fields[g, "baseline", q]["characters"][part][j, u]
                        - fields[g, arm, q]["characters"][part][j, u]
                    )
                    for part in M.PARTS
                },
            )
        )
    sectors, contexts = [], {}
    for group in groups:
        contexts.setdefault(tuple(group["coordinates"][:3]), {})[
            group["coordinates"][3]
        ] = group["group"]
    for context, by_action in contexts.items():
        ordered = [by_action[a] for a in range(6)]
        for arm in spec["arms"]:
            if not all((g, arm, q) in fields for g in ordered for q in range(6)):
                continue
            labels = (arm,) if arm == "baseline" else (arm, "delta_" + arm)
            for label, j, u in itertools.product(labels, range(4), range(2)):
                parts = {}
                for part in M.PARTS:
                    matrix = np.array(
                        [
                            [
                                fields[g, arm, q]["characters"][part][j, u]
                                for q in range(6)
                            ]
                            for g in ordered
                        ]
                    )
                    if label.startswith("delta_"):
                        matrix = (
                            np.array(
                                [
                                    [
                                        fields[g, "baseline", q]["characters"][part][
                                            j, u
                                        ]
                                        for q in range(6)
                                    ]
                                    for g in ordered
                                ]
                            )
                            - matrix
                        )
                    parts[part] = A.matrix_sectors(matrix)
                for sector in parts["native"]:
                    sectors.append(
                        dict(
                            bank=context[0],
                            bridge=context[1],
                            carrier=context[2],
                            arm=label,
                            character=J.CHARACTERS[j],
                            unit=M.UNITS[u],
                            sector=sector,
                            **M.signed_account(*(parts[p][sector] for p in M.PARTS)),
                        )
                    )
    banks, target_accounts = {}, []
    for bank in range(3):
        selected = [g for g in baseline if groups[g]["coordinates"][0] == bank]
        causal = [g for g in matched if groups[g]["coordinates"][0] == bank]
        item = {
            "baseline_groups": len(selected),
            "matched_groups": len(causal),
            "units": {},
        }
        for u, unit in enumerate(M.UNITS):
            item["units"][unit] = {
                "baseline": account(selected, "baseline", 3, u),
                "effects": {arm: account(causal, arm, 3, u) for arm in spec["arms"][1:]}
                if causal
                else None,
            }
            for q, j in itertools.product(range(6), range(4)):
                target_accounts.append(
                    dict(
                        bank=bank,
                        arm="baseline",
                        target=spec["targets"][q],
                        character=J.CHARACTERS[j],
                        unit=unit,
                        **account(selected, "baseline", j, u, q),
                    )
                )
                for arm in spec["arms"][1:] if causal else ():
                    target_accounts.append(
                        dict(
                            bank=bank,
                            arm="delta_" + arm,
                            target=spec["targets"][q],
                            character=J.CHARACTERS[j],
                            unit=unit,
                            **account(causal, arm, j, u, q),
                        )
                    )
        banks[str(bank)] = item
    for k, value in checks.items():
        gate = (
            spec["quadrature_tolerance"]
            if k == "quadrature"
            else spec["arithmetic_tolerance"]
        )
        if value > gate:
            raise ValueError(f"Fixed {k} gate failed: {value} > {gate}")
    summary = {
        "status": "FINITE_HM108_EVALUATED_NOT_GLOBAL",
        "classification": spec["classification"],
        "new_forwards": 0,
        "predictor_fits": 0,
        "original_scope_complete": False,
        "primary_HM107_acceptance": False,
        "actual_HM107_state": status,
        "baseline_groups": len(baseline),
        "matched_groups": len(matched),
        "complete_arm_target_fields": len(fields),
        "banks": banks,
        "checks": checks,
        "inherited_checks": old_checks,
        "missing_fresh_causal_controls": banks["2"]["matched_groups"] == 0,
        "unit_scope": "native candidate observations,not human traits or information shares",
    }
    return (
        summary,
        dict(
            zip(
                TABLES,
                (rows, effects, authors, coverage, sectors, target_accounts),
                strict=True,
            )
        ),
        arrays,
    )


def alternate(arrays, table):
    """Recompute the four-corner and candidate accounts by independent explicit sums."""
    errors = []
    signs = ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))
    lookup = {
        (int(r["group"]), r["arm"], r["target"], r["character"], r["unit"]): r
        for r in table
    }
    targets = H.load(SPEC)["targets"]
    for name, raw in arrays.items():
        g, arm, q, _ = name.split("__")
        g, q = int(g[1:]), int(q[1:])
        z = np.array([[float(x) - sum(row) / 10 for x in row] for row in raw])
        natural = np.array(
            [sum(sign[c] * z[c] for c in range(4)) / 4 for sign in signs]
        )
        mains = np.array(
            [
                natural[0] + signs[1][c] * natural[1] + signs[2][c] * natural[2]
                for c in range(4)
            ]
        )

        def observe(value):
            w = np.exp(value - max(value))
            w /= sum(w)
            return np.array(
                [sum(w[i] * i / 9 for i in range(10)), np.log(sum(w[5:]) / sum(w[:5]))]
            )

        full, main = (
            np.array([observe(row) for row in z]),
            np.array([observe(row) for row in mains]),
        )
        for part, corner in zip(M.PARTS, (full, main, full - main), strict=True):
            for j, u in itertools.product(range(4), range(2)):
                value = sum(signs[j][c] * corner[c, u] for c in range(4)) / 4
                actual = float(
                    lookup[g, arm, targets[q], J.CHARACTERS[j], M.UNITS[u]][part]
                )
                errors.append(abs(value - actual))
    return max(errors)


def execute(phase, out):
    """Run or replay a finite, exclusive-output, no-forward packet."""
    if phase == "prepare":
        prepare(out)
        return
    started = time.monotonic()
    if phase == "verify":
        H.verify_hashes(out, H.load(out / "output_seal.json"))
    summary, tables, arrays = compute(out)
    if phase == "analyze":
        for name, rows in tables.items():
            write_table(out / f"{name}.csv", rows)
        with (out / "logits.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        H.write(out / "summary.json", summary)
        H.write(out / "output_seal.json", H.seals(out, out))
        print(
            "HM108 EVALUATED",
            {
                "checks": summary["checks"],
                "fields": len(arrays),
                "seconds": time.monotonic() - started,
            },
            flush=True,
        )
        return
    if summary != H.load(out / "summary.json"):
        raise ValueError("Summary did not replay exactly")
    for name, expected in tables.items():
        with (out / f"{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        columns = sorted({k for r in expected for k in r})
        rendered = [
            {k: "" if r.get(k) is None else str(r.get(k, "")) for k in columns}
            for r in expected
        ]
        if actual != rendered:
            raise ValueError("Table replay failed: " + name)
    with np.load(out / "logits.npz", allow_pickle=False) as stored:
        if set(stored.files) != set(arrays) or any(
            not np.array_equal(stored[k], v) for k, v in arrays.items()
        ):
            raise ValueError("Raw complete logit fields changed")
    error = alternate(arrays, tables["characters"])
    if error > H.load(SPEC)["arithmetic_tolerance"]:
        raise ValueError("Independent corner/readout reconstruction failed")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM108_NOT_GLOBAL",
            "alternate_error": error,
            "checks": summary["checks"],
            "tables": {k: len(v) for k, v in tables.items()},
            "seconds": time.monotonic() - started,
            "same_agent_not_third_party": True,
            "new_forwards": 0,
            "original_scope_complete": False,
        },
    )
    print(
        "HM108 VERIFIED",
        {"alternate_error": error, "seconds": time.monotonic() - started},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    try:
        execute(args.phase, args.output)
    except Exception as exc:
        if args.output.exists() and not (args.output / "failure.json").exists():
            H.write(
                args.output / "failure.json",
                {"phase": args.phase, "error": str(exc), "new_forwards": 0},
            )
        raise
