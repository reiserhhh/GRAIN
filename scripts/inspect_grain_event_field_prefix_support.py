#!/usr/bin/env python3
"""Explain HM72 order effects through actual causal-prefix factor availability."""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_event_field import (
    CHARACTERS,
    EVENTS,
    characters,
    cosine,
)
from scripts.diagnose_grain_event_field_cross_order import (
    PRIMARY,
    export,
    read,
    sha,
    write,
)

OUT = PRIMARY / "05_prefix_support"
FEATURES = ("event_end", "event_mean", "action_mean")
CODE = (
    "docs/GRAIN_EVENT_FIELD_PREFIX_SUPPORT_COMPANION_HM72.md",
    "scripts/inspect_grain_event_field_prefix_support.py",
    "tests/test_grain_event_field_prefix_support.py",
    "scripts/diagnose_grain_event_field_cross_order.py",
    "grain_research/construction/math_event_field.py",
    "grain_research/construction/math_cold_relational.py",
)


def availability(prefixes):
    """Return factors which actually change a prefix on the complete cube."""
    if len(prefixes) != 8:
        raise ValueError("Complete eight-cell prefix support required")
    return tuple(
        any(prefixes[i] != prefixes[i ^ (1 << (2 - j))] for i in range(8))
        for j in range(3)
    )


def allowed(mask, support):
    """Check the exact Boolean interaction support of a prefix-dependent field."""
    return all(support[j] for j in range(3) if mask & (1 << j))


def prepare():
    """Seal a result-after mechanism question and all actual observations."""
    if OUT.exists():
        raise FileExistsError("Preserve an existing support analysis")
    if (
        read(PRIMARY / "verification.json")["status"]
        != "VERIFIED_FINITE_HM72_NOT_GLOBAL"
    ):
        raise ValueError("Verified primary required")
    (OUT / "00/code").mkdir(parents=True)
    code = {}
    for name in CODE:
        destination = OUT / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = sha(ROOT / name)
    names = list(read(PRIMARY / "03/output_seal.json")) + [
        "00/rows.json",
        "00/manifest.json",
        "01/raw_seal.json",
        "03/output_seal.json",
        "verification.json",
    ]
    write(
        OUT / "00/manifest.json",
        {
            "result_after": True,
            "code": code,
            "inputs": {
                str((PRIMARY / n).relative_to(ROOT)): sha(PRIMARY / n) for n in names
            },
            "new_calls_fits": [0, 0],
        },
    )


def check():
    """Refuse changed scientific inputs or analysis code."""
    manifest = read(OUT / "00/manifest.json")
    for name, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(ROOT / name) != digest:
            raise ValueError("Drift: " + name)


def compute():
    """Derive all structural zeros and compare them with frozen actual fields."""
    rows = read(PRIMARY / "00/rows.json")
    lookup = {(r["bank"], r["pair"], r["order"], r["cell"]): r for r in rows}
    supports, coefficients, overlaps = [], [], []
    support_map = {}
    max_identical_prefix_error, max_alternate_error, checks = 0.0, 0.0, 0
    with np.load(PRIMARY / "03/field_tables.npz", allow_pickle=False) as data:
        for bank, pair, order, event_id in itertools.product(
            ("source", "new"), range(30), range(2), range(3)
        ):
            event = EVENTS[event_id]
            cells = [lookup[bank, pair, order, cell] for cell in range(8)]
            prefixes = [
                tuple(r["input_ids"][: r["spans"][event]["event"][-1] + 1])
                for r in cells
            ]
            support = availability(prefixes)
            support_map[bank, pair, order, event_id] = support
            identity = {"bank": bank, "pair": pair, "order": order, "event": event}
            supports.append(
                identity | {"x": support[0], "y": support[1], "g": support[2]}
            )
            for feature in FEATURES:
                field = data[f"{bank}_{pair:02d}_{feature}"][order, :, event_id]
                values = characters() @ field / 8
                for mask, label in enumerate(CHARACTERS, 1):
                    alternate = (
                        np.sum(
                            np.asarray(characters()[mask - 1], np.longdouble)[:, None]
                            * np.asarray(field, np.longdouble),
                            axis=0,
                            dtype=np.longdouble,
                        )
                        / 8
                    )
                    error = float(np.max(abs(values[mask - 1] - alternate)))
                    max_alternate_error = max(max_alternate_error, error)
                    checks += field.shape[1]
                    coefficients.append(
                        identity
                        | {
                            "feature": feature,
                            "character": label,
                            "allowed": allowed(mask, support),
                            "coefficient_norm": float(np.linalg.norm(values[mask - 1])),
                        }
                    )
                for j in range(3):
                    if support[j]:
                        continue
                    for cell in range(8):
                        other = cell ^ (1 << (2 - j))
                        if cell < other:
                            max_identical_prefix_error = max(
                                max_identical_prefix_error,
                                float(np.max(abs(field[cell] - field[other]))),
                            )
        for bank, pair, feature, mask in itertools.product(
            ("source", "new"), range(30), FEATURES, range(1, 8)
        ):
            field = data[f"{bank}_{pair:02d}_{feature}"]
            a, b = [np.tensordot(characters()[mask - 1] / 8, h, axes=1) for h in field]
            slots = [
                [
                    i
                    for i in range(3)
                    if allowed(mask, support_map[bank, pair, order, i])
                ]
                for order in range(2)
            ]
            overlaps.append(
                {
                    "bank": bank,
                    "pair": pair,
                    "feature": feature,
                    "character": CHARACTERS[mask - 1],
                    "forward_allowed_events": "|".join(EVENTS[i] for i in slots[0]),
                    "reverse_allowed_events": "|".join(EVENTS[i] for i in slots[1]),
                    "overlap_count": len(set(slots[0]) & set(slots[1])),
                    "within_order_cosine": cosine(a, b),
                }
            )
    forbidden = [r for r in coefficients if not r["allowed"]]
    total_energy = sum(r["coefficient_norm"] ** 2 for r in coefficients)
    forbidden_energy = sum(r["coefficient_norm"] ** 2 for r in forbidden)
    summary = {
        "result_after_not_primary": True,
        "new_calls_fits": [0, 0],
        "support_coefficient_overlap_rows": [
            len(supports),
            len(coefficients),
            len(overlaps),
        ],
        "forbidden_coefficient_rows": len(forbidden),
        "forbidden_max_norm": max(r["coefficient_norm"] for r in forbidden),
        "forbidden_energy_fraction": forbidden_energy / total_energy,
        "identical_prefix_max_coordinate_discrepancy": max_identical_prefix_error,
        "alternate_scalar_checks_max_error": [checks, max_alternate_error],
        "support_patterns": sorted(
            {(r["order"], r["event"], r["x"], r["y"], r["g"]) for r in supports}
        ),
        "all_banks_pairs_same_support": len(
            {(r["order"], r["event"], r["x"], r["y"], r["g"]) for r in supports}
        )
        == 6,
        "original_scope_complete": False,
    }
    summary["disjoint_characters"] = sorted(
        {r["character"] for r in overlaps if r["overlap_count"] == 0}
    )
    disjoint = [
        r["within_order_cosine"]
        for r in overlaps
        if r["overlap_count"] == 0 and r["within_order_cosine"] is not None
    ]
    summary["disjoint_within_order_max_abs_cosine"] = max(abs(v) for v in disjoint)
    return supports, coefficients, overlaps, summary


def run():
    """Save every support/interaction account without changing primary results."""
    check()
    if (OUT / "output_seal.json").exists():
        raise FileExistsError("Do not overwrite analyzed support results")
    supports, coefficients, overlaps, summary = compute()
    for name, values in (
        ("prefix_support.csv", supports),
        ("coefficients.csv", coefficients),
        ("support_overlap.csv", overlaps),
    ):
        export(OUT / name, values)
    write(OUT / "summary.json", summary)
    write(
        OUT / "output_seal.json",
        {
            name: sha(OUT / name)
            for name in (
                "prefix_support.csv",
                "coefficients.csv",
                "support_overlap.csv",
                "summary.json",
            )
        },
    )
    print("HM72 prefix support: all contexts/characters retained", flush=True)


def verify():
    """Recompute support and alternate coefficients from sealed actual tables."""
    check()
    for name, digest in read(OUT / "output_seal.json").items():
        if sha(OUT / name) != digest:
            raise ValueError("Output drift")
    *_, summary = compute()
    if json_normalize(summary) != read(OUT / "summary.json"):
        raise ValueError("Summary replay mismatch")
    write(
        OUT / "verification.json",
        {
            "status": "VERIFIED_RESULT_AFTER_SUPPORT_LAW",
            "same_agent_not_third_party": True,
            "scalar_checks_max_error": summary["alternate_scalar_checks_max_error"],
            "original_scope_complete": False,
        },
    )
    print("HM72 prefix support verified", flush=True)


def json_normalize(value):
    """Normalize tuple-based generated records to their saved JSON types."""
    import json

    return json.loads(json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    {"prepare": prepare, "run": run, "verify": verify}[parser.parse_args().action]()
