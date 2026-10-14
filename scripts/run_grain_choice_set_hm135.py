#!/usr/bin/env python3
"""Finite unchosen-option mechanism experiment; never accuracy optimization."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_choice_evidence as M
from scripts import run_grain_causal_valuation_hm134 as R

H = R.H
SPEC = ROOT / "configs/grain_choice_set_hm135.json"
CODE = (
    "configs/grain_choice_set_hm135.json",
    "docs/GRAIN_CHOICE_SET_PLAN_HM135.md",
    "grain_research/construction/math_choice_evidence.py",
    "scripts/run_grain_choice_set_hm135.py",
    "tests/test_grain_choice_evidence.py",
)


@contextmanager
def engine():
    """Reuse the immutable addressed engine with explicit, restored config binding."""
    old_spec, old_code = R.SPEC, R.CODE
    R.SPEC, R.CODE = SPEC, tuple(dict.fromkeys((*old_code, *CODE)))
    try:
        yield
    finally:
        R.SPEC, R.CODE = old_spec, old_code


def derive(cube):
    """Keep alternative/availability effects separate from factual observations."""
    natural = cube[:, :, :, :, 0]
    effects = np.stack(
        [
            np.stack([M.opportunity_effects(natural[b, f]) for f in range(6)])
            for b in range(2)
        ]
    )
    parts = np.moveaxis(R.M.cause_parts(np.moveaxis(cube, 4, 0)), 0, 4)
    fields = []
    for bank, person, unit in itertools.product(range(2), range(2), range(4)):
        reference = effects[bank, :, 3, person, :6, unit]
        part_field = (
            parts[bank, :, 1, 0, :, person, :6, unit]
            - parts[bank, :, 0, 0, :, person, :6, unit]
        )
        denominator = float(np.sum(reference**2))
        fields.append(
            {
                "bank": bank,
                "person": "AB"[person],
                "unit": R.UNITS[unit],
                "closed_option_contrast_rms": float(
                    np.sqrt(np.mean(effects[bank, :, 0, person, :6, unit] ** 2))
                ),
                "open_option_contrast_rms": float(
                    np.sqrt(np.mean(effects[bank, :, 1, person, :6, unit] ** 2))
                ),
                "opportunity_interaction_rms": float(np.sqrt(np.mean(reference**2))),
                "interaction_diagonal": np.diag(reference).tolist(),
                "fact_availability_max": float(
                    np.max(
                        abs(
                            natural[bank, :, :, 1, person, 6, unit]
                            - natural[bank, :, :, 0, person, 6, unit]
                        )
                    )
                ),
                "K_V_joint_signed_projections": [
                    float(np.sum(part_field[:, j] * reference) / denominator)
                    if denominator
                    else None
                    for j in range(3)
                ],
                "part_reconstruction_error": float(
                    np.max(abs(part_field.sum(axis=1) - reference))
                ),
            }
        )
    return effects, parts, fields


def tables(effects, parts):
    """Export every registered person/target/unit, not only desirable diagonals."""
    names = H.load(SPEC)["targets"] + ["action_fact"]
    result = {"opportunity_effects": [], "selective_parts": []}
    for bank, family, person, target, unit in itertools.product(
        range(2), range(6), range(2), range(7), range(4)
    ):
        base = {
            "bank": bank,
            "family": names[family],
            "person": "AB"[person],
            "target": names[target],
            "unit": R.UNITS[unit],
        }
        result["opportunity_effects"].append(
            {
                **base,
                **dict(
                    zip(
                        (
                            "closed_option_contrast",
                            "open_option_contrast",
                            "availability_level",
                            "option_availability_interaction",
                        ),
                        map(float, effects[bank, family, :, person, target, unit]),
                        strict=True,
                    )
                ),
            }
        )
        for b, c, part in itertools.product(range(2), range(2), range(3)):
            result["selective_parts"].append(
                {
                    **base,
                    "alternative_polarity": 2 * b - 1,
                    "available": c,
                    "part": ("k", "v", "kv_interaction")[part],
                    "value": float(
                        parts[bank, family, b, c, part, person, target, unit]
                    ),
                }
            )
    return result


def example():
    """Declared choice-law illustration, not an empirical LLM/person result."""
    return {
        "status": "ASSUMED_EXPONENTIAL_CHOICE_EXAMPLE_NOT_DISCOVERY",
        "theta_low_high": [-1, 1],
        "chosen_feature": 0,
        "low_option_open": M.choice_log_bayes_factor(
            [[0], [-1]], [-1], [1], 0, [True, True]
        ),
        "high_option_open": M.choice_log_bayes_factor(
            [[0], [1]], [-1], [1], 0, [True, True]
        ),
        "low_option_closed": M.choice_log_bayes_factor(
            [[0], [-1]], [-1], [1], 0, [True, False]
        ),
        "high_option_closed": M.choice_log_bayes_factor(
            [[0], [1]], [-1], [1], 0, [True, False]
        ),
    }


def analyze(out):
    """Compute finite causal fields without fitting or choosing a better observer."""
    with engine():
        cube, checks = R.observations(out)
    effects, parts, fields = derive(cube)
    folder = out / "02"
    folder.mkdir()
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube, effects=effects, parts=parts)
    for name, records in tables(effects, parts).items():
        H.table(folder / (name + ".csv"), records)
    H.write(folder / "choice_example.json", example())
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_OPPORTUNITY_OPERATION_NOT_HUMAN_VALIDITY",
            "checks": checks,
            "fields": fields,
            "batches": 196,
            "endpoints": 2744,
            "model_seconds": H.load(out / "status.json")["model_seconds"],
            "original_scope_complete": False,
        },
    )
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print("ANALYZED HM135", {"checks": checks, "fields": fields}, flush=True)


def verify(out):
    """Independently re-count finite characters, logits and all persisted tables."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    with engine():
        cube, checks = R.observations(out)
    effects, parts, fields = derive(cube)
    with np.load(out / "02/observations.npz", allow_pickle=False) as z:
        for name, value in (("cube", cube), ("effects", effects), ("parts", parts)):
            if not np.array_equal(z[name], value):
                raise ValueError("Saved field changed: " + name)
    error = 0.0
    for bank, family in itertools.product(range(2), range(6)):
        x = cube[bank, family, :, :, 0]
        mixed = sum(
            b * c * x[(b + 1) // 2, (c + 1) // 2]
            for b, c in itertools.product((-1, 1), repeat=2)
        )
        error = max(error, float(np.max(abs(mixed - effects[bank, family, 3]))))
    error = max(
        error,
        float(
            np.max(abs(parts.sum(axis=4) - (cube[:, :, :, :, 3] - cube[:, :, :, :, 0])))
        ),
    )
    with engine():
        for d in H.load(out / "00/design.json"):
            raw = R.committed(out, d)
            logits = raw["logits"]
            w = np.exp(logits - logits.max(axis=1)[:, None])
            p = w / w.sum(axis=1)[:, None]
            alternate = np.stack(
                (
                    p @ np.arange(10) / 9,
                    np.log(w[:, 5:].sum(axis=1) / w[:, :5].sum(axis=1)),
                    np.exp(
                        logits.max(axis=1) + np.log(w.sum(axis=1)) - raw["normalizer"]
                    ),
                    -np.sum(p * np.log(p), axis=1),
                ),
                axis=1,
            )
            error = max(error, float(np.max(abs(alternate - raw["readings"]))))
    if error > 1e-10:
        raise ValueError("Independent finite-field/primitive reconstruction failed")
    saved = H.load(out / "02/summary.json")
    if (
        saved["checks"] != checks
        or saved["fields"] != fields
        or H.load(out / "02/choice_example.json") != example()
    ):
        raise ValueError("Saved scientific summary changed")
    counts = {}
    for name, records in tables(effects, parts).items():
        with (out / f"02/{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        if actual != [{n: str(v) for n, v in r.items()} for r in records]:
            raise ValueError("Saved scientific table changed")
        counts[name] = len(records)
    print(
        "VERIFIED HM135",
        {"checks": checks, "independent_error": error, "table_rows": counts},
        flush=True,
    )


def main():
    """Run isolated phases with no automatic retry, cap renewal or new fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "collect", "analyze", "verify")
    )
    parser.add_argument("--out", type=Path, default=ROOT / H.load(SPEC)["out"])
    args = parser.parse_args()
    M.validate_material_spec(H.load(SPEC))
    print("EXPERIMENT HM135; immutable HM134 addressed-KV engine", flush=True)
    if args.phase in ("prepare", "collect"):
        with engine():
            getattr(R, args.phase)(args.out)
    else:
        {"analyze": analyze, "verify": verify}[args.phase](args.out)


if __name__ == "__main__":
    main()
