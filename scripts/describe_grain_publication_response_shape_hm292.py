#!/usr/bin/env python3
"""Decompose a recorded response field into retained personal level and shape."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_publication as C
from scripts import run_grain_conditional_publication_hm292 as R


def decompose(rule, measured):
    """Retain both personal additive deviation and address-specific zero-sum shape."""
    f = measured["response"]
    units = np.sqrt(rule["address_mass"])[None, None, :, None]
    raw = f.reshape(*f.shape[:2], 8, 3) * units
    delta = raw.sum(2)
    level = measured["address_probability"][..., None] * delta[:, :, None, :]
    return {
        "level": (level / units).reshape(f.shape),
        "shape": ((raw - level) / units).reshape(f.shape),
    }


def advantage(value, donor):
    """Compare fixed measured fields,not outputs of a tuned forecasting model."""
    a, b = value[:, 0], value[:, 1]
    return np.sum((a - b[donor]) ** 2, 1) - np.sum((a - b) ** 2, 1)


def main():
    """Append one separate descriptive/scientific application of unchanged fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    manifest = R.check(args.parent, ("computed_seal.json", "verification_seal.json"))
    out = args.parent / "postoutcome_shape_r1"
    out.mkdir(exist_ok=False)
    code = (
        "scripts/describe_grain_publication_response_shape_hm292.py",
        "docs/GRAIN_PUBLICATION_RESPONSE_SHAPE_FOLLOWUP_HM292.md",
        "tests/test_grain_publication_response_shape.py",
    )
    R.R.write(
        out / "manifest.json",
        {
            "classification": "POSTOUTCOME level/shape measurement,not changed primary",
            "code": {p: R.R.digest(ROOT / p) for p in code},
            "inputs": {
                p: R.R.digest(args.parent / p)
                for p in (
                    "observations.npz",
                    "source_rules.json",
                    "measured_fields.npz",
                )
            },
            "same_covectors_and_bootstrap_seed": True,
            "compute_seconds_cap": 180,
        },
    )
    z = R.R.load(args.parent / "observations.npz")
    rules = {
        ref: {k: np.array(v) for k, v in rule.items()}
        for ref, rule in R.R.read(args.parent / "source_rules.json").items()
    }
    start, rows, errors, arrays = time.monotonic(), [], [], {}
    for bank in (0, 1):
        rule = rules[f"bank{bank}"]
        test = z["bank"] != bank
        measured = C.fields(rule, z["source_states"][test], z["source_outputs"][test])
        parts = decompose(rule, measured)
        errors.append(
            float(np.max(abs(parts["level"] + parts["shape"] - measured["response"])))
        )
        for part, value in parts.items():
            scores = np.einsum("uwf,aqf->uwaq", value, rule["covectors"])
            for view in (0, 1):
                for assign in (0, 1):
                    for q, trait in enumerate(R.TRAITS):
                        y = z["source_labels"][0, test, q]
                        score = scores[:, view, assign, q]
                        rows.append(
                            {
                                "calibration_bank": bank,
                                "view": view,
                                "calibration_assignment": ("actual", "old_permuted")[
                                    assign
                                ],
                                "part": part,
                                "trait": trait,
                                "r_actual_binary": C.correlation(score, y),
                                "signed_covariance_actual": float(
                                    np.mean((score - score.mean()) * (y - y.mean()))
                                ),
                            }
                        )
    rule = rules["pooled"]
    measured = C.fields(rule, z["held_states"], z["held_outputs"])
    parts = decompose(rule, measured)
    donor = z["donors"]
    cfg = manifest["config"]
    boot = np.random.default_rng(cfg["seed"]).integers(0, 120, (cfg["bootstrap"], 120))
    result = {}
    gains = {name: advantage(value, donor) for name, value in parts.items()}
    cross = advantage(measured["response"], donor) - gains["level"] - gains["shape"]
    for name, value in parts.items():
        arrays[name] = value
        scores = np.einsum("uwf,aqf->uwaq", value, rule["covectors"])
        arrays[f"{name}_scores"] = scores
        g = gains[name]
        result[name] = {
            "matched_advantage": float(g.mean()),
            "descriptive_CI": np.quantile(g[boot].mean(1), [0.025, 0.975]).tolist(),
            "positive_people": int(np.sum(g > 0)),
            "score_continuity_ENACO": [
                C.correlation(scores[:, 0, 0, q], scores[:, 1, 0, q]) for q in range(5)
            ],
        }
    result["retained_cross_term"] = {
        "mean": float(cross.mean()),
        "descriptive_CI": np.quantile(cross[boot].mean(1), [0.025, 0.975]).tolist(),
    }
    # Direct mixed-distance calculation,not subtraction of already computed gains.
    a_l, b_l = parts["level"][:, 0], parts["level"][:, 1]
    a_s, b_s = parts["shape"][:, 0], parts["shape"][:, 1]
    literal_cross = 2 * np.sum(
        (a_l - b_l[donor]) * (a_s - b_s[donor]) - (a_l - b_l) * (a_s - b_s), 1
    )
    errors.append(float(np.max(abs(literal_cross - cross))))
    errors.append(
        float(np.max(abs(parts["level"] + parts["shape"] - measured["response"])))
    )
    if max(errors) > 1e-10 or time.monotonic() - start > 180:
        raise ArithmeticError("Frozen response-field account failed;no rescue")
    R.R.save(out / "measured_parts.npz", arrays)
    with (out / "criterion_contacts.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "held": result,
        "literal_error": max(errors),
        "seconds": time.monotonic() - start,
        "primary_unchanged": True,
        "not_full_individual_causal_response": True,
        "original_scope_complete": False,
    }
    R.R.write(out / "summary.json", summary)
    R.R.seal(
        out,
        "seal.json",
        [
            "manifest.json",
            "measured_parts.npz",
            "criterion_contacts.csv",
            "summary.json",
        ],
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
