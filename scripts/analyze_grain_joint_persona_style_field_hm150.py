#!/usr/bin/env python3
"""Complete declared HM150 response controls and signed cross-seed geometry."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_persona_style_hm150 as J
from scripts import run_grain_semantic_anchor_hm147 as H
from scripts.run_grain_joint_persona_style_enumeration_hm150 import ScientificOutputPath


def crossed_product(left, right):
    """Average different-seed cross-products, retaining sign without clipping."""
    x, y = np.asarray(left, float), np.asarray(right, float)
    if x.shape != y.shape or x.ndim != 2 or len(x) != 3:
        raise ValueError("Three aligned seed vectors in one fixed unit required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite supported seed vectors required")
    return sum(float(x[i] @ y[j]) for i in range(3) for j in range(3) if i != j) / 6


def response_account(neutral, styled):
    """Separate supported directions from raw interaction energy, not accuracy."""
    x, y = np.asarray(neutral, float), np.asarray(styled, float)
    ux, uy = crossed_product(x, x), crossed_product(y, y)
    alignment = crossed_product(x, y)
    mixed = crossed_product(y - x, y - x)
    error = abs(mixed - (ux + uy - 2 * alignment))
    if error > 1e-8 * max(1, abs(mixed), abs(ux), abs(uy), abs(alignment)):
        raise ArithmeticError("Cross-seed polarization account failed")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    denominator = np.linalg.norm(xm) * np.linalg.norm(ym)
    return {
        "neutral_support_per_coordinate": ux / x.shape[1],
        "styled_support_per_coordinate": uy / x.shape[1],
        "alignment_per_coordinate": alignment / x.shape[1],
        "mixed_support_per_coordinate": mixed / x.shape[1],
        "mean_direction_cosine": float(xm @ ym / denominator) if denominator else None,
        "polarization_error": error,
    }


def analyze(out):
    """Retain all original roles and styles with no observer or target refitting."""
    out = ScientificOutputPath(out)
    manifest = J.check(out)
    rows, _ = J.generated(out)
    for seal in ("observation_seal.json", "output_seal.json"):
        for name, checksum in H.load(out / seal).items():
            if H.digest(out / name) != checksum:
                raise ValueError("Frozen observation/output changed: " + name)
    source = H.archive(out / "numeric_observations.npz", manifest["representations"])
    source = {name: np.asarray(values, float) for name, values in source.items()}
    program = H.archive(
        H.DEFAULT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )
    z = (source["grain4_slices"] - program["origin"]) / program["scale"][None, :, None]
    source["grain4_slices"] = z.reshape(len(z), -1)
    lookup = {
        tuple(r[k] for k in ("axis", "family", "situation", "seed", "style", "pole")): i
        for i, r in enumerate(rows)
    }
    tasks = list(dict.fromkeys(r["situation"] for r in rows))
    roles = {r["situation"]: r["role"] for r in rows}
    traits, controls, fields = [], [], []
    for representation, values in source.items():
        for axis, family, task in itertools.product(J.M.TARGETS, J.M.FAMILIES, tasks):
            responses = {}
            for style in J.M.STYLES:
                contrasts = []
                for seed in J.M.SEEDS:
                    high = lookup[axis, family, task, seed, style, 1]
                    low = lookup[axis, family, task, seed, style, -1]
                    delta = values[high] - values[low]
                    ok = bool(np.isfinite(delta).all())
                    contrasts.append(delta)
                    traits.append(
                        {
                            "representation": representation,
                            "axis": axis,
                            "family": family,
                            "situation": task,
                            "role": roles[task],
                            "seed": seed,
                            "style": style,
                            "supported": ok,
                            "response_rms": float(np.sqrt(np.mean(delta**2)))
                            if ok
                            else None,
                        }
                    )
                responses[style] = np.asarray(contrasts)
            for style in J.M.STYLES[1:]:
                ok = bool(
                    np.isfinite(responses[style]).all()
                    and np.isfinite(responses["neutral"]).all()
                )
                fields.append(
                    {
                        "representation": representation,
                        "axis": axis,
                        "family": family,
                        "situation": task,
                        "role": roles[task],
                        "style": style,
                        "supported": ok,
                        **(
                            response_account(responses["neutral"], responses[style])
                            if ok
                            else {
                                key: None
                                for key in (
                                    "neutral_support_per_coordinate",
                                    "styled_support_per_coordinate",
                                    "alignment_per_coordinate",
                                    "mixed_support_per_coordinate",
                                    "mean_direction_cosine",
                                    "polarization_error",
                                )
                            }
                        ),
                    }
                )
        settings = list(itertools.product(J.M.TARGETS, J.M.FAMILIES, (-1, 1))) + [
            ("none", "none", 0)
        ]
        for axis, family, pole in settings:
            for task, style in itertools.product(tasks, J.M.STYLES[1:]):
                deltas = []
                for seed in J.M.SEEDS:
                    a = lookup[axis, family, task, seed, style, pole]
                    b = lookup[axis, family, task, seed, "neutral", pole]
                    deltas.append(values[a] - values[b])
                deltas = np.asarray(deltas)
                ok = bool(np.isfinite(deltas).all())
                controls.append(
                    {
                        "representation": representation,
                        "axis": axis,
                        "family": family,
                        "pole": pole,
                        "situation": task,
                        "role": roles[task],
                        "style": style,
                        "supported": ok,
                        "cross_seed_support_per_coordinate": crossed_product(
                            deltas, deltas
                        )
                        / values.shape[1]
                        if ok
                        else None,
                        "mean_response_rms": float(
                            np.sqrt(np.mean(deltas.mean(axis=0) ** 2))
                        )
                        if ok
                        else None,
                    }
                )
    destination = out / "response_field_companion"
    destination.mkdir(exist_ok=False)
    H.write_csv(destination / "persona_responses.csv", traits)
    H.write_csv(destination / "style_controls.csv", controls)
    H.write_csv(destination / "response_fields.csv", fields)
    summary = {}
    for representation in source:
        valid = [
            r
            for r in fields
            if r["representation"] == representation and r["supported"]
        ]
        summary[representation] = {
            "supported_groups": len(valid),
            "positive_mixed_support": sum(
                r["mixed_support_per_coordinate"] > 0 for r in valid
            ),
            "both_persona_directions_supported": sum(
                r["neutral_support_per_coordinate"] > 0
                and r["styled_support_per_coordinate"] > 0
                for r in valid
            ),
            "negative_cross_style_alignment": sum(
                r["alignment_per_coordinate"] < 0 for r in valid
            ),
            "both_supported_negative_alignment": sum(
                r["neutral_support_per_coordinate"] > 0
                and r["styled_support_per_coordinate"] > 0
                and r["alignment_per_coordinate"] < 0
                for r in valid
            ),
            "max_polarization_error": max(
                (r["polarization_error"] for r in valid), default=None
            ),
        }
    H.write_json(
        destination / "summary.json",
        {
            "representation_summaries": summary,
            "persona_rows": len(traits),
            "style_control_groups": len(controls),
            "response_field_groups": len(fields),
            "scope": "Signed finite generator replicate geometry,not authors/psychological information/accuracy",
            "new_model_calls": 0,
        },
    )
    code = [
        Path(__file__),
        ROOT / "tests/test_grain_joint_persona_style_field.py",
        ROOT / "docs/GRAIN_JOINT_RESPONSE_FIELD_COMPANION_HM150.md",
    ]
    H.write_json(
        destination / "seal.json",
        {
            "codes": {str(p.relative_to(ROOT)): H.digest(p) for p in code},
            "outputs": {p.name: H.digest(p) for p in sorted(destination.iterdir())},
            "parent_output_seal": H.digest(out / "output_seal.json"),
        },
    )
    print(summary, flush=True)


def main():
    """Analyze only a complete sealed joint collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/joint_persona_style_r2",
    )
    analyze(parser.parse_args().output)


if __name__ == "__main__":
    main()
