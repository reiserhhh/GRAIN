#!/usr/bin/env python3
"""P1O: exact four-world vector geometry and finite conditional response gain."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def sha(path):
    """Return a saved-source identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(worlds):
    """Exact balanced Boolean vector expansion; rows are mu, B, R and BR."""
    if set(worlds) != set(SIGNS):
        raise ValueError("Incomplete four-world rectangle")
    x = np.array([worlds[k] for k in SIGNS], dtype=float)
    design = np.array([[1, b, r, b * r] for b, r in SIGNS], dtype=float)
    coefficients = design.T @ x / 4
    error = float(np.max(np.abs(design @ coefficients - x)))
    return coefficients, error


def describe(coefficients):
    """Rotation-invariant finite geometry, without interpreting rank as human factors."""
    mu, u, v, w = coefficients
    un, vn, wn = [float(np.linalg.norm(x)) for x in (u, v, w)]
    cosine = float(np.dot(u, v) / (un * vn)) if un * vn > 1e-12 else np.nan
    return {
        "mean_norm": float(np.linalg.norm(mu)),
        "B_norm": un,
        "R_norm": vn,
        "BR_norm": wn,
        "B_R_cosine": cosine,
        "B_R_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        if np.isfinite(cosine)
        else np.nan,
        "mixed_to_main": wn / np.hypot(un, vn) if np.hypot(un, vn) > 1e-12 else np.nan,
    }


def run(args):
    """Use every declared cached rectangle and all B/R anchor-response rows."""
    start = time.monotonic()
    parent = args.parent_dir
    for name, digest in json.loads((parent / "output_hashes.json").read_text()).items():
        if sha(parent / name) != digest:
            raise ValueError("Parent artifact changed")
    if json.loads((parent / "run_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Complete P1N parent required")
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "docs/GRAIN_ATTRIBUTE_GEOMETRY_P1O_PLAN.md",
        ROOT / "tests/test_grain_attribute_geometry.py",
    ]
    manifest = {
        "parent_identity": parent_manifest["identity"],
        "new_neural_calls": 0,
        "max_seconds": 60,
        "code": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        material = json.loads((parent / "materials.json").read_text())["rows"]
        layouts = json.loads((parent / "token_layouts.json").read_text())
        donors = json.loads((parent / "donors.json").read_text())
        base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
        effects = pd.read_csv(parent / "effects.csv", keep_default_na=False)
        cases = {r["case_id"]: r for r in material}
        states = {
            r["case_id"]: np.load(parent / "activations" / (r["case_id"] + ".npz"))[
                "block_16"
            ][layouts[r["case_id"]]["anchor"][0]]
            for r in material
        }
        groups = {}
        for r in material:
            if r["criterion"] not in ("event", "venue_object"):
                continue
            key = (r["group_id"], r["entity"])
            groups.setdefault(key, {})[r["role_sign"], r["room_sign"]] = states[
                r["case_id"]
            ]
        if len(groups) != 12 or any(len(v) != 4 for v in groups.values()):
            raise ValueError("Expected12 full rectangles")
        geometry, coefficients, order = [], [], []
        reconstruction = 0.0
        by_key = {}
        for key, worlds in sorted(groups.items()):
            coeff, error = expand(worlds)
            reconstruction = max(reconstruction, error)
            coefficients.append(coeff)
            by_key[key] = coeff
            order.append({"group_id": key[0], "entity": key[1]})
            geometry.append(
                {
                    "group_id": key[0],
                    "entity": key[1],
                    "target_kind": "object" if key[1] == "room" else "person",
                    **describe(coeff),
                }
            )
        np.savez_compressed(
            out / "coefficients.npz",
            coefficients=np.array(coefficients),
            signs=np.array(SIGNS),
        )
        (out / "coefficient_index.json").write_text(json.dumps(order, indent=2) + "\n")
        pd.DataFrame(geometry).to_csv(out / "geometry.csv", index=False)
        displacement, mixed = [], []
        for case, r in cases.items():
            coeff = by_key[r["group_id"], r["entity"]]
            n0, nb, nr, nbr = (
                states[case],
                *(states[donors[case][axis]["anchor"]] for axis in ("B", "R", "BR")),
            )
            add = nb + nr - n0
            expected = 4 * r["role_sign"] * r["room_sign"] * coeff[3]
            mixed.append(
                {
                    "case_id": case,
                    "group_id": r["group_id"],
                    "entity": r["entity"],
                    "criterion": r["criterion"],
                    "fp32_additive_replay_max": float(
                        np.max(abs(nbr.astype(float) - add.astype(float) - expected))
                    ),
                }
            )
            for axis in ("B", "R"):
                source = donors[case][axis]["anchor"]
                delta = states[source].astype(float) - states[case].astype(float)
                effect = effects[
                    (effects.case_id == case)
                    & (effects.patch == "anchor")
                    & (effects.intervention == axis)
                ]
                if len(effect) != 1:
                    raise ValueError("Missing anchor response")
                displacement.append(
                    {
                        "case_id": case,
                        "group_id": r["group_id"],
                        "entity": r["entity"],
                        "criterion": r["criterion"],
                        "axis": axis,
                        "state_displacement": float(np.linalg.norm(delta)),
                        "logit_shift": float(
                            effect.margin.iloc[0] - base.loc[case, "margin"]
                        ),
                    }
                )
        pd.DataFrame(mixed).to_csv(out / "mixed_state_replay.csv", index=False)
        displacement = pd.DataFrame(displacement)
        displacement.to_csv(out / "displacement_rows.csv", index=False)
        gain = []
        for key, sub in displacement.groupby(
            ["group_id", "entity", "criterion", "axis"]
        ):
            norm = float(np.sqrt(np.mean(sub.state_displacement**2)))
            effect = float(np.sqrt(np.mean(sub.logit_shift**2)))
            gain.append(
                dict(zip(["group_id", "entity", "criterion", "axis"], key))
                | {
                    "cases": len(sub),
                    "state_rms": norm,
                    "effect_rms": effect,
                    "finite_gain": effect / norm if norm > 1e-12 else np.nan,
                }
            )
        pd.DataFrame(gain).to_csv(out / "gain_by_rectangle.csv", index=False)
        summary = []
        for (criterion, axis), sub in displacement.groupby(["criterion", "axis"]):
            norm = float(np.sqrt(np.mean(sub.state_displacement**2)))
            effect = float(np.sqrt(np.mean(sub.logit_shift**2)))
            summary.append(
                {
                    "criterion": criterion,
                    "axis": axis,
                    "cases": len(sub),
                    "state_rms": norm,
                    "effect_rms": effect,
                    "finite_gain": effect / norm if norm > 1e-12 else np.nan,
                }
            )
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
        maximum = max(r["fp32_additive_replay_max"] for r in mixed)
        if (
            len(displacement) != 160
            or len(gain) != 40
            or reconstruction > 1e-10
            or maximum > 1e-5
        ):
            raise ValueError("Incomplete panel or exact expansion failed")
        if time.monotonic() - start > 60:
            raise RuntimeError("P1O CPU budget exceeded")
        status = {
            "status": "COMPLETE",
            "new_neural_calls": 0,
            "rectangles": 12,
            "canonical_vectors": 48,
            "response_rows": 160,
            "seconds": time.monotonic() - start,
            "float64_reconstruction_max": reconstruction,
            "fp32_additive_replay_max": maximum,
            "interpretation": "Finite encoding geometry versus conditional response; not psychological factors or global derivatives.",
        }
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        (out / "output_hashes.json").write_text(
            json.dumps(
                {
                    str(p.relative_to(out)): sha(p)
                    for p in out.rglob("*")
                    if p.is_file() and not p.name.startswith("._")
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(status))
    except Exception as exc:
        (out / "run_status.json").write_text(
            json.dumps({"status": "FAILED", "new_neural_calls": 0, "error": str(exc)})
            + "\n"
        )
        raise


def main():
    """Run one saved-state packet with no neural dependency or new text input."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
