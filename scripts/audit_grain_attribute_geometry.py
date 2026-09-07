#!/usr/bin/env python3
"""Independently solve and replay P1O vector rectangles and directional gains."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Return the exact artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent):
    """Use raw states, a direct solve and raw candidate logits; no neural calls."""
    manifest = json.loads((result / "manifest.json").read_text())
    for root in (result, parent):
        if json.loads((root / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete source")
        for name, digest in json.loads(
            (root / "output_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise ValueError("Changed artifact")
        m = json.loads((root / "manifest.json").read_text())
        with zipfile.ZipFile(root / "source_snapshot.zip") as z:
            for name, digest in m["code"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Changed frozen source")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Wrong source identity")
    material = json.loads((parent / "materials.json").read_text())["rows"]
    layouts = json.loads((parent / "token_layouts.json").read_text())
    donors = json.loads((parent / "donors.json").read_text())
    states = {
        r["case_id"]: np.load(parent / "activations" / (r["case_id"] + ".npz"))[
            "block_16"
        ][layouts[r["case_id"]]["anchor"][0]]
        for r in material
    }
    rectangles = {}
    for r in material:
        if r["criterion"] in ("event", "venue_object"):
            key = (r["group_id"], r["entity"])
            sign = (r["role_sign"], r["room_sign"])
            if sign in rectangles.setdefault(key, {}):
                raise ValueError("Duplicate canonical vector")
            rectangles[key][sign] = states[r["case_id"]].astype(float)
    coefficient_order = json.loads((result / "coefficient_index.json").read_text())
    saved = np.load(result / "coefficients.npz")["coefficients"]
    geometry = pd.read_csv(result / "geometry.csv").set_index(["group_id", "entity"])
    if len(rectangles) != 12 or len(saved) != 12 or len(geometry) != 12:
        raise ValueError("Rectangle population")
    coeffs = {}
    coefficient_error = geometry_error = 0.0
    for i, key in enumerate(coefficient_order):
        key = (key["group_id"], key["entity"])
        x = rectangles[key]
        signs = sorted(x)
        if signs != [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            raise ValueError("Missing world")
        matrix = np.array([[1, b, r, b * r] for b, r in signs], dtype=float)
        coefficients = np.linalg.solve(matrix, np.stack([x[s] for s in signs]))
        coeffs[key] = coefficients
        coefficient_error = max(
            coefficient_error,
            float(abs(coefficients - saved[i]).max()),
            float(abs(matrix @ coefficients - np.stack([x[s] for s in signs])).max()),
        )
        mu, u, v, w = coefficients
        norm = [np.sqrt(np.dot(a, a)) for a in (mu, u, v, w)]
        cosine = np.dot(u, v) / (norm[1] * norm[2])
        values = dict(zip(["mean_norm", "B_norm", "R_norm", "BR_norm"], norm))
        values.update(
            B_R_cosine=cosine,
            B_R_angle_degrees=np.arccos(np.clip(cosine, -1, 1)) * 180 / np.pi,
            mixed_to_main=norm[3] / np.sqrt(norm[1] ** 2 + norm[2] ** 2),
        )
        geometry_error = max(
            geometry_error, *(abs(geometry.loc[key, k] - v) for k, v in values.items())
        )
    base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
    effects = pd.read_csv(parent / "effects.csv").set_index(
        ["case_id", "patch", "intervention"]
    )
    rows = []
    mixed_saved = pd.read_csv(result / "mixed_state_replay.csv").set_index("case_id")
    mixed_error = mixed_replay = 0.0
    for r in material:
        case = r["case_id"]
        n0 = states[case]
        nb, nr, nbr = [states[donors[case][k]["anchor"]] for k in ("B", "R", "BR")]
        expected = (
            4 * r["role_sign"] * r["room_sign"] * coeffs[r["group_id"], r["entity"]][3]
        )
        error = float(
            abs(nbr.astype(float) - (nb + nr - n0).astype(float) - expected).max()
        )
        mixed_error = max(mixed_error, error)
        mixed_replay = max(
            mixed_replay, abs(error - mixed_saved.loc[case, "fp32_additive_replay_max"])
        )
        for axis, n1 in (("B", nb), ("R", nr)):
            patch = effects.loc[(case, "anchor", axis)]
            raw_shift = (patch.high_logit - patch.low_logit) - (
                base.loc[case, "high_logit"] - base.loc[case, "low_logit"]
            )
            delta = n1.astype(float) - n0.astype(float)
            rows.append(
                {
                    "case_id": case,
                    "group_id": r["group_id"],
                    "entity": r["entity"],
                    "criterion": r["criterion"],
                    "axis": axis,
                    "state_displacement": np.sqrt(np.dot(delta, delta)),
                    "logit_shift": raw_shift,
                }
            )
    fresh = pd.DataFrame(rows)
    stored = pd.read_csv(result / "displacement_rows.csv")
    if (
        len(fresh) != 160
        or len(stored) != 160
        or stored.duplicated(["case_id", "axis"]).any()
    ):
        raise ValueError("Response population")
    a = fresh.set_index(["case_id", "axis"]).sort_index()
    b = stored.set_index(["case_id", "axis"]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Response identities")
    response_error = float(
        abs(
            a[["state_displacement", "logit_shift"]]
            - b[["state_displacement", "logit_shift"]]
        )
        .to_numpy()
        .max()
    )
    gain_error = 0.0
    for file, keys, expected_rows in (
        ("gain_by_rectangle.csv", ["group_id", "entity", "criterion", "axis"], 40),
        ("summary.csv", ["criterion", "axis"], 6),
    ):
        table = pd.read_csv(result / file).set_index(keys)
        if len(table) != expected_rows or table.index.duplicated().any():
            raise ValueError("Summary population")
        for key, sub in fresh.groupby(keys):
            norm = np.sqrt(
                np.dot(sub.state_displacement, sub.state_displacement) / len(sub)
            )
            effect = np.sqrt(np.dot(sub.logit_shift, sub.logit_shift) / len(sub))
            row = table.loc[key]
            if row.cases != len(sub):
                raise ValueError("Summary denominator")
            gain_error = max(
                gain_error,
                abs(norm - row.state_rms),
                abs(effect - row.effect_rms),
                abs(effect / norm - row.finite_gain),
            )
    if (
        max(coefficient_error, geometry_error, response_error, gain_error, mixed_replay)
        > 1e-9
        or mixed_error > 1e-5
    ):
        raise ValueError("Independent replay failed")
    return {
        "status": "VERIFIED",
        "new_neural_calls": 0,
        "rectangles": 12,
        "canonical_vectors": 48,
        "response_rows": 160,
        "direct_solve_max": coefficient_error,
        "geometry_max": geometry_error,
        "raw_logit_response_max": response_error,
        "gain_summary_max": gain_error,
        "mixed_fp32_max": mixed_error,
        "mixed_replay_max": mixed_replay,
        "parent_identity": manifest["parent_identity"],
    }


def main():
    """Store the separate audit receipt, keeping analysis outputs immutable."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.result_dir, a.parent_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
