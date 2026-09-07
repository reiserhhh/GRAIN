#!/usr/bin/env python3
"""Independent two-by-two product-grid replay of P1S amplitude/gain accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash immutable artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, original, reversed_dir):
    """Reconstruct amplitudes from arrays and gains from raw logits, not stored ratios."""
    for root in (result, original, reversed_dir):
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
                    raise ValueError("Changed source")
    manifest = json.loads((result / "manifest.json").read_text())
    for key, root in (
        ("original_identity", original),
        ("reversed_identity", reversed_dir),
    ):
        if (
            manifest[key]
            != json.loads((root / "manifest.json").read_text())["identity"]
        ):
            raise ValueError("Parent chain")
    metadata = json.loads((reversed_dir / "materials.json").read_text())["rows"]
    directions = np.load(original / "directions.npz")
    stores = []
    for root in (original, reversed_dir):
        stores.append(
            (
                root,
                pd.read_csv(root / "baselines.csv").set_index("case_id"),
                pd.read_csv(root / "effects.csv").set_index(
                    ["case_id", "axis", "patch"]
                ),
                json.loads((root / "donors.json").read_text()),
                json.loads((root / "token_layouts.json").read_text()),
            )
        )
    saved = pd.read_csv(result / "paired_components.csv").set_index("case_id")
    saved_rest = pd.read_csv(result / "remainder_components.csv").set_index(
        ["case_id", "order"]
    )
    if (
        len(saved) != 64
        or len(saved_rest) != 128
        or saved.index.duplicated().any()
        or saved_rest.index.duplicated().any()
    ):
        raise ValueError("Complete pair population required")
    fresh, rest = [], []
    error = 0.0
    for r in metadata:
        items = []
        for order, case, store in zip(
            ("original", "reversed"), (r["parent_case"], r["case_id"]), stores
        ):
            root, base, effect, donors, layouts = store
            source = (
                donors[case]["B"]["anchor"]
                if order == "original"
                else donors[case]["anchor"]
            )
            x = np.load(root / "activations" / (case + ".npz"))["block_16"][
                layouts[case]["anchor"][0]
            ].astype(float)
            y = np.load(root / "activations" / (source + ".npz"))["block_16"][
                layouts[source]["anchor"][0]
            ].astype(float)
            a = float(
                np.linalg.lstsq(
                    directions[r["entity"] + "_direction"][:, None], y - x, rcond=None
                )[0][0]
            )
            b = base.loc[case]
            zero = np.longdouble(b.high_logit) - np.longdouble(b.low_logit)
            scores = []
            for patch in ("natural", "projected", "complement"):
                v = effect.loc[(case, "B", patch)]
                scores.append(
                    np.longdouble(v.high_logit) - np.longdouble(v.low_logit) - zero
                )
            n, p, c = scores
            if abs(a) < 1e-6:
                raise ValueError("Undefined gain")
            items.append((np.longdouble(a), p, p / a))
            row = {
                "case_id": case,
                "order": order,
                "group_id": r["group_id"],
                "criterion": r["criterion"],
                "complement": float(c),
                "interaction": float(n - p - c),
                "projection_error": float(n - p),
            }
            rest.append(row)
            for k in ("complement", "interaction", "projection_error"):
                error = max(error, abs(row[k] - saved_rest.loc[(case, order), k]))
        (a0, y0, g0), (a1, y1, g1) = items
        grid = np.outer(np.array([g0, g1]), np.array([a0, a1]))
        values = {
            "original_amplitude": a0,
            "reversed_amplitude": a1,
            "original_shift": y0,
            "reversed_shift": y1,
            "original_gain": g0,
            "reversed_gain": g1,
            "amplitude_term": np.mean(grid[:, 1] - grid[:, 0]),
            "gain_term": np.mean(grid[1, :] - grid[0, :]),
            "observed_change": y1 - y0,
            "original_gain_prediction": grid[0, 1],
        }
        fresh.append(
            {
                "group_id": r["group_id"],
                "criterion": r["criterion"],
                **{k: float(v) for k, v in values.items()},
            }
        )
        for k, v in values.items():
            error = max(error, abs(float(v) - saved.loc[r["case_id"], k]))
        if saved.loc[r["case_id"], "parent_case"] != r["parent_case"]:
            raise ValueError("Wrong paired original")
    fresh = pd.DataFrame(fresh)
    rest = pd.DataFrame(rest)
    summary = pd.read_csv(result / "gain_summary.csv", keep_default_na=False)
    remainder = pd.read_csv(result / "remainder_energy.csv", keep_default_na=False)
    if len(summary) != 10 or len(remainder) != 20:
        raise ValueError("Summary population")
    for r in summary.itertuples():
        s = fresh[fresh.criterion == r.criterion]
        if r.scope == "setting":
            s = s[s.group_id == r.group_id]
        d = s.reversed_shift - s.original_shift
        prediction = s.reversed_shift - s.original_gain_prediction
        values = {
            "cases": len(s),
            "change_sse": np.dot(d, d),
            "amplitude_sse": np.dot(s.amplitude_term, s.amplitude_term),
            "gain_sse": np.dot(s.gain_term, s.gain_term),
            "signed_cross_term": 2 * np.dot(s.amplitude_term, s.gain_term),
            "original_gain_prediction_sse": np.dot(prediction, prediction),
            "zero_prediction_sse": np.dot(s.reversed_shift, s.reversed_shift),
            "prediction_sse_over_unchanged": np.dot(prediction, prediction)
            / np.dot(d, d),
        }
        error = max(error, *(abs(v - getattr(r, k)) for k, v in values.items()))
    for r in remainder.itertuples():
        s = rest[(rest.criterion == r.criterion) & (rest.order == r.order)]
        if r.scope == "setting":
            s = s[s.group_id == r.group_id]
        values = {
            "cases": len(s),
            "projection_error_sse": np.dot(s.projection_error, s.projection_error),
            "complement_sse": np.dot(s.complement, s.complement),
            "interaction_sse": np.dot(s.interaction, s.interaction),
            "signed_cross_term": 2 * np.dot(s.complement, s.interaction),
        }
        error = max(error, *(abs(v - getattr(r, k)) for k, v in values.items()))
    if error > 1e-9:
        raise ValueError("Independent numerical replay")
    return {
        "status": "VERIFIED",
        "pairs": 64,
        "gain_summaries": 10,
        "remainder_summaries": 20,
        "new_neural_calls": 0,
        "raw_array_logit_grid_and_energy_max": error,
        "original_identity": manifest["original_identity"],
        "reversed_identity": manifest["reversed_identity"],
        "interpretation": "Two-by-two product-grid accounting and original-only gain prediction; no personality or independent causal-share claim.",
    }


def main():
    """Store a separate audit receipt and the script that generated it."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "original-dir", "reversed-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.result_dir, a.original_dir, a.reversed_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
