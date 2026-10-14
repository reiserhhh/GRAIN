#!/usr/bin/env python3
"""Offline alternative-formula HM124 audit, with no torch or model execution."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(out):
    """Rebuild source coefficients and all native scalars independently."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    for name in ("inputs",):
        H.verify_hashes(out, manifest[name])
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "02/seal.json"))
    state = H.load(out / "status.json")
    if (
        state["state"] != "COLLECTED"
        or state["completed"] != 240
        or state["pending"] is not None
    ):
        raise ValueError("Full safely terminal capture required for full readback")
    error = {
        n: 0.0
        for n in (
            "source_projection",
            "source_actions",
            "native_units",
            "native_field",
            "effect_csv",
            "interaction",
        )
    }
    source_rows = H.load(out / "00/source_rows.json")
    with np.load(out / "00/decomposition.npz", allow_pickle=False) as z:
        a, x = z["design"], z["values"]
        represented = a @ np.linalg.lstsq(a, x, rcond=None)[0]
        error["source_projection"] = float(np.max(abs(represented - z["represented"])))
        reconstructed = {
            n: np.zeros((6, 2, x.shape[1]))
            for n in ("grain_part", "remainder", "whole")
        }
        for i, row in enumerate(source_rows):
            f, c = row["family"], (row["condition"] + 1) // 2
            sign = row["owner"] * row["polarity"] / 8
            for name, value in (
                ("grain_part", represented),
                ("remainder", x - represented),
                ("whole", x),
            ):
                reconstructed[name][f, c] += sign * value[i]
        for name, value in reconstructed.items():
            error["source_actions"] = max(
                error["source_actions"], float(np.max(abs(value - z[name])))
            )
    rows, design = H.load(out / "00/rows.json"), H.load(out / "00/design.json")
    fields = np.zeros((5, 2, 6, 6, 4))
    arm_ids = {n: i for i, n in enumerate(manifest["spec"]["arms"])}
    for item in design:
        stem = out / f"01/units/b{item['batch']:03d}"
        meta = H.load(stem.with_suffix(".json"))
        if H.sha(stem.with_suffix(".npz")) != meta["sha256"]:
            raise ValueError("Native batch hash changed")
        row = rows[item["row"]]
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            logit, normalizer = z["logits"].astype(float), z["normalizer"].astype(float)
            lognorm = logsumexp(logit, axis=1)
            probabilities = np.exp(logit - lognorm[:, None])
            units = np.column_stack(
                (
                    probabilities @ (np.arange(10) / 9),
                    logsumexp(logit[:, 5:], axis=1) - logsumexp(logit[:, :5], axis=1),
                    np.exp(lognorm - normalizer),
                    -(probabilities * (logit - lognorm[:, None])).sum(1),
                )
            )
            error["native_units"] = max(
                error["native_units"], float(np.max(abs(units - z["readings"])))
            )
        for focus in range(2):
            fields[arm_ids[item["arm"]], row["wording"], row["family"]] += (
                row["polarity"]
                * row["owner"]
                * (1 - 2 * focus)
                / 8
                * units[focus * 6 : (focus + 1) * 6]
            )
    effects = fields[0:1] - fields
    interaction = effects[3] - effects[1] - effects[2]
    with np.load(out / "02/native_fields.npz", allow_pickle=False) as z:
        error["native_field"] = float(np.max(abs(fields - z["fields"])))
        error["interaction"] = float(np.max(abs(interaction - z["interaction"])))
    with (out / "02/all_target_effects.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    if len(cells) != 1440:
        raise ValueError("Missing per-target cells")
    for row in cells:
        ix = tuple(int(row[n]) for n in ("wording", "family", "target", "unit"))
        arm = arm_ids[row["arm"]]
        error["effect_csv"] = max(
            error["effect_csv"],
            abs(float(row["baseline"]) - fields[0][ix]),
            abs(float(row["effect"]) - effects[arm][ix]),
        )
    if max(error.values()) > 1e-8:
        raise ValueError(f"Independent numerical readback failed: {error}")
    result = dict(
        status="VERIFIED_FINITE_NATIVE_REMAINDER_NOT_FULL",
        errors=error,
        raw_native_batches=240,
        raw_native_endpoints=2880,
        source_records=96,
        per_target_rows=1440,
        same_agent_alternative_formula=True,
        new_model_calls=0,
        original_scope_complete=False,
    )
    H.write(out / "readback.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/grain_native_remainder_r1",
    )
    audit(parser.parse_args().out.resolve())
