#!/usr/bin/env python3
"""Rebuild the partial choice descriptions from raw logits by alternate formulas."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_text_anchor_hm164 as A
from scripts import run_grain_writer_history_bridge_hm165 as R


def csv_rows(path):
    """Read an immutable descriptive table for direct raw-observation replay."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    """Audit each unit/draw, each signed cell and every orthogonal coordinate."""
    out = R.DEFAULT
    manifest = R.check(out)
    partial = out / "partial_description_r1"
    observation = out / "observation_law_description_r1"
    for parent in (partial, observation):
        for name, expected in A.read(parent / "output_seal.json").items():
            if A.digest(parent / name) != expected:
                raise ValueError("Changed sealed output: " + name)
    units = A.read(out / "units.json")
    captures = R.results(out)
    tables, draw_errors = {}, 0
    for index, row in captures.items():
        unit = units[index]
        if row["unit_id"] != unit["unit_id"] or row["role"] != "discovery":
            raise ValueError("Captured unit/role mismatch")
        aa, bb = row["ab_logits"]
        qa = (1 + math.tanh((aa - bb) / 2)) / 2
        key = hashlib.sha256(
            f"{manifest['config']['draw_seed']}|{unit['unit_id']}".encode()
        ).digest()
        draw = int(
            np.random.default_rng(int.from_bytes(key[:8], "little")).random() < qa
        )
        draw_errors += draw != row["draw_a"] or row["protocol_output"] != (
            "A" if draw else "B"
        )
    for request in A.read(out / "requests.json"):
        if request["unit_index"] not in captures:
            continue
        row = captures[request["unit_index"]]
        aa, bb = row["ab_logits"]
        ell = (aa - bb) * (1 if request["order"] == 0 else -1)
        key = request["pair_id"], request["opportunity"], request["order"]
        tables.setdefault(key, {})[request["policy"], request["history"]] = (
            ell,
            (1 + math.tanh(ell / 2)) / 2,
        )
    full = {key: cells for key, cells in tables.items() if len(cells) == 9}
    errors = {
        "raw_cell": 0.0,
        "link_identity": 0.0,
        "orthogonal_coordinate": 0.0,
        "natural_identity": 0.0,
    }
    for row in csv_rows(observation / "link_accounts.csv"):
        key = row["pair_id"], row["opportunity"], int(row["order"])
        p, h = int(row["policy"]), int(row["history"])
        ell, q = full[key][p, h]
        cells = full[key]
        sep = cells[p, 0][0] + cells[0, h][0] - cells[0, 0][0]
        qs = (1 + math.tanh(sep / 2)) / 2
        qa = cells[p, 0][1] + cells[0, h][1] - cells[0, 0][1]
        errors["raw_cell"] = max(
            errors["raw_cell"],
            abs(q - float(row["actual_q"])),
            abs(ell - float(row["actual_log_odds"])),
        )
        for name, expected in (
            ("link_curvature", qs - qa),
            ("response_remainder", q - qs),
            ("log_odds_interaction", ell - sep),
            ("q_interaction", q - qa),
        ):
            errors["link_identity"] = max(
                errors["link_identity"], abs(expected - float(row[name]))
            )
    contrasts = {
        "constant": [1 / math.sqrt(3)] * 3,
        "polarity": [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
        "nonneutral_vs_neutral": [
            1 / math.sqrt(6),
            -2 / math.sqrt(6),
            1 / math.sqrt(6),
        ],
    }
    coordinates = csv_rows(observation / "contrast_coordinates.csv")
    for row in coordinates:
        cells = full[row["pair_id"], row["opportunity"], int(row["order"])]
        pi, hj = contrasts[row["policy_contrast"]], contrasts[row["history_contrast"]]
        index = 1 if row["measure"] == "q" else 0
        value = math.fsum(
            pi[i] * hj[j] * cells[i - 1, j - 1][index]
            for i in range(3)
            for j in range(3)
        )
        errors["orthogonal_coordinate"] = max(
            errors["orthogonal_coordinate"], abs(value - float(row["coefficient"]))
        )
    for row in csv_rows(partial / "complete_response_accounts.csv"):
        cells = full[row["pair_id"], row["opportunity"], int(row["order"])]
        for prefix, index in (("", 1), ("log_odds_", 0)):
            natural = cells[1, 1][index] - cells[-1, -1][index]
            reconstructed = math.fsum(
                float(row[prefix + k])
                for k in ("policy_effect", "history_effect", "interaction_effect")
            )
            errors["natural_identity"] = max(
                errors["natural_identity"], abs(natural - reconstructed)
            )
    if draw_errors or max(errors.values()) > 1e-12:
        raise ValueError("Raw observation replay failed")
    result = {
        "status": "PASS_ALTERNATE_RAW_LOGIT_REPLAY_NOT_BLIND",
        "captures_checked": len(captures),
        "complete_tables_checked": len(full),
        "contrast_coordinates_checked": len(coordinates),
        "draw_errors": draw_errors,
        "max_errors": errors,
        "new_model_forwards": 0,
        "source_program_compiled": False,
        "original_scope_complete": False,
        "script_sha256": A.digest(Path(__file__)),
    }
    A.write(observation / "alternate_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
