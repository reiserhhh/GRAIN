#!/usr/bin/env python3
"""Exact planned source-code x receiver-code contrasts, using collected HM53 only."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_answer_recoding_mechanism import OUT, check, write
from scripts.run_grain_math_response import write_csv
from scripts.verify_grain_math_relational_cut import csv_rows


def characters(cells):
    """Project four declared raw actions onto the four characters of C2 x C2."""
    if set(cells) != set(itertools.product((1, -1), repeat=2)):
        raise ValueError("Every source/receiver-code cell required")
    values = {}
    for name, i, j in (
        ("constant", 0, 0),
        ("source_code", 1, 0),
        ("receiver_code", 0, 1),
        ("code_interaction", 1, 1),
    ):
        values[name] = math.fsum(s**i * k**j * y for (s, k), y in cells.items()) / 4
    replay = {
        key: values["constant"]
        + key[0] * values["source_code"]
        + key[1] * values["receiver_code"]
        + key[0] * key[1] * values["code_interaction"]
        for key in cells
    }
    error = max(abs(cells[key] - replay[key]) for key in cells)
    return values | {"reconstruction_error": error}


def analyze(out):
    """Keep all signed cells/contrasts; no fitting, hidden representation claim or calls."""
    check(out)
    if (
        read_json(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM53_NOT_GLOBAL"
    ):
        raise ValueError("Verified main acquisition required")
    effects = csv_rows(out / "03/effects.csv")
    grouped = defaultdict(dict)
    for r in effects:
        if r["arm"] in ("source_forward", "source_reverse"):
            key = tuple(r[k] for k in ("content", "class", "person", "direction"))
            cell = (1 if r["arm"] == "source_forward" else -1, int(r["codec"]))
            if cell in grouped[key]:
                raise ValueError("Repeated source/code cell")
            grouped[key][cell] = float(r["raw_action"])
    rows = []
    for key, cells in grouped.items():
        values = characters(cells)
        # A second formula via the previously verified receiver-code orbit table.
        plus_even = (cells[1, 1] + cells[1, -1]) / 2
        minus_even = (cells[-1, 1] + cells[-1, -1]) / 2
        plus_odd = (cells[1, 1] - cells[1, -1]) / 2
        minus_odd = (cells[-1, 1] - cells[-1, -1]) / 2
        alternate = {
            "constant": (plus_even + minus_even) / 2,
            "source_code": (plus_even - minus_even) / 2,
            "receiver_code": (plus_odd + minus_odd) / 2,
            "code_interaction": (plus_odd - minus_odd) / 2,
        }
        if (
            max(abs(values[k] - alternate[k]) for k in alternate) > 1e-10
            or values["reconstruction_error"] > 1e-10
        ):
            raise ValueError("Finite character reconstruction failed")
        rows.append(
            dict(zip(("content", "class", "person", "direction"), key, strict=True))
            | {f"raw_{s}_{k}": y for (s, k), y in cells.items()}
            | values
        )
    if len(rows) != 48:
        raise ValueError("All48 full matrices required")
    groups = defaultdict(list)
    for r in rows:
        groups[r["content"], r["class"]].append(r)
    fields = ("constant", "source_code", "receiver_code", "code_interaction")
    summary = [
        {
            "content": key[0],
            "class": key[1],
            "rows": len(v),
            **{k + "_mean": math.fsum(r[k] for r in v) / len(v) for k in fields},
            **{
                k + "_rms": math.sqrt(math.fsum(r[k] ** 2 for r in v) / len(v))
                for k in fields
            },
        }
        for key, v in groups.items()
    ]
    write_csv(out / "04_code_characters/matrices.csv", rows)
    write_csv(out / "04_code_characters/target_class_summary.csv", summary)
    natives = csv_rows(out / "03/native.csv") + csv_rows(out / "03/source_native.csv")
    native_index = {
        (
            r["role"],
            r["content"],
            r["class"],
            r["person"],
            int(r["direction"]),
            int(r["codec"]),
        ): r
        for r in natives
    }
    native_contrasts = []
    for key, a in native_index.items():
        role, target, kind, person, direction, codec = key
        if direction != 0:
            continue
        b = native_index[role, target, kind, person, 1, codec]
        sa, sb = [native_index[role, target, kind, person, d, 1] for d in (0, 1)]
        contrast = float(b["raw_score"]) - float(a["raw_score"])
        standard_contrast = float(sb["raw_score"]) - float(sa["raw_score"])
        native_contrasts.append(
            {
                "role": role,
                "content": target,
                "class": kind,
                "person": person,
                "codec": codec,
                "raw_ownership_contrast": contrast,
                "semantic_ownership_contrast": codec * contrast,
                "standard_raw_contrast": standard_contrast,
                "raw_direction_matches_standard": contrast * standard_contrast > 0,
                "semantic_direction_matches_standard": codec
                * contrast
                * standard_contrast
                > 0,
            }
        )
    write_csv(
        out / "04_code_characters/native_ownership_contrasts.csv", native_contrasts
    )
    native_groups = defaultdict(list)
    for r in native_contrasts:
        native_groups[r["role"], r["class"], r["codec"]].append(r)
    native_summary = [
        {
            "role": k[0],
            "class": k[1],
            "codec": k[2],
            "contrasts": len(v),
            "raw_direction_matches_standard": sum(
                r["raw_direction_matches_standard"] for r in v
            ),
            "semantic_direction_matches_standard": sum(
                r["semantic_direction_matches_standard"] for r in v
            ),
            "semantic_contrast_rms": math.sqrt(
                math.fsum(r["semantic_ownership_contrast"] ** 2 for r in v) / len(v)
            ),
        }
        for k, v in native_groups.items()
    ]
    write_csv(out / "04_code_characters/native_summary.csv", native_summary)
    geometry = []
    with np.load(out / "01/source_vectors.npz", allow_pickle=False) as source:
        for target in read_json(out / "00/manifest.json")["spec"]["targets"]:
            for person in ("A", "B"):
                a, b = [
                    source[f"{target}__{person}__{codec}"].ravel() for codec in (1, -1)
                ]
                na, nb = (
                    math.sqrt(math.fsum(x * x for x in a)),
                    math.sqrt(math.fsum(x * x for x in b)),
                )
                dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
                gap = math.sqrt(
                    math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))
                )
                geometry.append(
                    {
                        "content": target,
                        "person": person,
                        "standard_norm": na,
                        "inverse_norm": nb,
                        "cosine": dot / (na * nb) if na and nb else None,
                        "relative_code_difference": gap / na if na else None,
                        "same_model_ordered_slot_frame_not_semantic_equivariance": True,
                    }
                )
    write_csv(out / "04_code_characters/source_code_geometry.csv", geometry)
    write(
        out / "04_code_characters/summary.json",
        {
            "rows": 48,
            "cells": 192,
            "maximum_reconstruction_error": max(
                r["reconstruction_error"] for r in rows
            ),
            "new_model_calls": 0,
            "standard_finite_fourier_not_new_hidden_group_theorem": True,
            "source_script_sha256": digest(Path(__file__)),
            "input_effects_sha256": digest(out / "03/effects.csv"),
            "groups": summary,
            "native_groups": native_summary,
            "source_code_geometry": geometry,
        },
    )
    print("CHARACTERS:48four-cell matrices,0newcalls", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    analyze(parser.parse_args().output)
