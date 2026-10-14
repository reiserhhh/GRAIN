#!/usr/bin/env python3
"""Descriptive evidence/reader routing, with all refusals and control changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_operator import metrics
from scripts.run_grain_conditional_operator import QUERIES, inputs, table
from scripts.run_grain_persona_generation_trace import write


def run(out):
    """Diagnose representation/response differences without changing fitted models."""
    rows, pairs = inputs()
    with np.load(out / "direct_model.npz") as m:
        scale = m["scale"]
    controls = {
        (r["situation"], r["seed"], r["condition"]): r
        for r in rows
        if r["axis"] == "none"
    }
    contrasts = []

    def add(h, l, kind, axis, family, task, seed):
        d = (np.array(h["g"]) - l["g"]) / scale
        row = {
            "kind": kind,
            "axis": axis,
            "family": family,
            "situation": task,
            "seed": seed,
            "evidence_norm": float(np.linalg.norm(d)),
            "word_delta": h["audit"]["words"] - l["audit"]["words"],
            "refusal": h["audit"]["refusal"] or l["audit"]["refusal"],
        }
        for qi, q in enumerate(QUERIES):
            row[q] = float(h["scores"][qi] - l["scores"][qi])
            row[f"flash_{q}"] = float(h["flash"][qi] - l["flash"][qi])
        contrasts.append(row)

    for p in pairs:
        add(
            p["high"],
            p["low"],
            "persona",
            p["axis"],
            p["family"],
            p["situation"],
            p["seed"],
        )
    for (task, seed, condition), r in controls.items():
        if condition != "neutral":
            add(
                r,
                controls[(task, seed, "neutral")],
                condition,
                "none",
                "none",
                task,
                seed,
            )
    table(out / "evidence_control_contrasts.csv", contrasts)
    summary = []
    for kind, axis in sorted({(r["kind"], r["axis"]) for r in contrasts}):
        for subset in ("all", "no_refusal_sensitivity"):
            rr = [
                r
                for r in contrasts
                if r["kind"] == kind
                and r["axis"] == axis
                and (subset == "all" or not r["refusal"])
            ]
            record = {
                "kind": kind,
                "axis": axis,
                "subset": subset,
                "n": len(rr),
                "mean_evidence_norm": float(np.mean([r["evidence_norm"] for r in rr])),
            }
            for q in QUERIES:
                record[f"mean_{q}"] = float(np.mean([r[q] for r in rr]))
                record[f"mean_flash_{q}"] = float(
                    np.mean([r[f"flash_{q}"] for r in rr])
                )
            summary.append(record)
    table(out / "evidence_control_summary.csv", summary)
    cell_map = {}
    for r in contrasts:
        if r["kind"] == "persona":
            cell_map.setdefault((r["axis"], r["family"], r["situation"]), []).append(r)
    cell_relations = {}
    for axis in QUERIES:
        cc = [rr for (a, _, _), rr in cell_map.items() if a == axis]
        e = np.array([np.mean([r["evidence_norm"] for r in rr]) for rr in cc])
        y = np.array([np.mean([r[axis] for r in rr]) for rr in cc])
        length = np.array([np.mean([r["word_delta"] for r in rr]) for rr in cc])
        z = np.column_stack([np.ones(len(cc)), length])
        residual_e = e - z @ np.linalg.lstsq(z, e, rcond=None)[0]
        residual_y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        cell_relations[axis] = {
            "cells": len(cc),
            "norm_vs_signed_readout_r": metrics(e, y)["r"],
            "length_partial_r": metrics(residual_e, residual_y)["r"],
            "scope": "post-result description, not a psychological validation or routing proof",
        }
    write(out / "cell_endpoint_relations.json", cell_relations)
    print(json.dumps({"controls": summary, "cell_relations": cell_relations}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    run(ap.parse_args().output)
