#!/usr/bin/env python3
"""Cached two-endpoint triage and frozen small evidence-action transfer study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_operator import (
    fit_action,
    make_pairs,
    metrics,
    predict_action,
)
from scripts.run_grain_persona_generation_trace import BASE, read_jsonl, sha, write

QUERIES = ("extraversion", "neuroticism")
DEFAULT = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r1"


def table(path, rows):
    """Write a nonempty flat CSV table deterministically."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def inputs():
    """Load immutable synthetic material and join cached endpoints by text ID."""
    folder = BASE / "expanded"
    rows = read_jsonl(folder / "generations.jsonl")
    g = json.loads((folder / "readout_grain.json").read_text())["grain"]
    scores = json.loads((folder / "readout_local_query.json").read_text())["raw"]
    flash = {
        r["text_id"]: r["rating"] for r in read_jsonl(folder / "judge_deepseek.jsonl")
    }
    if len({r["text_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate material IDs")
    for r in rows:
        rid = r["text_id"]
        r["g"] = g[rid]["g"]
        r["scores"] = [scores[rid][q] for q in QUERIES]
        r["flash"] = [flash[rid][q] for q in QUERIES]
        if len(r["g"]) != 4:
            raise ValueError("Missing evidence coordinate")
    return rows, make_pairs(rows)


def identity(out):
    """Refuse to mix changed inputs, source code or protocol in a run."""
    files = [
        BASE / "expanded" / p
        for p in (
            "generations.jsonl",
            "readout_grain.json",
            "readout_local_query.json",
            "judge_deepseek.jsonl",
        )
    ]
    files += [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/conditional_operator.py",
        ROOT / "docs/GRAIN_CONDITIONAL_OPERATOR_GOAL_20260915.md",
    ]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in files}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Run identity changed; choose another output")
    write(path, manifest)


def triage(out, rows, pairs):
    """Compare two endpoints without treating either as psychological truth."""
    source = [p for p in pairs if p["split"] == "source"]
    dx = np.array([np.subtract(p["high"]["g"], p["low"]["g"]) for p in pairs])
    src = np.array([p["split"] == "source" for p in pairs])
    sy = np.array([np.subtract(p["high"]["scores"], p["low"]["scores"]) for p in pairs])
    model = fit_action(dx[src], sy[src])
    pred = predict_action(model, dx)
    np.savez(out / "direct_model.npz", **model)
    pair_rows = []
    for i, p in enumerate(pairs):
        h, l = p["high"], p["low"]
        row = {
            k: p[k] for k in ("pair", "axis", "family", "situation", "seed", "split")
        }
        row.update(
            high_id=h["text_id"],
            low_id=l["text_id"],
            refusal=bool(h["audit"]["refusal"] or l["audit"]["refusal"]),
            word_delta=h["audit"]["words"] - l["audit"]["words"],
            evidence_norm=float(np.linalg.norm(dx[i] / model["scale"])),
        )
        row.update({f"g{k + 1}_delta": float(v) for k, v in enumerate(dx[i])})
        for qi, q in enumerate(QUERIES):
            row[f"local_{q}"] = float(sy[i, qi])
            row[f"direct_{q}"] = float(pred[i, qi])
            row[f"flash_{q}"] = float(h["flash"][qi] - l["flash"][qi])
        pair_rows.append(row)
    table(out / "pairs.csv", pair_rows)
    groups = {}
    for r in pair_rows:
        groups.setdefault((r["axis"], r["family"], r["situation"]), []).append(r)
    cells = []
    for (axis, family, task), rr in groups.items():
        cell = {
            "axis": axis,
            "family": family,
            "situation": task,
            "n_pairs": len(rr),
            "refusal_pairs": sum(r["refusal"] for r in rr),
        }
        for k in (
            "evidence_norm",
            "word_delta",
            "g1_delta",
            "g2_delta",
            "g3_delta",
            "g4_delta",
            "local_extraversion",
            "local_neuroticism",
            "flash_extraversion",
            "flash_neuroticism",
        ):
            cell[k] = float(np.mean([r[k] for r in rr]))
        cells.append(cell)
    table(out / "cells.csv", cells)
    summary = {
        "texts": len(rows),
        "pairs": len(pairs),
        "source_pairs": len(source),
        "cells": len(cells),
        "refusal_texts": sum(r["audit"]["refusal"] for r in rows),
        "direct": {},
    }
    for split in ("source", "new_task", "new_wording", "new_both", "application"):
        sel = np.array(
            [
                p["split"] != "source"
                if split == "application"
                else p["split"] == split
                for p in pairs
            ]
        )
        summary["direct"][split] = {
            q: metrics(pred[sel, qi], sy[sel, qi]) for qi, q in enumerate(QUERIES)
        }
    write(out / "triage_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    """Execute a finite cache analysis; neural stages are separate explicit commands."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT)
    args = ap.parse_args()
    identity(args.output)
    rows, pairs = inputs()
    triage(args.output, rows, pairs)


if __name__ == "__main__":
    main()
