#!/usr/bin/env python3
"""Distinguish answer-code logit offsets from genuine changes in evidence response."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    quality,
    read_json,
    write_json,
)
from scripts.analyze_grain_evidence_packets import instrument_matrix, response_rows


def run(packet, out):
    """Fit only a construction-world code offset; evaluate fixed held scenes."""
    if out.exists():
        raise FileExistsError("Use a new log-odds analysis ID")
    if read_json(packet / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete native packet first")
    worlds = pd.DataFrame(read_json(packet / "private_worlds.json")).drop(
        columns="texts"
    )
    rows = []
    worst = 0.0
    for d in read_json(packet / "private_design.json"):
        meta = read_json(packet / "private_packets" / f"unit_{d['index']:04d}.json")
        if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
            raise ValueError("Query/unit identity mismatch")
        if d["alphabet"]["values"] != [0.0, 1.0]:
            raise ValueError("Binary positive-code alignment required")
        negative, positive = meta["native"]["candidate_logits"]
        margin = positive - negative
        worst = max(
            worst, abs(float(expit(margin)) - meta["native"]["conditional_score"])
        )
        rows.append(
            {
                "world_id": d["world_id"],
                "query": d["query"],
                "readout": "log_odds",
                "score": margin,
            }
        )
    if worst > 1e-10:
        raise ValueError("Logit/probability identity failed")
    frame = pd.DataFrame(rows).merge(worlds, on="world_id", validate="many_to_one")
    contrasts = response_rows(frame, "identity")
    matrix = instrument_matrix(frame, contrasts, read_json(packet / "queries.json"))
    levels = frame.pivot(
        index=["world_id", "split"], columns="query", values="score"
    ).reset_index()
    records, offsets = [], {}
    for framing in ("big5_binary", "mbti_binary"):
        a, b = framing + "_digits", framing + "_letters"
        train = levels[levels.split == "construction"]
        shift = float((train[b] - train[a]).mean())
        offsets[framing] = {
            "letter_minus_digit_logit_offset": shift,
            "construction_worlds": len(train),
            "construction_scenes": 6,
            "operation": "letters_logit - fixed_offset",
        }
        for split, g in levels.groupby("split"):
            residual = g[b] - shift - g[a]
            records.append(
                {
                    "framing": framing,
                    "split": split,
                    "n": len(g),
                    "fixed_code_offset": shift,
                    "logit_r": quality(g[a], g[b])["r"],
                    "centered_logit_mae": float(abs(residual).mean()),
                    "uncorrected_probability_mae": float(
                        abs(expit(g[b]) - expit(g[a])).mean()
                    ),
                    "offset_corrected_probability_mae": float(
                        abs(expit(g[b] - shift) - expit(g[a])).mean()
                    ),
                    "mean_logit_residual": float(residual.mean()),
                    "sd_logit_residual": float(residual.std()),
                }
            )
    out.mkdir(parents=True)
    matrix.to_csv(out / "query_logodds_matrix.csv", index=False)
    contrasts.to_csv(out / "logodds_responses.csv", index=False)
    pd.DataFrame(records).to_csv(out / "code_offset_transport.csv", index=False)
    write_json(out / "code_offsets.json", offsets)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_QUERY_LOGODDS_DECOMPOSITION",
            "probability_identity_max_error": worst,
            "human_label_fit": False,
            "offset_fit_split": "construction",
            "target": "digits-coded instrument, not human truth",
            "raw_scores_preserved": True,
            "design_sha256": digest(packet / "private_design.json"),
            "code_sha256": digest(Path(__file__)),
        },
    )
    print(
        pd.DataFrame(records)
        .query("split=='held_scene'")
        .round(4)
        .to_string(index=False)
    )
    print(
        matrix.query("split=='held_scene' and change=='coding'")
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    source = BASE / "navigation_v1/NAV-05/query_identity_r1"
    run(source, source / "logodds_analysis_r2")
