#!/usr/bin/env python3
"""Strict local audit of the actually administered adapted mood composites."""

from pathlib import Path

import numpy as np
import pandas as pd

from suica_core.meps_layout import normalize_participant_code

from .factor_catalog import coefficient_alpha

PHASES = ("baseline", "pre", "mid", "post")
COMPOSITES = {
    "adapted_na8": tuple(f"mood_{i:02d}" for i in range(1, 9)),
    "adapted_dejected5": tuple(f"mood_{i:02d}" for i in range(9, 14)),
    "combined_negative13": tuple(f"mood_{i:02d}" for i in range(1, 14)),
}
DESCRIPTIONS = {
    "adapted_na8": "feeling jittery, afraid, upset, worried, on edge, distressed, ashamed or irritable",
    "adapted_dejected5": "feeling sad, unworthy of praise, discouraged and unmotivated, lonely or downhearted",
    "combined_negative13": "negative feelings including fear, worry, agitation, distress, shame, irritability, sadness, discouragement, loneliness and feeling low",
}


def recompute_mood(frame, pseudonyms):
    """Validate all occasions/items and return strict means, never inferred norms."""
    data = frame.copy()
    data["participant_code"] = data.participant_id.map(normalize_participant_code).map(
        pseudonyms
    )
    if data.participant_code.isna().any() or set(data.participant_code) != set(
        pseudonyms.values()
    ):
        raise ValueError("Mood/text membership mismatch")
    if data.duplicated(["participant_code", "phase", "item_id"]).any():
        raise ValueError("Duplicate mood response")
    if set(data.phase) != set(PHASES) or set(data.item_id) != set(
        COMPOSITES["combined_negative13"]
    ):
        raise ValueError("Unexpected mood phase or item")
    data["response"] = pd.to_numeric(data.response, errors="raise")
    if not data.response.dropna().between(1, 6).all():
        raise ValueError("Mood response outside administered1-6 range")
    expected = data.item_id.map(
        lambda k: "PANAS_NA" if int(k[-2:]) <= 8 else "POMS_SHORT"
    )
    if not expected.equals(data.source_scale):
        raise ValueError("Mood item/composite source mismatch")
    records, audit = [], []
    for phase in PHASES:
        wide = data[data.phase == phase].pivot(
            index="participant_code", columns="item_id", values="response"
        )
        wide = wide.reindex(
            index=sorted(pseudonyms.values()), columns=COMPOSITES["combined_negative13"]
        )
        for name, items in COMPOSITES.items():
            selected = wide[list(items)]
            score = selected.mean(axis=1).where(selected.notna().all(axis=1))
            records.extend(
                {
                    "participant_code": uid,
                    "phase": phase,
                    "composite": name,
                    "scale": f"{phase}_{name}",
                    "score": value,
                }
                for uid, value in score.items()
            )
            audit.append(
                {
                    "phase": phase,
                    "composite": name,
                    "items": len(items),
                    "n": len(score),
                    "valid": int(score.notna().sum()),
                    "missing": int(score.isna().sum()),
                    "mean": float(score.mean()),
                    "sd": float(score.std()),
                    "alpha": coefficient_alpha(selected),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(audit)


def audit_mood(root, authorized=False):
    """Read only processed numeric mood items under explicit branch permission."""
    if not authorized:
        raise PermissionError("Explicit local mood branch required")
    root = Path(root)
    folders = sorted(
        p
        for p in (root / "decrypted_output").iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    codes = [normalize_participant_code(p.name) for p in folders]
    if len(set(codes)) != len(codes) or not all(codes):
        raise ValueError("Ambiguous source folders")
    mapping = dict(
        zip(codes, (f"P{i:03d}" for i in range(1, len(codes) + 1)), strict=True)
    )
    path = root / "research2_anal/processed/mood_long.csv"
    frame = pd.read_csv(
        path,
        usecols=["participant_id", "phase", "item_id", "source_scale", "response"],
        dtype={"participant_id": str},
    )
    scores, audit = recompute_mood(frame, mapping)
    if not np.isfinite(scores.score.dropna()).all():
        raise ValueError("Nonfinite mood scores")
    return scores, audit, path
