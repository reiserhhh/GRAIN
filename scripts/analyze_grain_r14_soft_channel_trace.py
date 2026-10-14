#!/usr/bin/env python
"""Contrasts for the R14 soft-channel trace companion (protocol V1). Application authors; 2,000 seed-stratified draws."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/R14/soft_channel_trace_r1"
DRAWS, BSEED = 2000, 20260915
MET = ("brier", "log_loss", "nextp_se")
CONTRASTS = {
    "grainpole hard1 - soft.5 (thresholding bias)": ("grainpole_grainq", "grainpole_soft"),
    "grainpole soft.5 - noinfo (model recycling)": ("grainpole_soft", "grain_noinfo_reference"),
    "noinfo - oracle (prior-only cost)": ("grain_noinfo_reference", "oracle"),
    "hybrid soft - hard": ("hybrid_soft", "hybrid_nlipole_grainq"),
    "lex96 soft - hard": ("lex96_soft", "lex96_transported"),
    "lex80 soft - hard": ("lex80_soft", "lex80_refit"),
    "hybrid_soft - oracle": ("hybrid_soft", "oracle"),
    "lex96_soft - oracle": ("lex96_soft", "oracle"),
    "lex80_soft - oracle": ("lex80_soft", "oracle"),
    "grainpole_soft - oracle": ("grainpole_soft", "oracle"),
}


def main() -> int:
    m = pd.read_csv(OUT / "metrics_by_author.csv", float_precision="round_trip")
    m = m[m.role == "application"]
    rng = np.random.default_rng(BSEED)
    rows = []
    for world in sorted(m.world.unique()):
        base = m[(m.world == world) & (m.channel == "oracle") & (m.panel == "mixed") & (m.candidate == "b1")].sort_values(["seed", "author_id"])
        keys = base[["seed", "author_id"]].reset_index(drop=True)
        strata = [np.where(keys.seed.to_numpy() == s)[0] for s in sorted(keys.seed.unique())]
        draws = np.stack([np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata]) for _ in range(DRAWS)])

        def vec(ch, panel, cand, met):
            d = m[(m.world == world) & (m.channel == ch) & (m.panel == panel) & (m.candidate == cand)].merge(keys, on=["seed", "author_id"], how="right", validate="one_to_one")
            return d[met].to_numpy(float)

        for name, (a, b) in CONTRASTS.items():
            for panel in ("complete", "mixed"):
                for cand in ("b1", "b2", "m", "personal_mean"):
                    for met in MET:
                        d = vec(a, panel, cand, met) - vec(b, panel, cand, met)
                        bs = d[draws].mean(axis=1)
                        rows.append({"world": world, "contrast": name, "panel": panel, "candidate": cand, "metric": met, "mean": float(d.mean()),
                                     "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)), "n": len(d)})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "contrasts.csv", index=False)
    tr = pd.read_parquet(OUT / "event_trace.parquet", columns=["role", "panel", "channel", "t", "evidence_mass_after", "transition_mass_after"])
    last = tr[(tr.t == 47) & (tr.role == "application") & tr.channel.isin(["grainpole_soft", "grain_noinfo_reference", "oracle", "grainpole_grainq"])]
    recyc = last.groupby(["panel", "channel"])[["evidence_mass_after", "transition_mass_after"]].mean().reset_index()
    recyc.to_csv(OUT / "recycling_mass.csv", index=False)
    meta = {"draws": DRAWS, "bootstrap_seed": BSEED, "contrast_rows": len(c),
            "output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in ("contrasts.csv", "recycling_mass.csv", "metrics_by_author.csv")},
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "analysis.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(recyc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
