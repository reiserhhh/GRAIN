#!/usr/bin/env python
"""Analysis for R14 same-object channels (protocol V1). Application authors only.

The bootstrap uses 2,000 draws stratified by seed (seed 20260915). The same author draws are shared by every channel,
panel, candidate and metric within a world.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/R14/same_object_channels_r1"
DRAWS, BSEED = 2000, 20260915
METRICS = ("brier", "log_loss", "nextp_se")


def main() -> int:
    m = pd.read_csv(OUT / "metrics_by_author.csv", float_precision="round_trip")
    m = m[m.role == "application"]
    sec = pd.read_csv(OUT / "secondary_em_metrics_by_author.csv", float_precision="round_trip")
    levels = m.groupby(["world", "panel", "channel", "candidate"])[list(METRICS)].mean().reset_index()
    levels.to_csv(OUT / "levels.csv", index=False)
    rng = np.random.default_rng(BSEED)
    rows = []

    def arr(df, world):
        d = df[df.world == world].sort_values(["seed", "author_id"])
        return d

    for world in sorted(m.world.unique()):
        base = arr(m[(m.panel == "mixed") & (m.channel == "oracle") & (m.candidate == "b1")], world)
        seeds = base.seed.to_numpy()
        strata = [np.where(seeds == s)[0] for s in sorted(set(seeds))]
        draws = np.stack([np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata]) for _ in range(DRAWS)])
        authors = base[["seed", "author_id"]].reset_index(drop=True)

        def vec(df):
            d = df.merge(authors, on=["seed", "author_id"], how="right", validate="one_to_one")
            if d[list(METRICS)].isna().any().any():
                raise ValueError("author misalignment")
            return d

        cube = {}
        for (panel, ch, cand), g in m[m.world == world].groupby(["panel", "channel", "candidate"]):
            cube[(panel, ch, cand)] = vec(g)
        for (ch, cand), g in sec[sec.world == world].groupby(["channel", "candidate"]):
            cube[("mixed_em", ch, cand)] = vec(g)

        def add(kind, label, a_key, b_key):
            a, b = cube[a_key], cube[b_key]
            for met in METRICS:
                d = a[met].to_numpy() - b[met].to_numpy()
                bs = d[draws].mean(axis=1)
                rows.append({"world": world, "kind": kind, "contrast": label, "a": "|".join(a_key), "b": "|".join(b_key), "metric": met,
                             "mean": float(d.mean()), "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5)),
                             "n_authors": len(d)})

        channels = sorted({k[1] for k in cube if k[0] in ("mixed", "complete")})
        cands = sorted({k[2] for k in cube})
        for panel in ("complete", "mixed"):
            for ch in channels:
                if ch == "oracle":
                    continue
                for cand in cands:
                    add("channel_minus_oracle", f"{ch}-oracle|{panel}|{cand}", (panel, ch, cand), (panel, "oracle", cand))
        for cand in cands:
            add("layer_render", f"oracle mixed-complete|{cand}", ("mixed", "oracle", cand), ("complete", "oracle", cand))
            add("layer_qualification", f"hybrid-nli_hard (q: GRAIN vs NLI)|mixed|{cand}", ("mixed", "hybrid_nlipole_grainq", cand), ("mixed", "nli_hard", cand))
            add("layer_pole_readout", f"grainpole-hybrid (pole: GRAIN ridge vs NLI)|mixed|{cand}", ("mixed", "grainpole_grainq", cand), ("mixed", "hybrid_nlipole_grainq", cand))
            add("layer_pole_readout", f"grainpole-hybrid (pole: GRAIN ridge vs NLI)|complete|{cand}", ("complete", "grainpole_grainq", cand), ("complete", "hybrid_nlipole_grainq", cand))
            add("layer_lexical_transport", f"lex96-lex80|mixed|{cand}", ("mixed", "lex96_transported", cand), ("mixed", "lex80_refit", cand))
            add("layer_soft_vs_hard", f"nli_soft-nli_hard|mixed|{cand}", ("mixed", "nli_soft", cand), ("mixed", "nli_hard", cand))
            for ch in ("nli_hard", "hybrid_nlipole_grainq", "grainpole_grainq", "lex96_transported", "lex80_refit"):
                if ("mixed_em", ch, cand) in cube:
                    add("layer_source_fit", f"{ch} EM(pilot law)-oracle source|mixed|{cand}", ("mixed_em", ch, cand), ("mixed", ch, cand))
        for panel in ("complete", "mixed"):
            for cand in ("b2", "m", "source_mean", "personal_mean"):
                add("layer_estimator", f"oracle {cand}-b1|{panel}", (panel, "oracle", cand), (panel, "oracle", "b1"))
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "contrasts.csv", index=False)
    files = ["levels.csv", "contrasts.csv", "metrics_by_author.csv", "secondary_em_metrics_by_author.csv", "typed_reader_table.csv", "reader_accuracy_by_role.csv"]
    meta = {"draws": DRAWS, "bootstrap_seed": BSEED, "contrasts": len(c),
            "output_sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "analysis.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
