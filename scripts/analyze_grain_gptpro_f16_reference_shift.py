#!/usr/bin/env python3
"""Retained computation: how the choice of source score-reference moves text positions.

The T1/T2/T3 estimands are fixed. What changes with a reader is the scoring
function (the observation law used to update the posterior) and therefore the
distribution of source-author scores that defines a reference. This analysis
keeps raw-unit application scores fixed (the canonical bundle outputs) and
compares their positions against three source score-references:

H historical: old reader, old 96-text bank, old rendering, old channel law.
C control:    old reader cache and old source-calibrated law on the NEW bank,
              same new rendering stream, same mixture estimator family.
F fresh:      changed reader + pilot mixture law (the published reference).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as dyn_text
from grain_research.gptpro_foundation_v3 import f16_t1_text as t1_text
from grain_research.gptpro_foundation_v3 import f16_text_transport as text_law
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file

ROOT = Path(__file__).resolve().parents[1]
F16 = ROOT / "results/grain_gptpro_foundation_v3/F16"
RUN = F16 / "serialize_full_r1"
TEXT = F16 / "text_transport_full_r1"
SEEDS = (20260915, 20260916, 20260917)


def control_engine(bundle, bank, laws, seed):
    """Old reader cache + old source-calibrated mixture law on the new bank."""
    engine = copy.deepcopy(bundle["engine"])
    engine["materials"] = json.loads(
        json.dumps(text_law.observation_cache(bank, "old"))
    )
    draws = text_law.channels(
        laws["old"], np.random.SeedSequence([seed, 170, 0]), 200, "mixture"
    )
    engine["channel_draws"] = {k: draws.tolist() for k in ("hybrid_joint", "nli_joint")}
    return engine


def main():
    """Compute references H/C/F and application position shifts with paired CIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = pd.read_csv(TEXT / "reader_scores.csv").to_dict("records")
    laws = json.loads((TEXT / "channels.json").read_text())
    refs, rows = [], []
    for seed in SEEDS:
        for route, targets in (("text_t1", ("T1",)), ("text_dynamic", ("T2", "T3"))):
            first = json.loads(
                (
                    RUN / "bundles" / f"{route}_{seed}_mixture_{targets[0]}.json"
                ).read_text()
            )
            inputs = [
                json.loads(x)
                for x in (
                    RUN / "fresh_reference" / f"{route}_{seed}_source_inputs.jsonl"
                )
                .read_text()
                .splitlines()
            ]
            engine = control_engine(first, bank, laws, seed)
            if route == "text_t1":
                engine.update(reference_mu=0.0, reference_sigma=1.0)
                scores = {
                    "T1": np.array(
                        [s["score"] for s in t1_text.cold_scores(inputs, engine)]
                    )
                }
            else:
                engine["reference"] = {
                    t: {"mu": 0.0, "sigma": 1.0} for t in ("T2", "T3")
                }
                out = dyn_text.cold_scores(inputs, engine)
                scores = {
                    t: np.array([s["targets"][t]["score"] for s in out])
                    for t in ("T2", "T3")
                }
            for target in targets:
                bundle = json.loads(
                    (
                        RUN / "bundles" / f"{route}_{seed}_mixture_{target}.json"
                    ).read_text()
                )
                hist = bundle["reference_policy"]["historical_positions"][0]
                fresh = bundle["reference_policy"]
                control = {
                    "mu": float(scores[target].mean()),
                    "sigma": float(scores[target].std(ddof=1)),
                }
                refs.append(
                    {
                        "seed": seed,
                        "target": target,
                        "H_mu": hist["mu"],
                        "H_sigma": hist["sigma"],
                        "C_mu": control["mu"],
                        "C_sigma": control["sigma"],
                        "F_mu": fresh["mu"],
                        "F_sigma": fresh["sigma"],
                        "source_authors": len(inputs),
                    }
                )
                cold = [
                    json.loads(x)
                    for x in (
                        RUN / "cold" / f"{route}_{seed}_mixture_{target}.scores.jsonl"
                    )
                    .read_text()
                    .splitlines()
                ]
                for r in cold:
                    s = r["score"]
                    rows.append(
                        {
                            "seed": seed,
                            "target": target,
                            "author_id": r["author_id"],
                            "raw_score": s,
                            "pos_H": (s - hist["mu"]) / hist["sigma"],
                            "pos_C": (s - control["mu"]) / control["sigma"],
                            "pos_F": (s - fresh["mu"]) / fresh["sigma"],
                        }
                    )
            print(json.dumps({"seed": seed, "route": route}), flush=True)
    refs = pd.DataFrame(refs)
    frame = pd.DataFrame(rows)
    frame["F_minus_H"] = (frame.pos_F - frame.pos_H).abs()
    frame["F_minus_C"] = (frame.pos_F - frame.pos_C).abs()
    frame["C_minus_H"] = (frame.pos_C - frame.pos_H).abs()
    rng = np.random.default_rng(20260915)
    summary = []
    for target, g in frame.groupby("target"):
        for col, meaning in (
            ("F_minus_H", "total: inherited historical vs fresh"),
            ("C_minus_H", "bank/rendering/old-law re-estimation"),
            ("F_minus_C", "reader change + its pilot-calibrated law"),
        ):
            rep = np.zeros(2000)
            for seed in SEEDS:
                d = g[g.seed == seed][col].to_numpy()
                rep += d[rng.integers(len(d), size=(2000, len(d)))].mean(1) / len(SEEDS)
            summary.append(
                {
                    "target": target,
                    "contrast": col,
                    "meaning": meaning,
                    "mean_abs_position_shift": float(g[col].mean()),
                    "lower": float(np.quantile(rep, 0.025)),
                    "upper": float(np.quantile(rep, 0.975)),
                    "max": float(g[col].max()),
                    "authors": len(g),
                }
            )
    refs.to_csv(args.output / "source_references.csv", index=False)
    frame.to_csv(args.output / "application_positions.csv", index=False)
    pd.DataFrame(summary).to_csv(
        args.output / "position_shift_summary.csv", index=False
    )
    (args.output / "units.md").write_text(
        "Positions are z-scores against each reference (each in units of its own reference SD). Shifts are |pos_A - pos_B| for the same raw application score. Raw application scores are the canonical bundle outputs and do not change; raw-unit reference means/SDs are in source_references.csv. Estimands T1/T2/T3 are unchanged; only the scoring function (reader observation law) and the source score-reference differ. Paired author bootstrap within seed, 2000 reps, seed 20260915, averaged over seeds.\n"
    )
    manifest = {
        p.name: sha256_file(p)
        for p in sorted(args.output.iterdir())
        if p.is_file() and not p.name.startswith(".")
    }
    (args.output / "output_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(pd.DataFrame(summary).to_string())
    print(refs.to_string())


if __name__ == "__main__":
    main()
