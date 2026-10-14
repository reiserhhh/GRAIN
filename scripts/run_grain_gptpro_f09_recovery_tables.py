#!/usr/bin/env python3
"""F09-05 recovery tables from frozen author packets + exact extra-B positive control.

Main panel: family and beta recovery separated (marginal posterior argmax),
family confusion, high-confidence wrong picks (winning mass > .8 and wrong),
per-target Brier/log loss; author bootstrap (seeds stratified, 2000, seed
20260915). Positive control: exact Bayes-optimal process-class accuracy for the
F15-02 process experiments (corners / + intermediate levels / + observed B).
No new simulation of authors, no refit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f15_theory as th

ROOT = Path(__file__).resolve().parents[1]
PACKETS = (
    ROOT
    / "results/grain_gptpro_foundation_v3/F09/panels_full_r1/author_packets.jsonl.gz"
)
FAMILIES = ("M1", "M2", "M3")
BETAS = (0.5, 1.0, 1.5)


def boot_ci(values, seeds, reps=2000, seed=20260915):
    """Seed-stratified author bootstrap of a mean."""
    rng = np.random.default_rng(seed)
    parts = [values[seeds == s] for s in np.unique(seeds)]
    rep = np.zeros(reps)
    for p in parts:
        rep += p[rng.integers(len(p), size=(reps, len(p)))].mean(1) / len(parts)
    return float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))


def main():
    """Write recovery tables and the exact positive control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with gzip.open(PACKETS, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["panel"] != "main" or d["role"] != "application":
                continue
            score = next(s for s in d["scores"] if s["target"] == "even12_to_odd12")
            post = np.asarray(score["posterior"]).reshape(
                3, 3
            )  # family x beta (atom order k*3+b)
            fam_post, beta_post = post.sum(1), post.sum(0)
            fam_tied = np.flatnonzero(np.abs(fam_post - fam_post.max()) <= 1e-12)
            beta_tied = np.flatnonzero(np.abs(beta_post - beta_post.max()) <= 1e-12)
            fam_share = {FAMILIES[k]: 1 / len(fam_tied) for k in fam_tied}
            beta_share = {BETAS[k]: 1 / len(beta_tied) for k in beta_tied}
            rows.append(
                {
                    "family_credit": fam_share.get(d["family"], 0.0),
                    "beta_credit": beta_share.get(d["beta"], 0.0),
                    "family_tie": len(fam_tied) > 1,
                    "beta_tie": len(beta_tied) > 1,
                    **{f"fam_share_{f}": fam_share.get(f, 0.0) for f in FAMILIES},
                    **{f"beta_share_{b}": beta_share.get(b, 0.0) for b in BETAS},
                    "seed": d["seed"],
                    "author_id": d["author_id"],
                    "outside_M4": d["outside"],
                    "family_true": d["family"],
                    "beta_true": d["beta"],
                    "family_hat": FAMILIES[int(np.argmax(fam_post))],
                    "beta_hat": BETAS[int(np.argmax(beta_post))],
                    "family_mass_max": float(fam_post.max()),
                    "joint_mass_max": float(post.max()),
                    "joint_correct": bool(score["class_correct"]),
                    "brier": score["brier"],
                    "log_loss": score["log_loss"],
                    "rejected_out_of_family": bool(d["out_of_family"]["rejected"]),
                }
            )
    df = pd.DataFrame(rows)
    inf = df[~df.outside_M4].copy()
    inf["family_correct"] = inf.family_credit
    inf["beta_correct"] = inf.beta_credit
    inf["high_conf_wrong_family"] = (inf.family_mass_max > 0.8) & (
        inf.family_credit == 0
    )
    seeds = inf.seed.to_numpy()
    table = []
    for name, col in (
        ("family_accuracy", "family_correct"),
        ("beta_accuracy", "beta_correct"),
        ("joint_accuracy", "joint_correct"),
        ("high_conf_wrong_family_rate", "high_conf_wrong_family"),
        ("exact_family_tie_rate", "family_tie"),
        ("exact_beta_tie_rate", "beta_tie"),
        ("in_family_false_rejection", "rejected_out_of_family"),
    ):
        v = inf[col].to_numpy(float)
        lo, hi = boot_ci(v, seeds)
        table.append(
            {
                "metric": name,
                "value": float(v.mean()),
                "lower": lo,
                "upper": hi,
                "n": len(v),
                "chance": {
                    "family_accuracy": 1 / 3,
                    "beta_accuracy": 1 / 3,
                    "joint_accuracy": 1 / 9,
                }.get(name, ""),
            }
        )
    m4 = df[df.outside_M4]
    v = m4.rejected_out_of_family.to_numpy(float)
    lo, hi = boot_ci(v, m4.seed.to_numpy())
    table.append(
        {
            "metric": "M4_detection_rate",
            "value": float(v.mean()),
            "lower": lo,
            "upper": hi,
            "n": len(v),
            "chance": 0.05,
        }
    )
    pd.DataFrame(table).to_csv(args.output / "recovery_table.csv", index=False)
    confusion = inf.groupby("family_true")[[f"fam_share_{f}" for f in FAMILIES]].mean()
    confusion.to_csv(args.output / "family_confusion.csv")
    beta_conf = inf.groupby("beta_true")[[f"beta_share_{b}" for b in BETAS]].mean()
    beta_conf.to_csv(args.output / "beta_confusion.csv")
    control = []
    for reps in (1, 2, 3):
        e1 = th.process_experiment((1.0, -1.0), reps)
        e2 = th.process_experiment((1.0, -1.0, 1 / 3, -1 / 3), reps)
        e3 = th.process_experiment((1.0, -1.0), reps, observe_b=True)
        for name, e in (
            ("corners_only", e1),
            ("plus_intermediate_levels", e2),
            ("plus_observed_belief", e3),
        ):
            proc = np.array([a[0] for a in e["atoms"]])
            classes = np.array(
                [e["channel"][proc == c].mean(0) for c in ("belief", "direct")]
            )
            bayes = float(np.sum(classes.max(0)) / 2)
            control.append(
                {
                    "reps_per_level": reps,
                    "design": name,
                    "outcomes": e["channel"].shape[1],
                    "bayes_process_accuracy": bayes,
                }
            )
    pd.DataFrame(control).to_csv(
        args.output / "extra_b_positive_control.csv", index=False
    )
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            {
                "packets_sha256": hashlib.sha256(PACKETS.read_bytes()).hexdigest(),
                "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=1,
        )
        + "\n"
    )
    print(pd.DataFrame(table).to_string())
    print(confusion.round(3).to_string())
    print(beta_conf.round(3).to_string())
    print(pd.DataFrame(control).to_string())


if __name__ == "__main__":
    main()
