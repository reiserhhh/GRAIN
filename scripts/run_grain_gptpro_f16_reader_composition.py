#!/usr/bin/env python3
"""Replay fixed C2 histories with separate content/eligibility readers.

Post-result follow-up on existing F08 worlds, not replacement of F16's Gaussian
T1/T2/T3 targets. No source scorer fit, neural calls, prompts or model selection.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from grain_research.measurement_discovery.estimators import ordered_rows

ARMS = (
    "nli",
    "nli_pole_grain_qualification",
    "nli_pole_lexical_qualification",
    "grain",
    "oracle",
)
CANDIDATES = ("b1", "b2", "m")


def compose_readings(material_id, readers, pole_truth, qualification_truth):
    """Join exact materials; truth enters only the explicitly labeled oracle."""
    n, g, l = [readers[k].loc[material_id] for k in ("nli", "grain", "lexical")]
    return {
        "nli": (float(n.pole_score), float(n.qualification_score)),
        "nli_pole_grain_qualification": (
            float(n.pole_score),
            float(g.qualification_score),
        ),
        "nli_pole_lexical_qualification": (
            float(n.pole_score),
            float(l.qualification_score),
        ),
        "grain": (float(g.pole_score), float(g.qualification_score)),
        "oracle": (float(pole_truth), float(qualification_truth)),
    }


def histories(path):
    """Yield original mixed application histories in unaltered archive order."""
    with gzip.open(path, "rt") as stream:
        selected = (
            r
            for r in map(json.loads, stream)
            if r["role"] == "application" and r["panel"] == "mixed"
        )
        for _, group in itertools.groupby(
            selected, key=lambda r: (r["event_id"].rsplit("_", 1)[0], r["author_id"])
        ):
            rows = list(group)
            if len(rows) != 48 or [r["t"] for r in rows] != list(range(48)):
                raise ValueError("original mixed history missing or reordered")
            yield rows


def main():
    """Compute all archived mixed application panels, paired by identical author."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    base = ROOT / "results/grain_gptpro_foundation_v3"
    original = base / "F08/nli_downstream_full_r1"
    paths = {
        "nli": base / "F08/nli_reader_r1/reader_scores.csv",
        "lexical": base / "F08/reader_recovery_r1/reader_scores.csv",
        "grain": base / "F11/grain_comment_reader_r1/reader_scores.csv",
    }
    source_paths = sorted(original.glob("source_*.pkl"))
    inputs = [
        *paths.values(),
        original / "observations.jsonl.gz",
        original / "metrics_by_author.csv",
        *source_paths,
    ]
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f08_estimators.py",
        ROOT / "grain_research/gptpro_foundation_v3/f08_filter.py",
        ROOT / "grain_research/gptpro_foundation_v3/f02.py",
    ]
    contract = {
        "followup": "POST_RESULT_FIXED_ARCHITECTURE_NOT_INDEPENDENT_REPLICATION",
        "inputs": {str(k): sha256_file(k) for k in inputs},
        "code": {str(k): sha256_file(k) for k in code},
        "arms": ARMS,
        "estimators": CANDIDATES,
        "worlds": ["c2_xor", "c2_iid"],
        "seeds": [20260915, 20260916, 20260917],
        "application_authors_per_world_seed": 200,
        "source_scorer_refit": False,
        "history": "exact old48positions,mixed application panel,next response scoredbefore observation from t12",
        "no_gaussian_substitution": True,
        "no_neural_calls": True,
        "interval": "paired author bootstrap withinseed2000,seed160823;finite reused material/world scope",
        "mainline_relation": "first target-compatible text-reader composition on fixed binarylaw;does not complete originalT1/T2/T3text mapping",
    }
    (a.output / "protocol_before_run.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    readers = {k: pd.read_csv(v).set_index("material_id") for k, v in paths.items()}
    identity = [
        "role",
        "family",
        "pole",
        "subject_status",
        "qualification",
        "variant",
        "text",
    ]
    if any(not readers["nli"][identity].equals(f[identity]) for f in readers.values()):
        raise ValueError("same text bank and roles required")
    models = {}
    for world in contract["worlds"]:
        for seed in contract["seeds"]:
            path = original / f"source_{world}_{seed}.pkl"
            with path.open("rb") as f:
                fitted = pickle.load(f)
            records, _ = generate_world(
                world,
                seed,
                counts={"source": 300, "validation": 100, "application": 200},
            )
            source = [r for r in records if r["role"] == "source"]
            initial = np.ones((2, 2))
            for group in ordered_rows(source).values():
                initial[group[0]["observation"], group[1]["observation"]] += 1
            initial /= initial.sum()
            source_mean = float(np.mean([r["observation"] for r in source]))
            models[world, seed] = (fitted, initial, source_mean)
            del records, source
    metrics, panel_counts, total_steps = [], {}, 0
    with gzip.open(a.output / "predictions.jsonl.gz", "wt") as output:
        for panel in histories(original / "observations.jsonl.gz"):
            match = re.match(r"^(c2_(?:xor|iid))_(\d+)_", panel[0]["event_id"])
            if match is None:
                raise ValueError("unknown archived world/seed")
            world, seed = match[1], int(match[2])
            key = (world, seed)
            panel_counts[key] = panel_counts.get(key, 0) + 1
            states = {
                arm: QualifiedHistory(*models[key], mode="filtered") for arm in ARMS
            }
            losses = {arm: {c: [] for c in CANDIDATES} for arm in ARMS}
            for row in panel:
                material_id = row["text_ref"].split("#")[1]
                readings = compose_readings(
                    material_id, readers, row["material_pole"], row["qualification"]
                )
                if not np.allclose(
                    readings["nli"],
                    (row["pole_score"], row["qualification_score"]),
                    atol=1e-15,
                    rtol=0,
                ):
                    raise ValueError("old NLI readout mismatch")
                for arm, state in states.items():
                    if row["t"] >= 12:
                        prediction = state.predict()
                        y = row["observation"]
                        for candidate in CANDIDATES:
                            probability = prediction["probabilities"][candidate]
                            brier = (probability - y) ** 2
                            logloss = -(
                                y * np.log(probability)
                                + (1 - y) * np.log1p(-probability)
                            )
                            losses[arm][candidate].append((brier, logloss))
                            output.write(
                                json.dumps(
                                    {
                                        "world": world,
                                        "seed": seed,
                                        "author_id": row["author_id"],
                                        "event_id": row["event_id"],
                                        "t": row["t"],
                                        "arm": arm,
                                        "candidate": candidate,
                                        "p": probability,
                                        "y": y,
                                        "brier": brier,
                                        "log_loss": logloss,
                                    }
                                )
                                + "\n"
                            )
                    state.observe(*readings[arm])
                    total_steps += 1
            for arm in ARMS:
                for candidate in CANDIDATES:
                    average = np.mean(losses[arm][candidate], axis=0)
                    metrics.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": panel[0]["author_id"],
                            "arm": arm,
                            "candidate": candidate,
                            "brier": average[0],
                            "log_loss": average[1],
                        }
                    )
            if sum(panel_counts.values()) % 200 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "paired_target_score",
                            "authors": sum(panel_counts.values()),
                        }
                    ),
                    flush=True,
                )
    if set(panel_counts) != set(models) or any(n != 200 for n in panel_counts.values()):
        raise ValueError("incomplete archived application population")
    frame = pd.DataFrame(metrics)
    frame.to_csv(a.output / "metrics_by_author.csv", index=False)
    old = pd.read_csv(original / "metrics_by_author.csv")
    keys = ["world", "seed", "author_id", "candidate"]
    replay_errors = {}
    for arm, channel in (("nli", "soft"), ("oracle", "oracle")):
        prior = old.loc[
            old.role.eq("application")
            & old.panel.eq("mixed")
            & old.channel.eq(channel)
            & old.candidate.isin(CANDIDATES)
        ]
        joined = frame.loc[frame.arm.eq(arm)].merge(
            prior, on=keys, validate="one_to_one", suffixes=("_new", "_old")
        )
        if len(joined) != 3600:
            raise ValueError("baseline replay population mismatch")
        error = float(
            np.max(
                abs(
                    joined[["brier_new", "log_loss_new"]].to_numpy()
                    - joined[["brier_old", "log_loss_old"]].to_numpy()
                )
            )
        )
        if error > 1e-10:
            raise ValueError("frozen baseline changed")
        replay_errors[arm] = error
    means = (
        frame.groupby(["world", "arm", "candidate"])[["brier", "log_loss"]]
        .mean()
        .reset_index()
    )
    means.to_csv(a.output / "application_summary.csv", index=False)
    contrasts = []
    for world in contract["worlds"]:
        for candidate in CANDIDATES:
            for arm in ARMS[1:]:
                joined = frame.loc[
                    frame.world.eq(world)
                    & frame.candidate.eq(candidate)
                    & frame.arm.eq(arm)
                ].merge(
                    frame.loc[
                        frame.world.eq(world)
                        & frame.candidate.eq(candidate)
                        & frame.arm.eq("nli")
                    ],
                    on=keys,
                    validate="one_to_one",
                    suffixes=("_new", "_old"),
                )
                for metric in ("brier", "log_loss"):
                    differences = joined[f"{metric}_new"] - joined[f"{metric}_old"]
                    rng = np.random.default_rng(160823)
                    draws = np.zeros(2000)
                    for seed in contract["seeds"]:
                        d = differences.loc[joined.seed.eq(seed)].to_numpy()
                        draws += (
                            d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1) / 3
                        )
                    contrasts.append(
                        {
                            "world": world,
                            "candidate": candidate,
                            "arm": arm,
                            "metric": metric,
                            "authors": len(joined),
                            "delta_vs_nli": float(differences.mean()),
                            "ci_low": float(np.quantile(draws, 0.025)),
                            "ci_high": float(np.quantile(draws, 0.975)),
                        }
                    )
    pd.DataFrame(contrasts).to_csv(a.output / "paired_contrasts.csv", index=False)
    summary = {
        "status": "FIXED_BINARY_TEXT_INTERFACE_COMPOSITION_COMPLETE",
        "application_authors": sum(panel_counts.values()),
        "history_updates": total_steps,
        "prediction_rows": sum(panel_counts.values())
        * 36
        * len(ARMS)
        * len(CANDIDATES),
        "frozen_baseline_replay_max_error": replay_errors,
        "new_source_fits": 0,
        "new_neural_calls": 0,
        "scope": contract["followup"],
        "F16_parent_complete": False,
        "next": "retain content and eligibility as separate observed channels;develop explicit target-compatible Gaussian text observation law rather than relabeling these binary scores",
        "seconds": time.monotonic() - began,
    }
    (a.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (a.output / "output_manifest.json").write_text(
        json.dumps(
            {p.name: sha256_file(p) for p in sorted(a.output.iterdir()) if p.is_file()},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
