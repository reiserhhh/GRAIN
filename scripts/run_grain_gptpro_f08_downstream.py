#!/usr/bin/env python3
"""Actual reader-to-estimator competition on paired frozen C2 simulations."""

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import fit_observed_source
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.measurement_discovery.estimators import ordered_rows


def write_table(path, rows):
    """Write derived tabular results, not a placeholder acceptance marker."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Fit frozen source estimators, then update each application author causally."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reader_rows = list(csv.DictReader((args.reader / "reader_scores.csv").open()))
    cache = {
        (r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r
        for r in reader_rows
    }
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    metrics = []
    evidence_cards = []
    total_steps = 0
    common_records = 0
    frozen = {
        "counts": counts,
        "seeds": seeds,
        "prefix": 12,
        "events": 48,
        "source_fit": "unchanged oracle-observation B1/B2/M;all channels same source scorer",
        "reader_scores_sha256": hashlib.sha256(
            (args.reader / "reader_scores.csv").read_bytes()
        ).hexdigest(),
        "panels": ["complete", "mixed"],
        "channels": ["oracle", "soft", "hard"],
        "semantics": "unqualified report pole independent of hidden author observation;time positions retained",
    }
    (args.output / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    with (
        gzip.open(args.output / "predictions.jsonl.gz", "wt") as predictions,
        gzip.open(args.output / "observations.jsonl.gz", "wt") as observations,
    ):
        for world_index, world in enumerate(("c2_xor", "c2_iid")):
            for seed in seeds:
                records, _ = generate_world(world, seed, counts=counts)
                source = [r for r in records if r["role"] == "source"]
                fitted = fit_observed_source(source, world)
                with (args.output / f"source_{world}_{seed}.pkl").open("wb") as stream:
                    pickle.dump(fitted, stream)
                initial = np.ones((2, 2))
                for group in ordered_rows(source).values():
                    initial[group[0]["observation"], group[1]["observation"]] += 1
                initial /= initial.sum()
                source_mean = float(np.mean([r["observation"] for r in source]))
                groups = ordered_rows([r for r in records if r["role"] != "source"])
                for author_index, (author, rows) in enumerate(groups.items()):
                    role = rows[0]["role"]
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed, 8, world_index, author_index])
                    )
                    statuses = rng.integers(0, 4, 48)
                    variants = rng.integers(0, 4, 48)
                    ancillary = rng.integers(0, 2, 48)
                    for panel in ("complete", "mixed"):
                        histories = {
                            c: QualifiedHistory(
                                fitted,
                                initial,
                                source_mean,
                                mode="independent"
                                if panel == "complete"
                                else "filtered",
                            )
                            for c in ("oracle", "soft", "hard")
                        }
                        losses = {
                            c: {
                                k: []
                                for k in (
                                    "b1",
                                    "b2",
                                    "m",
                                    "source_mean",
                                    "personal_mean",
                                )
                            }
                            for c in histories
                        }
                        for t, row in enumerate(rows):
                            status = (
                                ("actual", "quoted", "hypothetical", "negated")[
                                    statuses[t]
                                ]
                                if panel == "mixed"
                                else ("actual" if statuses[t] % 2 == 0 else "negated")
                            )
                            q = int(status in ("actual", "negated"))
                            material_pole = (
                                row["observation"] if q else int(ancillary[t])
                            )
                            material = cache[
                                (role, material_pole, status, int(variants[t]))
                            ]
                            p = float(material["pole_score"])
                            quality = float(material["qualification_score"])
                            readings = {
                                "oracle": (float(material_pole), float(q)),
                                "soft": (p, quality),
                                "hard": (float(p >= 0.5), float(quality >= 0.5)),
                            }
                            if panel == "complete":
                                readings = {
                                    c: (p_read, 1.0)
                                    for c, (p_read, _) in readings.items()
                                }
                            observations.write(
                                json.dumps(
                                    {
                                        **row,
                                        "panel": panel,
                                        "text_ref": f"reader_scores.csv#{material['material_id']}",
                                        "material_pole": material_pole,
                                        "qualification": q,
                                        "pole_score": p,
                                        "qualification_score": quality,
                                    }
                                )
                                + "\n"
                            )
                            common_records += 1
                            for channel, history in histories.items():
                                before = history.predict() if t >= 2 else None
                                if t >= 12:
                                    for candidate, probability in before[
                                        "probabilities"
                                    ].items():
                                        y = row["observation"]
                                        loss = (probability - y) ** 2
                                        logloss = -(
                                            y * np.log(probability)
                                            + (1 - y) * np.log1p(-probability)
                                        )
                                        losses[channel][candidate].append(
                                            (loss, logloss)
                                        )
                                        predictions.write(
                                            json.dumps(
                                                {
                                                    "world": world,
                                                    "seed": seed,
                                                    "author_id": author,
                                                    "role": role,
                                                    "panel": panel,
                                                    "channel": channel,
                                                    "candidate": candidate,
                                                    "event_id": row["event_id"],
                                                    "t": t,
                                                    "y": y,
                                                    "p": probability,
                                                    "brier": loss,
                                                    "log_loss": logloss,
                                                    "evidence_mass": before[
                                                        "evidence_mass"
                                                    ],
                                                    "status": before["status"],
                                                }
                                            )
                                            + "\n"
                                        )
                                history.observe(*readings[channel])
                                total_steps += 1
                                if (
                                    world_index == 0
                                    and seed == seeds[0]
                                    and author_index == 0
                                    and panel == "mixed"
                                    and channel == "soft"
                                    and t >= 2
                                    and sum(
                                        c["subject_status"] == status
                                        for c in evidence_cards
                                    )
                                    < 2
                                ):
                                    evidence_cards.append(
                                        {
                                            "event_id": row["event_id"],
                                            "text_ref": f"reader_scores.csv#{material['material_id']}",
                                            "text": material["text"],
                                            "subject_status": status,
                                            "raw_pole": float(material["pole_raw"]),
                                            "qualification": quality,
                                            "before": before,
                                            "after": history.predict(),
                                            "target": "next latent binary response;before/after concern different time targets",
                                        }
                                    )
                        for channel, candidates in losses.items():
                            for candidate, values in candidates.items():
                                average = np.mean(values, axis=0)
                                metrics.append(
                                    {
                                        "world": world,
                                        "seed": seed,
                                        "author_id": author,
                                        "role": role,
                                        "panel": panel,
                                        "channel": channel,
                                        "candidate": candidate,
                                        "brier": float(average[0]),
                                        "log_loss": float(average[1]),
                                        "events": len(values),
                                    }
                                )
    write_table(args.output / "metrics_by_author.csv", metrics)
    summaries = []
    for world in ("c2_xor", "c2_iid"):
        for panel in ("complete", "mixed"):
            for channel in ("oracle", "soft", "hard"):
                for candidate in ("b1", "b2", "m", "source_mean", "personal_mean"):
                    subset = [
                        r
                        for r in metrics
                        if r["role"] == "application"
                        and (r["world"], r["panel"], r["channel"], r["candidate"])
                        == (world, panel, channel, candidate)
                    ]
                    summaries.append(
                        {
                            "world": world,
                            "panel": panel,
                            "channel": channel,
                            "candidate": candidate,
                            "authors": len(subset),
                            "brier": float(np.mean([r["brier"] for r in subset])),
                            "log_loss": float(np.mean([r["log_loss"] for r in subset])),
                        }
                    )
    write_table(args.output / "application_summary.csv", summaries)
    (args.output / "evidence_cards.json").write_text(
        json.dumps(evidence_cards, indent=2) + "\n"
    )
    result = {
        "status": "COMPUTED_AUDIT_PENDING",
        "updated_steps": total_steps,
        "observation_rows": common_records,
        "author_metric_rows": len(metrics),
        "evidence_cards": len(evidence_cards),
        "source_model_refit_per_world_seed": True,
        "reader_fit_reused": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
