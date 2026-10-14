#!/usr/bin/env python3
"""Compare frozen online heads with fixed-source joint-history refit heads."""

import argparse
import gzip
import hashlib
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f02 import c2_next_probability, generate_world
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_history_sampling import sample_histories
from grain_research.gptpro_foundation_v3.f13_path_heads import (
    path_head_scores,
    summarize_path_scores,
)
from scripts.audit_grain_gptpro_f08_states import source_replay


def main():
    """Use 200 dependent-history draws; checkpoint per author-panel group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config = json.loads((args.input / "protocol.json").read_text())
    models = {}
    files = [
        Path(__file__),
        Path("grain_research/gptpro_foundation_v3/f13_path_heads.py"),
        Path("grain_research/gptpro_foundation_v3/f13_history_sampling.py"),
        Path("grain_research/gptpro_foundation_v3/f08_estimators.py"),
        args.input / "protocol.json",
        args.input / "observations.jsonl.gz",
    ]
    for world in ("c2_xor", "c2_iid"):
        for seed in config["seeds"]:
            path = args.input / f"source_{world}_{seed}.pkl"
            files.append(path)
            with path.open("rb") as stream:
                model = pickle.load(stream)
            source, _ = generate_world(world, seed, counts=config["counts"])
            initial, mean = source_replay(
                [r for r in source if r["role"] == "source"], model
            )
            models[world, seed] = model, initial, mean
    protocol = {
        "draws": 200,
        "smoke": args.smoke,
        "scope": "fixed-source joint-history imputation; not finite-event parameter posterior or source bootstrap",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ],
    }
    protocol_path = args.output / "protocol_before_run.json"
    if args.resume:
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("resume code/input mismatch")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    started, done, planned, scored = time.monotonic(), 0, 0, 0
    with gzip.open(args.input / "observations.jsonl.gz", "rt") as stream:
        for _, group in itertools.groupby(
            (json.loads(line) for line in stream),
            key=lambda r: (r["author_id"], r["panel"]),
        ):
            observations = list(group)
            if observations[0]["role"] != "application":
                continue
            index = planned
            if args.smoke and index >= 2:
                break
            planned += 1
            path = args.output / f"group_{index:04d}.json"
            if path.exists():
                old = json.loads(path.read_text())
                assert (
                    old["author_id"] == observations[0]["author_id"]
                    and old["panel"] == observations[0]["panel"]
                )
                done += 1
                scored += len(old["rows"])
                continue
            if time.monotonic() - started >= args.max_seconds:
                continue
            parts = observations[0]["event_id"].split("_")
            world, seed = "_".join(parts[:2]), int(parts[2])
            model, initial, mean = models[world, seed]
            complete = observations[0]["panel"] == "complete"
            histories = {
                c: QualifiedHistory(
                    model, initial, mean, mode="independent" if complete else "filtered"
                )
                for c in ("oracle", "soft", "hard")
            }
            readings = {c: ([], []) for c in histories}
            rng_recipe = [seed, 13, 8, index, 3]
            rng = np.random.default_rng(np.random.SeedSequence(rng_recipe))
            hidden, results = [], []
            for row in observations:
                for channel, history in histories.items():
                    poles, qualifications = readings[channel]
                    if row["t"] >= 12:
                        paths = sample_histories(
                            poles,
                            qualifications,
                            model["b1"]["source_p"],
                            initial,
                            200,
                            rng,
                            complete=complete,
                        )
                        scores = path_head_scores(paths, qualifications, model)
                        old = history.predict()["probabilities"]
                        truth = c2_next_probability(
                            hidden[-2],
                            hidden[-1],
                            1,
                            "xor" if world == "c2_xor" else "iid",
                        )
                        for head, raw in scores.items():
                            result = summarize_path_scores(raw)
                            results.append(
                                {
                                    "world": world,
                                    "seed": seed,
                                    "t": row["t"],
                                    "channel": channel,
                                    "head": head,
                                    "online": old[head],
                                    "true_next_p": truth,
                                    "outcome": row["observation"],
                                    **result,
                                }
                            )
                    p, q = (
                        (row["material_pole"], row["qualification"])
                        if channel == "oracle"
                        else (row["pole_score"], row["qualification_score"])
                    )
                    if channel == "hard":
                        p, q = float(p >= 0.5), float(q >= 0.5)
                    q = 1 if complete else q
                    history.observe(p, q)
                    poles.append(p)
                    qualifications.append(q)
                hidden.append(row["observation"])
            payload = {
                "author_id": observations[0]["author_id"],
                "panel": observations[0]["panel"],
                "rng_recipe": rng_recipe,
                "rows": results,
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload) + "\n")
            temporary.replace(path)
            done += 1
            scored += len(results)
            if done % 10 == 0:
                checkpoint = {
                    "completed": done,
                    "rows": scored,
                    "elapsed": time.monotonic() - started,
                }
                (args.output / "checkpoint.json").write_text(
                    json.dumps(checkpoint) + "\n"
                )
                print(json.dumps(checkpoint), flush=True)
    summary = {
        "status": "COMPLETE" if done == planned else "TIME_BUDGET_CHECKPOINT",
        "completed": done,
        "planned": planned,
        "rows": scored,
        "elapsed": time.monotonic() - started,
        "scope": protocol["scope"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
