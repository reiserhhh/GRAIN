#!/usr/bin/env python3
"""F08 end-to-end: source B1/B2/M fitted on reader-read source histories.

The downstream packet fitted sources on oracle source observations. Here source
authors' events are rendered on source-role materials (declared stream
SeedSequence([seed, 8, 100+world, author])) and read by the same reader
(hard thresholds). Mixed-panel events read as ineligible are missing; fitting is
gap-aware (only triples inside contiguous read runs). Application rendering uses
the downstream streams, so the oracle-fitted/hard arm replays the archived
downstream hard-channel losses (built-in check).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import (
    fit_observed_source,
)
from grain_research.gptpro_foundation_v3.f08_estimators import (
    QualifiedHistory,
)
from grain_research.measurement_discovery.estimators import (
    fit_b1,
    fit_b2,
    ordered_rows,
)

F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"
READERS = {"lexical": F08 / "reader_recovery_r1", "nli": F08 / "nli_reader_r1"}
SEEDS = (20260915, 20260916, 20260917)
COUNTS = {"source": 300, "validation": 100, "application": 200}
STATUSES = ("actual", "quoted", "hypothetical", "negated")


def render(rng_seq, panel, observation_values, cache, role):
    """Readings for one author's events (hard pole, hard qualification, oracle)."""
    rng = np.random.default_rng(rng_seq)
    statuses = rng.integers(0, 4, 48)
    variants = rng.integers(0, 4, 48)
    ancillary = rng.integers(0, 2, 48)
    out = []
    for t, y in enumerate(observation_values):
        status = (
            STATUSES[statuses[t]]
            if panel == "mixed"
            else ("actual" if statuses[t] % 2 == 0 else "negated")
        )
        q = int(status in ("actual", "negated"))
        pole = y if q else int(ancillary[t])
        m = cache[(role, pole, status, int(variants[t]))]
        p, quality = float(m["pole_score"]), float(m["qualification_score"])
        hard = (float(p >= 0.5), 1.0 if panel == "complete" else float(quality >= 0.5))
        oracle = (float(pole), 1.0 if panel == "complete" else float(q))
        out.append({"hard": hard, "oracle": oracle})
    return out


def runs(rows_read):
    """Split an author's read history into contiguous runs of read-eligible events."""
    segments, current = [], []
    for row in rows_read:
        if row is None:
            if len(current):
                segments.append(current)
            current = []
        else:
            current.append(row)
    if current:
        segments.append(current)
    return segments


def fit_read_source(read_histories, world):
    """Gap-aware B1/B2/M: triples only inside contiguous read runs."""
    pseudo = {}
    tables = []
    for author, rows in read_histories.items():
        counts = np.zeros((4, 2))
        for k, seg in enumerate(runs(rows)):
            pseudo[f"{author}#{k}"] = seg
            values = [int(r["observation"]) for r in seg]
            for i in range(2, len(values)):
                counts[2 * values[i - 2] + values[i - 1], values[i]] += 1
        tables.append((counts[:, 1] + 1) / (counts.sum(axis=1) + 2))
    table = np.asarray(tables)
    mean = table.mean(axis=0)
    _, singular, right = np.linalg.svd(table - mean, full_matrices=False)
    return {
        "world": world,
        "b1": fit_b1(pseudo),
        "b2": fit_b2(pseudo),
        "m": {"mean": mean, "right": right[:2], "singular_values": singular, "rank": 2},
    }


def score_application(fitted, initial, source_mean, panel, readings, ys, channel):
    """Sequential predictions t>=12 for B1/B2/M through the frozen filter."""
    history = QualifiedHistory(
        fitted,
        initial,
        source_mean,
        mode="independent" if panel == "complete" else "filtered",
    )
    losses = {k: [] for k in ("b1", "b2", "m")}
    for t, y in enumerate(ys):
        before = history.predict() if t >= 2 else None
        if t >= 12:
            for cand, store in losses.items():
                pr = before["probabilities"][cand]
                store.append(
                    ((pr - y) ** 2, -(y * np.log(pr) + (1 - y) * np.log1p(-pr)))
                )
        history.observe(*readings[t][channel])
    return {k: np.mean(v, axis=0) for k, v in losses.items()}


def main():
    """Run readers x worlds x seeds x panels; compare oracle- vs reader-fitted sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows_out, fit_stats = [], []
    for reader, folder in READERS.items():
        path = folder / "reader_scores.csv"
        with path.open() as f:
            cache = {
                (r["role"], int(r["pole"]), r["subject_status"], int(r["variant"])): r
                for r in csv.DictReader(f)
            }
        for world_index, world in enumerate(("c2_xor", "c2_iid")):
            for seed in SEEDS:
                records, _ = generate_world(world, seed, counts=COUNTS)
                source = [r for r in records if r["role"] == "source"]
                oracle_fit = fit_observed_source(source, world)
                init0 = np.ones((2, 2))
                for g in ordered_rows(source).values():
                    init0[g[0]["observation"], g[1]["observation"]] += 1
                init0 /= init0.sum()
                mean0 = float(np.mean([r["observation"] for r in source]))
                src_groups = ordered_rows(source)
                apps = ordered_rows([r for r in records if r["role"] != "source"])
                for panel in ("complete", "mixed"):
                    read_hist, read_vals, errors, missing = {}, [], 0, 0
                    for s_index, (author, rows) in enumerate(src_groups.items()):
                        ys = [r["observation"] for r in rows]
                        rd = render(
                            np.random.SeedSequence(
                                [seed, 8, 100 + world_index, s_index]
                            ),
                            panel,
                            ys,
                            cache,
                            "source",
                        )
                        seq = []
                        for r, y, x in zip(rows, ys, rd, strict=True):
                            p, q = x["hard"]
                            if q < 0.5:
                                seq.append(None)
                                missing += 1
                            else:
                                seq.append(dict(r, observation=int(p)))
                                read_vals.append(int(p))
                                errors += int(int(p) != y)
                        read_hist[author] = seq
                    read_fit = fit_read_source(read_hist, world)
                    init1 = np.ones((2, 2))
                    for seq in read_hist.values():
                        if seq[0] is not None and seq[1] is not None:
                            init1[seq[0]["observation"], seq[1]["observation"]] += 1
                    init1 /= init1.sum()
                    mean1 = float(np.mean(read_vals))
                    fit_stats.append(
                        {
                            "reader": reader,
                            "world": world,
                            "seed": seed,
                            "panel": panel,
                            "source_events": 300 * 48,
                            "read_missing": missing,
                            "read_disagree_with_true_or_ancillary": errors,
                            "b1_source_p_oracle": oracle_fit["b1"]["source_p"].tolist(),
                            "b1_source_p_read": read_fit["b1"]["source_p"].tolist(),
                        }
                    )
                    for a_index, (author, rows) in enumerate(apps.items()):
                        if rows[0]["role"] != "application":
                            continue
                        ys = [r["observation"] for r in rows]
                        rd = render(
                            np.random.SeedSequence([seed, 8, world_index, a_index]),
                            panel,
                            ys,
                            cache,
                            rows[0]["role"],
                        )
                        for arm, fitted, init, mean, channel in (
                            (
                                "oracle_channel_oracle_source",
                                oracle_fit,
                                init0,
                                mean0,
                                "oracle",
                            ),
                            (
                                "hard_channel_oracle_source",
                                oracle_fit,
                                init0,
                                mean0,
                                "hard",
                            ),
                            (
                                "hard_channel_read_source",
                                read_fit,
                                init1,
                                mean1,
                                "hard",
                            ),
                        ):
                            res = score_application(
                                fitted, init, mean, panel, rd, ys, channel
                            )
                            for cand, (brier, logloss) in res.items():
                                rows_out.append(
                                    {
                                        "reader": reader,
                                        "world": world,
                                        "seed": seed,
                                        "panel": panel,
                                        "author_id": author,
                                        "arm": arm,
                                        "candidate": cand,
                                        "brier": float(brier),
                                        "log_loss": float(logloss),
                                    }
                                )
                    print(
                        json.dumps(
                            {
                                "reader": reader,
                                "world": world,
                                "seed": seed,
                                "panel": panel,
                                "missing": missing,
                                "errors": errors,
                            }
                        ),
                        flush=True,
                    )
    for name, data in (("metrics_by_author.csv", rows_out),):
        with (args.output / name).open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (args.output / "source_fit_stats.json").write_text(
        json.dumps(fit_stats, indent=1) + "\n"
    )
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            {
                r: hashlib.sha256((f / "reader_scores.csv").read_bytes()).hexdigest()
                for r, f in READERS.items()
            }
            | {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
