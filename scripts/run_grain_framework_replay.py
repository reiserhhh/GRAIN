#!/usr/bin/env python3
"""Replay the complete frozen framework from private states, with zero neural calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grain_research.vector_mechanisms.level_response import (
    paired_quality,
    response_ledger,
)
from grain_research.vector_mechanisms.reference_system import (
    FrozenReferenceSystem,
    file_digest,
    query_function_gram,
    reference_coordinates,
)

BASE = ROOT / "results/grain_vector_mechanisms"
LEVEL = BASE / "complete_level_response_v1"
QUERIES = ("TF_cont", "neuroticism", "confirmation_seeking")


def read(path):
    """Load a local artifact without emitting participant data."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Save reproducible JSON, rejecting NaN or unsupported scalar types."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_complete(frame, out):
    """Plot all frozen P+Q comparisons locally, never publishing participant points."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    labels = {
        "TF_cont": "T/F",
        "neuroticism": "Neuroticism",
        "confirmation_seeking": "Confirmation seeking",
    }
    for i, query in enumerate(QUERIES):
        for j, target in enumerate(("level", "natural", "anchor")):
            g = frame[(frame["query"] == query) & (frame.target == target)]
            ax = axes[i, j]
            bounds = [
                min(g.actual.min(), g.predicted.min()),
                max(g.actual.max(), g.predicted.max()),
            ]
            ax.plot(bounds, bounds, color="#999999", lw=1, ls="--")
            ax.scatter(
                g.actual,
                g.predicted,
                s=34,
                c="#206f8a",
                edgecolor="white",
                linewidth=0.5,
            )
            r = np.corrcoef(g.actual, g.predicted)[0, 1]
            ax.set(
                title=f"{labels[query]} / {target}\nr = {r:.3f}, n = {len(g)}",
                xlabel="Native observer",
                ylabel="Frozen mathematical readout",
            )
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Complete value and response replay | model agreement, not human-label accuracy",
        fontsize=13,
    )
    fig.savefig(out / "private_complete_readout_scatter.png", dpi=160)
    plt.close(fig)


def run(out):
    """Recompute scores, reference positions, exact ledgers and matched-query geometry."""
    out.mkdir(parents=True, exist_ok=False)
    system = FrozenReferenceSystem.load(LEVEL)
    design = read(LEVEL / "private_design.json")
    if len(design) != 36 or len({r["author_id"] for r in design}) != 12:
        raise ValueError("Exact complete12-author/three-query panel required")
    manifest = read(LEVEL / "manifest.json")
    if file_digest(LEVEL / "private_design.json") != manifest["input_sha256"]:
        raise ValueError("Frozen design changed")
    rows, profiles, ledger_error = [], {}, 0.0
    dependencies = {}
    for entry in design:
        query, index = entry["query"], entry["index"]
        path = LEVEL / "private_packets" / f"unit_{index:02d}_{query}.json"
        packet = read(path)
        if (
            packet["author_id"] != entry["author_id"]
            or file_digest(path.with_suffix(".npz")) != packet["arrays_sha256"]
        ):
            raise ValueError("Cache identity or state hash changed")
        dependencies[str(path.relative_to(ROOT))] = file_digest(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            prediction = system.response(
                query,
                a["full_P"],
                a["full_Q"],
                a["edited_P"],
                a["edited_Q"],
                np.asarray(entry["full_ids"])[entry["full_mask"]].tolist(),
                system.observer_id,
            )
        ledger = response_ledger(packet["full"], packet["edited"], **packet["cells"])
        total = sum(
            ledger[k]
            for k in ("rebase", "E_main", "P_main", "EP_interaction", "Q_conditional")
        )
        ledger_error = max(ledger_error, abs(total - ledger["natural"]))
        for target, actual, value, base in (
            (
                "level",
                packet["full"],
                prediction["full"]["score"],
                prediction["full"]["reference_score"],
            ),
            ("natural", packet["edited"] - packet["full"], prediction["natural"], 0.0),
            (
                "anchor",
                packet["anchor_score"] - packet["full"],
                prediction["anchor_candidate"],
                0.0,
            ),
        ):
            rows.append(
                {
                    "index": index,
                    "query": query,
                    "target": target,
                    "actual": actual,
                    "predicted": value,
                    "baseline": base,
                }
            )
        profiles.setdefault(index, {})[query] = (
            prediction["full"]["score"],
            prediction["natural"],
        )
    frame = pd.DataFrame(rows)
    old = pd.read_csv(LEVEL / "private_predictions.csv").query(
        "method == 'PQ' and target in ['level','natural','anchor']"
    )
    matched = frame.merge(
        old,
        on=["index", "query", "target"],
        suffixes=("", "_old"),
        validate="one_to_one",
    )
    if len(matched) != 108:
        raise ValueError("Exact108 paired outputs required")
    replay_error = max(
        float(np.max(np.abs(matched[k] - matched[k + "_old"])))
        for k in ("actual", "predicted", "baseline")
    )
    values = np.asarray(
        [[v for q in QUERIES for v in profiles[i][q]] for i in range(12)]
    )
    norm_path = LEVEL / "private_author_configuration.npz"
    with np.load(norm_path, allow_pickle=False) as a:
        coordinates = reference_coordinates(values, a["center"], a["scale"])
        coordinate_error = float(np.max(np.abs(coordinates - a["constructed_view0"])))
    if max(replay_error, coordinate_error, ledger_error) > 1e-11:
        raise ValueError("Mathematical API disagrees with frozen score/profile/ledger")
    frame.to_csv(out / "private_complete_predictions.csv", index=False)
    metrics = []
    for (query, target), group in frame.groupby(["query", "target"]):
        metrics.append(
            {
                "query": query,
                "target": target,
                **paired_quality(
                    group.actual.to_numpy(),
                    group.predicted.to_numpy(),
                    group.baseline.to_numpy(),
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "complete_metrics.csv", index=False)
    grams = {
        "query_order": list(QUERIES),
        "population": "same12 development authors",
        "level": query_function_gram(values[:, ::2]).tolist(),
        "response": query_function_gram(values[:, 1::2]).tolist(),
    }
    write(out / "functional_gram.json", grams)
    plot_complete(frame, out)
    files = [
        LEVEL / "manifest.json",
        LEVEL / "reference.npz",
        LEVEL / "observer.json",
        norm_path,
        BASE / "prefix_source_connection_v1/summary.json",
        BASE / "configuration_parity_v1_r2/summary.json",
        BASE / "odd_action_construction_v1/summary.json",
        BASE / "endpoint_action_constraint_v1_r2/summary.json",
    ]
    for path in files:
        dependencies[str(path.relative_to(ROOT))] = file_digest(path)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_REFERENCE_FRAMEWORK_REPLAY_V1",
            "source_artifacts": dependencies,
            "code_hashes": {
                str(p.relative_to(ROOT)): file_digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "grain_research/vector_mechanisms/reference_system.py",
                )
            },
            "human_labels_read": False,
            "neural_calls": 0,
        },
    )
    summary = {
        "status": "COMPLETE",
        "authors": 12,
        "queries": 3,
        "scored_pairs": 36,
        "compared_outputs": len(matched),
        "score_replay_error_max": replay_error,
        "reference_position_error_max": coordinate_error,
        "natural_ledger_error_max": ledger_error,
        "neural_calls": 0,
        "human_labels_read": False,
        "reference_population": "two construction authors; provisional coordinates, not population norms",
        "output_scope": "native-score and response API, not a clinical or questionnaire-calibrated predictor",
        "metrics": metrics,
    }
    write(out / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "metrics"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "reference_framework_replay_v1"
    )
    run(parser.parse_args().output)
