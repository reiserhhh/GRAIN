#!/usr/bin/env python3
"""Package M25/M26 coefficients and replay all author profiles without a model."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_bundle import FrozenFunctionalBundle
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def main():
    """Export only coefficients, query contract and reference moments; no texts/IDs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    analysis, states, out = (
        x.resolve() for x in (args.analysis_dir, args.state_dir, args.output_dir)
    )
    if not all(x.is_relative_to(BASE) for x in (analysis, states, out)):
        raise ValueError("Use ignored private artifact paths")
    m, s = read(analysis / "manifest.json"), read(states / "manifest.json")
    if (
        read(analysis / "run_status.json")["status"] != "COMPLETE"
        or sha(states / "manifest.json") != m["state_manifest_sha256"]
    ):
        raise ValueError("Completed matching analysis required")
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, key in (
        ("compact_functions.npz", "compact_sha256"),
        ("author_reference.json", "author_reference_sha256"),
        ("matched_query_projection.json", "shared_projection_sha256"),
    ):
        if sha(analysis / name) != m[key]:
            raise ValueError("Frozen construction changed")
        if name == "compact_functions.npz":
            with np.load(analysis / name, allow_pickle=False) as z:
                packed = dict(z)
            for query, rank in m["compact_ranks"].items():
                if rank == min(packed[f"{query}__p_ref"].shape):
                    packed[f"{query}__configuration"] = packed.pop(
                        f"{query}__left"
                    ) @ packed.pop(f"{query}__right")
            np.savez_compressed(out / name, **packed)
        else:
            shutil.copy2(analysis / name, out / name)
        files[name] = sha(out / name)
    observer = {
        k: v for k, v in read(states / "observer.json").items() if k != "unix_time"
    }
    write(
        out / "bundle.json",
        {
            "protocol": "GRAIN_FUNCTIONAL_BUNDLE_M26_V1",
            "queries": ["TF_cont", "neuroticism", "confirmation_seeking"],
            "question_tokens": s["question_tokens"],
            "observer_configuration": observer,
            "files": files,
            "analysis_manifest_sha256": sha(analysis / "manifest.json"),
            "units": "unclipped model-query function; not questionnaire probabilities",
            "reference": "12 fixed S1 authors; two views averaged before norm fitting",
            "state_requirement": "same frozen causal prefix, layer16, query template and E1/E2 material procedure",
            "human_labels_read": False,
        },
    )
    bundle = FrozenFunctionalBundle.load(out)
    lookup = {}
    for path in (states / "private_packets").glob("cell_*.json"):
        d = read(path)
        if sha(path.with_suffix(".npz")) != d["array_sha256"]:
            raise ValueError("State packet changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            lookup.setdefault((d["author_id"], d["view"]), {}).setdefault(
                d["query"], {}
            )[d["condition"]] = (z["P"], z["Q"])
    expected = pd.read_csv(analysis / "private_compact_parts.csv").set_index(
        ["author_id", "view", "query", "condition"]
    )
    rows, max_error = [], 0.0
    for (author, view), panel in lookup.items():
        result = bundle.profile(panel, s["question_tokens"], observer)
        for q, conditions in result["parts"].items():
            for c, parts in conditions.items():
                for k, value in parts.items():
                    max_error = max(
                        max_error,
                        abs(value - float(expected.loc[(author, view, q, c), k])),
                    )
        rows.append(
            {
                "author_id": author,
                "view": view,
                **dict(
                    zip(result["coordinate_names"], result["positions"], strict=True)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_replayed_positions.csv", index=False)
    raw = np.array(
        [
            bundle.profile(panel, s["question_tokens"], observer)["raw"]
            for panel in lookup.values()
        ]
    )
    batch_error = float(
        np.max(
            np.abs(
                np.vstack([bundle.coordinates(x[None]) for x in raw])
                - bundle.coordinates(raw)
            )
        )
    )
    if max_error > 1e-10 or batch_error > 1e-12:
        raise ValueError("Portable function replay failed")
    compact_count = 0
    full_count = 0
    for q in bundle.manifest["queries"]:
        get = lambda k, query=q: bundle.arrays[f"{query}__{k}"]
        if f"{q}__configuration" in bundle.arrays:
            compact_count += get("configuration").size
        else:
            compact_count += get("left").size + get("right").size
        full_count += get("p_ref").size
    write(
        out / "replay_audit.json",
        {
            "profiles": len(rows),
            "states": len(expected),
            "max_score_part_error": max_error,
            "batch_error": batch_error,
            "configuration_coefficient_count": compact_count,
            "full_configuration_count": full_count,
            "configuration_coefficient_ratio": compact_count / full_count,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "script_sha256": sha(__file__),
            "bundle_module_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/functional_bundle.py"
            ),
        },
    )
    print(read(out / "replay_audit.json"))


if __name__ == "__main__":
    main()
