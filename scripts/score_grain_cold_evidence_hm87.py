#!/usr/bin/env python3
"""HM87 numerical-only scoring process; exactly two declared data inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_cold_evidence import all_queries


def main():
    """Read a frozen bundle and numeric carriers, refusing hidden research inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    allowed = {folder / "bundle.json", folder / "carriers.npz"}
    reads = set()

    def audit(event, arguments):
        if event != "open" or not isinstance(arguments[0], (str, bytes, Path)):
            return
        path = Path(arguments[0]).resolve()
        mode = arguments[1]
        if mode and any(flag in mode for flag in "wax+"):
            return
        if ROOT / "results" in path.parents:
            if path not in allowed:
                raise RuntimeError("Undeclared research data read: " + str(path))
            reads.add(path.name)

    sys.addaudithook(audit)
    with (folder / "bundle.json").open() as stream:
        bundle = json.load(stream)
    with np.load(folder / "carriers.npz", allow_pickle=False) as data:
        vectors = []
        for i, length in enumerate(data["lengths"]):
            vectors.append(
                all_queries(
                    data["keys"][i, :length],
                    data["values"][i, :length],
                    data["event_ids"][i, :length],
                    int(data["endpoints"][i]),
                    bundle,
                )
            )
    forbidden = sorted(
        name for name in sys.modules if name.split(".")[0] in {"torch", "transformers"}
    )
    if forbidden or reads != {"bundle.json", "carriers.npz"}:
        raise RuntimeError("Cold execution dependency contract failed")
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, vectors=np.stack(vectors))
    with (folder / "cold_audit.json").open("x") as stream:
        json.dump(
            {
                "data_reads": sorted(reads),
                "forbidden_modules": forbidden,
                "bodies": len(vectors),
                "no_design_truth_or_native_queries": True,
            },
            stream,
            indent=2,
        )
    print("Cold numeric observations:", len(vectors), "bodies; zero neural calls")


if __name__ == "__main__":
    main()
