#!/usr/bin/env python3
"""Compute frozen joint SOURCE target laws from numeric evidence only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_source_target_law as M


def main():
    """Keep each event bound to an opaque author without target/model inputs."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "inputs", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    with np.load(a.bundle, allow_pickle=False) as z:
        if set(z.files) != {"source_means", "codes", "variance", "anchors"}:
            raise ValueError("Unexpected SOURCE bundle keys")
        bank = {k: z[k] for k in z.files}
    entries = json.loads(a.inputs.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Nonempty addressed author records required")
    seen_authors, seen_events, rows = set(), set(), []
    for entry in entries:
        result = M.score(bank, entry)
        if entry["author"] in seen_authors or seen_events & set(entry["event_ids"]):
            raise ValueError("Duplicate author/event addresses across records")
        seen_authors.add(entry["author"])
        seen_events.update(entry["event_ids"])
        rows.append(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
            }
        )
    with a.output.open("x") as f:
        json.dump(rows, f, allow_nan=False)
        f.write("\n")
    print(f"cold scored {len(rows)} authors;0 neural/application-label calls")


if __name__ == "__main__":
    main()
