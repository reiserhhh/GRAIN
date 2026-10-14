#!/usr/bin/env python3
"""Cold paired total/composition/within-configuration criterion readings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_components as M


def score(path, record, bank, space):
    """Accept numeric evidence only and exclude the evaluated SOURCE author."""
    if (
        set(record) != {"author_cache_id", "sampling_regime", "pairs"}
        or record["sampling_regime"] != "next_cached_record_within_registered_window"
        or bank not in (0, 1)
    ):
        raise ValueError("Qualified numeric evidence only, no evaluated labels")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    ix = data["folds"] == bank
    if record["author_cache_id"] in data["source_indices"][ix]:
        raise ValueError("Evaluated author occurs in the selected SOURCE bank")
    bundle = {
        k: data[k][ix] if k in ("prototypes", "codes", "relink") else data[k]
        for k in ("prototypes", "codes", "relink", "projection", "variance")
    }
    values = M.cold_components(bundle, record["pairs"], space)
    return {
        "channels": M.CHANNELS,
        "window_order": ["window0", "window1", "all22"],
        "target_order": ["E", "N", "A", "C", "O"],
        "values": values.tolist(),
        "source_bank": bank,
        "application_labels_used": False,
        "not_calibrated_human_trait_probabilities": True,
    }


def main():
    """Expose the separately usable source-qualified numerical measurement."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--source-bank", type=int, choices=(0, 1), required=True)
    p.add_argument("--space", choices=("E5", "GRAIN15"), required=True)
    a = p.parse_args()
    print(
        json.dumps(
            score(a.bundle, json.loads(a.record.read_text()), a.source_bank, a.space)
        )
    )


if __name__ == "__main__":
    main()
