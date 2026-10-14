#!/usr/bin/env python3
"""Cold numerical adjacent-pair scoring; upstream embedding remains neural."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_source_event_coupling as M


def score(bundle, record):
    """Return both frozen spaces without selecting a score or asking a model."""
    result = {}
    for space in ("E5", "GRAIN15"):
        current = copy.deepcopy(record)
        prototypes = bundle["prototypes"]
        if space == "GRAIN15":
            q = bundle["projection"]
            prototypes = prototypes @ q
            for e in current["events"]:
                e["vector"] = (np.asarray(e["vector"], float) @ q).tolist()
        result[space] = M.cold(bundle | {"prototypes": prototypes}, current)
    return {
        "spaces": result,
        "encoder_still_neural": True,
        "source_empirical_profile_reading_not_validated_human_trait_probability": True,
        "new_LLM_API_training_or_application_labels": False,
    }


def main():
    """Read only a frozen numerical bank and one qualified addressed record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bank = {
            k: z[k] for k in ("prototypes", "codes", "relink", "variance", "projection")
        }
    record = json.loads(Path(args.record).read_text())
    print(json.dumps(score(bank, record), allow_nan=False))


if __name__ == "__main__":
    main()
