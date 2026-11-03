#!/usr/bin/env python3
"""Score an early numeric episode against frozen observed-behavior SOURCE values."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_behavior_pushforward as P


def main():
    """Require only a immutable numeric source bundle and qualified early evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    result = P.cold(bundle, json.loads(args.input.read_text()))
    print(
        json.dumps(
            {
                "features": P.FEATURES,
                "readings": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in result.items()
                },
                "not_offline_behavior_or_human_trait_probability": True,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
