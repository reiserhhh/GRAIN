#!/usr/bin/env python3
"""Build review/acceptance cards (rules bound to machine-checked evidence)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.review_cards import build
from grain_research.gptpro_foundation_v3.review_specs import SPECS

ATOMIC = Path("/Volumes/mobile3/projects/project persona/configs/grain_gptpro_atomic_v3.json")


def main():
    """Build one or more cards into <package>/review_r<rev>/<card>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", nargs="+", required=True)
    parser.add_argument("--revision", default="review_r1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/grain_gptpro_foundation_v3")
    args = parser.parse_args()
    atomic = json.loads(ATOMIC.read_text())
    status = 0
    for card_id in args.card:
        card = SPECS[card_id]()
        out = args.output_root / card.package / args.revision / card_id
        res = build(card, ROOT, out, atomic)
        print(card_id, "checks", res["tests"], "failed", len(res["failed"]))
        for line in res["failed"]:
            print("  ", line)
        status |= bool(res["failed"])
    sys.exit(status)


if __name__ == "__main__":
    main()
