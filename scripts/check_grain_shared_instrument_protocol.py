#!/usr/bin/env python3
"""Check readers against the shared instrument protocol; a reader passes only if every component passes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/grain_shared_instrument_protocol_v1.json"
PASSING = {"PASS"}


def check(cfg):
    for paths in cfg["evidence"].values():
        for p in paths:
            if not (ROOT / p).exists():
                raise FileNotFoundError(p)
    out = {}
    for reader, st in cfg["readers"].items():
        missing = [c for c in cfg["components"] if c not in st]
        if missing:
            raise ValueError(f"{reader} missing {missing}")
        failing = [c for c in cfg["components"] if st[c] not in PASSING | {"NOT_APPLICABLE"}]
        out[reader] = {"passes_protocol": not failing, "non_passing_components": failing}
    return out


if __name__ == "__main__":
    print(json.dumps(check(json.loads(CFG.read_text())), indent=1))
