#!/usr/bin/env python3
"""Emit numeric entity--behaviour affinities from an already extracted source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def score(operator, slots):
    """Return both declared kernels, not probabilities or personality diagnoses."""
    u, v, s = (
        operator["entity_basis"],
        operator["attribute_basis"],
        operator["singular_values"],
    )
    e, c = slots[:3], slots[3:5]
    if slots.shape != (5, len(u)):
        raise ValueError("Three controlled people and two clause endpoints required")
    return {
        "source_entity_metric": ((e @ u * s[: u.shape[1]] ** 2) @ (c @ u).T).tolist(),
        "mixed_bilinear_kernel": ((e @ u * s[: u.shape[1]]) @ (c @ v).T).tolist(),
        "raw_inner_product": (e @ c.T).tolist(),
        "input_is_extracted_hidden_state_not_raw_text": True,
        "pretrained_encoder_upstream_required": True,
    }


def main():
    """Use no torch/tokenizer/transformer, query, answer or fitted score head."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("row", type=int)
    args = parser.parse_args()
    with (
        np.load(args.root / "01/operator.npz", allow_pickle=False) as operator,
        np.load(
            args.root / f"02/a{args.row:02d}_slots.npz", allow_pickle=False
        ) as data,
    ):
        slots = data["slots"]
        if not np.allclose(slots, slots[0], atol=1e-5):
            raise ValueError("Causal body slots unexpectedly vary by future query")
        print(json.dumps(score(operator, slots[0]), indent=2))


if __name__ == "__main__":
    main()
