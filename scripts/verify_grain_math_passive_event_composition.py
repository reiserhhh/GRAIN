#!/usr/bin/env python3
"""Independent scalar/ledger replay of HM55; no neural calls or fitting."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_passive_event_composition import (
    FIELDS,
    OUT,
    check,
    token_proofs,
)


def alternate_channels(history, atom):
    """Replay this explicit design without production active_records/channels."""
    active, mentions = {}, []
    for r in history:
        kind = r["kind"]
        if kind in ("new", "revise"):
            if kind == "revise":
                del active[r["parent_id"]]
            active[r["record_id"]] = r
        elif kind == "retract":
            del active[r["parent_id"]]
        elif kind != "alias":
            raise ValueError("Unexpected design operation")
        if kind in ("new", "alias", "revise") and r["actor"] == "A":
            mentions.append(atom[r["family"], r["polarity"]])
    events = [
        atom[r["family"], r["polarity"]] for r in active.values() if r["actor"] == "A"
    ]
    return {
        "event_profile": math.fsum(events) / len(events) if events else None,
        "mention_profile": math.fsum(mentions) / len(mentions) if mentions else None,
        "own_events": len(events),
        "own_mentions": len(mentions),
    }


def verify(out):
    """Validate every saved forward, scalar, profile and paired effect."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 408
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
    ):
        raise ValueError("All 408 actual forwards required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Result artifact drift " + name)
    rows = read_json(out / "00/rows.json")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if token_proofs(rows, tokenizer) != read_json(out / "00/token_proofs.json"):
        raise ValueError("Token-pair proof drift")
    scalars, atoms, maximum = {}, {}, 0.0
    for r in rows:
        meta, arrays = load_arrays(out / f"01/units/unit_{r['unit']:04d}.json")
        if any(meta[k] != r[k] for k in FIELDS):
            raise ValueError("Forward identity mismatch")
        full = arrays["full"]
        if len(full) != len(r["input_ids"]) or not np.isfinite(full).all():
            raise ValueError("Incomplete full-state capture")
        native = meta["native"]
        x = np.asarray(native["candidate_logits"], float)
        weights = np.exp(x - logsumexp(x))
        score = float(weights @ (np.arange(10) / 9))
        log_odds = float(logsumexp(x[5:]) - logsumexp(x[:5]))
        maximum = max(maximum, abs(score - native["conditional_score"]))
        scalars[r["unit"]] = {"score": score, "log_odds": log_odds}
        if r["role"] == "atom":
            atoms[r["target"], r["family"], r["polarity"]] = score
    lookup = {r["unit"]: r for r in rows}
    profiles = {}
    with (out / "03/native.csv").open() as stream:
        table = list(csv.DictReader(stream))
    if len(table) != 408:
        raise ValueError("Incomplete native table")
    for row in table:
        unit = int(row["unit"])
        r = lookup[unit]
        atom = {
            (f, p): atoms[r["target"], f, p]
            for f, p in itertools.product((0, 1), repeat=2)
        }
        profiles[unit] = alternate_channels(r["history"], atom)
        for k in (
            "score",
            "log_odds",
            "event_profile",
            "mention_profile",
            "own_events",
            "own_mentions",
        ):
            value = scalars[unit][k] if k in scalars[unit] else profiles[unit][k]
            if value is None:
                if row[k] != "":
                    raise ValueError("Missing evidence turned into a number")
            else:
                maximum = max(maximum, abs(float(row[k]) - value))
    with (out / "03/paired_effects.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 384:
        raise ValueError("Incomplete paired table")
    for r in pairs:
        hi, lo = int(r["hi"]), int(r["lo"])
        for key in ("score", "log_odds", "event_profile", "mention_profile"):
            source = scalars if key in ("score", "log_odds") else profiles
            a, b = source[hi][key], source[lo][key]
            if a is None or b is None:
                if r[key] != "":
                    raise ValueError("Unobserved contrast became numerical")
            else:
                maximum = max(maximum, abs(float(r[key]) - (a - b)))
    if maximum > m["spec"]["arithmetic_tolerance"]:
        raise ValueError("Alternate scalar/ledger replay differs")
    result = {
        "status": "VERIFIED_FINITE_HM55_NOT_GLOBAL",
        "actual_forward_artifacts": 408,
        "paired_scalar_replays": 384,
        "token_only_occasion_pairs": 48,
        "maximum_alternate_error": maximum,
        "new_forward_calls": 0,
        "independent_method_replication": False,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result)


def main():
    """Verify a retained run without accessing any private corpus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
