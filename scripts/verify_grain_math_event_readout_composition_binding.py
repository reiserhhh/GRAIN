#!/usr/bin/env python3
"""Cross-bind HM56 certificate components to raw queries, atoms and event profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_event_readout_composition import OUT, check, observation
from scripts.verify_grain_math_event_readout_composition import table


def verify(out):
    """Check table-to-raw identities that scalar/projection replay alone cannot prove."""
    m = check(out)
    rows = read_json(out / "00/rows.json")
    values, atoms, accounts = {}, {}, {}
    for r in rows:
        raw, _ = observation(out, r, m)
        logits = np.asarray(raw["candidate_logits"], float)
        score = float(np.exp(logits - logsumexp(logits)) @ (np.arange(10) / 9))
        values[r["unit"]] = score
        if r["role"] == "atom":
            atoms[r["content"], r["family"], r["polarity"], r["query"]] = score
        else:
            accounts[r["parent_unit"], r["query"]] = r
    profiles = {(int(r["parent_unit"]), r["query"]): r for r in table(out, "profiles")}
    maximum, seen = 0.0, set()
    for r in table(out, "components"):
        key = int(r["parent_unit"]), r["query"]
        if key in seen:
            raise ValueError("Duplicate account/query component")
        seen.add(key)
        source = accounts[key]
        c, f, p, q = (source[k] for k in ("content", "family", "polarity", "query"))
        expected = {
            "native": values[source["unit"]],
            "focal_atom": atoms[c, f, p, q],
            "companion_atom": atoms[c, 1 - f, 1 - p, q],
            "event_profile": float(profiles[key]["event_profile"]),
            "mention_profile": float(profiles[key]["mention_profile"]),
        }
        for k, v in expected.items():
            maximum = max(maximum, abs(float(r[k]) - v))
        for k in ("content", "regime", "family", "polarity", "order"):
            if str(source[k]) != r[k]:
                raise ValueError("Component identity mismatch")
    if seen != set(accounts):
        raise ValueError("Missing account/query component")
    for r in table(out, "update_components"):
        hi, lo, q = int(r["hi"]), int(r["lo"]), r["query"]
        for k, field in (
            ("event_change", "event_profile"),
            ("mention_change", "mention_profile"),
        ):
            expected = float(profiles[hi, q][field]) - float(profiles[lo, q][field])
            maximum = max(maximum, abs(float(r[k]) - expected))
    if maximum > m["spec"]["arithmetic_tolerance"]:
        raise ValueError("Cross-bound values differ")
    result = {
        "status": "VERIFIED_HM56_RAW_COMPONENT_BINDING",
        "account_query_components": len(seen),
        "maximum_binding_error": maximum,
        "new_neural_calls": 0,
        "companion_sha256": digest(Path(__file__)),
        "primary_frozen_verifier_unchanged": True,
        "independent_method_replication": False,
    }
    write(out / "binding_verification.json", result)
    print(result)


def main():
    """Audit an existing complete run without modifying any scientific result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
