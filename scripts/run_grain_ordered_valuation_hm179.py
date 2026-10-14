#!/usr/bin/env python3
"""One finite all-valuation mechanism certificate;no utility fit or model call."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_ordered_valuation as O
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_ordered_valuation_hm179.json"
DEFAULT = ROOT / "results/grain_math_response_v1/ordered_valuation_action_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ORDERED_VALUATION_PLAN_HM179.md",
    "grain_research/construction/math_ordered_valuation.py",
    "scripts/run_grain_ordered_valuation_hm179.py",
    "scripts/verify_grain_ordered_valuation_hm179.py",
    "tests/test_grain_ordered_valuation.py",
)


def scopes(source, native):
    """Keep two-view author means and both views without pseudoreplication."""
    return (
        (source.mean(1), native.mean(1)),
        (source[:, 0], native[:, 0]),
        (source[:, 1], native[:, 1]),
    )


def row_for(base, result, tails):
    """Keep unresolved SOURCE observations as missing certificates,not zero ones."""
    keys = O.certificate(np.zeros(10))
    row = base | dict.fromkeys(keys) | dict.fromkeys(("source_norm", "source_value"))
    row.update(result)
    row.pop("authors", None)
    for k in range(9):
        row[f"tail_{k + 1}"] = None if tails is None else float(tails[k])
    return row


def prepare(out):
    """Bind exact caches and complete finite scope before computing new bounds."""
    start = time.perf_counter()
    spec = R.read(ROOT / CONFIG)
    if spec["resolution"] != O.RESOLUTION:
        raise ValueError("Changed numerical resolution")
    code, inputs = {}, {}
    for name, stages in (
        ("relation_parent", ("scored_seal.json", "analysis_seal.json")),
        ("native_parent", ("scored_seal.json", "analysis_seal.json")),
    ):
        path = ROOT / spec[name]
        parent = R.check(path, stages)
        code.update(parent["code"])
        for seal in ("manifest.json", "prepared_seal.json", *stages):
            files = (seal,) if seal == "manifest.json" else (seal, *R.read(path / seal))
            for f in files:
                relative = str((path / f).relative_to(ROOT))
                inputs[relative] = R.digest(ROOT / relative)
    with np.load(
        ROOT / spec["relation_parent"] / "mechanism_accounts.npz", allow_pickle=False
    ) as z:
        components = z["components"]
        old_score, old_odds = z["native_ordinal_actions"], z["native_log_odds_actions"]
    with np.load(
        ROOT / spec["native_parent"] / "composition_accounts.npz", allow_pickle=False
    ) as z:
        logits = z["native_logits"]
    with np.load(
        ROOT / spec["relation_parent"] / "observations.npz", allow_pickle=False
    ) as z:
        author_ids = z["author_ids"]
    if components.shape != (2, 24, 2, 5, 5) or logits.shape != (24, 2, 5, 10):
        raise ValueError("Changed SAME-author/event scope")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": code | {f: R.digest(ROOT / f) for f in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
            "outcomes_reused_post_HM178": True,
            "native_task_count": 1,
        },
    )
    R.archive(
        out / "observations.npz",
        components=components,
        native_logits=logits,
        old_ordinal=old_score,
        old_log_odds=old_odds,
        author_ids=author_ids,
    )
    quality = {
        "authors": 24,
        "views": 48,
        "native_subsets": 240,
        "cached_digit_logits": int(logits.size),
        "target_covectors": 5,
        "native_queries": ["neuroticism"],
        "pooled_certificates": 150,
        "individual_certificates": 3600,
        "new_neural_calls": 0,
        "target_labels_read": False,
        "raw_private_text_read": False,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def analyze(out):
    """Exhaust the ordered function class algebraically,not by parameter trials."""
    start = time.perf_counter()
    spec = R.check(out)["config"]
    z = R.prepared(out)
    p, native, plus, minus = O.native_action(z["native_logits"])
    ordinal_error = float(
        np.max(np.abs(O.readout(native, np.arange(10) / 9) - z["old_ordinal"]))
    )
    log_odds = z["native_logits"][..., 1:] - z["native_logits"][..., :1]
    odds_action = (
        log_odds[..., 2, :]
        - log_odds[..., 1, :]
        - log_odds[..., 3, :]
        + log_odds[..., 4, :]
    )
    odds_error = float(np.max(np.abs(odds_action - z["old_log_odds"])))
    if max(ordinal_error, odds_error) > O.RESOLUTION:
        raise ValueError("Historical native accounts changed")
    rows, individuals = [], []
    pooled = np.full((2, 3, 5, 5, 10), np.nan)
    for arm, component, target in np.ndindex(2, 5, 5):
        source = z["components"][arm, :, :, component, target]
        for scope, (x, m) in enumerate(scopes(source, native)):
            b, cert = O.source_alignment(x, m)
            base = {
                "arm": spec["arms"][arm],
                "scope": spec["scopes"][scope],
                "component": spec["components"][component],
                "target": spec["targets"][target],
            }
            tails = None if b is None else O.tail_actions(b)
            if b is not None:
                pooled[arm, scope, component, target] = b
            rows.append(row_for(base, cert, tails))
            for u in range(24):
                c = O.oriented_individual(x[u], m[u])
                t = (
                    None
                    if c["status"] == "UNRESOLVED_SOURCE_ZERO"
                    else O.tail_actions(np.sign(x[u]) * m[u])
                )
                individuals.append(
                    row_for(
                        base | {"panel_index": u, "author_id": str(z["author_ids"][u])},
                        c,
                        t,
                    )
                )
    R.table(out / "pooled_certificates.csv", rows)
    R.table(out / "individual_certificates.csv", individuals)
    primary = next(
        r
        for r in rows
        if (r["arm"], r["scope"], r["component"], r["target"])
        == ("cue_mask", "author_mean", "specific_change", "N")
    )
    primary_individual = [
        r
        for r in individuals
        if all(r[k] == primary[k] for k in ("arm", "scope", "component", "target"))
    ]
    if primary["status"] == "SCALE_DEPENDENT":
        decision = "ORDINAL_ORDER_ALONE_DOES_NOT_IDENTIFY_TARGET_VALUATION_NO_FAVORABLE_SCALE_SELECTION"
    elif primary["all_generators_strictly_negative"]:
        decision = "STOP_MONOTONE_SPACING_RESCUE_TARGET_INTERPRETATION_BRIDGE_REQUIRED"
    elif primary["status"] == "ORDINALLY_NONNEGATIVE":
        decision = "KEEP_ORDERED_ALIGNMENT_AT_THIS_SCOPE_NOT_PSYCHOLOGICAL_ACCURACY"
    else:
        decision = "BOUNDARY_OR_ZERO_KEEP_EXACT_CERTIFICATE_NO_PROMOTION"
    identified = [r for r in rows if r["lower"] is not None]
    summary = {
        "classification": spec["classification"],
        "primary": primary,
        "primary_individual_counts": dict(
            Counter(r["status"] for r in primary_individual)
        ),
        "all_scope_counts": dict(Counter(r["status"] for r in rows)),
        "primary_decision": decision,
        "pooled_certificates": len(rows),
        "individual_certificates": len(individuals),
        "native_ordinal_replay_error": ordinal_error,
        "native_nine_odds_replay_error": odds_error,
        "tail_mean_identity_error": max(
            abs(r["equally_spaced"] - r["tail_mean"]) for r in identified
        ),
        "digit_conditional_distribution": True,
        "no_selected_utility": True,
        "weak_bounds_strict_infimum_supremum_distinguished": True,
        "original_scope_complete": False,
        "new_neural_calls": 0,
        "seconds": time.perf_counter() - start,
    }
    R.archive(
        out / "order_accounts.npz",
        probabilities=p,
        native_measure=native,
        positive_distribution=plus,
        negative_distribution=minus,
        native_tails=O.tail_actions(native),
        pooled_measure=pooled,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "pooled_certificates.csv",
            "individual_certificates.csv",
            "order_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Run one requested finite stage with exclusive outputs and no continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
