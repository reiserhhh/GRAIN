#!/usr/bin/env python3
"""Build/use one frozen author instrument, not another accuracy search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_pair_measurement as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_pair_measurement_hm194.json"
DEFAULT = ROOT / "results/grain_math_response_v1/paired_measurement_instrument_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PAIR_MEASUREMENT_PLAN_HM194.md",
    "grain_research/construction/math_pair_measurement.py",
    "scripts/run_grain_pair_measurement_hm194.py",
    "scripts/score_grain_pair_measurement.py",
    "tests/test_grain_pair_measurement.py",
    "grain_research/construction/math_observed_family.py",
    "grain_research/construction/math_route_relation.py",
)


def prepare(out):
    """Load only authorized numeric observations; exclude source author indices."""
    cfg = R.read(ROOT / CONFIG)
    laws, event = ROOT / cfg["law_parent"], ROOT / cfg["event_parent"]
    R.check(laws, ("computed_seal.json", "verification_seal.json"))
    R.check(event)
    with np.load(event / "observations.npz", allow_pickle=False) as z:
        held = z["held"]
        data = {
            k: z[k][held]
            for k in ("coordinates", "conditions", "times", "original_units")
        }
    with np.load(laws / "observations.npz", allow_pickle=False) as z:
        source, rho = z["source"], z["reference"]
    if len(held) != cfg["held_authors"] or np.intersect1d(source, held).size:
        raise ValueError("Changed or source-overlapping author cohort")
    files = [
        laws / k
        for k in (
            "manifest.json",
            "aggregate_three_law_bundle.json",
            "observations.npz",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    files += [
        event / k
        for k in (
            "manifest.json",
            "observations.npz",
            "source_law.json",
            "prepared_seal.json",
        )
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **data, held=held, reference=rho)
    R.write(
        out / "input_quality.json",
        {
            "held_authors": len(held),
            "source_authors": len(source),
            "source_held_overlap": 0,
            "application_labels_read": False,
            "historical_representation_overlap_retained": True,
            "semantic_actor_quotation_volition_not_available": True,
            "new_neural_private_API_or_training": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(R.read(out / "input_quality.json")))


def make_packet(z, i, cfg, instrument):
    """Keep same chronological source-family units; never infer actor ownership."""
    u = int(z["held"][i])
    events = []
    for window, (start, stop) in enumerate(cfg["event_indices"]):
        for j in range(start, stop):
            events.append(
                {
                    "event_id": f"author-{u}:comment-{int(z['original_units'][i, j])}",
                    "window": window,
                    "condition": int(z["conditions"][i, j]),
                    "timestamp": float(z["times"][i, j]),
                    "coordinates": z["coordinates"][i, j].tolist(),
                }
            )
    return {
        "author_key": f"historical-author-{u}",
        "regime": M.REGIME,
        "representation_id": instrument["representation_id"],
        "events": events,
    }


def run(out):
    """Apply existing SOURCE functions, retain all author/event action accounts."""
    manifest = R.check(out)
    cfg, z = manifest["config"], R.prepared(out)
    start = time.perf_counter()
    source = ROOT / cfg["law_parent"]
    event = ROOT / cfg["event_parent"]
    instrument = M.compile_instrument(
        R.read(source / "aggregate_three_law_bundle.json"),
        z["reference"],
        R.read(event / "source_law.json")["thresholds"],
        R.digest(event / "source_law.json"),
    )
    R.write(out / "instrument.json", instrument)
    readings, effects, mixed, empirical, rows, statuses = [], [], [], [], [], []
    for i in range(len(z["held"])):
        packet = make_packet(z, i, cfg, instrument)
        q = M.score(instrument, packet)
        statuses.append(
            {
                "author_key": packet["author_key"],
                "status": q["status"],
                "counts": q["counts"],
            }
        )
        if q["status"] != "SUPPORTED":
            continue
        readings.append(q["relative"])
        effects.append(q["event_deletion_delta"])
        mixed.append(q["joint_deletion_interaction"])
        empirical.append(q["empirical_pair_law"])
        for k, model in enumerate(M.M.MODELS):
            for t, target in enumerate(M.TARGETS):
                rows.append(
                    {
                        "author_key": packet["author_key"],
                        "model": model,
                        "target": target,
                        "relative_pair_functional": float(q["relative"][k, t]),
                        "reference_route_composition": float(
                            q["reference_route_composition"][k, t]
                        ),
                        "within_route_departure": float(
                            q["within_route_departure"][k, t]
                        ),
                        "mean_pair_reading_NOT_history_posterior": float(
                            q["pair_average_NOT_history_posterior"][k, t]
                        ),
                    }
                )
        if i == 0:
            R.write(out / "example_packet.json", packet)
    R.write(out / "packet_status.json", statuses)
    if len(readings) != cfg["held_authors"]:
        R.write(
            out / "partial_status.json",
            {
                "decision": "UNSUPPORTED_PACKETS_RETAINED_NO_SCOPE_SHRINK",
                "supported": len(readings),
            },
        )
        raise ValueError("Not all120 packets supported; retained partial statuses")
    readings, effects, mixed, empirical = map(
        np.asarray, (readings, effects, mixed, empirical)
    )
    R.archive(
        out / "author_actions.npz",
        readings=readings,
        event_actions=effects,
        paired_actions=mixed,
        empirical_pair_laws=empirical,
        held=z["held"],
    )
    R.table(out / "author_accounts.csv", rows)
    by_target = []
    for t, target in enumerate(M.TARGETS):
        actual = np.array(
            [
                r["relative_pair_functional"]
                for r in rows
                if r["model"] == "same_author" and r["target"] == target
            ]
        )
        choice = np.array(
            [
                r["reference_route_composition"]
                for r in rows
                if r["model"] == "same_author" and r["target"] == target
            ]
        )
        within = np.array(
            [
                r["within_route_departure"]
                for r in rows
                if r["model"] == "same_author" and r["target"] == target
            ]
        )
        by_target.append(
            {
                "target": target,
                "relative_min": float(actual.min()),
                "relative_max": float(actual.max()),
                "route_within_opposite_sign_authors": int(np.sum(choice * within < 0)),
                "maximum_event_deletion_abs": float(np.max(np.abs(effects[:, 2, t]))),
                "maximum_paired_deletion_interaction_abs": float(
                    np.max(np.abs(mixed[:, 2, t]))
                ),
                "routed_law_change_RMS": float(
                    np.sqrt(np.mean((readings[:, 1, t] - readings[:, 0, t]) ** 2))
                ),
                "within_law_change_RMS": float(
                    np.sqrt(np.mean((readings[:, 2, t] - readings[:, 1, t]) ** 2))
                ),
            }
        )
    R.table(out / "measurement_summary.csv", by_target)
    summary = {
        "packet": "HM194",
        "classification": "DEVELOPMENT_USE_NOT_NEW_THEOREM_OR_ACCURACY",
        "supported_authors": len(readings),
        "distinct_current_events": len(readings) * 34,
        "pair_evaluations_NOT_independent_observations": len(readings) * 17 * 17,
        "target_accounts": by_target,
        "runtime_neural_reasoning_or_labels": 0,
        "new_mathematical_instrument_parameters": 0,
        "compute_seconds": time.perf_counter() - start,
        "source_world_native_semantic_bridge_complete": False,
        "original_scope_complete": False,
        "decision": "USE_FROZEN_REFERENCE_AND_KEEP_CHOICE_WITHIN_AND_CONTEXTUAL_EVENT_ACTIONS",
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "instrument.json",
            "example_packet.json",
            "packet_status.json",
            "author_actions.npz",
            "author_accounts.csv",
            "measurement_summary.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One alternate literal deletion audit; no numeric precision campaign."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z = R.prepared(out)
    instrument = R.read(out / "instrument.json")
    psi, _, ref, _, _ = M.instrument_arrays(instrument)
    errors = {
        "literal_pair_average": 0.0,
        "literal_event_deletion": 0.0,
        "literal_paired_deletion": 0.0,
        "decompositions": 0.0,
    }
    checks = 0
    for i in range(len(z["held"])):
        p = make_packet(z, i, cfg, instrument)
        q = M.score(instrument, p)
        _, (a, b) = M.addressed_cells(instrument, p)
        table = np.array([[psi[:, :, x, y] for y in b] for x in a]).transpose(
            2, 3, 0, 1
        )
        mean = table.sum((2, 3)) / (len(a) * len(b))
        errors["literal_pair_average"] = max(
            errors["literal_pair_average"],
            float(np.max(np.abs(mean - ref - q["relative"]))),
        )
        singles = []
        for axis, n in ((2, len(a)), (3, len(b))):
            for j in range(n):
                changed = np.delete(table, j, axis=axis).mean((2, 3)) - mean
                singles.append(changed)
        singles = np.stack(singles, axis=2)
        errors["literal_event_deletion"] = max(
            errors["literal_event_deletion"],
            float(np.max(np.abs(singles - q["event_deletion_delta"]))),
        )
        for j in range(len(a)):
            for k in range(len(b)):
                both = (
                    np.delete(np.delete(table, j, axis=2), k, axis=3).mean((2, 3))
                    - mean
                )
                changed = both - singles[..., j] - singles[..., len(a) + k]
                errors["literal_paired_deletion"] = max(
                    errors["literal_paired_deletion"],
                    float(
                        np.max(
                            np.abs(changed - q["joint_deletion_interaction"][..., j, k])
                        )
                    ),
                )
                checks += 1
        errors["decompositions"] = max(
            errors["decompositions"],
            float(
                np.max(
                    np.abs(
                        q["relative"]
                        - q["reference_route_composition"]
                        - q["within_route_departure"]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        q["law_change_total"]
                        - q["law_change_routed"]
                        - q["law_change_within"]
                    )
                )
            ),
        )
    if max(errors.values()) > cfg["audit_tolerance"]:
        raise ValueError("Measurement calculation mismatch; no threshold rescue")
    R.write(
        out / "audit_receipt.json",
        {
            "errors": errors,
            "paired_deletion_cases": checks,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("audit_receipt.json",))
    print(json.dumps(R.read(out / "audit_receipt.json")))


def main():
    """Exclusive one-pass stages; preserve all old scientific artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "verify"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "verify": verify}[a.stage](a.out)


if __name__ == "__main__":
    main()
