#!/usr/bin/env python3
"""Finite observation-law discrimination, not a predictor or precision search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_evidence_sampling as S
from grain_research.construction import math_joint_evidence as J
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_evidence_sampling_hm196.json"
DEFAULT = ROOT / "results/grain_math_response_v1/evidence_sampling_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVIDENCE_SAMPLING_PLAN_HM196.md",
    "grain_research/construction/math_evidence_sampling.py",
    "scripts/run_grain_evidence_sampling_hm196.py",
    "tests/test_grain_evidence_sampling.py",
    "grain_research/construction/math_joint_evidence.py",
)


def check(out, stages=()):
    """Read seals without modifying or reexecuting any parent science."""
    m = R.read(out / "manifest.json")
    for section in ("code", "inputs"):
        for name, digest in m[section].items():
            if R.digest(ROOT / name) != digest:
                raise ValueError("Changed sealed method or input: " + name)
    for stage in ("prepared_seal.json", *stages):
        for name, digest in R.read(out / stage).items():
            if R.digest(out / name) != digest:
                raise ValueError("Changed sealed result: " + name)
    return m


def prepare(out):
    """Bind existing authorized numeric records before the new regime comparison."""
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["parent"]
    R.check(parent, ("scored_seal.json",))
    bundle = R.read(parent / "source_bundle.json")
    p, y = J.validate_bundle(bundle)
    n = S.empirical_counts(p, spec["source_events_per_author"])
    if len(n) != spec["source_authors"]:
        raise ValueError("Changed complete SOURCE author family")
    with (parent / "cold_input.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != spec["cold_families"]:
        raise ValueError("Changed complete application evidence family")
    for r in records:
        if set(r) != {"author_key", "events"}:
            raise ValueError("Cold evidence includes undeclared fields")
    names = (
        "source_bundle.json",
        "cold_input.jsonl",
        "cold_output.jsonl",
        "manifest.json",
        "prepared_seal.json",
        "scored_seal.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {name: R.digest(ROOT / name) for name in SCIENCE},
            "inputs": {
                str((parent / name).relative_to(ROOT)): R.digest(parent / name)
                for name in names
            },
            "status": "new_sampling_regime_comparison_not_natural_law_identification",
            "old_law": bundle["observation_law"],
            "new_law": "counterfactual_finite_source_inventory_sampling_without_replacement",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", counts=n, targets=y)
    R.write(out / "cold_records.json", records)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": len(n),
            "historical_records_per_source_author": int(n.sum(1)[0]),
            "cold_families": len(records),
            "cold_authors": len({r["author_key"].split("/")[0] for r in records}),
            "new_neural_calls": 0,
            "new_target_or_native_fit": False,
            "sampling_regime_is_counterfactual_not_proven_human_generation": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_records.json", "input_quality.json"),
    )
    print(json.dumps(R.read(out / "input_quality.json")))


def observations(out):
    """Load this packet's frozen arrays only."""
    with np.load(out / "observations.npz", allow_pickle=False) as z:
        return z["counts"], z["targets"]


def compute(out):
    """Compare every fixed singleton/pair and all historical evidence families."""
    spec = check(out)["config"]
    n, y = observations(out)
    start = time.perf_counter()
    rows, word_arrays = [], []
    for a in range(8):
        for b in (-1, *range(8)):
            word = [a] if b == -1 else [a, b]
            events = [{"event_id": str(i), "cell": c} for i, c in enumerate(word)]
            values = []
            for regime in S.REGIMES:
                _, state = S.replay(n, y, events, regime)
                values.append(state["mu"])
            if any(v is None for v in values):
                raise ValueError(
                    "Missing declared singleton/pair support; retain failure"
                )
            word_arrays.append(values)
            for q, target in enumerate(spec["targets"]):
                rows.append(
                    {
                        "cell_a": a,
                        "cell_b": b,
                        "target": target,
                        "replacement": float(values[0][q]),
                        "finite_inventory": float(values[1][q]),
                        "difference": float(values[1][q] - values[0][q]),
                    }
                )
    R.table(out / "singleton_pair_comparison.csv", rows)
    R.archive(out / "singleton_pair_values.npz", probabilities=np.asarray(word_arrays))
    records = R.read(out / "cold_records.json")
    final = np.full((len(records), 2, 5), np.nan)
    supported = np.zeros((len(records), 2), bool)
    steps, endpoint_rows = [], []
    continuation_error = 0.0
    for i, record in enumerate(records):
        for m, regime in enumerate(S.REGIMES):
            trajectory, state = S.replay(n, y, record["events"], regime)
            supported[i, m] = state["supported"]
            if state["supported"]:
                final[i, m] = state["mu"]
            for q, target in enumerate(spec["targets"]):
                endpoint_rows.append(
                    {
                        "family": record["author_key"],
                        "regime": regime,
                        "target": target,
                        "supported": bool(state["supported"]),
                        "probability": None
                        if state["mu"] is None
                        else float(state["mu"][q]),
                    }
                )
            for t, step in enumerate(trajectory):
                if step["supported"] and step["predicted"] is not None:
                    continuation_error = max(
                        continuation_error,
                        float(np.max(np.abs(step["after"] - step["predicted"]))),
                    )
                for q, target in enumerate(spec["targets"]):
                    before, after = step["before"], step["after"]
                    steps.append(
                        {
                            "family": record["author_key"],
                            "regime": regime,
                            "step": t,
                            "event_id": step["event_id"],
                            "cell": step["cell"],
                            "target": target,
                            "supported": bool(step["supported"]),
                            "remaining_source_authors": step[
                                "remaining_source_authors"
                            ],
                            "before": None if before is None else float(before[q]),
                            "after": None if after is None else float(after[q]),
                            "increment": None
                            if before is None or after is None
                            else float(after[q] - before[q]),
                        }
                    )
    R.table(out / "target_trajectories.csv", steps)
    R.table(out / "cold_posteriors.csv", endpoint_rows)
    R.archive(out / "cold_values.npz", probabilities=final, supported=supported)
    prior = y.mean(0)
    common = supported.all(1)
    delta = final[common, 1] - final[common, 0]
    signs = (final[common, 1] - prior) * (final[common, 0] - prior) < -(
        spec["audit_tolerance"] ** 2
    )
    by_target = {
        target: {
            "rms_probability_difference": float(np.sqrt(np.mean(delta[:, q] ** 2))),
            "maximum_absolute_probability_difference": float(
                np.max(np.abs(delta[:, q]))
            ),
            "opposite_prior_relative_directions": int(signs[:, q].sum()),
        }
        for q, target in enumerate(spec["targets"])
    }
    singleton_error = max(abs(r["difference"]) for r in rows if r["cell_b"] == -1)
    distinct_error = max(
        abs(r["difference"])
        for r in rows
        if r["cell_b"] >= 0 and r["cell_a"] != r["cell_b"]
    )
    summary = {
        "packet": "HM196",
        "source_authors": len(n),
        "cold_families": len(records),
        "supported_by_regime": supported.sum(0).tolist(),
        "common_supported_families": int(common.sum()),
        "finite_unsupported_families": [
            records[i]["author_key"] for i in np.flatnonzero(~supported[:, 1])
        ],
        "singleton_agreement_error": singleton_error,
        "distinct_category_pair_agreement_error": distinct_error,
        "repeated_category_pair_maximum_difference": max(
            abs(r["difference"]) for r in rows if r["cell_a"] == r["cell_b"]
        ),
        "continuation_identity_error": continuation_error,
        "overall_rms_probability_difference": float(np.sqrt(np.mean(delta**2))),
        "by_target": by_target,
        "cpu_seconds": time.perf_counter() - start,
        "not_native_reasoning_or_human_personality_change": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "singleton_pair_comparison.csv",
            "singleton_pair_values.npz",
            "target_trajectories.csv",
            "cold_posteriors.csv",
            "cold_values.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One alternate count-product audit, with exact replay of the old iid reader."""
    m = check(out, ("computed_seal.json",))
    n, y = observations(out)
    records = R.read(out / "cold_records.json")
    with np.load(out / "cold_values.npz", allow_pickle=False) as z:
        saved, supported = z["probabilities"], z["supported"]
    parent = ROOT / m["config"]["parent"]
    with (parent / "cold_output.jsonl").open() as stream:
        old = {
            r["author_key"]: np.asarray(r["probabilities"])[2]
            for r in map(json.loads, stream)
        }
    error, replay_error, checks = 0.0, 0.0, 0
    for i, record in enumerate(records):
        seen = {}
        for event in record["events"]:
            if event["event_id"] in seen and seen[event["event_id"]] != event["cell"]:
                raise ValueError("Changed event identity")
            seen[event["event_id"]] = event["cell"]
        cells = list(seen.values())
        for k, regime in enumerate(S.REGIMES):
            alternate = S.direct_posterior(n, y, cells, regime)
            if (alternate is not None) != supported[i, k]:
                raise ValueError("Support disagreement")
            if alternate is not None:
                error = max(error, float(np.max(np.abs(alternate - saved[i, k]))))
                checks += 5
        replay_error = max(
            replay_error, float(np.max(np.abs(saved[i, 0] - old[record["author_key"]])))
        )
    if max(error, replay_error) > m["config"]["audit_tolerance"]:
        raise ValueError("Alternate audit failed; do not tune tolerance or rerun")
    result = {
        "status": "PASS",
        "direct_count_product_checks": checks,
        "maximum_alternate_error": error,
        "old_shared_author_replay_error": replay_error,
        "audit_is_same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    print(json.dumps(result))


def main():
    """Execute one declared stage, refusing already-existing stage outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    markers = {
        "prepare": "manifest.json",
        "compute": "computed_seal.json",
        "verify": "verification.json",
    }
    if (args.out / markers[args.stage]).exists():
        raise FileExistsError("Frozen stage already exists; no overwrite or rerun")
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
