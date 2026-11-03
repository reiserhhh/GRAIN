#!/usr/bin/env python3
"""Run one frozen SOURCE-to-observed-metadata measuring construction, not a grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as B
from grain_research.construction import math_behavior_pushforward as P

CONFIG = "configs/grain_behavior_pushforward_hm291.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BEHAVIOR_PUSHFORWARD_PLAN_HM291.md",
    "grain_research/construction/math_behavior_pushforward.py",
    "grain_research/construction/math_author_belief.py",
    "scripts/run_grain_behavior_pushforward_hm291.py",
    "scripts/score_grain_behavior_pushforward_cold.py",
    "tests/test_grain_behavior_pushforward.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/behavior_pushforward_r1"
MODELS = ("full_mode", "actual_profile", "old_permuted_profile")


def read(path):
    """Read a numeric-science JSON; no credential or private material access."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Stream immutable file identity."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Write a new result once, never replace existing evidence."""
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def save(path, values):
    """Retain only authorized numerical derivatives."""
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **values)


def load(path):
    """Open numeric arrays without pickle or raw-text reconstruction."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def seal(out, name, files):
    """Bind all completed stage artifacts."""
    write(out / name, {p: digest(out / p) for p in files})


def check(out, stages=()):
    """Authenticate scientific scope and unchanged old/new artifacts."""
    m = read(out / "manifest.json")
    for category in ("code", "inputs"):
        for path, expected in m[category].items():
            if digest(ROOT / path) != expected:
                raise ValueError("Changed science input " + path)
    for stage in ("prepared_seal.json",) + tuple(stages):
        for path, expected in read(out / stage).items():
            if digest(out / path) != expected:
                raise ValueError("Changed scientific artifact " + path)
    return m


def observations(metadata, indices):
    """Bind later18 events by inherited author position, not fuzzy ID matching."""
    return np.array(
        [
            P.behavioral_observation(
                metadata["lengths"][u, 18:],
                metadata["times"][u, 18:],
                metadata["venue_codes"][u, 18:],
            )
            for u in indices
        ]
    )


def prepare(out):
    """Freeze rules before outcome measurements and preserve complete author scope."""
    c = read(ROOT / CONFIG)
    parent = ROOT / c["parent"]
    check(
        parent, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    inputs = [c["metadata"], c["parent"] + "/observations.npz"]
    inputs += [c["parent"] + f"/source_{r}.npz" for r in c["references"]]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "config": c,
            "code": {p: digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: digest(ROOT / p) for p in inputs},
            "original_scope_complete": False,
            "cohort": "historically reused public derivatives,not an untouched lockbox",
        },
    )
    z, metadata = load(parent / "observations.npz"), load(ROOT / c["metadata"])
    if len(z["held"]) != 120 or len(z["source_indices"]) != 386:
        raise ValueError("Cannot silently shrink the declared authors")
    if np.intersect1d(z["held"], z["source_indices"]).size:
        raise ValueError("SOURCE/application author overlap")
    if not np.array_equal(
        metadata["times"][z["held"]].reshape(120, 2, 18), z["held_times"]
    ):
        raise ValueError("Early/late vector and metadata events disagree")
    if not np.array_equal(
        metadata["original_units"][z["held"]].reshape(120, 2, 18), z["held_units"]
    ):
        raise ValueError("Vector and metadata comment addresses disagree")
    save(
        out / "observations.npz",
        {
            "held": z["held"],
            "source_indices": z["source_indices"],
            "held_paths": z["held_paths"],
            "donors": z["donors"],
            "actual_later_behavior": observations(metadata, z["held"]),
        },
    )
    for ref in c["references"]:
        bundle = load(parent / f"source_{ref}.npz")
        bundle["behavior"] = observations(metadata, bundle["source_indices"])
        if np.any(bundle["behavior"].var(0) <= 0):
            raise ValueError(
                "SOURCE endpoint constant: retain failure,do not drop feature"
            )
        save(out / f"source_{ref}.npz", bundle)
    record = read(parent / "example_record.json")
    record["episodes"] = record["episodes"][:1]
    write(out / "example_record.json", record)
    quality = {
        "source_authors": 386,
        "held_authors": 120,
        "later_records": 2160,
        "behaviors": list(P.FEATURES),
        "source_application_overlap": 0,
        "event_binding": "EXACT",
        "source_rows_use_old_alpha_codes_prior": True,
        "length_unit": "inherited end-start span,not newly certified tokens",
        "gap_unit": "log1p seconds between sampled records,not response latency",
        "held_labels_raw_API_neural_train_reads": 0,
        "original_scope_complete": False,
    }
    write(out / "input_quality.json", quality)
    seal(
        out,
        "prepared_seal.json",
        ["observations.npz", "example_record.json", "input_quality.json"]
        + [f"source_{r}.npz" for r in c["references"]],
    )
    print(json.dumps(quality))


def interval(value, indices):
    """Describe paired resampling with fixed SOURCE; no new population guarantee."""
    value = np.asarray(value)
    means = value[indices].mean(1)
    return {
        "mean": float(value.mean()),
        "ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "positive_people": int(np.sum(value > 0)),
    }


def compute(out):
    """Construct all three references once, scoring no application personality labels."""
    m = check(out)
    c, z = m["config"], load(out / "observations.npz")
    start = time.monotonic()
    truth, paths, donors = (
        z["actual_later_behavior"],
        z["held_paths"][:, 0],
        z["donors"],
    )
    if (
        np.any(donors < 0)
        or np.any(donors == np.arange(120))
        or len(np.unique(donors)) != 120
    ):
        raise ValueError("Old matched-donor law incomplete")
    indices = np.random.default_rng(c["seed"]).integers(0, 120, (c["bootstrap"], 120))
    summary, arrays, rows = {}, {}, []
    for ref in c["references"]:
        bundle = load(out / f"source_{ref}.npz")
        readings = [P.infer(bundle, path) for path in paths]
        prior, variance = bundle["behavior"].mean(0), bundle["behavior"].var(0)
        arrays[f"{ref}_prior"], arrays[f"{ref}_variance"] = prior, variance
        arrays[f"{ref}_profiles"] = np.array([r["profiles"] for r in readings])
        arrays[f"{ref}_marginals"] = np.array([r["marginals"] for r in readings])
        arrays[f"{ref}_within"] = np.array([r["within_profile"] for r in readings])
        gains = {}
        cells = {}
        for name in MODELS:
            pred = np.array([r[name] for r in readings])
            arrays[f"{ref}_{name}"] = pred
            gain = P.risk_gain(truth, pred, prior, variance)
            stranger = P.risk_gain(truth, pred[donors], prior, variance)
            gains[name] = gain
            cells[name] = {
                "versus_prior": interval(gain.mean(1), indices),
                "own_versus_matched_stranger": interval(
                    (gain - stranger).mean(1), indices
                ),
            }
            for j, feature in enumerate(P.FEATURES):
                row = {
                    "reference": ref,
                    "model": name,
                    "feature": feature,
                    **interval(gain[:, j], indices),
                }
                row["ci_lower"], row["ci_upper"] = row.pop("ci")
                row["own_stranger_gain"] = float((gain[:, j] - stranger[:, j]).mean())
                rows.append(row)
        cells["actual_profile_minus_old_permuted"] = interval(
            (gains["actual_profile"] - gains["old_permuted_profile"]).mean(1), indices
        )
        cells["full_minus_actual_profile"] = interval(
            (gains["full_mode"] - gains["actual_profile"]).mean(1), indices
        )
        # A complete five-target by seven-observation SOURCE description,not selection.
        bits = ((bundle["codes"][0, :, None] >> np.arange(5)) & 1).astype(float)
        cells["source_target_behavior_covariance_ENACO"] = (
            (bits - bits.mean(0)).T @ (bundle["behavior"] - prior) / len(bits)
        ).tolist()
        summary[ref] = cells
    elapsed = time.monotonic() - start
    if elapsed > c["compute_seconds_cap"]:
        raise TimeoutError("Bounded computation exceeded;no automatic renewed cap")
    save(out / "predictions.npz", arrays)
    with (out / "metrics_by_observation.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write(
        out / "summary.json",
        {
            "results": summary,
            "compute_seconds": elapsed,
            "actual_people": 120,
            "new_neural_API_private_train": 0,
            "original_scope_complete": False,
        },
    )
    seal(
        out,
        "computed_seal.json",
        ["predictions.npz", "metrics_by_observation.csv", "summary.json"],
    )
    print(json.dumps({"compute_seconds": elapsed, "pooled": summary["pooled"]}))


def verify(out):
    """Audit the new functional directly; no rerun,selection or precision search."""
    c = check(out, ("computed_seal.json",))["config"]
    z, p = load(out / "observations.npz"), load(out / "predictions.npz")
    errors = []
    for ref in c["references"]:
        bundle = load(out / f"source_{ref}.npz")
        # Direct scalar conditional means audit every author/feature/profile assignment.
        for u in range(120):
            ll = B.component_log_likelihood(bundle, [z["held_paths"][u, 0]], True)
            weights = np.exp(ll - logsumexp(ll))
            full = sum(weights[i] * bundle["behavior"][i] for i in range(len(weights)))
            errors.append(float(np.max(abs(full - p[f"{ref}_full_mode"][u]))))
            for a, model in enumerate(MODELS[1:]):
                value = np.zeros(7)
                for g in range(32):
                    ix = bundle["codes"][a] == g
                    if np.any(ix):
                        value += weights[ix].sum() * bundle["behavior"][ix].mean(0)
                errors.append(float(np.max(abs(value - p[f"{ref}_{model}"][u]))))
        np.testing.assert_allclose(
            p[f"{ref}_full_mode"][:, None, :] - p[f"{ref}_within"],
            np.stack([p[f"{ref}_{name}"] for name in MODELS[1:]], axis=1),
            atol=1e-12,
        )
    record = read(out / "example_record.json")
    b = load(out / "source_pooled.npz")
    cold = P.cold(b, record)
    # The inherited example belongs to held position0; explicit path binding required.
    first_path = np.array([e["cell"] for e in record["episodes"][0]["events"]])
    if not np.array_equal(first_path, z["held_paths"][0, 0]):
        raise ValueError("Cold example is not the registered first held episode")
    cold_error = float(np.max(abs(cold["full_mode"] - p["pooled_full_mode"][0])))
    verdict = {
        "status": "PASS",
        "literal_max_error": max(errors),
        "cold_max_error": cold_error,
        "auditor": "same_agent_not_blind",
        "original_scope_complete": False,
    }
    if max(errors + [cold_error]) > 1e-10:
        raise ArithmeticError("Calculation changes the new measuring construction")
    write(out / "verification.json", verdict)
    seal(out, "verification_seal.json", ["verification.json"])
    print(json.dumps(verdict))


def main():
    """Run exclusive prepare/compute/verify stages or a read-only seal check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "run":
        compute(args.out)
        verify(args.out)
    else:
        check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("PASS")


if __name__ == "__main__":
    main()
