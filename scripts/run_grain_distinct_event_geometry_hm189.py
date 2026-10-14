#!/usr/bin/env python3
"""One source mechanism discrimination; no new deployment or precision grid."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_choice_geometry as M
from grain_research.construction import math_distinct_condition_geometry as U
from scripts import run_grain_choice_geometry_hm188 as H
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_distinct_event_geometry_hm189.json"
DEFAULT = ROOT / "results/grain_math_response_v1/distinct_event_geometry_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_DISTINCT_EVENT_GEOMETRY_PLAN_HM189.md",
    "grain_research/construction/math_distinct_condition_geometry.py",
    "scripts/run_grain_distinct_event_geometry_hm189.py",
    "tests/test_grain_distinct_condition_geometry.py",
)


def prepare(out):
    """Seal unchanged source events and a fixed mechanism question."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    prior = R.check(
        parent, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    z = R.prepared(parent)
    if z["coordinates"].shape != (386, 2, 12, 15):
        raise ValueError("Changed SAME-event geometry")
    out.mkdir(parents=True, exist_ok=False)
    inputs = (
        "manifest.json",
        "observations.npz",
        "construction_accounts.npz",
        "prepared_seal.json",
        "compiled_seal.json",
        "scored_seal.json",
        "verification_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": prior["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in inputs
            },
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z)
    counts = np.stack(
        ((z["conditions"] == 0).sum(-1), (z["conditions"] == 1).sum(-1)), -1
    )
    s = (counts >= 2).all(-1)
    R.write(
        out / "quality.json",
        {
            "authors": len(s),
            "events": int(z["conditions"].size),
            "eligible_per_view": s.sum(0).tolist(),
            "eligible_both": int(s.all(1).sum()),
            "identical_support_all_three_geometries": True,
            "all_authors_retained": True,
            "no_new_cold_or_neural_runs": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "quality.json"))
    print(json.dumps(R.read(out / "quality.json")))


def banks(m, y, folds, literal=False):
    """Build all seven signed accounts for all three geometries."""
    result = []
    for k in range(3):
        g = m["geometry"][:, :, k]
        if literal:
            local = m | {
                "geometry": g,
                "weighted": np.where(
                    m["support"][..., None, None], m["a"][..., None, None] * g, 0
                ),
            }
            result.append(H.literal_fields(local, y, folds))
        else:
            result.append(M.fields(m["a"], g, m["support"], y, folds)["fields"])
    return np.asarray(result)


def statistics(f):
    """Keep each geometry/target/account, never choose the best score."""
    rows = []
    for k, name in enumerate(U.GEOMETRIES):
        rows.extend([row | {"geometry": name} for row in M.statistics(f[k])])
    return rows


def values(f):
    """Return fixed complete statistic order for reference replay."""
    return np.asarray([row["signed_product"] for row in statistics(f)])


def geometry_cross(f):
    """Retain all nine signed terms for K_squared=K_distinct+K_within."""
    rows = []
    for target in list(range(5)) + [None]:
        ids = [target] if target is not None else list(range(5))
        for i, left in enumerate(U.GEOMETRIES):
            for j, right in enumerate(U.GEOMETRIES):
                a = f[i, 0][:, ids, 6].mean(0)
                b = f[j, 1][:, ids, 6].mean(0)
                rows.append(
                    {
                        "target": M.P.TARGETS[target] if target is not None else "all",
                        "left": left,
                        "right": right,
                        "signed_product": float(np.sum(a * b)),
                    }
                )
    return rows


def compute(out):
    """Run one frozen full-source discrimination and all declared references."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    start = time.monotonic()
    parent = ROOT / R.read(ROOT / cfg["parent"] / "manifest.json")["config"]["parent"]
    with np.load(parent / "construction_accounts.npz", allow_pickle=False) as data:
        yp, cp = data["label_permutations"], data["pairing_permutations"]
    m = U.observations(z["coordinates"], z["conditions"])
    f = banks(m, z["labels"], z["folds"])
    rows, n = statistics(f), cfg["null_replicates"]
    null = np.empty((3, n, len(rows)))
    orders, diag = [], []
    rng = np.random.default_rng(cfg["pairing_seed"])
    for r in range(n):
        if time.monotonic() - start > cfg["max_compute_seconds"]:
            R.write(
                out / "compute_failure.json",
                {"reason": "fixed_elapsed_cap", "completed": r},
            )
            raise RuntimeError("No renewal or larger cap")
        order, info = M.coupling_permutation(z["labels"], z["folds"], m["support"], rng)
        orders.append(order)
        diag.append(info)
        swapped = m | {"a": m["a"][order]}
        null[0, r] = values(banks(swapped, z["labels"], z["folds"]))
        null[1, r] = values(banks(m, yp[r], z["folds"]))
        linked = U.observations(z["coordinates"], cp[r])
        null[2, r] = values(banks(linked, z["labels"], z["folds"]))
    table = []
    for i, row in enumerate(rows):
        item = row.copy()
        for k, name in enumerate(("pairing", "profile", "condition")):
            ref = null[k, :, i]
            invariant = name == "pairing" and row["kind"] in (
                "choice",
                "geometry",
                "product",
            )
            item[name + "_null_mean"] = float(ref.mean())
            item[name + "_range"] = float(np.ptp(ref))
            item[name + "_upper_tail"] = (
                None
                if invariant or np.ptp(ref) == 0
                else float((1 + np.sum(ref >= row["signed_product"])) / (n + 1))
            )
            item[name + "_invariant"] = bool(invariant or np.ptp(ref) == 0)
        table.append(item)
    primary = next(
        r
        for r in table
        if (r["geometry"], r["kind"], r["target"])
        == ("distinct_event", "joint_minus_product", "all")
    )
    terms = geometry_cross(f)
    source_terms = []
    for k, name in enumerate(U.GEOMETRIES):
        source_terms.extend([row | {"geometry": name} for row in M.cross_terms(f[k])])
    supported = m["geometry"][m["support"]]
    negative = {
        name: int((np.linalg.eigvalsh(supported[:, k]).min(-1) < -1e-12).sum())
        for k, name in enumerate(U.GEOMETRIES)
    }
    errors = {
        "operator_additivity": float(
            np.max(np.abs(supported[:, 0] - supported[:, 1] - supported[:, 2]))
        ),
        "field_additivity": float(np.max(np.abs(f[0] - f[1] - f[2]))),
        "four_field_reconstruction": float(
            np.max(np.abs(f[:, :, :, :, :4].sum(4) - f[:, :, :, :, 4]))
        ),
    }
    raw = next(
        r["signed_product"]
        for r in rows
        if (r["geometry"], r["kind"], r["target"])
        == ("squared", "joint_minus_product", "all")
    )
    composed = sum(
        r["signed_product"]
        for r in terms
        if r["target"] == "all" and r["left"] != "squared" and r["right"] != "squared"
    )
    errors["geometry_contraction_account"] = abs(raw - composed)
    receipt = {
        "primary": primary,
        "elapsed_seconds": time.monotonic() - start,
        "source_authors": len(m["a"]),
        "eligible_per_view": m["support"].sum(0).tolist(),
        "negative_operator_counts": negative,
        "pairing_groups": diag[0]["groups"],
        "singleton_groups": diag[0]["singleton_groups"],
        "movable_authors": diag[0]["movable_authors"],
        "fixed_points_range": [
            min(v["fixed_points"] for v in diag),
            max(v["fixed_points"] for v in diag),
        ],
        "errors": errors,
        "same_agent_not_blind": True,
        "new_neural_API_private_training": 0,
    }
    if max(errors.values()) > cfg["comparison_tolerance"]:
        R.write(out / "compute_failure.json", receipt)
        raise ValueError("Account mismatch, no precision rescue")
    R.archive(
        out / "accounts.npz",
        geometry=m["geometry"],
        support=m["support"],
        a=m["a"],
        fields=f,
        null=null,
        orders=np.asarray(orders),
    )
    R.table(out / "target_accounts.csv", table)
    R.table(out / "geometry_cross_terms.csv", terms)
    R.table(out / "component_cross_terms.csv", source_terms)
    R.write(out / "computation_receipt.json", receipt)
    R.seal(
        out,
        "computed_seal.json",
        (
            "accounts.npz",
            "target_accounts.csv",
            "geometry_cross_terms.csv",
            "component_cross_terms.csv",
            "computation_receipt.json",
        ),
    )
    print(json.dumps(receipt))


def verify(out):
    """Check source/ref geometry by unequal pairs, not centered covariance formulas."""
    cfg, z = R.check(out, ("computed_seal.json",))["config"], R.prepared(out)
    parent = ROOT / R.read(ROOT / cfg["parent"] / "manifest.json")["config"]["parent"]
    with np.load(parent / "construction_accounts.npz", allow_pickle=False) as data:
        yp, cp = data["label_permutations"], data["pairing_permutations"]
    with np.load(out / "accounts.npz", allow_pickle=False) as data:
        saved = {k: data[k] for k in data.files}
    m = U.literal_observations(z["coordinates"], z["conditions"])
    if not np.array_equal(m["support"], saved["support"]):
        raise ValueError("Changed eligibility")
    errors = {
        "source_geometry": float(
            np.max(
                np.abs(m["geometry"][m["support"]] - saved["geometry"][m["support"]])
            )
        ),
        "source_fields": 0.0,
        "reference_statistics": 0.0,
        "reference_invariant": 0.0,
    }
    f = banks(m, z["labels"], z["folds"], True)
    errors["source_fields"] = float(np.max(np.abs(f - saved["fields"])))
    for r in range(cfg["null_replicates"]):
        order = saved["orders"][r]
        if not (
            np.array_equal(z["labels"][order], z["labels"])
            and np.array_equal(z["folds"][order], z["folds"])
            and np.array_equal(m["support"][order], m["support"])
        ):
            raise ValueError("Wrong pairing strata")
        swapped = m | {"a": m["a"][order]}
        all_banks = (
            banks(swapped, z["labels"], z["folds"], True),
            banks(m, yp[r], z["folds"], True),
            banks(
                U.literal_observations(z["coordinates"], cp[r]),
                z["labels"],
                z["folds"],
                True,
            ),
        )
        for k, bank in enumerate(all_banks):
            errors["reference_statistics"] = max(
                errors["reference_statistics"],
                float(np.max(np.abs(values(bank) - saved["null"][k, r]))),
            )
        errors["reference_invariant"] = max(
            errors["reference_invariant"],
            float(
                np.max(
                    np.abs(
                        all_banks[0][:, :, :, :, [1, 2, 5]] - f[:, :, :, :, [1, 2, 5]]
                    )
                )
            ),
        )
    receipt = {
        "all597_reference_banks_replayed": True,
        "errors": errors,
        "same_agent_not_blind": True,
        "independence_not_assumed_for_real_comments": True,
    }
    if max(errors.values()) > cfg["comparison_tolerance"]:
        R.write(out / "verification_failure.json", receipt)
        raise ValueError("One-time account audit failed; no rescue")
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Execute one exclusive stage, never spawn a next research package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](
        args.output_dir
    )


if __name__ == "__main__":
    main()
