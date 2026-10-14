#!/usr/bin/env python3
"""Post-outcome scalar/traceless accounts of every fixed HM187 operator."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_condition_shape as M
from scripts import run_grain_observed_family_hm170 as R

PARENT = ROOT / "results/grain_math_response_v1/condition_shape_response_r1"
PLAN = "docs/GRAIN_CONDITION_SHAPE_ISOTROPIC_DESCRIPTION_HM187.md"
DEFAULT = PARENT / "post_outcome_geometry_r1"


def split(values):
    """Retain both orthogonal symmetric-matrix components in the fixed metric."""
    a = np.asarray(values, float)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError("Finite square operators required")
    isotropic = (
        np.trace(a, axis1=-2, axis2=-1)[..., None, None]
        * np.eye(a.shape[-1])
        / a.shape[-1]
    )
    return isotropic, a - isotropic


def run(out):
    """Describe existing results once without changing any original scientific result."""
    m = R.check(
        PARENT, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    z = R.prepared(PARENT)
    with np.load(PARENT / "construction_accounts.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    cfg = m["config"]
    out.mkdir(parents=True, exist_ok=False)
    files = (
        PLAN,
        str(Path(__file__).relative_to(ROOT)),
        "tests/test_grain_shape_isotropic_description.py",
    )
    inputs = {
        str((PARENT / p).relative_to(ROOT)): R.digest(PARENT / p)
        for p in (
            "manifest.json",
            "observations.npz",
            "construction_accounts.npz",
            "compiled_seal.json",
            "verification_seal.json",
        )
    }
    R.write(
        out / "manifest.json",
        {
            "code": m["code"] | {p: R.digest(ROOT / p) for p in files},
            "inputs": inputs,
            "config": {
                "packet": "HM187_POSTOUTCOME_DESCRIPTION",
                "original_primary_unchanged": True,
                "new_confirmatory_claim": False,
            },
        },
    )
    fparts = split(saved["fold_fields"])
    oparts = split(saved["operators"])
    n = cfg["null_replicates"]
    label_null = np.empty((2, n, 24))
    pairing_null = np.empty_like(label_null)
    author_null = np.empty((2, n, 4))
    author_refs = [
        [M.author_comparison(part[:, :, k], saved["counts"][:, 1, 1]) for k in range(4)]
        for part in oparts
    ]
    for r in range(n):
        cp = saved["pairing_permutations"][r]
        paired = M.moments(z["coordinates"], cp)["operators"]
        paired_parts = split(paired)
        for s in (0, 1):
            f_label = M.fields(oparts[s], saved["label_permutations"][r], z["folds"])[
                "fields"
            ]
            f_pair = M.fields(paired_parts[s], z["labels"], z["folds"])["fields"]
            label_null[s, r] = [v["signed_product"] for v in M.statistics(f_label)]
            pairing_null[s, r] = [v["signed_product"] for v in M.statistics(f_pair)]
            order = saved["author_permutations"][r]
            for k, ref in enumerate(author_refs[s]):
                self_values = np.sum(
                    oparts[s][:, 0, k] * oparts[s][order, 1, k], axis=(-1, -2)
                )
                author_null[s, r, k] = np.mean(
                    (self_values - ref["stranger"])[ref["support"]]
                )
    target_rows, author_rows = [], []
    full = M.statistics(saved["fold_fields"])
    target_error = author_error = trace_error = 0.0
    part_stats = [M.statistics(part) for part in fparts]
    for s, name in enumerate(("isotropic", "traceless")):
        for i, row in enumerate(part_stats[s]):
            target_rows.append(
                row
                | {
                    "component": name,
                    "profile_upper_tail": float(
                        (
                            1
                            + np.count_nonzero(
                                label_null[s, :, i] >= row["signed_product"]
                            )
                        )
                        / (n + 1)
                    ),
                    "pairing_upper_tail": float(
                        (
                            1
                            + np.count_nonzero(
                                pairing_null[s, :, i] >= row["signed_product"]
                            )
                        )
                        / (n + 1)
                    ),
                }
            )
        for k, ref in enumerate(author_refs[s]):
            author_rows.append(
                {
                    "kind": M.KINDS[k],
                    "component": name,
                    "self_minus_stranger": ref["primary"],
                    "upper_tail": float(
                        (1 + np.count_nonzero(author_null[s, :, k] >= ref["primary"]))
                        / (n + 1)
                    ),
                }
            )
    for i, row in enumerate(full):
        target_error = max(
            target_error,
            abs(
                row["signed_product"]
                - part_stats[0][i]["signed_product"]
                - part_stats[1][i]["signed_product"]
            ),
        )
    for k in range(4):
        ref = M.author_comparison(saved["operators"][:, :, k], saved["counts"][:, 1, 1])
        author_error = max(
            author_error,
            abs(
                ref["primary"]
                - author_refs[0][k]["primary"]
                - author_refs[1][k]["primary"]
            ),
        )
    f = saved["fold_fields"]
    for k in range(4):
        for q in list(range(5)) + [None]:
            ids = [q] if q is not None else list(range(5))
            literal = float(
                np.sum(
                    np.trace(f[0, k][:, ids].mean(0), axis1=-2, axis2=-1)
                    * np.trace(f[1, k][:, ids].mean(0), axis1=-2, axis2=-1)
                )
                / 15
            )
            i = 6 * k + (q if q is not None else 5)
            trace_error = max(
                trace_error, abs(literal - part_stats[0][i]["signed_product"])
            )
    errors = {
        "target_additivity": target_error,
        "author_additivity": author_error,
        "trace_product_replay": trace_error,
        "operator_reconstruction": float(
            np.max(np.abs(oparts[0] + oparts[1] - saved["operators"]))
        ),
    }
    if max(errors.values()) > cfg["comparison_tolerance"]:
        raise ValueError(
            "Fixed geometry account failed; retain outputs/no precision rescue"
        )
    R.table(out / "target_components.csv", target_rows)
    R.table(out / "author_components.csv", author_rows)
    R.archive(
        out / "accounts.npz",
        label_null=label_null,
        pairing_null=pairing_null,
        author_null=author_null,
    )
    receipt = {
        "errors": errors,
        "all_target_rows": len(target_rows),
        "all_author_rows": len(author_rows),
        "null_assignments": "inherited_199_each_unchanged",
        "post_outcome_not_confirmatory": True,
        "same_agent_not_blind": True,
        "all_target_summaries": [r for r in target_rows if r["target"] == "all"],
        "author_shape": [r for r in author_rows if r["kind"] == "shape"],
    }
    R.write(out / "summary.json", receipt)
    R.seal(
        out,
        "prepared_seal.json",
        (
            "summary.json",
            "target_components.csv",
            "author_components.csv",
            "accounts.npz",
        ),
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    run(p.parse_args().output_dir)
