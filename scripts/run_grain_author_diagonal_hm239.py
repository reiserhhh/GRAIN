#!/usr/bin/env python3
"""Finite source identity/moment discrimination, without prediction tuning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_diagonal as M
from grain_research.construction import math_joint_target_atlas as A
from grain_research.construction import math_target_complement as C
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_author_diagonal_hm239.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_diagonal_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_DIAGONAL_PLAN_HM239.md",
    "grain_research/construction/math_author_diagonal.py",
    "scripts/run_grain_author_diagonal_hm239.py",
    "tests/test_grain_author_diagonal.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "grain_research/construction/math_target_complement.py",
    "scripts/run_grain_observed_family_hm170.py",
)
PARENT_STAGES = ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
STAGES = ("computed_seal.json", "verification_seal.json")
SOURCE_KEYS = ("means", "codes", "folds", "source_indices", "mapping")
NAMES = ("V", "D", "R", "U")
BANKS = ("pooled", "fold0", "fold1")


def selected(path, keys):
    """Decompress only explicitly authorized source arrays, never other NPZ keys."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def prepare(out):
    """Authenticate the old complete source atlas before any new outcome."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, PARENT_STAGES)
    z = selected(parent / "observations.npz", SOURCE_KEYS)
    old = selected(parent / "construction_accounts.npz", ("fields", "fold_fields"))
    if (
        z["means"].shape != (386, 2, 1024)
        or z["mapping"].shape != (1024, 15)
        or len(np.unique(z["source_indices"])) != 386
        or not np.array_equal(np.bincount(z["folds"]), [203, 183])
        or np.bincount(z["codes"], minlength=32).min() < 2
    ):
        raise ValueError("Changed fixed source scope or unavailable full U moment")
    if not np.array_equal(
        z["folds"],
        A.author_folds(
            z["codes"], z["source_indices"], "HM185_source_profile_split_v1"
        ),
    ):
        raise ValueError("Changed inherited author partition")
    C.fixed_basis(z["mapping"])
    quality = {
        "source_authors": 386,
        "views": 2,
        "dimensions": 1024,
        "profile_counts": np.bincount(z["codes"], minlength=32).tolist(),
        "fold_profile_counts": [
            np.bincount(z["codes"][z["folds"] == f], minlength=32).tolist()
            for f in (0, 1)
        ],
        "source_keys_read": list(SOURCE_KEYS),
        "old_account_keys_read": ["fields", "fold_fields"],
        "query_fields_application_labels_rawtext_model_answers_read": False,
        "historical_encoder_neural": True,
        "reference": "uniform over the same32 observed target profiles",
        "new_neural_API_private_application_labels_training": 0,
    }
    out.mkdir(parents=True, exist_ok=False)
    files = [
        parent / p
        for p in (
            "manifest.json",
            "observations.npz",
            "construction_accounts.npz",
            "all_modes.csv",
            "quality.json",
            "prepared_seal.json",
            *PARENT_STAGES,
        )
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "previous_turn": "NO_PROGRESS_EXECUTION:read-only direction review;no new scientific state",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z, **old)
    R.write(out / "quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "quality.json"))
    print(json.dumps(quality), flush=True)


def target_name(mask):
    """Use inherited target addresses, never new construct names."""
    return "+".join(t for q, t in enumerate(A.TARGETS) if mask & (1 << q)) or "common"


def diagonal_summary(account, keep):
    """Keep signed moment sums and counts, including unavailable full estimators."""
    return {
        name: None
        if account[name] is None
        else {
            "sum": float(np.diag(account[name])[keep].sum()),
            "positive": int(np.count_nonzero(np.diag(account[name])[keep] > 0)),
            "count": int(keep.sum()),
        }
        for name in NAMES
    }


def compute(out):
    """Resolve fixed source moments and one declared identity audit, once."""
    cfg = R.check(out)["config"]
    z = R.prepared(out)
    start = time.perf_counter()
    basis, _ = C.fixed_basis(z["mapping"])
    kernels = M.kernel_spaces(z["means"], basis)
    groups = (
        np.ones(386, bool),
        z["folds"] == 0,
        z["folds"] == 1,
    )
    grams = np.full((2, 3, 3, 4, 32, 32), np.nan)
    rows, summaries = [], []
    for o, origin in enumerate(cfg["origins"]):
        for s, space in enumerate(cfg["spaces"]):
            for b, (bank, ix) in enumerate(zip(BANKS, groups, strict=True)):
                a = M.accounts(kernels[o, s][np.ix_(ix, ix)], z["codes"][ix])
                for v, name in enumerate(NAMES):
                    if a[name] is not None:
                        grams[o, s, b, v] = a[name]
                summaries.append(
                    {
                        "origin": origin,
                        "space": space,
                        "bank": bank,
                        "singleton_profiles": a["singleton_profiles"].tolist(),
                        "single": diagonal_summary(a, A.orders() == 1),
                        "joint": diagonal_summary(a, A.orders() >= 2),
                        "all_nonconstant": diagonal_summary(a, A.orders() >= 1),
                    }
                )
                for mask in range(32):
                    rows.append(
                        {
                            "origin": origin,
                            "space": space,
                            "bank": bank,
                            "mask": mask,
                            "order": mask.bit_count(),
                            "targets": target_name(mask),
                            **{
                                name: None
                                if a[name] is None
                                else float(a[name][mask, mask])
                                for name in NAMES
                            },
                        }
                    )
    pairing = M.cyclic_pairing(z["codes"])
    repaired = np.empty((2, 3, 4, 32, 32))
    for o in range(2):
        for s in range(3):
            a = M.accounts(kernels[o, s][:, pairing], z["codes"])
            repaired[o, s] = np.array([a[name] for name in NAMES])
    nulls = np.empty((cfg["null_replicates"], 3, 32))
    rng = np.random.default_rng(cfg["null_seed"])
    for n in range(len(nulls)):
        c = A.permute_profiles(z["codes"], z["folds"], rng)
        for s in range(3):
            nulls[n, s] = np.diag(M.accounts(kernels[0, s], c)["U"])
    cross = np.empty((3, 32, 32))
    opposite = np.empty_like(cross)
    within = np.empty((3, 32))
    for s in range(3):
        f = z["fold_fields"][:, :, :, s]
        cross[s] = f[0].mean(0) @ f[1].mean(0).T
        opposite[s] = (f[0, 0] @ f[1, 1].T + f[0, 1] @ f[1, 0].T) / 2
        within[s] = np.mean([np.diag(grams[0, s, b, 0]) for b in (1, 2)], axis=0)
    null_rows = []
    for s, space in enumerate(cfg["spaces"]):
        for label, keep in (
            ("single", A.orders() == 1),
            ("joint", A.orders() >= 2),
            ("all", A.orders() >= 1),
        ):
            observed = float(np.diag(grams[0, s, 0, 3])[keep].sum())
            values = nulls[:, s, keep].sum(axis=1)
            null_rows.append(
                {
                    "space": space,
                    "group": label,
                    "U_observed": observed,
                    "null_mean": float(values.mean()),
                    "null_min": float(values.min()),
                    "null_max": float(values.max()),
                    "upper_rank": float(
                        (1 + np.count_nonzero(values >= observed)) / 200
                    ),
                }
            )
    origin_error = max(
        float(np.nanmax(abs(grams[0, :, :, n, 1:, 1:] - grams[1, :, :, n, 1:, 1:])))
        for n in (0, 3)
    )
    pairing_v_error = float(np.max(abs(repaired[:, :, 0] - grams[:, :, 0, 0])))
    if max(origin_error, pairing_v_error) > cfg["comparison_tolerance"]:
        raise ValueError("Declared translation/pairing invariance failed")
    seconds = time.perf_counter() - start
    if seconds > cfg["max_cpu_seconds"]:
        raise TimeoutError("Fixed finite computation exceeded cap;no retry")
    R.archive(
        out / "accounts.npz",
        grams=grams,
        kernels=kernels,
        repaired=repaired,
        pairing=pairing,
        nulls=nulls,
        cross=cross,
        opposite=opposite,
        within=within,
    )
    R.table(out / "all_modes.csv", rows)
    R.table(out / "null_groups.csv", null_rows)
    R.table(
        out / "null_modes.csv",
        [
            {
                "space": space,
                "mask": mask,
                "order": mask.bit_count(),
                "targets": target_name(mask),
                "U_observed": float(grams[0, s, 0, 3, mask, mask]),
                "null_mean": float(nulls[:, s, mask].mean()),
                "upper_rank": float(
                    (
                        1
                        + np.count_nonzero(
                            nulls[:, s, mask] >= grams[0, s, 0, 3, mask, mask]
                        )
                    )
                    / 200
                ),
            }
            for s, space in enumerate(cfg["spaces"])
            for mask in range(32)
        ],
    )
    R.table(
        out / "pairing.csv",
        [
            {
                "viewA_author": int(z["source_indices"][i]),
                "viewB_actual_author": int(z["source_indices"][i]),
                "viewB_counterfactual_author": int(z["source_indices"][p]),
                "profile": int(z["codes"][i]),
            }
            for i, p in enumerate(pairing)
        ],
    )
    R.table(
        out / "cross_bank_modes.csv",
        [
            {
                "space": space,
                "mask": mask,
                "order": mask.bit_count(),
                "targets": target_name(mask),
                "old_within_fold_view_product": float(within[s, mask]),
                "old_cross_bank_four_view_product": float(cross[s, mask, mask]),
                "cross_bank_opposite_view_product": float(opposite[s, mask, mask]),
            }
            for s, space in enumerate(cfg["spaces"])
            for mask in range(32)
        ],
    )
    summary = {
        "primary": next(
            r for r in null_rows if r["space"] == "full" and r["group"] == "joint"
        ),
        "moments": summaries,
        "diagonal_terms_by_origin_and_space": grams[:, :, 0, 1, 1, 1].tolist(),
        "U_minus_V_diagonal_by_origin_and_space": (
            grams[:, :, 0, 3, 1, 1] - grams[:, :, 0, 0, 1, 1]
        ).tolist(),
        "origin_nonconstant_V_U_error": origin_error,
        "pairing_V_error": pairing_v_error,
        "pairing_U_joint_shift_by_space": (
            np.diagonal(repaired[0, :, 3] - grams[0, :, 0, 3], axis1=-2, axis2=-1)[
                :, A.orders() >= 2
            ].sum(axis=1)
        ).tolist(),
        "old_cross_bank_positive_modes_by_space": [
            int(np.count_nonzero(np.diag(v)[1:] > 0)) for v in cross
        ],
        "all_targets_and_signed_off_diagonal_Grams_retained": True,
        "CPU_wall_seconds": seconds,
        "U_is_new_source_estimand_not_individual_accuracy": True,
        "D_not_pure_author_fingerprint_or_removable_noise": True,
        "pairing_is_audit_counterfactual_not_new_human_data": True,
        "new_neural_API_private_application_labels_training": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "accounts.npz",
            "all_modes.csv",
            "null_groups.csv",
            "null_modes.csv",
            "pairing.csv",
            "cross_bank_modes.csv",
            "summary.json",
        ),
    )
    print(json.dumps({"primary": summary["primary"], "seconds": seconds}), flush=True)


def verify(out):
    """Use literal author-pair enumeration and independent old-atlas readback."""
    cfg = R.check(out, STAGES[:1])["config"]
    z, saved = (
        R.prepared(out),
        selected(
            out / "accounts.npz",
            ("grams", "kernels", "repaired", "pairing", "cross", "within", "nulls"),
        ),
    )
    groups = (np.ones(386, bool), z["folds"] == 0, z["folds"] == 1)
    literal_error = 0.0
    count = unavailable = 0
    for o in range(2):
        for s in range(3):
            for b, ix in enumerate(groups):
                a = M.literal_diagonal(
                    saved["kernels"][o, s][np.ix_(ix, ix)], z["codes"][ix]
                )
                for n, name in enumerate(NAMES):
                    actual = np.diag(saved["grams"][o, s, b, n])
                    if a[name] is None:
                        if not np.isnan(actual).all():
                            raise ValueError("Missing U full estimator was imputed")
                        unavailable += 32
                    else:
                        literal_error = max(
                            literal_error, float(np.max(abs(actual - a[name])))
                        )
                        count += 32
            paired = M.literal_diagonal(
                saved["kernels"][o, s][:, saved["pairing"]], z["codes"]
            )
            for n, name in enumerate(NAMES):
                literal_error = max(
                    literal_error,
                    float(
                        np.max(abs(np.diag(saved["repaired"][o, s, n]) - paired[name]))
                    ),
                )
                count += 32
    basis, _ = C.fixed_basis(z["mapping"])
    fields = C.split_fields(A.construct(z["means"], z["codes"])["fields"][0], basis)
    fold_fields = C.split_fields(
        A.construct(z["means"], z["codes"], z["folds"])["fields"], basis
    )
    atlas_error = max(
        float(np.max(abs(fields - z["fields"]))),
        float(np.max(abs(fold_fields - z["fold_fields"]))),
    )
    for s in range(3):
        atlas_error = max(
            atlas_error,
            float(
                np.max(
                    abs(
                        fields[0, :, s] @ fields[1, :, s].T - saved["grams"][0, s, 0, 0]
                    )
                )
            ),
        )
    with (ROOT / cfg["parent"] / "all_modes.csv").open() as f:
        for row in csv.DictReader(f):
            s, mask = cfg["spaces"].index(row["space"]), int(row["mask"])
            atlas_error = max(
                atlas_error,
                abs(float(row["signed_inner_product"]) - saved["cross"][s, mask, mask]),
                abs(
                    float(row["within_fold_view_inner_product"])
                    - saved["within"][s, mask]
                ),
            )
    partition_error = float(
        np.nanmax(
            abs(saved["grams"][:, 0] - saved["grams"][:, 1] - saved["grams"][:, 2])
        )
    )
    origin_error = max(
        float(
            np.nanmax(
                abs(
                    saved["grams"][0, :, :, n, 1:, 1:]
                    - saved["grams"][1, :, :, n, 1:, 1:]
                )
            )
        )
        for n in (0, 3)
    )
    pairing_error = float(
        np.max(abs(saved["repaired"][:, :, 0] - saved["grams"][:, :, 0, 0]))
    )
    rng = np.random.default_rng(cfg["null_seed"])
    null_error = 0.0
    for r in range(cfg["null_replicates"]):
        c = A.permute_profiles(z["codes"], z["folds"], rng)
        if r in (0, 99, 198):
            a = M.literal_diagonal(saved["kernels"][0, 0], c)
            null_error = max(
                null_error, float(np.max(abs(a["U"] - saved["nulls"][r, 0])))
            )
            count += 32
    errors = {
        "literal_author_pair_error": literal_error,
        "old_atlas_replay_error": atlas_error,
        "space_partition_error": partition_error,
        "nonconstant_origin_error": origin_error,
        "identity_pairing_V_error": pairing_error,
        "selected_null_literal_error": null_error,
    }
    if max(errors.values()) > cfg["comparison_tolerance"]:
        raise ValueError("Same-agent literal source audit failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            **errors,
            "literal_diagonal_values": count,
            "unavailable_U_values_retained": unavailable,
            "same_agent_not_blind_replication": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps({"PASS": True, **errors, "values": count}), flush=True)


def main():
    """Run immutable finite stages; check only reads existing seals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, STAGES)
        print("ALL_SEALS_PASS")
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
