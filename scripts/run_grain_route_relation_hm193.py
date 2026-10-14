#!/usr/bin/env python3
"""One fixed source experiment separating routed marginals from pair relations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_route_relation as M
from grain_research.construction import math_transfer_update as T
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_route_relation_hm193.json"
DEFAULT = ROOT / "results/grain_math_response_v1/route_preserving_relation_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ROUTE_RELATION_PLAN_HM193.md",
    "grain_research/construction/math_route_relation.py",
    "scripts/run_grain_route_relation_hm193.py",
    "tests/test_grain_route_relation.py",
    "grain_research/construction/math_transfer_update.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Copy only sealed SOURCE laws/events and pre-existing exchange addresses."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    old = R.check(parent, ("computed_seal.json",))
    receipt = R.read(parent / "audit_receipt.json")
    if (
        receipt["pair_query_cases"] != 256
        or max(
            receipt["direct_event_pair_law_error"],
            receipt["covariance_update_and_query_error"],
        )
        > cfg["audit_tolerance"]
    ):
        raise ValueError("Parent audit did not validate the frozen law")
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        data = {
            k: z[k]
            for k in (
                "source",
                "cells",
                "units",
                "codes",
                "folds",
                "probabilities",
                "prior",
                "reference",
            )
        }
    with np.load(parent / "mechanism_accounts.npz", allow_pickle=False) as z:
        permutations = z["null_permutations"]
        old_laws, old_fields = z["laws"], z["fields"]
    if data["probabilities"].shape != (386, 2, 2, 8):
        raise ValueError("Changed source bank")
    if permutations.shape != (199, 386):
        raise ValueError("Changed prescribed exchange reference")
    names = (
        "manifest.json",
        "observations.npz",
        "mechanism_accounts.npz",
        "prepared_seal.json",
        "computed_seal.json",
        "audit_receipt.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "parent_quarter_indices": old["config"]["quarter_indices"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in names
            },
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **data,
        permutations=permutations,
        old_laws=old_laws,
        old_fields=old_fields,
    )
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "banks": np.bincount(data["folds"]).tolist(),
            "events": int(data["units"].size),
            "profiles": 32,
            "quarters": old["config"]["quarter_indices"],
            "condition": cfg["condition"],
            "semantic_opportunity_actor_volition_not_available": True,
            "new_neural_private_API_application_native_or_training": 0,
            "new_raw_text_or_human_observations": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(R.read(out / "input_quality.json")))


def controls(laws, fields):
    """Check scientific matching constraints, not optimize measurement precision."""
    route_error = 0.0
    for a in (0, 1):
        for b in (0, 1):
            blocks = laws[..., 4 * a : 4 * a + 4, 4 * b : 4 * b + 4]
            for axis in (-1, -2):
                route_error = max(
                    route_error,
                    float(
                        np.max(
                            np.abs(
                                blocks[:, :, 1].sum(axis) - blocks[:, :, 2].sum(axis)
                            )
                        )
                    ),
                )
    return {
        "old_singleton_matching_error": float(
            max(
                np.max(np.abs(laws[:, :, 0].sum(-1) - laws[:, :, 1].sum(-1))),
                np.max(np.abs(laws[:, :, 0].sum(-2) - laws[:, :, 1].sum(-2))),
            )
        ),
        "route_conditional_marginal_error": route_error,
        "target_action_telescope_error": float(
            np.nanmax(np.abs(fields[:, :, 2] - fields[:, :, :2].sum(2)))
        ),
    }


def compute(out):
    """Run three fixed laws;never pick new traits,weights,bins or smoothing."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.perf_counter()
    p, c, f, pi, rho = (
        z[k] for k in ("probabilities", "codes", "folds", "prior", "reference")
    )
    laws, fields, support = M.compile_fields(p, c, f, pi)
    observed = M.transport(fields, support, rho)
    checks = controls(laws, fields)
    checks["unchanged_HM192_law_error"] = float(
        np.max(np.abs(laws[:, :, [0, 2]] - z["old_laws"]))
    )
    checks["unchanged_HM192_target_action_error"] = float(
        np.nanmax(np.abs(fields[:, :, 2] - z["old_fields"]))
    )
    checks["signed_cross_account_error"] = float(
        np.max(
            np.abs(
                observed["crossed_accounts"].sum((1, 2))
                - observed["components"][2]["by_target"]
            )
        )
    )
    kl_rows = []
    for s in (0, 1):
        for v in (0, 1):
            d = M.divergences(laws[s, v])
            kl_rows.append(
                {
                    "bank": s,
                    "view": v,
                    **d,
                    "pythagorean_error": d["total"] - d["within"] - d["routing"],
                }
            )
    checks["kl_pythagorean_error"] = max(abs(d["pythagorean_error"]) for d in kl_rows)
    null_values = []
    for permutation in z["permutations"]:
        if not np.array_equal(c[permutation], c) or not np.array_equal(
            f[permutation], f
        ):
            raise ValueError("Unqualified inherited exchange")
        x = p.copy()
        x[:, :, 1] = p[permutation, :, 1]
        _, nh, ns = M.compile_fields(x, c, f, pi)
        n = M.transport(nh, ns, rho)
        null_values.append([a["by_target"] for a in n["components"]])
    null_values = np.array(null_values)
    result = np.array([a["by_target"] for a in observed["components"]])
    tails = (1 + (null_values >= result).sum(0)) / (1 + len(null_values))
    rows = [
        {
            "component": name,
            "target": cfg["targets"][q],
            "cross_author_disjoint_text": float(result[i, q]),
            "within_bank_disjoint_text": float(a["within_bank_disjoint_text"][q]),
            "coherence_NOT_r": float(a["coherence"][q]),
            "inherited_exchange_mean": float(null_values[:, i, q].mean()),
            "inherited_exchange_upper_rank_NOT_route_conditional_p": float(tails[i, q]),
        }
        for i, (name, a) in enumerate(
            zip(M.COMPONENTS, observed["components"], strict=True)
        )
        for q in range(5)
    ]
    R.table(out / "target_components.csv", rows)
    R.table(
        out / "signed_cross_accounts.csv",
        [
            {
                "target": cfg["targets"][q],
                "first_component": M.COMPONENTS[i],
                "second_component": M.COMPONENTS[j],
                "cross_author_account": float(observed["crossed_accounts"][q, i, j]),
            }
            for q in range(5)
            for i in (0, 1)
            for j in (0, 1)
        ],
    )
    R.table(
        out / "condition_components.csv",
        [
            {
                "component": M.COMPONENTS[i],
                "target": cfg["targets"][q],
                "first_condition": a,
                "second_condition": b,
                "cross_author_account": float(
                    observed["components"][i]["by_condition"][2 * a + b, q]
                ),
            }
            for i in range(3)
            for q in range(5)
            for a in (0, 1)
            for b in (0, 1)
        ],
    )
    R.table(out / "law_information_accounts.csv", kl_rows)
    R.archive(
        out / "mechanism_accounts.npz",
        laws=laws,
        fields=fields,
        support=support,
        null_values=null_values,
        crossed_accounts=observed["crossed_accounts"],
    )
    full_p = T.distributions(z["cells"])
    bundle = M.make_bundle(M.three_laws(T.source_laws(full_p, c, pi)))
    R.write(out / "aggregate_three_law_bundle.json", bundle)
    if max(checks.values()) > cfg["audit_tolerance"]:
        raise ValueError("Scientific comparison control failed;no rescue")
    summary = {
        "packet": "HM193",
        "components": rows,
        "all_five_cross_author_accounts": result.mean(1).tolist(),
        "all_five_signed_cross_accounts": observed["crossed_accounts"].mean(0).tolist(),
        "information_accounts_NOT_psychological_fractions": kl_rows,
        "common_supported_pairs": observed["components"][2]["supported_pair_count"],
        "common_reference_mass": observed["components"][2]["reference_mass"],
        "controls": checks,
        "compute_seconds": time.perf_counter() - start,
        "decision": "KEEP_ROUTING_AND_WITHIN_ROUTE_ACTIONS_SEPARATE_NO_SCORE_PROMOTION",
        "routing_is_conditional_marginal_selection_NOT_pure_choice": True,
        "null_does_not_preserve_routes_NOT_conditional_independence_test": True,
        "historical_source_representation_and_sparse_profiles_retained": True,
        "new_model_API_private_training_native_application_or_cold_CLI": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "target_components.csv",
            "signed_cross_accounts.csv",
            "condition_components.csv",
            "law_information_accounts.csv",
            "mechanism_accounts.npz",
            "aggregate_three_law_bundle.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One direct source-event/block/posterior sum audit,not a precision project."""
    manifest = R.check(out, ("computed_seal.json",))
    z = R.prepared(out)
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        laws, fields, accounts = a["laws"], a["fields"], a["crossed_accounts"]
    errors = {
        "source_event_law": 0.0,
        "route_block": 0.0,
        "direct_target_posterior": 0.0,
        "signed_cross_sum": 0.0,
        "aggregate_query": 0.0,
    }
    labels = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
    for s in (0, 1):
        for v, ids in enumerate(manifest["parent_quarter_indices"]):
            direct = np.zeros((32, 8, 8))
            for u in np.flatnonzero(z["folds"] == s):
                g = z["codes"][u]
                count = np.sum((z["codes"] == g) & (z["folds"] == s))
                w = z["prior"][g] / (count * len(ids) ** 2)
                for a in z["cells"][u, 0, ids]:
                    for b in z["cells"][u, 1, ids]:
                        direct[g, a, b] += w
            errors["source_event_law"] = max(
                errors["source_event_law"],
                float(np.max(np.abs(direct - laws[s, v, 2]))),
            )
            middle = np.zeros_like(direct)
            for g in range(32):
                for ra in (0, 1):
                    for rb in (0, 1):
                        block = direct[g, 4 * ra : 4 * ra + 4, 4 * rb : 4 * rb + 4]
                        t = block.sum()
                        if t:
                            for i in range(4):
                                for j in range(4):
                                    middle[g, 4 * ra + i, 4 * rb + j] = (
                                        block[i].sum() * block[:, j].sum() / t
                                    )
            errors["route_block"] = max(
                errors["route_block"], float(np.max(np.abs(middle - laws[s, v, 1])))
            )
            for a in range(8):
                for b in range(8):
                    after = []
                    for law in laws[s, v]:
                        cell = law[:, a, b]
                        after.append(
                            labels.T @ cell / cell.sum()
                            if cell.sum()
                            else np.full(5, np.nan)
                        )
                    expected = np.array(
                        (after[1] - after[0], after[2] - after[1], after[2] - after[0])
                    )
                    errors["direct_target_posterior"] = max(
                        errors["direct_target_posterior"],
                        float(np.nanmax(np.abs(expected - fields[s, v, :, :, a, b]))),
                    )
    rho = z["reference"]
    for q in range(5):
        total = 0.5 * np.sum(
            rho
            * (
                fields[0, 0, 2, q] * fields[1, 1, 2, q]
                + fields[0, 1, 2, q] * fields[1, 0, 2, q]
            )
        )
        errors["signed_cross_sum"] = max(
            errors["signed_cross_sum"], abs(total - accounts[q].sum())
        )
    bundle = R.read(out / "aggregate_three_law_bundle.json")
    for a in range(8):
        for b in range(8):
            result = M.query_pair(
                bundle,
                [
                    {"event_id": "early", "window": 0, "cell": a},
                    {"event_id": "late", "window": 1, "cell": b},
                ],
            )
            errors["aggregate_query"] = max(
                errors["aggregate_query"],
                float(
                    np.max(
                        np.abs(
                            result["components"][2] - result["components"][:2].sum(0)
                        )
                    )
                ),
            )
    if max(errors.values()) > manifest["config"]["audit_tolerance"]:
        raise ValueError("Finite audit failed;do not rerun or optimize")
    R.write(
        out / "audit_receipt.json",
        {"errors": errors, "query_pairs": 64, "same_agent_alternate_not_blind": True},
    )
    R.seal(out, "verification_seal.json", ("audit_receipt.json",))
    print(json.dumps(R.read(out / "audit_receipt.json")))


def main():
    """Explicit write-once scientific stages,no automatic old queue recovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.output)


if __name__ == "__main__":
    main()
