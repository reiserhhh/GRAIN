#!/usr/bin/env python3
"""One finite cross-author/disjoint-evidence target-update experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_target_atlas as A
from grain_research.construction import math_transfer_update as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_transfer_update_hm192.json"
DEFAULT = ROOT / "results/grain_math_response_v1/transferable_pair_update_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TRANSFER_UPDATE_PLAN_HM192.md",
    "grain_research/construction/math_transfer_update.py",
    "scripts/run_grain_transfer_update_hm192.py",
    "tests/test_grain_transfer_update.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Authenticate source-only numeric cells, identities and inherited splits."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("scored_seal.json", "analysis_seal.json"))
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        source = z["source"]
        cells, units, labels = (
            z[k] for k in ("source_cells", "source_units", "source_targets")
        )
    if cells.shape != (cfg["source_authors"], 2, cfg["source_events_per_window"]):
        raise ValueError("Changed fixed source family")
    if units.shape != cells.shape or labels.shape != (len(cells), 5):
        raise ValueError("Changed event/label addresses")
    for row in units:
        if len(set(row.ravel().tolist())) != row.size:
            raise ValueError("Copied source event across windows")
    quarter = cfg["quarter_indices"]
    if sorted(quarter[0] + quarter[1]) != list(range(17)):
        raise ValueError("Missing/overlapping prescribed event quarters")
    codes = A.profile_codes(labels)
    folds = A.author_folds(codes, source, cfg["fold_salt"])
    p = np.stack([M.distributions(cells[:, :, ids]) for ids in quarter], axis=1)
    prior = np.bincount(codes, minlength=32) / len(cells)
    full = M.distributions(cells)
    reference = full[:, 0].T @ full[:, 1] / len(cells)
    names = (
        "manifest.json",
        "observations.npz",
        "prepared_seal.json",
        "scored_seal.json",
        "analysis_seal.json",
    )
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in names
            },
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source=source,
        cells=cells,
        units=units,
        codes=codes,
        folds=folds,
        probabilities=p,
        prior=prior,
        reference=reference,
    )
    counts = np.stack([np.bincount(codes[folds == s], minlength=32) for s in (0, 1)])
    R.write(
        out / "input_quality.json",
        {
            "source_authors": len(cells),
            "fold_authors": np.bincount(folds).tolist(),
            "profile_counts_by_fold": counts.tolist(),
            "minimum_profile_count": int(counts.min()),
            "quarter_sizes": [len(q) for q in quarter],
            "distinct_source_events": int(units.size),
            "quarter_event_overlap": 0,
            "pair_reference_uses_labels": False,
            "historical_encoder_overlap_retained": True,
            "new_neural_private_API_training_native_or_application_labels": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(R.read(out / "input_quality.json")))


def finite(x):
    """Represent unavailable scalar diagnostics as null, not scientific zeros."""
    return float(x) if np.isfinite(x) else None


def compute(out):
    """Compare shared update actions, not optimized final personality scores."""
    cfg = R.check(out)["config"]
    z, start = R.prepared(out), time.perf_counter()
    p, c, f, pi, rho = (
        z[k] for k in ("probabilities", "codes", "folds", "prior", "reference")
    )
    laws, h, support = M.compile_fields(p, c, f, pi)
    observed = M.transport(h, support, rho)
    rng = np.random.default_rng(cfg["null_seed"])
    nulls, permutations, mass_errors = [], [], []
    for _ in range(cfg["null_replicates"]):
        mixed, permutation = M.permute_late(p, c, f, rng)
        nl, nh, ns = M.compile_fields(mixed, c, f, pi)
        nulls.append(M.transport(nh, ns, rho))
        permutations.append(permutation)
        mass_errors.append(
            max(
                float(np.max(np.abs(nl.sum(-1) - laws.sum(-1)))),
                float(np.max(np.abs(nl.sum(-2) - laws.sum(-2)))),
            )
        )
    null_values = np.array([np.r_[a["primary"], a["by_target"]] for a in nulls])
    values = np.r_[observed["primary"], observed["by_target"]]
    tails = (1 + (null_values >= values).sum(0)) / (1 + len(null_values))
    rows = []
    for q, name in enumerate(cfg["targets"]):
        rows.append(
            {
                "target": name,
                "cross_author_cross_text": float(observed["by_target"][q]),
                "within_bank_disjoint_text": float(
                    observed["within_bank_disjoint_text"][q]
                ),
                "cross_bank_same_text_pattern": float(
                    observed["cross_bank_same_text_pattern"][q]
                ),
                "coherence": finite(observed["coherence"][q]),
                "null_mean": float(null_values[:, q + 1].mean()),
                "null_upper_rank": float(tails[q + 1]),
                "secondary_unadjusted": True,
            }
        )
    R.table(out / "target_transport.csv", rows)
    R.table(
        out / "condition_transport.csv",
        [
            {
                "first_condition": a,
                "second_condition": b,
                "target": cfg["targets"][q],
                "cross_author_cross_text": float(
                    observed["by_condition"][2 * a + b, q]
                ),
                "reference_mass": float(
                    rho[4 * a : 4 * a + 4, 4 * b : 4 * b + 4].sum()
                ),
                "secondary_unadjusted": True,
            }
            for a in (0, 1)
            for b in (0, 1)
            for q in range(5)
        ],
    )
    action_rows = []
    for bank in (0, 1):
        for v in (0, 1):
            action = M.update_actions(laws[bank, v])
            for q, name in enumerate(cfg["targets"]):
                for a in range(8):
                    for b in range(8):
                        action_rows.append(
                            {
                                "bank": bank,
                                "experiment": v,
                                "target": name,
                                "first_cell": a,
                                "second_cell": b,
                                "supported": bool(action["supported"][a, b]),
                                "before": finite(action["before"][1, q, a]),
                                "product_after": finite(action["after"][0, q, a, b]),
                                "same_after": finite(action["after"][1, q, a, b]),
                                "relation_increment": finite(
                                    action["relation"][q, a, b]
                                ),
                            }
                        )
    R.table(out / "update_actions.csv", action_rows)
    R.table(
        out / "matched_reference.csv",
        [
            {
                "replicate": r,
                "primary": float(null_values[r, 0]),
                **{
                    name: float(null_values[r, q + 1])
                    for q, name in enumerate(cfg["targets"])
                },
                "fixed_authors": int(np.sum(permutations[r] == np.arange(len(p)))),
                "supported_reference_mass": nulls[r]["reference_mass"],
            }
            for r in range(len(null_values))
        ],
    )
    R.archive(
        out / "mechanism_accounts.npz",
        laws=laws,
        fields=h,
        support=support,
        null_values=null_values,
        null_permutations=np.array(permutations),
        signed_cross_gram=observed["signed_cross_gram"],
        by_condition=observed["by_condition"],
    )
    pooled_p = M.distributions(z["cells"])
    bundle_law = M.source_laws(pooled_p, c, pi)
    R.write(out / "aggregate_update_bundle.json", M.make_bundle(bundle_law))
    bundle_actions = M.update_actions(bundle_law)
    relation_rms = float(np.sqrt(np.nanmean(bundle_actions["relation"] ** 2)))
    decision = (
        "CANDIDATE_COMMON_UPDATE_ACTION"
        if values[0] > 0 and tails[0] <= 0.05
        else "NO_COMMON_UPDATE_BANK_PROMOTION_KEEP_LOCAL_RELATIONS"
    )
    summary = {
        "packet": "HM192",
        "scientific_decision": decision,
        "primary": float(values[0]),
        "primary_null_mean": float(null_values[:, 0].mean()),
        "primary_upper_rank": float(tails[0]),
        "target_results": rows,
        "common_supported_pairs": observed["supported_pair_count"],
        "common_supported_reference_mass": observed["reference_mass"],
        "null_marginal_error": max(mass_errors),
        "singleton_error": float(
            max(
                np.max(np.abs(laws[:, :, 0].sum(-1) - laws[:, :, 1].sum(-1))),
                np.max(np.abs(laws[:, :, 0].sum(-2) - laws[:, :, 1].sum(-2))),
            )
        ),
        "aggregate_update_relation_rms": relation_rms,
        "signed_cross_gram_eigenvalues": np.linalg.eigvalsh(
            observed["signed_cross_gram"]
        ).tolist(),
        "null_fixed_authors_range": [
            int(np.min(np.sum(np.array(permutations) == np.arange(len(p)), axis=1))),
            int(np.max(np.sum(np.array(permutations) == np.arange(len(p)), axis=1))),
        ],
        "compute_seconds": time.perf_counter() - start,
        "not_prediction_accuracy_native_mediation_or_psychological_factor_count": True,
        "no_answer_fit_new_neural_API_private_training": True,
        "no_new_cold_CLI_or_480_replay": True,
        "original_scope_complete": False,
    }
    if (
        max(summary["singleton_error"], summary["null_marginal_error"])
        > cfg["audit_tolerance"]
    ):
        raise ValueError("Observation-law control failed; no tolerance rescue")
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "target_transport.csv",
            "condition_transport.csv",
            "update_actions.csv",
            "matched_reference.csv",
            "mechanism_accounts.npz",
            "aggregate_update_bundle.json",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One different direct pair-count audit, enough to validate the decision."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    z = R.prepared(out)
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        laws, fields = a["laws"], a["fields"]
    law_error, action_error, queries = 0.0, 0.0, 0
    for bank in (0, 1):
        for v, addresses in enumerate(cfg["quarter_indices"]):
            direct = np.zeros((32, 8, 8))
            for g in range(32):
                ids = np.flatnonzero((z["folds"] == bank) & (z["codes"] == g))
                for u in ids:
                    early, late = (
                        z["cells"][u, 0, addresses],
                        z["cells"][u, 1, addresses],
                    )
                    weight = z["prior"][g] / (len(ids) * len(addresses) ** 2)
                    for a in early:
                        for b in late:
                            direct[g, a, b] += weight
            law_error = max(law_error, float(np.max(np.abs(direct - laws[bank, v, 1]))))
            acts = M.update_actions(laws[bank, v])
            gamma_step = np.divide(
                acts["gamma_numerator"],
                acts["pair_mass"][:, None],
                out=np.full_like(acts["increment"], np.nan),
                where=acts["pair_mass"][:, None] > 0,
            )
            action_error = max(
                action_error, float(np.nanmax(np.abs(gamma_step - acts["increment"])))
            )
            bundle = M.make_bundle(laws[bank, v])
            for a in range(8):
                for b in range(8):
                    row = M.query_pair(
                        bundle,
                        [
                            {"event_id": "early", "window": 0, "cell": a},
                            {"event_id": "late", "window": 1, "cell": b},
                        ],
                    )
                    if row["supported"]:
                        action_error = max(
                            action_error,
                            float(
                                np.max(
                                    np.abs(
                                        row["relation_increment"]
                                        - fields[bank, v, :, a, b]
                                    )
                                )
                            ),
                        )
                    queries += 1
    if max(law_error, action_error) > cfg["audit_tolerance"]:
        raise ValueError("Alternate count/update audit failed")
    R.write(
        out / "audit_receipt.json",
        {
            "direct_event_pair_law_error": law_error,
            "covariance_update_and_query_error": action_error,
            "pair_query_cases": queries,
            "same_agent_not_blind": True,
            "no_scientific_rerun": True,
        },
    )
    print(json.dumps(R.read(out / "audit_receipt.json")))


def main():
    """Require an explicit finite stage;never rerun an already sealed stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.output)


if __name__ == "__main__":
    main()
