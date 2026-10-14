#!/usr/bin/env python3
"""Describe sealed HM235 secondary results;do not repair or replace its primary."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_path_mixture as D
from scripts import describe_grain_observed_path_memory_partial_hm235 as C
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

BASE = C.PARENT / "partial_markov_accounts_r1"
OUT = C.PARENT / "mixture_decomposition_r1"
TARGETS = ("E", "N", "A", "C", "O")
TERMS = ("total", "remaining", "mixture", "remaining_x_mixture", "mixture_x_remaining")


def main():
    """Read existing arrays once,then seal all signed descriptive accounts."""
    C.check(BASE)
    OUT.mkdir(parents=True, exist_ok=False)
    code = (
        "grain_research/construction/math_observed_path_mixture.py",
        "scripts/describe_grain_observed_path_memory_mixture_hm235.py",
        "tests/test_grain_observed_path_mixture.py",
    )
    inputs = (
        BASE / "accounts.npz",
        BASE / "manifest.json",
        BASE / "computed_seal.json",
        BASE / "verification.json",
        C.PARENT / "observations.npz",
        C.PARENT / "compute_failure.json",
    )
    R.write(
        OUT / "manifest.json",
        {
            "code": {name: R.digest(ROOT / name) for name in code},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "post_outcome_description": True,
            "purpose": "mixture covariance and signed interactions;not primary rescue",
            "new_observations_or_fit": False,
            "primary_remains_NOT_COMPUTED": True,
        },
    )
    start = time.perf_counter()
    a = S.selected(
        BASE / "accounts.npz", ("author_mass", "tables", "fields", "reference")
    )
    z = R.prepared(C.PARENT)
    kl_rows, field_rows, covariance_rows = [], [], []
    covariance_error = table_error = field_error = kl_error = 0.0
    crossed = np.empty((2, 5, 5))
    covariance_mass = np.zeros_like(a["tables"][:, :, :, 0])
    for label_law in range(2):
        law_name = ("actual", "old_permutation")[label_law]
        for bank in range(2):
            used = z["bank"] == bank
            for window in range(2):
                value = D.mixture_covariance(
                    a["author_mass"][used, window], z["targets"][label_law, used]
                )
                codes = value["profiles"] @ (1 << np.arange(5))
                for g, code in enumerate(codes):
                    covariance_mass[label_law, bank, window, code] = value[
                        "covariance_mass"
                    ][g]
                table = a["tables"][label_law, bank, window]
                table_error = max(
                    table_error,
                    float(
                        np.max(
                            abs(
                                covariance_mass[label_law, bank, window]
                                - table[1]
                                + table[0]
                            )
                        )
                    ),
                )
                covariance_error = max(covariance_error, value["identity_error"])
                covariance_rows.append(
                    {
                        "label_law": law_name,
                        "bank": bank,
                        "window": window,
                        "covariance_identity_error": value["identity_error"],
                        "profiles_present": len(codes),
                    }
                )
                k = D.kl_decomposition(table[2], table[1], table[0])
                kl_error = max(kl_error, k["identity_error"])
                kl_rows.append(
                    {"label_law": law_name, "bank": bank, "window": window, **k}
                )
        pairs = (((0, 0), (1, 1)), ((0, 1), (1, 0)))
        values = []
        for left, right in pairs:
            f = a["fields"][label_law]
            value = D.field_decomposition(f[left], f[right], a["reference"])
            if value["identity_error"] is None:
                raise ValueError("No joint observed support;do not substitute zeros")
            field_error = max(field_error, value["identity_error"])
            values.append(value["terms"])
            for term, row in zip(TERMS, value["terms"], strict=True):
                for target, number in zip(TARGETS, row, strict=True):
                    field_rows.append(
                        {
                            "label_law": law_name,
                            "left_bank": left[0],
                            "left_window": left[1],
                            "right_bank": right[0],
                            "right_window": right[1],
                            "term": term,
                            "target": target,
                            "signed_probability_field_product": float(number),
                            "reference_mass_coverage": value["reference_mass_coverage"],
                        }
                    )
        crossed[label_law] = np.mean(values, axis=0)
    R.archive(OUT / "accounts.npz", covariance_mass=covariance_mass, crossed=crossed)
    R.table(OUT / "kl_signed_decomposition.csv", kl_rows)
    R.table(OUT / "field_signed_decomposition.csv", field_rows)
    R.table(OUT / "covariance_identity.csv", covariance_rows)
    summary = {
        "post_outcome_description_not_preregistered_new_hypothesis": True,
        "primary_remains_NOT_COMPUTED": True,
        "secondary_science_unchanged": True,
        "field_terms": TERMS,
        "targets": TARGETS,
        "actual_permuted_field_terms_ENACO": crossed.tolist(),
        "mean_KL_accounts": {
            law: {
                key: float(np.mean([r[key] for r in kl_rows if r["label_law"] == law]))
                for key in (
                    "total_KL_nats",
                    "remaining_KL_nats",
                    "mixture_KL_nats",
                    "signed_cross_nats",
                )
            }
            for law in ("actual", "old_permutation")
        },
        "checks": {
            "covariance_identity_error": covariance_error,
            "sealed_table_error": table_error,
            "KL_signed_identity_error": kl_error,
            "field_signed_identity_error": field_error,
            "same_agent_not_blind": True,
        },
        "units": "KL nats and signed probability-field products;not psychological fractions or prediction r",
        "new_model_API_private_training_calls": 0,
        "description_seconds": time.perf_counter() - start,
        "original_scope_complete": False,
    }
    R.write(OUT / "summary.json", summary)
    if max(covariance_error, table_error, field_error, kl_error) > 1e-12:
        raise ValueError("Descriptor identity failure;outputs retained")
    R.seal(
        OUT,
        "descriptor_seal.json",
        (
            "accounts.npz",
            "kl_signed_decomposition.csv",
            "field_signed_decomposition.csv",
            "covariance_identity.csv",
            "summary.json",
        ),
    )
    for group in ("code", "inputs"):
        for name, digest in R.read(OUT / "manifest.json")[group].items():
            if R.digest(ROOT / name) != digest:
                raise ValueError("Changed descriptor provenance:" + name)
    C.check(BASE)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
