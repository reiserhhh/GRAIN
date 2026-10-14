#!/usr/bin/env python3
"""Describe the frozen chain's support failure, without changing its law."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import run_grain_overlap_chain_hm238 as C

M, R, ROOT = C.M, C.R, C.ROOT
PARENT = C.DEFAULT


def pooled_law(bundle):
    """Erase target assignment solely to test unlabelled path availability."""
    cliques = bundle["cliques"].sum(1, keepdims=True)
    pairs = cliques.sum(-1)
    transitions = np.divide(
        cliques,
        pairs[..., None],
        out=np.zeros_like(cliques),
        where=pairs[..., None] > 0,
    )
    return pairs[0], transitions


def main():
    """Run once, preserving every original missing path and cold score."""
    R.check(PARENT, C.STAGES)
    out = PARENT / "support_description_r1"
    out.mkdir(exist_ok=False)
    files = (
        "scripts/describe_grain_overlap_support_hm238.py",
        "docs/GRAIN_OVERLAP_SUPPORT_DESCRIPTION_HM238.md",
    )
    inputs = (
        PARENT / "cold_bundle.npz",
        PARENT / "cold_inputs.jsonl",
        PARENT / "cold_outputs.jsonl",
        PARENT / "summary.json",
    )
    R.write(
        out / "manifest.json",
        {
            "post_outcome_description": True,
            "code": {p: R.digest(ROOT / p) for p in files},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_science_changed": False,
            "prediction_selection": False,
        },
    )
    laws = C.bundles(PARENT / "cold_bundle.npz")
    records = [
        json.loads(s) for s in (PARENT / "cold_inputs.jsonl").read_text().splitlines()
    ]
    rows, prefixes, maxerror = [], [], 0.0
    for i, b in enumerate(laws):
        initial, factors = pooled_law(b)
        for r in records:
            if r["case_key"].startswith("empty"):
                continue
            observed = M.observed_vector(r, 8, 18)
            profile, logmass = M.propagate(b["initial"], b["transitions"], observed)
            _, pooled_logmass = M.propagate(initial, factors, observed)
            exact = C.literal_masses(initial, factors, observed).sum()
            if (exact > 0) != np.isfinite(pooled_logmass):
                raise ValueError("Pooled support arithmetic mismatch")
            if exact > 0:
                maxerror = max(maxerror, abs(float(np.log(exact)) - pooled_logmass))
            rows.append(
                {
                    "law": ("actual", "old_permutation")[i],
                    "case_key": r["case_key"],
                    "profile_chain_supported": bool(np.isfinite(logmass)),
                    "pooled_supported": bool(np.isfinite(pooled_logmass)),
                    "positive_profiles_if_supported": int(np.count_nonzero(profile))
                    if np.isfinite(logmass)
                    else None,
                }
            )
            if not r["case_key"].endswith("/full"):
                continue
            prefix = np.full(18, -1, int)
            for event in sorted(r["events"], key=lambda e: e["position"]):
                pos = event["position"]
                prefix[pos] = event["cell"]
                p, z = M.propagate(b["initial"], b["transitions"], prefix)
                prefixes.append(
                    {
                        "law": ("actual", "old_permutation")[i],
                        "case_key": r["case_key"],
                        "revealed_position": pos,
                        "supported": bool(np.isfinite(z)),
                        "positive_profiles": int(np.count_nonzero(p)),
                    }
                )
    summary = []
    for law in ("actual", "old_permutation"):
        for cohort in ("held", "native"):
            for arm in ("full", "support", "counter"):
                selected = [
                    r
                    for r in rows
                    if r["law"] == law
                    and r["case_key"].startswith(cohort + "/")
                    and r["case_key"].endswith("/" + arm)
                ]
                summary.append(
                    {
                        "law": law,
                        "cohort": cohort,
                        "arm": arm,
                        "cases": len(selected),
                        "profile_supported": sum(
                            r["profile_chain_supported"] for r in selected
                        ),
                        "pooled_supported": sum(
                            r["pooled_supported"] for r in selected
                        ),
                        "pooled_only": sum(
                            r["pooled_supported"] and not r["profile_chain_supported"]
                            for r in selected
                        ),
                    }
                )
    if maxerror > 1e-10:
        raise ValueError("Descriptor alternate arithmetic failed")
    R.table(out / "support_cases.csv", rows)
    R.table(out / "prefix_support.csv", prefixes)
    R.write(
        out / "summary.json",
        {
            "support": summary,
            "direct_logmass_error": maxerror,
            "pooled_law_discards_target_assignment": True,
            "psych_accuracy": False,
            "same_agent_not_blind": True,
        },
    )
    R.seal(
        out,
        "description_seal.json",
        ("support_cases.csv", "prefix_support.csv", "summary.json"),
    )
    print(json.dumps({"support": summary, "max_error": maxerror}), flush=True)


if __name__ == "__main__":
    main()
