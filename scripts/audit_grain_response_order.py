#!/usr/bin/env python3
"""Independently verify fixed-material context-order metrics and permutations."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.audit_grain_meps_material import verify_outputs
from scripts.audit_grain_response_depth import compare, interval
from scripts.run_grain_vector_mechanism_controls import write_json

MODES, SEEDS = ("block8", "token"), (731, 1731)


def replay_permutation(code, mode, seed, count):
    """Reconstruct the predeclared index order without text or token values."""
    width = 8 if mode == "block8" else 1
    groups = [list(range(i, min(i + width, count))) for i in range(0, count, width)]
    salt = hashlib.sha256(f"RC3|{code}|{mode}|{seed}".encode()).digest()
    order = np.random.default_rng(int.from_bytes(salt[:8], "big")).permutation(
        len(groups)
    )
    if len(groups) > 1 and np.array_equal(order, np.arange(len(groups))):
        order = np.roll(order, 1)
    flattened = [j for i in order for j in groups[i]]
    return material_hash(flattened), len(groups)


def audit(run, parent_dir):
    """Check paired evidence and its whole arithmetic ledger, not just positive means."""
    result, parent = verify_outputs(run), verify_outputs(parent_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        result["status"] != "COMPLETE"
        or parent["status"] != "COMPLETE"
        or result["parent_identity"] != manifest["identity"]
        or result["rc1_identity"] != parent["parent_identity"]
        or manifest["rc1_identity"] != parent["parent_identity"]
        or manifest["identity"]
        != material_hash({k: v for k, v in manifest.items() if k != "identity"})
        or tuple(manifest["modes"]) != MODES
        or tuple(manifest["seeds"]) != SEEDS
    ):
        raise ValueError("Changed or incomplete experiment identity")
    material = json.loads((run / "private_material_manifest.json").read_text())
    if (
        material
        != json.loads((parent_dir / "private_material_manifest.json").read_text())
        or material_hash(material["cases"]) != material["identity"]
    ):
        raise ValueError("Changed frozen responses/donors")
    cases = pd.DataFrame(material["cases"]).set_index("participant_code")
    codes = sorted(cases.index)
    metrics = pd.read_csv(run / "private_order_metrics.csv")
    meta = pd.read_csv(run / "private_order_manifest.csv")
    conditions = [r + "_native" for r in ("actual", "matched")] + [
        f"{r}_{m}_{s}" for r in ("actual", "matched") for m in MODES for s in SEEDS
    ]
    if (
        cases.index.has_duplicates
        or len(metrics) != 10 * len(cases)
        or metrics.duplicated(["participant_code", "condition"]).any()
        or set(zip(metrics.participant_code, metrics.condition))
        != {(c, q) for c in codes for q in conditions}
        or result["participants"] != len(cases)
        or result["forwards"] != 6 + 10 * len(cases)
    ):
        raise ValueError("Incorrect person/condition coverage")
    if (
        not np.isfinite(
            metrics.drop(columns=["participant_code", "condition"]).to_numpy()
        ).all()
        or not metrics.input_embedding_max_error.eq(0).all()
        or not (metrics[["mean_logp", "tail_logp"]] <= 0).all().all()
    ):
        raise ValueError("Invalid scalar or current-input invariant")
    for row in metrics.itertuples():
        if row.response_tokens != cases.loc[row.participant_code, "response_tokens"]:
            raise ValueError("Response tokens differ from frozen material")
    expected_meta = {
        (c, r, m, s)
        for c in codes
        for r in ("actual", "matched")
        for m in MODES
        for s in SEEDS
    }
    if (
        len(meta) != len(expected_meta)
        or meta.duplicated(["participant_code", "role", "mode", "seed"]).any()
        or set(zip(meta.participant_code, meta.role, meta["mode"], meta.seed))
        != expected_meta
    ):
        raise ValueError("Missing or duplicate order metadata")
    if (
        not meta.inverse_max_token_error.eq(0).all()
        or not meta.outside_span_changed.eq(False).all()
        or not meta.isolated_context_tokens.between(8, 768).all()
        or not meta.fixed_boundary_crossing_tokens.ge(0).all()
        or not meta.prefix_tokens.between(1, 768).all()
        or not meta.changed_tokens.ge(0).all()
        or not meta.changed_tokens.le(meta.isolated_context_tokens).all()
    ):
        raise ValueError("Order/boundary invariant failed")
    for row in meta.itertuples():
        owner = (
            row.participant_code
            if row.role == "actual"
            else cases.loc[row.participant_code, "matched_donor_code"]
        )
        expected_hash, groups = replay_permutation(
            owner, row.mode, row.seed, row.isolated_context_tokens
        )
        if (
            row.context_code != owner
            or row.permutation_sha256 != expected_hash
            or row.permutation_groups != groups
            or row.isolated_context_tokens > row.prefix_tokens
            or row.response_tokens != cases.loc[row.participant_code, "response_tokens"]
        ):
            raise ValueError("Donor permutation or length failed replay")
    invariant = [
        "bag_sha256",
        "native_context_order_sha256",
        "altered_context_order_sha256",
        "permutation_sha256",
        "changed_tokens",
        "isolated_context_tokens",
    ]
    grouped = meta.groupby(["context_code", "mode", "seed"])
    if (
        not grouped.size().eq(2).all()
        or not grouped[invariant].nunique().eq(1).all().all()
    ):
        raise ValueError("Same context received recipient-dependent corruption")
    origin = [
        "bag_sha256",
        "native_context_order_sha256",
        "isolated_context_tokens",
        "fixed_boundary_crossing_tokens",
    ]
    if not meta.groupby("context_code")[origin].nunique().eq(1).all().all():
        raise ValueError("Native context identity differs across order conditions")
    old = pd.read_csv(parent_dir / "private_response_metrics.csv").set_index(
        "participant_code"
    )
    effects, rows, replay_error = {}, [], 0.0
    for score in ("mean_logp", "tail_logp"):
        pivot = metrics.pivot(
            index="participant_code", columns="condition", values=score
        ).loc[codes]
        for role in ("actual", "matched"):
            error = float(
                abs(pivot[role + "_native"] - old.loc[codes, role + "_" + score]).max()
            )
            replay_error = max(replay_error, error)
            if error > 1e-5:
                raise ValueError("Native RC1 response scores failed replay")
        native = pivot.actual_native - pivot.matched_native
        effects[score + "/native"] = interval(native)
        for mode in MODES:
            diffs, penalties = [], []
            for seed in SEEDS:
                va, vm = pivot[f"actual_{mode}_{seed}"], pivot[f"matched_{mode}_{seed}"]
                diffs.append(va - vm)
                penalties.append(
                    ((pivot.actual_native - va) + (pivot.matched_native - vm)) / 2
                )
            for seed, difference, penalty in [
                *zip(SEEDS, diffs, penalties),
                ("seed_mean", sum(diffs) / 2, sum(penalties) / 2),
            ]:
                stats = {
                    "actual_minus_matched": interval(difference),
                    "order_interaction": interval(native - difference),
                    "common_degradation": interval(penalty),
                }
                effects[f"{score}/{mode}/{seed}"] = stats
                for name, values in stats.items():
                    rows.append(
                        {
                            "score": score,
                            "mode": mode,
                            "seed": seed,
                            "contrast": name,
                            **values,
                        }
                    )
    compare(result["effects"], effects)
    numerical = json.loads((run / "technical_numerical_gate.json").read_text())
    technical = json.loads((run / "technical_order_gate.json").read_text())
    if len(numerical) != 2 or len(technical) != 8:
        raise ValueError("Incomplete technical controls")
    expected_technical = {
        (lang, m, s) for lang in ("en", "ja") for m in MODES for s in SEEDS
    }
    if {(x["language"], x["mode"], x["seed"]) for x in technical} != expected_technical:
        raise ValueError("Synthetic language/order coverage differs")
    for x in technical:
        expected, groups = replay_permutation(
            x["language"], x["mode"], x["seed"], x["isolated_context_tokens"]
        )
        if x["permutation_sha256"] != expected or x["permutation_groups"] != groups:
            raise ValueError("Synthetic order permutation did not replay")
    if any(
        not x["passed"]
        or x["self_patch_max_logp_error"] != 0
        or x["compact_fixed_mean_logp_difference"] > 0.02
        or x["minimum_site_cosine"] < 0.999
        for x in numerical
    ):
        raise ValueError("Numerical instrument failed")
    if any(
        x["outside_span_changed"] or x["inverse_max_token_error"] != 0
        for x in technical
    ):
        raise ValueError("Synthetic order control failed")
    unique = meta.drop_duplicates(["context_code", "mode", "seed"])
    dose = {}
    for mode in MODES:
        part = unique.loc[unique["mode"] == mode]
        dose[mode] = {
            "contexts": len(cases),
            "unique_conditions": len(part),
            "mean_changed_token_fraction": float(
                (part.changed_tokens / part.isolated_context_tokens).mean()
            ),
            "unchanged_orders": int(part.changed_tokens.eq(0).sum()),
            "contexts_with_identical_orders_across_seeds": int(
                part.groupby("context_code")
                .altered_context_order_sha256.nunique()
                .eq(1)
                .sum()
            ),
        }
    return {
        "status": "COMPLETE_ORDER_SCALAR_AND_ASSIGNMENT_AUDIT",
        "parent_identity": manifest["identity"],
        "rc1_identity": parent["parent_identity"],
        "participants": len(cases),
        "model_rows": len(metrics),
        "metadata_rows": len(meta),
        "forwards": result["forwards"],
        "parent_replay_max_error": replay_error,
        "effects": effects,
        "perturbation_dose": dose,
        "boundary_crossing_contexts": int(
            unique.groupby("context_code")
            .fixed_boundary_crossing_tokens.max()
            .gt(0)
            .sum()
        ),
        "new_model_forwards": 0,
        "raw_text_reopened": False,
        "labels_read": False,
        "scope": "scalar/hash/permutation audit, not independent neural replay or semantic validity",
    }, rows


def main():
    """Create an independent audit without changing the original sealed run."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(x for x in a.output_dir.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    result, rows = audit(a.run_dir, a.parent_dir)
    pd.DataFrame(rows).to_csv(a.output_dir / "order_effects.csv", index=False)
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in [
            Path(__file__),
            ROOT / "scripts/audit_grain_response_depth.py",
            ROOT / "scripts/audit_grain_meps_material.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ]:
            z.write(path, path.relative_to(ROOT))
    write_json(a.output_dir / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
