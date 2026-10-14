#!/usr/bin/env python3
"""One full-vector relational evidence constructor, not a model-selection loop."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_relational_diagnostic_update as M
from grain_research.construction.math_joint_target_atlas import characters
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_relational_diagnostic_update_hm209.json"
DEFAULT = ROOT / "results/grain_math_response_v1/relational_diagnostic_update_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_RELATIONAL_DIAGNOSTIC_UPDATE_PLAN_HM209.md",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_relational_diagnostic_update_hm209.py",
    "tests/test_grain_relational_diagnostic_update.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "scripts/run_grain_observed_family_hm170.py",
)
FRAME_NAMES = (
    "full",
    "view0_remove_selected",
    "view0_remove_control",
    "view0_selected",
    "view0_control",
    "view1_remove_selected",
    "view1_remove_control",
    "view1_selected",
    "view1_control",
)


def slots(z, author, frame):
    """Keep the companion full; do not invent a model-read companion intervention."""
    groups = [np.arange(12), np.arange(12)]
    if frame:
        view, arm = divmod(frame - 1, 4)
        selected, control = z["selected"][author, view], z["control"][author, view]
        choices = (
            groups[view][~np.isin(groups[view], selected)],
            groups[view][~np.isin(groups[view], control)],
            selected,
            control,
        )
        groups[view] = choices[arm]
    return groups


def record(z, author, frame):
    """Expose only cached event addresses, vectors and inherited conditions."""
    return {
        "frame_key": f"{z['author_ids'][author]}/{FRAME_NAMES[frame]}",
        "windows": [
            [
                {
                    "event_id": f"{z['author_ids'][author]}:{v}:{t}",
                    "vector": z["panel_vectors"][author, v, t].tolist(),
                    "condition": int(z["panel_conditions"][author, v, t]),
                }
                for t in group
            ]
            for v, group in enumerate(slots(z, author, frame))
        ],
    }


def prepare(out):
    """Bind exact HM208 events and a previously frozen source scale before results."""
    cfg = R.read(ROOT / CONFIG)
    parent, variance_parent = ROOT / cfg["parent"], ROOT / cfg["variance_parent"]
    R.check(
        parent,
        (
            "prepared_seal.json",
            "built_seal.json",
            "analyzed_seal.json",
            "verified_seal.json",
        ),
    )
    R.check(variance_parent, ("compiled_seal.json", "verification_seal.json"))
    z, reference = R.prepared(parent), R.read(variance_parent / "source_reference.json")
    scale_obs = R.prepared(variance_parent)
    if (
        z["source_means"].shape != (386, 2, 1024)
        or z["panel_vectors"].shape != (24, 2, 12, 1024)
        or not np.array_equal(z["source_indices"], scale_obs["source_indices"])
        or not np.array_equal(z["source_means"], scale_obs["source_vectors"].mean(2))
        or reference["variance"] != 0.05525111588037168
    ):
        raise ValueError("Different source/evidence/scale; do not silently re-fit")
    points = np.array(
        [[M.cold_point(record(z, u, f), 1024) for f in range(9)] for u in range(24)]
    )
    files = [
        parent / "observations.npz",
        parent / "manifest.json",
        parent / "verified_seal.json",
        variance_parent / "source_reference.json",
        variance_parent / "observations.npz",
        variance_parent / "manifest.json",
        variance_parent / "compiled_seal.json",
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        **z,
        panel_points=points,
        variance=np.array(reference["variance"]),
    )
    R.write(out / "cold_example.json", record(z, 0, 0))
    quality = {
        "source_authors": 386,
        "source_folds": np.bincount(z["folds"]).tolist(),
        "application_authors": 24,
        "cold_frames": 216,
        "dimensions": 1024,
        "scale": reference["variance"],
        "scale_scope": "reused_label_free_HM204_all386_source_calibration_not_new_nested_CV",
        "prior": "uniform32profiles_NOT_HM204_uniformauthors",
        "native_answers_application_labels_new_neural_API_private_training": 0,
        "conditions_choice_state_full_vectors_retained": True,
        "historical_encoder_overlap_N_selection_retained": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_example.json", "input_quality.json"),
    )
    print(json.dumps(quality))


def banks(out):
    """Load the two frozen author-fold source instruments without rebuilding."""
    with np.load(out / "source_banks.npz", allow_pickle=False) as z:
        return [
            {k[3:]: z[k] for k in z.files if k.startswith(f"b{b}_")} for b in (0, 1)
        ]


def build(out):
    """Compile source pairing laws, before opposite-source or application scoring."""
    R.check(out, ("prepared_seal.json",))
    z, start = R.prepared(out), time.perf_counter()
    bs = [
        M.compile_bank(
            z["source_means"][z["folds"] == b],
            z["codes"][z["folds"] == b],
            z["variance"],
        )
        for b in (0, 1)
    ]
    R.archive(
        out / "source_banks.npz",
        **{f"b{b}_{k}": v for b in (0, 1) for k, v in bs[b].items()},
    )
    receipt = {
        "all_profiles_supported": True,
        "profile_counts": [np.bincount(b["codes"], minlength=32).tolist() for b in bs],
        "likelihoods": list(M.LAWS),
        "application_or_native_inputs_to_build": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "build_summary.json", receipt)
    R.seal(out, "built_seal.json", ("source_banks.npz", "build_summary.json"))
    print(json.dumps(receipt))


def log_scores(result, codes):
    """Proper signed diagnostics on opposite SOURCE labels, not model selection."""
    lp, g = result["log_profile"], np.asarray(codes, int)
    joint = lp[np.arange(len(g)), :, g]
    binary = []
    for q in range(5):
        from scipy.special import logsumexp

        sides = np.stack(
            [logsumexp(lp[..., M.BITS[:, q] == bit], -1) for bit in (0, 1)], -1
        )
        binary.append(sides[np.arange(len(g)), :, (g >> q) & 1])
    return np.stack(binary, -1), joint


def analyze(out):
    """Check reusable target dependence; never rescue the sign by tuning scale."""
    cfg = R.check(out, ("built_seal.json",))["config"]
    z, bs, start = R.prepared(out), banks(out), time.perf_counter()
    held, application, fields, summaries, source_rows, app_rows = [], [], [], [], [], []
    for b in (0, 1):
        ix = z["folds"] != b
        r = M.read(bs[b], z["source_means"][ix])
        logs, full_logs = log_scores(r, z["codes"][ix])
        change, full_change = logs[:, 0] - logs[:, 1], full_logs[:, 0] - full_logs[:, 1]
        counts = np.bincount(z["codes"][ix], minlength=32)
        balanced = 1 / (32 * counts[z["codes"][ix]])
        for u, (source, code) in enumerate(
            zip(z["source_indices"][ix], z["codes"][ix], strict=True)
        ):
            for q, name in enumerate(M.TARGETS):
                source_rows.append(
                    {
                        "source_bank": b,
                        "opposite_source_author": int(source),
                        "target": name,
                        "inherited_bit": int((code >> q) & 1),
                        "paired_log_score": float(logs[u, 0, q]),
                        "marginal_log_score": float(logs[u, 1, q]),
                        "signed_change": float(change[u, q]),
                    }
                )
            source_rows.append(
                {
                    "source_bank": b,
                    "opposite_source_author": int(source),
                    "target": "joint32",
                    "inherited_bit": int(code),
                    "paired_log_score": float(full_logs[u, 0]),
                    "marginal_log_score": float(full_logs[u, 1]),
                    "signed_change": float(full_change[u]),
                }
            )
        p = M.read(bs[b], z["panel_points"].reshape(-1, 2, 1024))
        for u, f, q in np.ndindex(24, 9, 5):
            pos = u * 9 + f
            app_rows.append(
                {
                    "source_bank": b,
                    "application_author": str(z["author_ids"][u]),
                    "frame": FRAME_NAMES[f],
                    "target": M.TARGETS[q],
                    "paired_reading": float(p["targets"][pos, 0, q]),
                    "marginal_reading": float(p["targets"][pos, 1, q]),
                    "log_odds_update": float(p["target_log_odds_update"][pos, q]),
                }
            )
        f = M.read(bs[b], z["panel_points"][:, 0], fields=True)
        fields.append(f["relational_target_field"])
        held.append(r)
        application.append(p)
        summaries.append(
            {
                "source_bank": b,
                "opposite_source_authors": int(ix.sum()),
                "target_log_score_mean_change": change.mean(0).tolist(),
                "target_log_score_profile_balanced_change": (
                    balanced @ change
                ).tolist(),
                "target_log_score_positive_authors": (change > 0).sum(0).tolist(),
                "joint32_log_score_mean_change": float(full_change.mean()),
                "joint32_log_score_profile_balanced_change": float(
                    balanced @ full_change
                ),
                "held_target_probability_change_RMS": np.sqrt(
                    np.mean((r["targets"][:, 0] - r["targets"][:, 1]) ** 2, axis=0)
                ).tolist(),
                "cold_target_update_RMS": np.sqrt(
                    np.mean(p["target_log_odds_update"] ** 2, axis=0)
                ).tolist(),
                "cold_profile_factor_common_RMS": float(
                    np.sqrt(np.mean(p["factor_characters"][:, 0] ** 2))
                ),
                "cold_profile_factor_nonconstant_RMS": float(
                    np.sqrt(np.mean(p["factor_characters"][:, 1:] ** 2))
                ),
                "source_single_author_profiles": int(
                    (np.bincount(bs[b]["codes"], minlength=32) == 1).sum()
                ),
            }
        )
        if time.perf_counter() - start > cfg["max_stage_seconds"]:
            R.write(out / "analysis_failure.json", {"stage": "finite_cap", "bank": b})
            raise RuntimeError("Finite cap reached; no automatic rerun")
    R.table(out / "held_source_log_scores.csv", source_rows)
    R.table(out / "cold_readings.csv", app_rows)
    R.archive(
        out / "analysis.npz",
        **{f"held{b}_{k}": v for b in (0, 1) for k, v in held[b].items()},
        **{f"panel{b}_{k}": v for b in (0, 1) for k, v in application[b].items()},
        relational_fields=np.array(fields),
    )
    field_product = np.mean(np.sum(fields[0] * fields[1], axis=-1), axis=(0, 2))
    summary = {
        "banks": summaries,
        "cross_source_relational_field_signed_products_by_target": field_product.tolist(),
        "targets": list(M.TARGETS),
        "source_authors": 386,
        "application_authors": 24,
        "no_new_predictions_selected_or_tuned": True,
        "new_capability": "positive_full_vector_relation_update_and_target_local_fields_on_SAME_evidence",
        "maturity": "Bayes_and_mixture_dependence_known_HM176177_categorical_ancestry",
        "qualification": "declared_reused_Gaussian_channel_not_natural_author_law_or_neural_mechanism_recovery",
        "original_scope_complete": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "held_source_log_scores.csv",
            "cold_readings.csv",
            "analysis.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """Audit the new instrument once; do not optimize numerical agreement."""
    cfg = R.check(out, ("analyzed_seal.json",))["config"]
    z, bs, start = R.prepared(out), banks(out), time.perf_counter()
    error, checks = 0.0, 0
    with np.load(out / "analysis.npz", allow_pickle=False) as arrays:
        a = {k: arrays[k] for k in arrays.files}
    for b in (0, 1):
        for prefix, points in (
            ("held", z["source_means"][z["folds"] != b]),
            ("panel", z["panel_points"].reshape(-1, 2, 1024)),
        ):
            lp, p = a[f"{prefix}{b}_log_profile"], a[f"{prefix}{b}_profile"]
            d, factor = (
                a[f"{prefix}{b}_log_factor"],
                a[f"{prefix}{b}_factor_characters"],
            )
            ratio = a[f"{prefix}{b}_log_evidence_ratio"]
            error = max(
                error,
                float(np.max(np.abs(p.sum(-1) - 1))),
                float(np.max(np.abs(factor @ characters().T - d))),
                float(np.max(np.abs(lp[:, 0] - lp[:, 1] - d + ratio[:, None]))),
            )
            for i in range(min(8, len(points))):
                v = M.literal(bs[b], points[i])
                for key in ("profile", "targets", "log_factor"):
                    error = max(
                        error,
                        float(np.max(np.abs(v[key] - a[f"{prefix}{b}_{key}"][i]))),
                    )
                    checks += int(np.size(v[key]))
    r = M.cold(bs[0], R.read(out / "cold_example.json"))
    for key, arr in (
        ("targets_by_law", "panel0_targets"),
        ("joint_profiles_by_law", "panel0_profile"),
        ("target_log_odds_update", "panel0_target_log_odds_update"),
    ):
        error = max(error, float(np.max(np.abs(np.array(r[key]) - a[arr][0]))))
    if not np.isfinite(error) or error > cfg["comparison_tolerance"]:
        R.write(
            out / "verification_failure.json", {"max_error": error, "no_rerun": True}
        )
        raise ValueError("Finite mixture audit failed; do not tune precision")
    receipt = {
        "PASS": True,
        "alternate_max_error": error,
        "literal_values": checks,
        "all_profile_updates_and_character_inversions_checked": True,
        "same_agent_not_independent": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verified_seal.json", ("verification.json",))
    print(json.dumps(receipt))


def main():
    """Execute one explicit stage; refuse overwriting completed stage artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "build", "analyze", "verify", "cold")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "cold":
        R.check(
            args.out, ("built_seal.json", "analyzed_seal.json", "verified_seal.json")
        )
        print(
            json.dumps(
                M.cold(banks(args.out)[0], R.read(args.out / "cold_example.json"))
            )
        )
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
