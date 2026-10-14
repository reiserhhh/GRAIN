#!/usr/bin/env python3
"""One actual-prefix memory intervention, not an accuracy or compression search."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_target_atlas as A
from grain_research.construction import math_vector_memory as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_vector_memory_hm211.json"
DEFAULT = ROOT / "results/grain_math_response_v1/vector_memory_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_VECTOR_MEMORY_PLAN_HM211.md",
    "grain_research/construction/math_vector_memory.py",
    "scripts/run_grain_vector_memory_hm211.py",
    "tests/test_grain_vector_memory.py",
    "grain_research/construction/math_joint_target_atlas.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Load only registered local numerical derivatives, without pickle."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def block_means(vectors):
    """Use every event once in three8-comment blocks balanced across old views."""
    z = np.asarray(vectors, float)
    if z.ndim != 4 or z.shape[1:3] != (2, 12) or not np.isfinite(z).all():
        raise ValueError("Two12-event registered views required")
    return np.stack([z[:, :, 4 * t : 4 * (t + 1)].mean((1, 2)) for t in range(3)], 1)


def event_record(vectors, conditions, frame_key, block):
    """Preserve genuine event addresses/conditions, not labels or predicted scores."""
    return {
        "frame_key": frame_key,
        "events": [
            {
                "event_id": f"{frame_key}:{view}:{slot}",
                "view": view,
                "slot": slot,
                "condition": int(conditions[view, slot]),
                "vector": vectors[view, slot].tolist(),
            }
            for view in (0, 1)
            for slot in range(4 * block, 4 * (block + 1))
        ],
    }


def prepare(out):
    """Authenticate actual comments; freeze banks before any comparison outcome."""
    cfg = R.read(ROOT / CONFIG)
    vp, rp, cp, pp = (
        ROOT / cfg[k]
        for k in (
            "vector_parent",
            "relation_parent",
            "condition_parent",
            "panel_condition_parent",
        )
    )
    parents = (
        (vp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")),
        (rp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json")),
        (cp, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")),
        (pp, ("built_seal.json", "analyzed_seal.json", "verified_seal.json")),
    )
    inputs = {}
    for p, stages in parents:
        R.check(p, stages)
        for name in (
            "observations.npz",
            "manifest.json",
            "prepared_seal.json",
            *stages,
        ):
            inputs[str((p / name).relative_to(ROOT))] = R.digest(p / name)
    v, h, c, p = (arrays(x / "observations.npz") for x in (vp, rp, cp, pp))
    if (
        not np.array_equal(v["source_vectors"], c["vectors"])
        or not np.array_equal(c["source_indices"], h["source_indices"])
        or not np.array_equal(c["folds"], h["folds"])
        or not np.array_equal(A.profile_codes(c["labels"]), h["codes"])
        or not np.array_equal(p["panel_vectors"], h["panel_vectors"])
        or not np.array_equal(v["panel_vectors"][:, 0], h["panel_vectors"])
        or not np.array_equal(p["author_ids"], h["author_ids"])
        or set(v["source_indices"]) & set(v["panel_indices"])
    ):
        raise ValueError("Changed author/profile/vector/condition event linkage")
    if c["vectors"].shape != (386, 2, 12, 1024) or h["panel_vectors"].shape != (
        24,
        2,
        12,
        1024,
    ):
        raise ValueError("Changed declared finite scope")
    if (
        not np.isin(c["conditions"], (0, 1)).all()
        or not np.isin(p["panel_conditions"], (0, 1)).all()
    ):
        raise ValueError("Unsupported observed conditions")
    prototypes, panel = block_means(c["vectors"]), block_means(h["panel_vectors"])
    s2 = cfg["historical_12_mean_variance"] * 12 / 8
    bs = [
        M.compile_bank(
            prototypes[h["folds"] == b],
            h["codes"][h["folds"] == b],
            s2,
            np.array([f"source-{i}" for i in h["source_indices"][h["folds"] == b]]),
        )
        for b in (0, 1)
    ]
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_blocks=prototypes,
        source_conditions=c["conditions"],
        source_indices=h["source_indices"],
        codes=h["codes"],
        folds=h["folds"],
        panel_blocks=panel,
        panel_vectors=h["panel_vectors"],
        panel_conditions=p["panel_conditions"],
        panel_indices=v["panel_indices"],
    )
    R.archive(
        out / "bank_bundle.npz",
        **{f"b{b}_{k}": a for b, bank in enumerate(bs) for k, a in bank.items()},
    )
    quality = {
        "source_authors": 386,
        "source_fold_counts": np.bincount(h["folds"]).tolist(),
        "panel_authors": 24,
        "source_events_reused": 9264,
        "panel_events_reused": 576,
        "new_vectors_or_model_forwards": 0,
        "blocks_per_author": 3,
        "events_per_block": 8,
        "all24events_disjoint_no_fabricated_vector": True,
        "source_profile_counts_by_fold": [bank["counts"].tolist() for bank in bs],
        "block_variance": s2,
        "variance_declared_not_natural_error_estimate": True,
        "panel_target_labels_native_answers_read": False,
        "conditions_provenance_not_separately_identified": True,
        "historical_encoder_overlap_and_cohorts_retained": True,
        "new_API_private_training": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "bank_bundle.npz", "input_quality.json"),
    )
    print(json.dumps(quality))


def banks(out):
    """Read immutable source banks rather than rebuilding from current subjects."""
    z = arrays(out / "bank_bundle.npz")
    return [{k[3:]: a for k, a in z.items() if k.startswith(f"b{b}_")} for b in (0, 1)]


def metrics(result, ix):
    """Report magnitudes and both directions, never select a positive witness."""
    d, odds = result["delta_targets"][ix], result["delta_log_odds"][ix]
    return {
        "histories": len(d),
        "current_profile_equality_max_error": float(
            np.max(
                np.abs(
                    result["before_profile"][ix] - result["before_reset_profile"][ix]
                )
            )
        ),
        "memory_TV_mean": float(np.mean(result["prefix_memory_TV"][ix])),
        "observation_log_evidence_ratio_mean": float(
            np.mean(result["observation_log_evidence_ratio"][ix])
        ),
        "delta_target_RMS": np.sqrt(np.mean(d**2, axis=0)).tolist(),
        "delta_target_mean": d.mean(0).tolist(),
        "delta_target_min": d.min(0).tolist(),
        "delta_target_max": d.max(0).tolist(),
        "delta_log_odds_RMS": np.sqrt(np.mean(odds**2, axis=0)).tolist(),
        "positive_target_changes": np.sum(d > 0, 0).tolist(),
        "negative_target_changes": np.sum(d < 0, 0).tolist(),
        "above_registered_numeric_tolerance": np.sum(np.abs(d) > 2e-12, 0).tolist(),
        "residual_identity_max_error": float(
            np.max(np.abs(d - result["residual_identity"][ix]))
        ),
    }


def compute(out):
    """Compare two states after actual prefixes, then evaluate held labels only."""
    cfg, start = R.check(out)["config"], time.perf_counter()
    z, bs = R.prepared(out), banks(out)
    saved, summaries, rows = {}, [], []
    for b, bank in enumerate(bs):
        held = np.flatnonzero(z["folds"] != b)
        points = np.concatenate((z["source_blocks"][held], z["panel_blocks"]))
        r = M.experiment(bank, points)
        saved.update({f"b{b}_{k}": a for k, a in r.items()})
        nh = len(held)
        truth = z["codes"][held]
        bits = M.BITS[truth].astype(int)
        gain = np.take_along_axis(
            r["fine_target_logs"][:nh] - r["reset_target_logs"][:nh],
            bits[:, :, None],
            axis=-1,
        )[..., 0]
        fine_lp = M.readout(bank, r["fine_next"][:nh])["log_profile"]
        reset_lp = M.readout(bank, r["reset_next"][:nh])["log_profile"]
        joint_gain = (fine_lp - reset_lp)[np.arange(nh), truth]
        counts = np.bincount(truth, minlength=32)
        if counts.min() == 0:
            raise ValueError("Held profile missing; cannot silently reweight")
        balanced = 1 / (32 * counts[truth])
        summaries.append(
            {
                "bank": b,
                "source_components": len(bank["codes"]),
                "held": metrics(r, np.arange(nh)),
                "panel": metrics(r, np.arange(nh, len(points))),
                "held_binary_log_score_gain_observed": gain.mean(0).tolist(),
                "held_binary_log_score_gain_profile_balanced": (
                    balanced @ gain
                ).tolist(),
                "held_joint_log_score_gain_observed": float(joint_gain.mean()),
                "held_joint_log_score_gain_profile_balanced": float(
                    balanced @ joint_gain
                ),
            }
        )
        for u, q in np.ndindex(len(points), 5):
            rows.append(
                {
                    "bank": b,
                    "role": "held_source" if u < nh else "panel_no_labels",
                    "subject_index": int(z["source_indices"][held[u]])
                    if u < nh
                    else int(z["panel_indices"][u - nh]),
                    "target": M.TARGETS[q],
                    "fine_minus_reset_target": float(r["delta_targets"][u, q]),
                    "fine_minus_reset_log_odds": float(r["delta_log_odds"][u, q]),
                    "prefix_memory_TV": float(r["prefix_memory_TV"][u]),
                    "observation_log_evidence_ratio": float(
                        r["observation_log_evidence_ratio"][u]
                    ),
                    "binary_log_score_gain": float(gain[u, q]) if u < nh else "",
                }
            )
    d0, d1 = saved["b0_delta_targets"][-24:], saved["b1_delta_targets"][-24:]
    summary = {
        "packet": "HM211",
        "banks": summaries,
        "panel_cross_bank_same_sign_counts": np.sum(d0 * d1 > 0, 0).tolist(),
        "panel_cross_bank_products_mean": np.mean(d0 * d1, 0).tolist(),
        "panel_people": 24,
        "comparison_is_memory_intervention_not_two_reachable_natural_prefixes": True,
        "new_neural_API_private_training_answer_fit_parameter_search": 0,
        "original_scope_complete": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if summary["elapsed_seconds"] > cfg["max_compute_seconds"]:
        raise RuntimeError("Finite compute budget exceeded; no automatic retry")
    R.archive(out / "analysis.npz", **saved)
    R.table(out / "actual_prefix_accounts.csv", rows)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        ("analysis.npz", "actual_prefix_accounts.csv", "summary.json"),
    )
    print(json.dumps(summary))


def verify(out):
    """Replay actual histories with direct distances and literal component sums."""
    cfg, start = R.check(out, ("computed_seal.json",))["config"], time.perf_counter()
    z, old, bs = R.prepared(out), arrays(out / "analysis.npz"), banks(out)
    errors, checks = [], 0
    for b, bank in enumerate(bs):
        held = np.flatnonzero(z["folds"] != b)
        points = np.concatenate((z["source_blocks"][held], z["panel_blocks"]))
        codes, prototypes = bank["codes"], bank["prototypes"]
        counts = np.bincount(codes, minlength=32)
        log_prior = -np.log(32 * counts[codes])
        for u, x in enumerate(points):
            k = -np.sum((prototypes - x[None]) ** 2, axis=2) / (
                2 * float(bank["variance"])
            )
            prefix = log_prior + k[:, 0] + k[:, 1]
            prefix -= logsumexp(prefix)
            a = np.exp(prefix)
            p = np.array(
                [
                    sum(float(a[i]) for i in np.flatnonzero(codes == g))
                    for g in range(32)
                ]
            )
            reset = p[codes] / counts[codes]
            fine_log = prefix + k[:, 2]
            reset_log = np.log(reset) + k[:, 2]
            ratio = logsumexp(fine_log) - logsumexp(reset_log)
            fine, coarse = (
                np.exp(fine_log - logsumexp(fine_log)),
                np.exp(reset_log - logsumexp(reset_log)),
            )
            fp = np.array(
                [
                    sum(float(fine[i]) for i in np.flatnonzero(codes == g))
                    for g in range(32)
                ]
            )
            rp = np.array(
                [
                    sum(float(coarse[i]) for i in np.flatnonzero(codes == g))
                    for g in range(32)
                ]
            )
            mu = rp @ M.BITS
            kernel = np.exp(k[:, 2] - max(k[:, 2]))
            denominator = sum(float(aa * ll) for aa, ll in zip(a, kernel, strict=True))
            identity = np.array(
                [
                    sum(
                        float(
                            (a[i] - reset[i])
                            * kernel[i]
                            * (M.BITS[codes[i], q] - mu[q])
                        )
                        for i in range(len(a))
                    )
                    / denominator
                    for q in range(5)
                ]
            )
            expected = {
                "before_profile": p,
                "fine_profile": fp,
                "reset_profile": rp,
                "delta_targets": (fp - rp) @ M.BITS,
                "residual_identity": identity,
                "observation_log_evidence_ratio": ratio,
                "prefix_memory_TV": 0.5
                * sum(float(abs(x - y)) for x, y in zip(a, reset, strict=True)),
            }
            for name, values in expected.items():
                errors.append(float(np.max(np.abs(values - old[f"b{b}_{name}"][u]))))
                checks += np.size(values)
            errors.append(float(np.max(np.abs((fp - rp) @ M.BITS - identity))))
            checks += 5
    result = {
        "PASS": bool(max(errors) <= cfg["comparison_tolerance"]),
        "alternate_max_error": max(errors),
        "literal_account_values": int(checks),
        "actual_histories_replayed": 386 + 2 * 24,
        "same_agent_not_independent": True,
        "elapsed_seconds": time.perf_counter() - start,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verified_seal.json", ("verification.json",))
    print(json.dumps(result))
    if not result["PASS"]:
        raise ValueError("Scientific numerical replay failed; keep failure/no retry")


def resume(out, bank_index):
    """Continue one saved prefix in a fresh process from next vectors only."""
    R.check(out, ("computed_seal.json", "verified_seal.json", "cold_request_seal.json"))
    bank = banks(out)[bank_index]
    checkpoint = R.read(out / f"cold_checkpoint_bank{bank_index}.json")
    next_record = R.read(out / f"cold_next_bank{bank_index}.json")
    states = [checkpoint[k] for k in ("fine", "reset")]
    before = [M.describe(bank, state) for state in states]
    after = [M.describe(bank, M.ingest(bank, state, next_record)) for state in states]
    response = {
        "bank": bank_index,
        "before": before,
        "after": after,
        "fine_minus_reset_targets": (
            np.array(list(after[0]["target_readings"].values()))
            - np.array(list(after[1]["target_readings"].values()))
        ).tolist(),
        "new_process_no_prior_blocks_or_current_labels": True,
    }
    R.write(out / f"cold_response_bank{bank_index}.json", response)
    print(json.dumps(response))


def cold(out):
    """Save actual16-event memory; a separate process ingests the remaining8."""
    R.check(out, ("computed_seal.json", "verified_seal.json"))
    z, bs = R.prepared(out), banks(out)
    files = []
    for b, bank in enumerate(bs):
        state = M.initialize(bank, "panel-0")
        for t in (0, 1):
            state = M.ingest(
                bank,
                state,
                event_record(
                    z["panel_vectors"][0], z["panel_conditions"][0], "panel-0", t
                ),
            )
        checkpoint = {"fine": state, "reset": M.reset_memory(bank, state)}
        cp, nr = f"cold_checkpoint_bank{b}.json", f"cold_next_bank{b}.json"
        R.write(out / cp, checkpoint)
        R.write(
            out / nr,
            event_record(z["panel_vectors"][0], z["panel_conditions"][0], "panel-0", 2),
        )
        files.extend((cp, nr))
    R.seal(out, "cold_request_seal.json", files)
    old, errors = arrays(out / "analysis.npz"), []
    for b in (0, 1):
        run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "resume",
                "--out",
                str(out),
                "--bank",
                str(b),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        response = R.read(out / f"cold_response_bank{b}.json")
        expected = old[f"b{b}_delta_targets"][-24]
        errors.append(
            float(
                np.max(
                    np.abs(np.array(response["fine_minus_reset_targets"]) - expected)
                )
            )
        )
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.array(response["before"][0]["profiles"])
                        - response["before"][1]["profiles"]
                    )
                )
            )
        )
        if any(
            d["event_count"] != 24 or d["next_block"] != 3 for d in response["after"]
        ):
            raise ValueError("Cold resumed event history incomplete")
        files.append(f"cold_response_bank{b}.json")
        print(run.stdout.strip())
    receipt = {
        "PASS": max(errors) <= 2e-12,
        "max_error": max(errors),
        "saved_prefix_events": 16,
        "new_events_per_resume": 8,
        "fresh_processes": 2,
        "current_labels_answers_or_new_model_calls": 0,
    }
    R.write(out / "cold_verification.json", receipt)
    files.append("cold_verification.json")
    R.seal(out, "cold_seal.json", files)
    print(json.dumps(receipt))
    if not receipt["PASS"]:
        raise ValueError("Cold memory resume disagrees; no retry")


def main():
    """Run one immutable finite stage; never silently append or restart."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compute", "verify", "cold", "resume")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--bank", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.stage == "resume":
        if args.bank is None:
            parser.error("resume requires a source bank")
        resume(args.out, args.bank)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
