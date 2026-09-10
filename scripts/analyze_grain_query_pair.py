#!/usr/bin/env python3
"""Assess complete and local conditional functions separately on frozen authors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_pair import ARMS, PAIRS, error_account
from grain_research.vector_mechanisms.joint_readout import centered_logits
from grain_research.vector_mechanisms.query_pair import scores
from scripts.analyze_grain_conditional_energy import author_draws, ci
from scripts.analyze_grain_query_baseline import corr
from scripts.run_grain_joint_carrier_readout import verify
from scripts.run_grain_query_pair import OLD, PARENT, verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def variation_account(actual, predicted):
    """Separate common offsets from author variation, condition by condition."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim < 2 or not np.isfinite([y, p]).all():
        raise ValueError("Finite aligned author-first arrays required")
    yc, pc = y - y.mean(0), p - p.mean(0)
    if np.all(np.ptp(p, axis=0) == 0):
        pc = np.zeros_like(p)
    mse = float(np.mean((y - p) ** 2))
    offset = float(np.mean((y - p).mean(0) ** 2))
    variation = float(np.mean((yc - pc) ** 2))
    energy = float(np.mean(yc**2))
    if abs(mse - offset - variation) > 1e-12:
        raise ValueError("Offset/variation decomposition failed")
    return {
        "mse": mse,
        "offset_mse": offset,
        "variation_mse": variation,
        "variation_mse_over_target": variation / energy if energy else float("nan"),
        "author_centered_r": corr(yc.ravel(), pc.ravel()),
    }


def scalar_metrics(y, p, draws):
    """Evaluate fixed scalar arrays, bootstrapping authors not independent pairs."""
    lo, hi, n = ci([corr(y[ix].ravel(), p[ix].ravel()) for ix in draws])
    return {
        "r": corr(y.ravel(), p.ravel()),
        "r_low": lo,
        "r_high": hi,
        "bootstrap_n": n,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(abs(y - p))),
        "actual_sd": float(y.std()),
        "predicted_sd": float(p.std()),
    }


def analyze(out, name):
    """Read actual pair endpoints only after all complete predictions were frozen."""
    verified(out)
    verify(PARENT)
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "analyzer_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "human_labels_read": False,
            "new_neural_calls": 0,
            "bootstrap": "2000 author draws/seed2501, pointwise fixed-prediction intervals",
        },
    )
    membership = read(out / "private_membership.json")
    authors = membership["evaluation"]
    if {r["author_id"] for r in authors} & {
        r["author_id"] for r in membership["sources"]
    }:
        raise ValueError("Source/evaluation authors overlap")
    with np.load(out / "private_predictions.npz") as z:
        pred = dict(z)
    with np.load(out / "private_model.npz") as z:
        values = z["values"].copy()
    base, inc, full = [], [], []
    for r in authors:
        p = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if (
            p["author_id"] != r["author_id"]
            or p["query"] != "neuroticism"
            or not np.array_equal(p["alphabet"]["values"], values)
        ):
            raise ValueError("Endpoint identity/alphabet changed")
        s = p["scores"]
        z0 = centered_logits(s["zero"]["candidate_logits"])
        base.append(z0)
        inc.append(
            [centered_logits(s[f"s{i}"]["candidate_logits"]) - z0 for i in (0, 1, 11)]
        )
        full.append(
            [centered_logits(s[k]["candidate_logits"]) for k in ("adjacent", "distant")]
        )
    base, inc, full = map(np.asarray, (base, inc, full))
    true_config = full - base[:, None]
    mixed = true_config - np.stack([inc[:, i] + inc[:, j] for i, j in PAIRS], 1)
    y, y0 = scores(full, values), scores(base, values)
    draws = author_draws([r["author_id"] for r in authors])
    whole, paired, components, local, errors, private = [], [], [], [], [], []
    endpoint_identity, error_identity, source_denied_score_replay = 0.0, 0.0, 0.0
    geometry = []
    for pair_idx, pair in enumerate(("adjacent", "distant")):
        configuration = true_config[:, pair_idx]
        geometry.append(
            {
                "pair": pair,
                "authors": len(authors),
                "native_base_full_score_r": corr(y0, y[:, pair_idx]),
                "native_score_action_rms": float(
                    np.sqrt(np.mean((y[:, pair_idx] - y0) ** 2))
                ),
                "full_score_sd": float(y[:, pair_idx].std()),
                "base_score_sd": float(y0.std()),
                "configuration_logit_rms": float(np.sqrt(np.mean(configuration**2))),
                "mixed_logit_rms": float(np.sqrt(np.mean(mixed[:, pair_idx] ** 2))),
            }
        )
    for arm in ARMS:
        pinc, pmixed = pred[arm + "_singleton"], pred[arm + "_mixed"]
        configuration = (
            np.stack([pinc[:, i] + pinc[:, j] for i, j in PAIRS], 1) + pmixed
        )
        assemblies = {
            "affine_complete": pred[arm + "_affine_full"],
            "kernel_complete": pred[arm + "_full"],
            "affine_base_only": np.repeat(pred["affine_base"][:, None], 2, 1),
            "oracle_base_predicted_configuration": base[:, None] + configuration,
            "affine_base_oracle_configuration": pred["affine_base"][:, None]
            + true_config,
        }
        score_arrays = {k: scores(v, values) for k, v in assemblies.items()}
        source_denied_score_replay = max(
            source_denied_score_replay,
            float(
                np.max(
                    abs(score_arrays["affine_complete"] - pred[arm + "_affine_score"])
                )
            ),
        )
        for scope, sl in (
            ("adjacent", slice(0, 1)),
            ("distant", slice(1, 2)),
            ("pooled", slice(None)),
        ):
            target = y[:, sl]
            for label, z in assemblies.items():
                p = score_arrays[label][:, sl]
                row = {
                    "arm": arm,
                    "pair": scope,
                    "assembly": label,
                    "authors": len(authors),
                    "pairs": p.size,
                }
                row.update(
                    {
                        "score_" + k: v
                        for k, v in scalar_metrics(target, p, draws).items()
                    }
                )
                row.update(
                    {
                        "logit_" + k: v
                        for k, v in variation_account(full[:, sl], z[:, sl]).items()
                    }
                )
                row.update(
                    {"score_" + k: v for k, v in variation_account(target, p).items()}
                )
                whole.append(row)
            for other in ("kernel_complete", "affine_base_only"):
                delta = (score_arrays["affine_complete"][:, sl] - target) ** 2 - (
                    score_arrays[other][:, sl] - target
                ) ** 2
                lo, hi, _ = ci([delta[ix].mean() for ix in draws])
                paired.append(
                    {
                        "arm": arm,
                        "pair": scope,
                        "comparator": other,
                        "mse_delta": float(delta.mean()),
                        "low": lo,
                        "high": hi,
                        "authors": len(authors),
                    }
                )
        compare = {
            f"singleton_{i}": (inc[:, j], pinc[:, j]) for j, i in enumerate((0, 1, 11))
        }
        for pidx, pair in enumerate(("adjacent", "distant")):
            compare["mixed_" + pair] = (mixed[:, pidx], pmixed[:, pidx])
            compare["configuration_" + pair] = (
                true_config[:, pidx],
                configuration[:, pidx],
            )
            dy = y[:, pidx] - y0
            for name0, b0 in (
                ("affine", pred["affine_base"]),
                ("kernel", pred["base"]),
                ("oracle", base),
            ):
                dp = scores(b0 + configuration[:, pidx], values) - scores(b0, values)
                row = {
                    "arm": arm,
                    "pair": pair,
                    "base": name0,
                    "mse_over_zero": float(np.sum((dy - dp) ** 2) / np.sum(dy**2)),
                    **scalar_metrics(dy, dp, draws),
                }
                local.append(row)
            i, j = PAIRS[pidx]
            terms = error_account(
                {
                    "base": pred["affine_base"] - base,
                    "i": pinc[:, i] - inc[:, i],
                    "j": pinc[:, j] - inc[:, j],
                    "mixed": pmixed[:, pidx] - mixed[:, pidx],
                }
            )
            error_identity = max(
                error_identity, float(np.max(abs(terms["identity_error"])))
            )
            endpoint_identity = max(
                endpoint_identity,
                float(
                    np.max(
                        abs(
                            terms["full_mse"]
                            - np.mean(
                                (assemblies["affine_complete"][:, pidx] - full[:, pidx])
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
            )
            errors.append(
                {
                    "arm": arm,
                    "pair": pair,
                    **{k: float(v.mean()) for k, v in terms.items()},
                }
            )
            for label, p in score_arrays.items():
                private.extend(
                    {
                        "author_id": r["author_id"],
                        "arm": arm,
                        "pair": pair,
                        "assembly": label,
                        "actual": y[u, pidx],
                        "predicted": p[u, pidx],
                    }
                    for u, r in enumerate(authors)
                )
        for key, (actual, p) in compare.items():
            components.append(
                {
                    "arm": arm,
                    "component": key,
                    "mse_over_zero": float(
                        np.sum((actual - p) ** 2) / np.sum(actual**2)
                    ),
                    "actual_rms": float(np.sqrt(np.mean(actual**2))),
                    "predicted_rms": float(np.sqrt(np.mean(p**2))),
                    "norm_r": corr(
                        np.linalg.norm(actual, axis=1), np.linalg.norm(p, axis=1)
                    ),
                    **variation_account(actual, p),
                }
            )
    if max(error_identity, endpoint_identity, source_denied_score_replay) > 1e-12:
        raise ValueError(
            "Score or component accounting changed frozen complete predictions"
        )
    tables = {
        "score_comparison": whole,
        "paired_score_mse": paired,
        "local_components": components,
        "configuration_scores": local,
        "error_account": errors,
        "target_geometry": geometry,
        "private_score_predictions": private,
    }
    for name0, records in tables.items():
        pd.DataFrame(records).to_csv(dest / (name0 + ".csv"), index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "evaluation_authors": len(authors),
            "source_authors": 4,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "endpoint_accounting_max": endpoint_identity,
            "error_identity_max": error_identity,
            "source_denied_score_replay_max": source_denied_score_replay,
            "files": {k + ".csv": sha(dest / (k + ".csv")) for k in tables},
        },
    )
    print(
        {
            "status": "ANALYSIS_COMPLETE",
            "comparisons": len(whole),
            "component_rows": len(components),
            "authors": len(authors),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_query_pair_v1")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)
