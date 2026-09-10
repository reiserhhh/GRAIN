#!/usr/bin/env python3
"""Explain unsigned strength versus signed response; no new predictor or fitting."""

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
from scripts.analyze_grain_query_baseline import corr
from scripts.run_grain_query_pair import OLD, verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def score_gradient(logits, values):
    """Analytic gradient of the original softmax-weighted candidate score."""
    z = np.asarray(logits, float)
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p * (np.asarray(values) - scores(z, values)[..., None])


def score_path(base, delta, values):
    """Integrate only the known score link using fixed16-node Gauss quadrature."""
    base, delta = np.asarray(base, float), np.asarray(delta, float)
    if base.shape != delta.shape or base.ndim != 2:
        raise ValueError("Aligned author-candidate matrices required")
    nodes, weights = np.polynomial.legendre.leggauss(16)
    grad = np.einsum(
        "a,nac->nc",
        weights / 2,
        score_gradient(
            base[:, None] + ((nodes + 1) / 2)[None, :, None] * delta[:, None], values
        ),
    )
    actual = scores(base + delta, values) - scores(base, values)
    prediction = np.sum(grad * delta, axis=1)
    discrepancy = float(np.max(abs(actual - prediction)))
    if discrepancy > 1e-12:
        raise ValueError("Output-link integral did not reconstruct score action")
    norms = np.linalg.norm(delta, axis=1)
    unit = np.divide(
        delta, norms[:, None], out=np.zeros_like(delta), where=norms[:, None] > 0
    )
    return {
        "gradient": grad,
        "action": actual,
        "norm": norms,
        "unit": unit,
        "identity_max": discrepancy,
    }


def direction_account(base, actual, predicted, values):
    """Exact ordered magnitude/orientation/link error account at a common true base.

    This is an oracle diagnostic. Reordering the product telescoping changes the
    allocation; terms are not independent causal or variance contributions.
    """
    t, p = score_path(base, actual, values), score_path(base, predicted, values)
    vt, vp = t["gradient"], p["gradient"]
    terms = {
        "magnitude": (p["norm"] - t["norm"]) * np.sum(t["unit"] * vt, 1),
        "orientation": p["norm"] * np.sum((p["unit"] - t["unit"]) * vt, 1),
        "score_link": p["norm"] * np.sum(p["unit"] * (vp - vt), 1),
    }
    error = p["action"] - t["action"]
    discrepancy = float(np.max(abs(sum(terms.values()) - error)))
    if discrepancy > 1e-12:
        raise ValueError("Magnitude/orientation/link account failed")
    return (
        t,
        p,
        terms,
        {
            "quadrature_max": max(t["identity_max"], p["identity_max"]),
            "ordered_account_max": discrepancy,
        },
    )


def analyze(out, name):
    """Inspect fixed model outputs and endpoints without selecting any new formula."""
    verified(out)
    dest = out / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "script_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "new_fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "allocation_order": ["magnitude", "orientation", "score_link"],
            "base": "oracle native zero, same for both arms",
        },
    )
    authors = read(out / "private_membership.json")["evaluation"]
    with np.load(out / "private_model.npz") as z:
        values = z["values"].copy()
    with np.load(out / "private_predictions.npz") as z:
        pred = dict(z)
    base, targets = [], []
    for r in authors:
        p = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if p["author_id"] != r["author_id"] or p["query"] != "neuroticism":
            raise ValueError("Endpoint identity changed")
        s = p["scores"]
        b = centered_logits(s["zero"]["candidate_logits"])
        base.append(b)
        targets.append(
            [
                centered_logits(s[k]["candidate_logits"]) - b
                for k in ("adjacent", "distant")
            ]
        )
    base, targets = np.array(base), np.array(targets)
    rows, accounts = [], []
    for arm in ARMS:
        inc = pred[arm + "_singleton"]
        configuration = (
            np.stack([inc[:, i] + inc[:, j] for i, j in PAIRS], 1)
            + pred[arm + "_mixed"]
        )
        for j, pair in enumerate(("adjacent", "distant")):
            t, p, parts, identities = direction_account(
                base, targets[:, j], configuration[:, j], values
            )
            vg = np.linalg.norm(t["gradient"], axis=1)
            at = np.sum(t["unit"] * t["gradient"], 1) / vg
            ap = np.sum(p["unit"] * t["gradient"], 1) / vg
            linear = np.sum(score_gradient(base, values) * targets[:, j], 1)
            row = {
                "arm": arm,
                "pair": pair,
                "authors": len(authors),
                "norm_r": corr(t["norm"], p["norm"]),
                "signed_alignment_r": corr(at, ap),
                "native_linear_action_r": corr(t["action"], linear),
                "native_link_linear_error_rms": float(
                    np.sqrt(np.mean((linear - t["action"]) ** 2))
                ),
                "native_action_rms": float(np.sqrt(np.mean(t["action"] ** 2))),
                "configuration_action_r": corr(t["action"], p["action"]),
                "action_error_rms": float(
                    np.sqrt(np.mean((p["action"] - t["action"]) ** 2))
                ),
                **identities,
            }
            row.update(
                {
                    k + "_error_rms": float(np.sqrt(np.mean(v**2)))
                    for k, v in parts.items()
                }
            )
            rows.append(row)
            e = error_account({k: v[:, None] for k, v in parts.items()})
            accounts.append(
                {"arm": arm, "pair": pair, **{k: float(v.mean()) for k, v in e.items()}}
            )
    pd.DataFrame(rows).to_csv(dest / "direction_account.csv", index=False)
    pd.DataFrame(accounts).to_csv(dest / "signed_error_terms.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "new_fits": 0,
            "new_neural_calls": 0,
            "diagnostic_only": True,
            "quadrature_max": max(r["quadrature_max"] for r in rows),
            "ordered_account_max": max(r["ordered_account_max"] for r in rows),
            "files": {
                k: sha(dest / k)
                for k in ("direction_account.csv", "signed_error_terms.csv")
            },
        },
    )
    print({"status": "DIRECTION_ACCOUNT_COMPLETE", "rows": len(rows), "new_fits": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_query_pair_v1")
    p.add_argument("--name", default="direction_analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.name)
