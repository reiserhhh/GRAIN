#!/usr/bin/env python3
"""Recalculate pairing, source scaling, scores and exact coupling attribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, correlation_row, load_texts
from scripts.run_grain_native_reference import E1


def coefficients(model):
    """Undo standardization, retaining score units and the full intercept."""
    f, h = model["features"], model["head"]
    beta = h.coef_.T / f.scaler_.scale_[:, None]
    return beta, h.intercept_ - f.scaler_.mean_ @ beta


def run(args):
    """Rebuild from units and raw addresses; no new target fits or neural calls."""
    datasets = load_texts()
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    essay = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = essay.loc[datasets["essays"]["ids"], list(TRAITS)].to_numpy()
    runtime = joblib.load(E1 / "artifacts/common_source_comment_representation.joblib")
    contract = json.loads((args.output / "contract.json").read_text())
    for key, path in (
        ("code_sha256", ROOT / "scripts/run_grain_local_bilinear.py"),
        ("pool_code_sha256", ROOT / "grain_research/local_bilinear.py"),
    ):
        if digest(path) != contract[key]:
            raise ValueError("Current producer differs from run")
    checks, decomposition, full_metrics = [], [], []
    for domain in contract["domains"]:
        root = args.output / domain
        summary = json.loads((root / "summary.json").read_text())
        for name, sha in summary["files"].items():
            if digest(root / name) != sha:
                raise ValueError("Changed produced file " + name)
        with np.load(root / "private_blocks.npz", allow_pickle=False) as z:
            blocks = {k: z[k] for k in z.files}
        with np.load(root / "private_units.npz", allow_pickle=False) as z:
            g, h, s, authors = (z[k] for k in ("g", "h", "semantic", "authors"))
        with np.load(root / "private_predictions.npz", allow_pickle=False) as z:
            predictions = {k: z[k] for k in z.files}
        addresses = pd.read_csv(root / "private_addresses.csv")
        docs = datasets[domain]["documents"]
        units = [docs[r.author_index][r.start : r.end] for r in addresses.itertuples()]
        if len(units) != len(g) or not np.array_equal(addresses.author_index, authors):
            raise ValueError("Unit-address identity mismatch")
        np.testing.assert_allclose(runtime.transform(units), g, atol=1e-12)
        if domain == "essays":
            for u, group in addresses.groupby("author_index", sort=True):
                assert group.start.iloc[0] == 0
                assert group.end.iloc[-1] == len(docs[u])
                np.testing.assert_array_equal(
                    group.start.to_numpy()[1:], group.end.to_numpy()[:-1]
                )
            cache = root / "encoder_cache"
            state = json.loads((cache / "status.json").read_text())
            assert state["status"] == "COMPLETE" and state["completed"] == len(g)
            assert state["contract_sha256"] == digest(cache / "contract.json")
            position = 0
            for path in sorted(cache.glob("batch_*.npz")):
                meta = json.loads(path.with_suffix(".json").read_text())
                assert meta["sha256"] == digest(path) and meta["start"] == position
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(
                        z["vectors"], h[position : meta["stop"]]
                    )
                position = meta["stop"]
            assert position == len(h)
        else:
            with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["comments"].reshape(h.shape), h)
        source = np.sort(np.r_[blocks["train"], blocks["val"]])
        held = blocks["held"]
        assert not set(source) & set(held)
        np.testing.assert_array_equal(held, predictions["held_indices"])
        pooled_error, identity_error = 0.0, 0.0
        for u in range(len(docs)):
            ix = np.flatnonzero(authors == u)
            a, b = np.asarray(g[ix], float), np.asarray(s[ix], float)
            outer = np.outer(a.mean(0), b.mean(0))
            joint = sum(np.outer(x, y) for x, y in zip(a, b, strict=True)) / len(ix)
            centered = (a - a.mean(0)).T @ (b - b.mean(0)) / len(ix)
            np.testing.assert_allclose(joint - outer, centered, atol=1e-12)
            identity_error = max(
                identity_error, float(np.max(abs(joint - outer - centered)))
            )
            rebuilt = {
                "semantic": np.asarray(h[ix], float).mean(0),
                "grain": np.r_[a.mean(0), a.std(0)],
                "affinity": b.mean(0),
                "outer": outer.ravel(),
                "paired": joint.ravel(),
            }
            for seed in (924, 925, 926):
                order = np.random.default_rng(
                    np.random.SeedSequence([seed, u])
                ).permutation(len(ix))
                rebuilt[f"shuffle{seed}"] = sum(
                    np.outer(a[t], b[order[t]]) for t in range(len(ix))
                ).ravel() / len(ix)
            for key, value in rebuilt.items():
                np.testing.assert_allclose(value, blocks[key][u], atol=1e-12)
                pooled_error = max(
                    pooled_error, float(np.max(abs(value - blocks[key][u])))
                )
        table = pd.read_csv(root / "metrics_by_trait.csv")
        head_error = 0.0
        for arm in contract["arms"]:
            model = joblib.load(root / f"private_model_{arm}.joblib")
            feature, head = model["features"], model["head"]
            np.testing.assert_array_equal(model["source_indices"], source)
            np.testing.assert_array_equal(feature.fit_indices_, source)
            if arm == "word_char":
                pred = (
                    np.asarray(feature.transform(blocks, docs, held) @ head.coef_.T)
                    + head.intercept_
                )
            else:
                raw = feature._raw(blocks, source)
                np.testing.assert_allclose(
                    raw.mean(0), feature.scaler_.mean_, atol=1e-12
                )
                beta, intercept = coefficients(model)
                pred = feature._raw(blocks, held) @ beta + intercept
            np.testing.assert_allclose(pred, predictions[arm], atol=1e-10)
            head_error = max(head_error, float(np.max(abs(pred - predictions[arm]))))
            for j, trait in enumerate(TRAITS):
                record = table[(table.arm == arm) & (table.trait == trait)].iloc[0]
                expected = np.corrcoef(ys[domain][held, j], pred[:, j])[0, 1]
                assert abs(record.r - expected) < 1e-10
        companion = args.output / "projection_control"
        control_summary = json.loads((companion / "summary.json").read_text())
        for name, sha in control_summary["files"].items():
            if digest(companion / name) != sha:
                raise ValueError("Changed projection control")
        with np.load(
            companion / f"private_{domain}_scores.npz", allow_pickle=False
        ) as z:
            np.testing.assert_array_equal(z["held"], held)
            control_prediction = z["prediction"]
            np.testing.assert_allclose(
                blocks["semantic"][held] @ z["beta"] + z["intercept"],
                control_prediction,
                atol=1e-10,
            )
        for j, trait in enumerate(TRAITS):
            observed, predicted = ys[domain][held, j], control_prediction[:, j]
            err = predicted - observed
            full_metrics.append(
                {
                    "domain": domain,
                    "arm": "full_semantic_ridge",
                    "trait": trait,
                    **correlation_row(observed, predicted),
                    "mae": float(np.mean(abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "auc": float(roc_auc_score(observed, predicted))
                    if domain == "essays"
                    else None,
                }
            )
        # Same fitted head, exactly decomposed; these are not separately fitted predictors.
        model = joblib.load(root / "private_model_paired.joblib")
        feature = model["features"]
        beta, intercept = coefficients(model)
        raw = feature._raw(blocks, held)
        width = g.shape[1] * s.shape[1]
        cut = raw.shape[1] - width
        additive = raw[:, :cut] @ beta[:cut] + intercept
        marginal = blocks["outer"][held] @ beta[cut:]
        covariance = (blocks["paired"][held] - blocks["outer"][held]) @ beta[cut:]
        np.testing.assert_allclose(
            additive + marginal + covariance, predictions["paired"], atol=1e-10
        )
        unit_weights = beta[cut:].reshape(g.shape[1], s.shape[1], len(TRAITS))
        contributions = np.einsum("ij,jlk,il->ik", g, unit_weights, s)
        for u in held:
            ix = np.flatnonzero(authors == u)
            expected = blocks["paired"][u] @ beta[cut:]
            np.testing.assert_allclose(contributions[ix].mean(0), expected, atol=1e-10)
        np.savez_compressed(
            root / "private_paired_attribution.npz",
            additive=additive,
            marginal=marginal,
            covariance=covariance,
            unit_interaction=contributions,
            unit_authors=authors,
            held_indices=held,
        )
        for j, trait in enumerate(TRAITS):
            decomposition.append(
                {
                    "domain": domain,
                    "trait": trait,
                    "held_n": len(held),
                    "additive_sd": float(additive[:, j].std()),
                    "marginal_sd": float(marginal[:, j].std()),
                    "covariance_sd": float(covariance[:, j].std()),
                    "covariance_criterion_r": float(
                        np.corrcoef(covariance[:, j], ys[domain][held, j])[0, 1]
                    ),
                    "scope": "posthoc_exact_score_component_not_causal_or_independent_prediction",
                }
            )
        checks.append(
            {
                "domain": domain,
                "authors": len(docs),
                "units": len(g),
                "held": len(held),
                "pairing_max_error": pooled_error,
                "covariance_identity_max_error": identity_error,
                "cold_head_max_error": head_error,
                "summary_sha256": digest(root / "summary.json"),
            }
        )
    pd.DataFrame(decomposition).to_csv(
        args.output / "paired_score_accounting.csv", index=False
    )
    pd.DataFrame(full_metrics).to_csv(
        args.output / "full_semantic_metrics.csv", index=False
    )
    result = {
        "status": "PASS",
        "checks": checks,
        "code_sha256": digest(Path(__file__)),
        "new_neural_calls": 0,
        "new_fits": 0,
    }
    write(args.output / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/local_bilinear_r2"
    )
    with threadpool_limits(limits=2):
        run(p.parse_args())
