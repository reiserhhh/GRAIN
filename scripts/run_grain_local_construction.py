#!/usr/bin/env python3
"""Build two local structural-reference cases with frozen, interpretable scorers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.common_factors import R_DRIVER, WordFactorReference
from grain_research.local_measure import (
    essay_blocks,
    explain_document,
    export_reference,
    material_interval,
    score_documents,
)
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write

PLAN = ROOT / "docs/GRAIN_LOCAL_CONSTRUCTION_REFERENCE_PLAN_V1.md"
OLD = BASE / "predecessor_integration_v1/common_factor_r1/private_models"
ESSAYS = Path(
    "/Volumes/mobile3/projects/project persona/data_sets/prepared/big5/essays_original_prepared.csv"
)


def load_texts():
    """Verify text/author splits without reading questionnaire arrays or columns."""
    manifest = json.loads((DEFAULT / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(DEFAULT / name) != sha:
            raise ValueError("Changed predecessor material")
    material = json.loads((DEFAULT / "private_material.json").read_text())
    if any(b["source_group"] != "big5_only" for b in material["bundles"]):
        raise ValueError("Only Big5-only author material is allowed")
    ids = [b["author_id"] for b in material["bundles"]]
    folds = np.array([b["official_folds"]["big5"] for b in material["bundles"]])
    p_units = [[c["body"] for v in b["views"] for c in v] for b in material["bundles"]]
    docs = material["documents"]
    if any("\n\n".join(u) != d for u, d in zip(p_units, docs, strict=True)):
        raise ValueError("Original comment aggregation changed")
    essay = pd.read_csv(
        ESSAYS, usecols=["user_id", "text", "essay_fold", "user_id_source"]
    )
    if (
        essay.user_id.duplicated().any()
        or not essay.user_id_source.eq("original_author_id").all()
    ):
        raise ValueError("Essays requires original unique author IDs")
    if len(set(ids)) != 624 or len(essay) != 2467 or essay.text.isna().any():
        raise ValueError("Material cohort changed")
    return {
        "pandora": {
            "ids": ids,
            "documents": docs,
            "units": p_units,
            "train": np.flatnonzero(folds != 0),
            "held": np.flatnonzero(folds == 0),
            "label_type": "continuous_0_1",
            "condition": "observed_reddit_36_comments",
            "unit": "comment",
        },
        "essays": {
            "ids": essay.user_id.astype(str).tolist(),
            "documents": essay.text.tolist(),
            "units": [essay_blocks(t) for t in essay.text],
            "train": np.flatnonzero(essay.essay_fold != 9),
            "held": np.flatnonzero(essay.essay_fold == 9),
            "label_type": "binary_high_low",
            "condition": "one_prompted_writing_occasion",
            "unit": "two_consecutive_regex_sentence_spans",
        },
    }


def correlation_row(x, y):
    """Two-sided Pearson/Fisher and Spearman, explicitly missing for degenerate data."""
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": len(x),
            "r": None,
            "p": None,
            "rho": None,
            "ci_low": None,
            "ci_high": None,
        }
    r, p = pearsonr(x, y)
    z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    ci = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) / np.sqrt(len(x) - 3))
    return {
        "n": len(x),
        "r": float(r),
        "p": float(p),
        "rho": float(spearmanr(x, y).statistic),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def bh_adjust(p):
    """Benjamini-Hochberg adjustment of the complete declared hypothesis family."""
    p = np.asarray(p, float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    result[order] = np.minimum(
        1,
        np.minimum.accumulate(
            (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
        )[::-1],
    )
    return result


def run(args):
    """Construct first; freeze the four reference artifacts before opening outcomes."""
    began = time.monotonic()
    if args.output.exists():
        raise ValueError("Fresh output required; preserve prior runs")
    args.output.mkdir(parents=True)
    material = load_texts()
    contract = {
        "plan_sha256": digest(PLAN),
        "essays_sha256": digest(ESSAYS),
        "pandora_manifest_sha256": digest(DEFAULT / "material_manifest.json"),
        "methods": ["common_factor", "pca"],
        "factors": 5,
        "max_terms": 200,
        "source_models": {
            m: digest(OLD / f"{m}_fold0.joblib") for m in ("common_factor", "pca")
        },
        "seed": 923,
        "material_draws": 200,
        "code": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                R_DRIVER,
                ROOT / "grain_research/local_measure.py",
                ROOT / "grain_research/common_factors.py",
            )
        },
        "data_use": "PANDORA Big5-only, original Essays; neither new lockbox nor MEPS/bridge",
    }
    write(args.output / "contract.json", contract)
    objects, outputs, numerical, manifests, intervals, length_rows = (
        {},
        {},
        [],
        [],
        [],
        [],
    )
    for domain, d in material.items():
        train, held = d["train"], d["held"]
        docs = np.asarray(d["documents"])
        for method in contract["methods"]:
            key = domain + "_" + method
            if domain == "pandora":
                cache = joblib.load(OLD / f"{method}_fold0.joblib")
                np.testing.assert_array_equal(cache["source_indices"], train)
                feature = cache["feature"]
            else:
                feature = WordFactorReference(method=method).fit(docs[train])
            bundle = export_reference(
                feature,
                docs[train].tolist(),
                domain=domain,
                source_ids=[d["ids"][i] for i in train],
            )
            bundle_path = args.output / f"private_{key}_scorer.json"
            write(bundle_path, bundle)
            objects[key] = bundle
            native = (feature.transform(docs[held]) - bundle["score_center"]) / bundle[
                "score_scale"
            ]
            cold = score_documents(
                json.loads(bundle_path.read_text()), docs[held].tolist()
            )
            np.testing.assert_allclose(native, cold["z"], atol=1e-9)
            single = np.array(
                [score_documents(bundle, [doc])["z"][0] for doc in docs[held]]
            )
            np.testing.assert_allclose(single, cold["z"], atol=1e-12)
            outputs[key] = cold
            frame = pd.DataFrame(cold["z"], columns=bundle["factor_ids"])
            frame.insert(0, "author_id", [d["ids"][i] for i in held])
            frame.to_csv(args.output / f"private_{key}_held_scores.csv", index=False)
            pd.DataFrame(cold["percentile"], columns=bundle["factor_ids"]).to_csv(
                args.output / f"private_{key}_held_percentiles.csv", index=False
            )
            for h, i in enumerate(held):
                bound = material_interval(bundle, d["units"][i], seed=923 + int(i))
                for j, factor in enumerate(bundle["factor_ids"]):
                    intervals.append(
                        {
                            "domain": domain,
                            "method": method,
                            "author_id": d["ids"][i],
                            "factor": factor,
                            "blocks": len(d["units"][i]),
                            "z": float(cold["z"][h, j]),
                            "lower": None if bound is None else bound["lower"][j],
                            "upper": None if bound is None else bound["upper"][j],
                        }
                    )
            example = min(held, key=lambda i: d["ids"][i])
            explanation = explain_document(bundle, docs[example])
            write(
                args.output / f"private_{key}_explanation.json",
                {
                    "author_id": d["ids"][example],
                    "document": docs[example],
                    **explanation,
                },
            )
            pd.DataFrame(
                feature.loadings_,
                index=bundle["vocabulary"],
                columns=bundle["factor_ids"],
            ).to_csv(args.output / f"private_{key}_pattern.csv")
            for j, factor in enumerate(bundle["factor_ids"]):
                length_rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        "factor": factor,
                        **correlation_row(cold["z"][:, j], np.log1p(cold["tokens"])),
                    }
                )
            numerical.append(
                {
                    "domain": domain,
                    "method": method,
                    "source_n": len(train),
                    "held_n": len(held),
                    "word_count": len(bundle["vocabulary"]),
                    "scorer_sha256": digest(bundle_path),
                    "native_json_max_error": float(np.max(np.abs(native - cold["z"]))),
                    "single_batch_max_error": float(np.max(np.abs(single - cold["z"]))),
                    "explanation_max_error": explanation["max_error"],
                    "diagnostics": feature.diagnostics_,
                    "warnings": feature.model_.warnings_
                    if method == "common_factor"
                    else [],
                    "admissible": feature.model_.admissible_
                    if method == "common_factor"
                    else None,
                }
            )
            print(
                json.dumps(
                    {
                        "constructed": key,
                        "source": len(train),
                        "held": len(held),
                        "seconds": round(time.monotonic() - began, 2),
                    }
                ),
                flush=True,
            )
        manifests.extend(
            {
                "domain": domain,
                "author_id": uid,
                "split": "source" if i in train else "held",
                "condition": d["condition"],
                "units": len(d["units"][i]),
                "unit_type": d["unit"],
            }
            for i, uid in enumerate(d["ids"])
        )
    pd.DataFrame(manifests).to_csv(
        args.output / "private_author_manifest.csv", index=False
    )
    pd.DataFrame(intervals).to_csv(
        args.output / "private_material_intervals.csv", index=False
    )
    pd.DataFrame(length_rows).to_csv(
        args.output / "length_associations.csv", index=False
    )
    freeze = {k: digest(args.output / f"private_{k}_scorer.json") for k in objects}
    write(
        args.output / "construction_freeze.json",
        {"scorers": freeze, "criterion_arrays_opened": False},
    )

    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as arrays:
        if arrays["author_ids"].astype(str).tolist() != material["pandora"]["ids"]:
            raise ValueError("PANDORA criterion author order mismatch")
        outcomes = {"pandora": arrays["y"]}
    essay_labels = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index(
        "user_id"
    )
    outcomes["essays"] = essay_labels.loc[
        material["essays"]["ids"], list(TRAITS)
    ].to_numpy()
    if (
        not np.isin(outcomes["essays"], [0, 1]).all()
        or not ((outcomes["pandora"] >= 0) & (outcomes["pandora"] <= 1)).all()
    ):
        raise ValueError("Unexpected criterion scale")
    rows = []
    for key, bundle in objects.items():
        domain, method = bundle["domain"], bundle["method"]
        d, values = material[domain], outputs[key]["z"]
        y = outcomes[domain][d["held"]]
        for j, factor in enumerate(bundle["factor_ids"]):
            for k, trait in enumerate(TRAITS):
                rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        "factor": factor,
                        "criterion": trait,
                        "label_type": d["label_type"],
                        **correlation_row(values[:, j], y[:, k]),
                        "auc_fixed_factor_direction": float(
                            roc_auc_score(y[:, k], values[:, j])
                        )
                        if domain == "essays"
                        else None,
                    }
                )
    associations = pd.DataFrame(rows)
    associations["q_bh_100"] = bh_adjust(associations.p)
    associations.to_csv(args.output / "criterion_associations.csv", index=False)
    if (
        len(associations) != 100
        or associations.duplicated(["domain", "method", "factor", "criterion"]).any()
    ):
        raise ValueError("Incomplete planned criterion family")
    for key, sha in freeze.items():
        if digest(args.output / f"private_{key}_scorer.json") != sha:
            raise ValueError("Scorer changed after criterion access")
    summary = {
        "status": "REFERENCE_CONSTRUCTION_COMPLETE",
        "scope": "Four local structural reference scorers; not complete Study B or psychological instrument acceptance",
        "numeric": numerical,
        "criterion_connections": len(associations),
        "q_below_0_05": int((associations.q_bh_100 < 0.05).sum()),
        "wall_seconds": time.monotonic() - began,
        "new_neural_calls": 0,
        "new_fits": 2,
        "reused_source_fits": 2,
    }
    summary["files"] = {
        p.name: digest(p)
        for p in args.output.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "summary.json"
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/local_construction_r1",
    )
    with threadpool_limits(limits=2):
        run(parser.parse_args())
