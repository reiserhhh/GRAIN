#!/usr/bin/env python3
"""Bounded BLT-style common-factor versus matched PCA comparison and structure audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.common_factors import R_DRIVER, WordFactorReference
from grain_research.predecessors import mean_pearson
from grain_research.vector_mechanisms.external_heads import metrics
from scripts.audit_grain_predecessor_structure import loading_alignment
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_predecessor_reference import ALPHAS, paired_delta


def fit_reference(documents, train, method, output):
    """Fit and persist an outcome-blind transform; retain inadmissible-fit warnings."""
    feature = WordFactorReference(method=method).fit(documents[train])
    scaler = StandardScaler().fit(feature.transform(documents[train]))
    joblib.dump({"feature": feature, "scaler": scaler, "source_indices": train}, output)
    return feature, scaler


def structure_audit(documents, halves, train, held, reference, method, output):
    """Use vocabulary alignment from discovery data; do not take any criterion labels."""
    records = []
    base = reference.transform(documents[held])
    half = [reference.transform(h[held]) for h in halves]
    half_rows = [
        {
            "method": method,
            "factor": j + 1,
            "half_r": float(pearsonr(half[0][:, j], half[1][:, j]).statistic),
        }
        for j in range(base.shape[1])
    ]
    rng = np.random.default_rng(731)
    warnings = []
    for repeat in range(8):
        keep = rng.choice(train, int(0.8 * len(train)), replace=False)
        alternative = WordFactorReference(method=method).fit(documents[keep])
        row, col, sign, phi = loading_alignment(reference, alternative)
        inferred = alternative.transform(documents[held])
        for j, k, s, congruence in zip(row, col, sign, phi, strict=True):
            records.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "factor": int(j + 1),
                    "loading_congruence": float(congruence),
                    "held_score_r": float(
                        pearsonr(base[:, j], inferred[:, k] * s).statistic
                    ),
                }
            )
        if method == "common_factor":
            warnings.append(
                {
                    "repeat": repeat,
                    "warnings": alternative.model_.warnings_,
                    "admissible": alternative.model_.admissible_,
                }
            )
    pd.DataFrame(records).to_csv(output / f"{method}_dropout.csv", index=False)
    pd.DataFrame(half_rows).to_csv(output / f"{method}_split_half.csv", index=False)
    terms = reference.vectorizer_.get_feature_names_out()
    pd.DataFrame(reference.loadings_, index=terms).to_csv(
        output / f"private_{method}_pattern.csv"
    )
    result = {
        "labels_used_in_structure_or_alignment": False,
        "discovery_authors": len(train),
        "held_authors": len(held),
        "median_loading_congruence": float(
            pd.DataFrame(records).loading_congruence.median()
        ),
        "median_held_score_r": float(pd.DataFrame(records).held_score_r.median()),
        "median_half_r": float(pd.DataFrame(half_rows).half_r.median()),
        "dropout_diagnostics": warnings,
    }
    if method == "common_factor":
        pd.DataFrame(reference.model_.phi_).to_csv(
            output / "factor_phi.csv", index=False
        )
        pd.DataFrame(
            {"term": terms, "uniqueness": reference.model_.uniquenesses_}
        ).to_csv(output / "private_uniqueness.csv", index=False)
        result["source_fit"] = reference.model_.diagnostics_
        result["warnings"] = reference.model_.warnings_
    return result


def run(args):
    """Execute same-author folds, inner-only alpha selection and label-free structure."""
    start = time.monotonic()
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output directory; old results are immutable")
    out.mkdir(parents=True)
    model_dir = out / "private_models"
    model_dir.mkdir()
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if digest(args.material / name) != checksum:
            raise ValueError("Material checksum mismatch")
    data = json.loads((args.material / "private_material.json").read_text())
    documents = np.asarray(data["documents"])
    halves = [np.asarray(data[k]) for k in ("half_a", "half_b")]
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as z:
        y, folds, author_ids = z["y"], z["folds"], z["author_ids"]
    if len(set(author_ids)) != len(y) or len(documents) != len(y):
        raise ValueError("Author-level input alignment failed")
    if any(b["source_group"] != "big5_only" for b in data["bundles"]):
        raise ValueError("Only authorized source authors may be used")
    contract = {
        "status": "STARTED",
        "material_sha256": digest(args.material / "material_manifest.json"),
        "source_authors": len(y),
        "comments_per_author": 36,
        "methods": ["common_factor", "pca"],
        "factors": 5,
        "max_terms": 200,
        "term_value": "frequency/all_alphabetic_tokens; source_selected_English_vocabulary",
        "cv": "official_source_author_5fold; inner_author_KFold3_seed731",
        "alphas": list(ALPHAS),
        "dropout_repeats": 8,
        "seed": 731,
        "adaptation": "BLT-style; native MINRES/psych promax, not original equamax sequence",
        "code_sha256": {
            p.name: digest(p)
            for p in [
                Path(__file__),
                R_DRIVER,
                ROOT / "grain_research/common_factors.py",
            ]
        },
    }
    write(out / "contract.json", contract)
    all_pred, selection, fits, reference, summaries = {}, [], [], {}, []
    for method in contract["methods"]:
        pred = np.full_like(y, np.nan)
        for fold in range(5):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            inner = {alpha: np.full_like(y[train], np.nan) for alpha in ALPHAS}
            for k, (ii, jj) in enumerate(
                KFold(3, shuffle=True, random_state=731).split(train)
            ):
                feature, scale = fit_reference(
                    documents,
                    train[ii],
                    method,
                    model_dir / f"{method}_fold{fold}_inner{k}.joblib",
                )
                x = scale.transform(feature.transform(documents[train[ii]]))
                v = scale.transform(feature.transform(documents[train[jj]]))
                for alpha in ALPHAS:
                    inner[alpha][jj] = (
                        Ridge(alpha=alpha).fit(x, y[train[ii]]).predict(v)
                    )
                if method == "common_factor":
                    fits.append(
                        {
                            "fold": fold,
                            "inner": k,
                            "admissible": feature.model_.admissible_,
                            "warnings": feature.model_.warnings_,
                            **feature.model_.diagnostics_,
                        }
                    )
            scores = {a: mean_pearson(y[train], p) for a, p in inner.items()}
            if not np.isfinite(list(scores.values())).all():
                raise ValueError("Undefined inner CV score")
            chosen = max(ALPHAS, key=lambda a: scores[a])
            feature, scale = fit_reference(
                documents, train, method, model_dir / f"{method}_fold{fold}.joblib"
            )
            x, v = [
                scale.transform(feature.transform(documents[ix]))
                for ix in (train, test)
            ]
            head = Ridge(alpha=chosen).fit(x, y[train])
            pred[test] = head.predict(v)
            joblib.dump(head, model_dir / f"{method}_fold{fold}_head.joblib")
            selection.append(
                {
                    "method": method,
                    "fold": fold,
                    "alpha": chosen,
                    "inner_mean_r": scores[chosen],
                }
            )
            if method == "common_factor":
                fits.append(
                    {
                        "fold": fold,
                        "inner": "outer",
                        "admissible": feature.model_.admissible_,
                        "warnings": feature.model_.warnings_,
                        **feature.model_.diagnostics_,
                    }
                )
            if fold == 0:
                reference[method] = feature
            print(
                json.dumps(
                    {
                        "method": method,
                        "fold": fold,
                        "alpha": chosen,
                        "seconds": round(time.monotonic() - start, 1),
                    }
                ),
                flush=True,
            )
        all_pred[method] = pred
        summaries.extend({"method": method, **m} for m in metrics(y, pred, "big5"))
        np.savez_compressed(
            out / f"private_{method}_oof.npz",
            author_ids=author_ids,
            folds=folds,
            y=y,
            predictions=pred,
        )
    pd.DataFrame(summaries).to_csv(out / "metrics_by_trait.csv", index=False)
    pd.DataFrame(selection).to_csv(out / "inner_selection.csv", index=False)
    write(out / "native_fit_diagnostics.json", fits)
    structure = {}
    for method in contract["methods"]:
        structure[method] = structure_audit(
            documents,
            halves,
            np.flatnonzero(folds != 0),
            np.flatnonzero(folds == 0),
            reference[method],
            method,
            out,
        )
    write(out / "structure_summary.json", structure)
    result = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - start,
        "source_authors": len(y),
        "mean_pearson": {m: mean_pearson(y, p) for m, p in all_pred.items()},
        "paired_common_minus_pca": paired_delta(
            y, all_pred["common_factor"], all_pred["pca"]
        ),
        "all_20_common_fits_admissible": all(f["admissible"] for f in fits),
        "common_fit_count": len(fits),
        "scope": "Exploratory local adaptation; rank fixed before run, no psychological factor names.",
    }
    write(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Use fixed-budget local source data only; never open MEPS/bridge/lockboxes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "predecessor_integration_v1/common_factor_r1",
    )
    args = p.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
