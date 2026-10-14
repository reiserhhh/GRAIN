#!/usr/bin/env python3
"""Bounded same-sentence GRAIN/semantic-affinity experiment using local caches."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import verify_frozen_top1
from grain_research.semantic_link import (
    SemanticLink,
    affinity_targets,
    author_mean,
    author_weights,
    opportunity_numeric,
    paired_author_mse,
    validate_addresses,
    validate_reference_exposure,
    weighted_metrics,
)
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_native_reference import E1

PRODUCER = BASE / "predecessor_integration_v1/paraphrase_r1"
PLAN = ROOT / "docs/GRAIN_SAME_ADDRESS_SEMANTIC_LINK_PLAN_V1.md"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
MODES = ("opportunity", "grain_linear", "joint_linear", "joint_quadratic")


def prepare(out):
    """Verify lineage/addresses and construct paired coordinates without labels."""
    manifest = json.loads((DEFAULT / "material_manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if digest(DEFAULT / name) != checksum:
            raise ValueError("Changed source material")
    data = json.loads((DEFAULT / "private_material.json").read_text())
    comments = validate_addresses(data)
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        # Questionnaire outcomes are intentionally never deserialized.
        ids, folds = z["author_ids"], z["folds"]
    if ids.tolist() != [b["author_id"] for b in data["bundles"]]:
        raise ValueError("Author material/array order differs")
    if folds.tolist() != [b["official_folds"]["big5"] for b in data["bundles"]]:
        raise ValueError("Author folds differ")
    previous = json.loads((BASE / "vm05_preflight_r1/manifest.json").read_text())
    runtime_path = E1 / "artifacts/common_source_comment_representation.joblib"
    if digest(runtime_path) != previous["runtime_sha256"]:
        raise ValueError("Frozen GRAIN representation changed")
    exposure = pd.read_csv(BASE / "vm05_preflight_r1/budget_by_author.csv")
    validate_reference_exposure(exposure, ids)
    evidence = pd.read_csv(PRODUCER / "private_sentence_evidence.csv")
    verified = verify_frozen_top1(DEFAULT, PRODUCER, evidence, data)
    blocks = []
    for path in sorted((PRODUCER / "paraphrase_cache").glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            blocks.append(z["vectors"])
    vectors = np.concatenate(blocks)
    items = json.loads((DEFAULT / "ipip_neo300.json").read_text())["items"]
    target, names = affinity_targets(
        vectors[len(items) :], vectors[: len(items)], items
    )
    runtime = joblib.load(runtime_path)
    g = runtime.transform([s["text"] for s in data["sentences"]])
    z = opportunity_numeric(data["sentences"], comments)
    addresses = pd.DataFrame(data["sentences"]).drop(columns="text")
    authors = addresses.author_index.to_numpy(int)
    sentence_folds = folds[authors]
    sub = np.asarray(
        [
            comments[s["author_index"]][s["comment_index"]]["subreddit"]
            for s in data["sentences"]
        ],
        str,
    )
    if set(np.unique(authors)) != set(range(len(ids))) or g.shape[1] != 24:
        raise ValueError("Unexpected represented population or GRAIN dimensions")
    write(
        out / "contract.json",
        {
            "plan_sha256": digest(PLAN),
            "script_sha256": digest(Path(__file__)),
            "module_sha256": digest(ROOT / "grain_research/semantic_link.py"),
            "material_manifest_sha256": digest(DEFAULT / "material_manifest.json"),
            "runtime_sha256": digest(runtime_path),
            "cache_verification": verified,
            "coordinate_scope": "Frozen external-discovery TF-IDF/SVD24 on sentences; not full GRAIN/native author geometry/H16 factors",
            "target_scope": "60 model-derived keyed affinity contrasts, NOT personality outcomes or verified evidence",
            "source_authors": int(np.sum(folds != 0)),
            "held_authors": int(np.sum(folds == 0)),
            "sentences": len(authors),
            "source_sentences": int(np.sum(sentence_folds != 0)),
            "held_sentences": int(np.sum(sentence_folds == 0)),
            "target_names": names,
            "source_fit_folds": [2, 3, 4],
            "validation_fold": 1,
            "held_fold": 0,
            "alphas": ALPHAS,
            "models": MODES,
            "negative_control_alpha": 100,
            "target_scaling": "author-weighted source SD, separately within each fit",
            "selection": "equal-author mean normalized MSE on source fold1 only",
            "held_scope": "historically explored development authors; new target/reader held-out, not a fresh lockbox",
            "questionnaire_arrays_accessed": False,
            "new_inference_calls": 0,
            "subreddit_min_source_authors": 10,
            "bootstrap": {"draws": 500, "seed": 913},
            "time_limit_seconds": 1200,
            "python": platform.python_version(),
        },
    )
    addresses["fold"] = sentence_folds
    addresses["accepted_top1"] = evidence.accepted.to_numpy()
    addresses["item_id"] = evidence.item_id.to_numpy()
    addresses.to_csv(out / "private_addresses.csv", index=False)
    np.savez_compressed(
        out / "private_inputs.npz",
        g=g,
        z=z,
        target=target,
        authors=authors,
        folds=sentence_folds,
        subreddits=sub,
        author_ids=ids,
    )
    return g, z, target, authors, sentence_folds, sub, names, addresses


def run(args):
    """Fit declared readers, evaluate held authors once and export bounded results."""
    start = time.monotonic()
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output directory; results are not overwritten")
    out.mkdir(parents=True)
    try:
        with threadpool_limits(limits=2):
            g, z, target, authors, folds, sub, names, addresses = prepare(out)
            train, val, source, held = folds >= 2, folds == 1, folds != 0, folds == 0

            def fit(mode, alpha, mask, values):
                """Fit only the selected, author-disjoint portion of inputs."""
                if time.monotonic() - start > 1200:
                    raise TimeoutError("Declared CPU tranche wall-time budget reached")
                return SemanticLink(mode, alpha).fit(
                    g[mask], z[mask], sub[mask], authors[mask], values
                )

            tuning, predictions, fitted = [], {}, {}
            # No held target enters the following selection/refit loop.
            for mode in MODES:
                losses = []
                for alpha in ALPHAS:
                    reader = fit(mode, alpha, train, target[train])
                    prediction = reader.predict(g[val], z[val], sub[val])
                    loss = weighted_metrics(
                        target[val], prediction, authors[val], reader.y_scaler_.scale_
                    )[2].mean()
                    tuning.append(
                        {
                            "model": mode,
                            "alpha": alpha,
                            "validation_normalized_mse": float(loss),
                        }
                    )
                    losses.append(loss)
                alpha = ALPHAS[int(np.argmin(losses))]
                reader = fit(mode, alpha, source, target[source])
                fitted[mode] = reader
                joblib.dump(reader, out / f"{mode}.joblib")
                predictions[mode] = reader.predict(g[held], z[held], sub[held])
                print(
                    json.dumps({"model": mode, "source_selected_alpha": alpha}),
                    flush=True,
                )
            rng = np.random.default_rng(913)
            control = fit(
                "joint_quadratic", 100.0, source, rng.permutation(target[source])
            )
            joblib.dump(control, out / "permuted_source.joblib")
            fitted["permuted_source"] = control
            predictions["permuted_source"] = control.predict(
                g[held], z[held], sub[held]
            )
            pd.DataFrame(tuning).to_csv(out / "source_tuning.csv", index=False)
            scale = (
                StandardScaler()
                .fit(target[source], sample_weight=author_weights(authors[source]))
                .scale_
            )
            metrics, contrasts, aggregates = [], [], []
            ay_ids, ay = author_mean(target[held], authors[held])
            for mode, pred in predictions.items():
                reloaded = joblib.load(out / f"{mode}.joblib").predict(
                    g[held], z[held], sub[held]
                )
                if not np.array_equal(reloaded, pred):
                    raise ValueError("Frozen model reload changed predictions")
                _, ap = author_mean(pred, authors[held])
                r, mse, nmse = weighted_metrics(
                    target[held], pred, authors[held], scale
                )
                ar = weighted_metrics(ay, ap, ay_ids, scale)[0]
                for j, name in enumerate(names):
                    kind, facet = name.split(":")
                    metrics.append(
                        {
                            "model": mode,
                            "function": kind,
                            "facet": facet,
                            "sentence_affinity_r": r[j],
                            "author_affinity_r": ar[j],
                            "mse": mse[j],
                            "normalized_mse": nmse[j],
                        }
                    )
                for kind in ("mean", "max"):
                    ix = np.array([n.startswith(kind + ":") for n in names])
                    aggregates.append(
                        {
                            "model": mode,
                            "function": kind,
                            "mean_sentence_affinity_r": float(np.nanmean(r[ix])),
                            "mean_author_affinity_r": float(np.nanmean(ar[ix])),
                            "mean_normalized_mse": float(np.mean(nmse[ix])),
                        }
                    )
            for mode, reference in (
                ("joint_linear", "opportunity"),
                ("joint_quadratic", "joint_linear"),
                ("joint_linear", "grain_linear"),
            ):
                for kind in ("mean", "max"):
                    ix = np.array([n.startswith(kind + ":") for n in names])
                    contrasts.append(
                        {
                            "model": mode,
                            "reference": reference,
                            "function": kind,
                            **paired_author_mse(
                                target[held][:, ix],
                                predictions[mode][:, ix],
                                predictions[reference][:, ix],
                                authors[held],
                                scale[ix],
                            ),
                        }
                    )
            pd.DataFrame(metrics).to_csv(out / "affinity_by_facet.csv", index=False)
            pd.DataFrame(aggregates).to_csv(out / "affinity_summary.csv", index=False)
            pd.DataFrame(contrasts).to_csv(out / "paired_comparisons.csv", index=False)
            np.savez_compressed(
                out / "private_held_predictions.npz",
                target=target[held],
                authors=authors[held],
                sentence_index=np.flatnonzero(held),
                **predictions,
            )
            # Attach existing small source-only annotations; never fit/tune on them.
            annotated_path = (
                BASE
                / "predecessor_integration_v1/evidence_audit_r3/private_pair_assessment.csv"
            )
            annotations = pd.read_csv(annotated_path)
            indices = annotations.sentence_index.to_numpy(int)
            if (
                np.any(folds[indices] == 0)
                or annotations.sentence_index.duplicated().any()
            ):
                raise ValueError("Annotation source split/identity changed")
            old_data = json.loads((DEFAULT / "private_material.json").read_text())
            for row in annotations.itertuples():
                if (
                    row.sentence != old_data["sentences"][row.sentence_index]["text"]
                    or row.author_index != authors[row.sentence_index]
                ):
                    raise ValueError("Annotation address/text mismatch")
                if row.item_id != addresses.iloc[row.sentence_index].item_id:
                    raise ValueError("Annotation item mismatch")
            annotation_join = annotations.merge(
                addresses.reset_index(names="sentence_index"),
                on=["sentence_index", "author_index", "item_id"],
                validate="one_to_one",
            )
            for j in range(g.shape[1]):
                annotation_join[f"g{j:02d}"] = g[indices, j]
            annotation_join.to_csv(
                out / "private_annotation_coordinate_join.csv", index=False
            )
            counts = (
                annotation_join.groupby(
                    ["self_attribution", "scope", "relation", "nli"]
                )
                .size()
                .rename("pairs")
            )
            counts.to_csv(out / "annotation_join_counts.csv")
            # Export every learned weight, without public text or subreddit names.
            coeffs = []
            for mode in MODES:
                reader = fitted[mode]
                for j, name in enumerate(names):
                    for k, value in enumerate(reader.head_.coef_[j]):
                        coeffs.append(
                            {
                                "model": mode,
                                "target": name,
                                "input_code": f"X{k:03d}",
                                "standardized_coefficient": float(value),
                            }
                        )
            pd.DataFrame(coeffs).to_csv(out / "coefficients.csv", index=False)
            summary = {
                "status": "SCOPED_INTERFACE_COMPLETE",
                "aggregates": aggregates,
                "paired_comparisons": contrasts,
                "annotation_pairs_joined": len(annotation_join),
                "annotation_authors": annotation_join.author_index.nunique(),
                "annotation_scope": "existing source-only model/NLI judgments, not expert truth or held performance",
                "reloaded_prediction_max_error": 0.0,
                "wall_seconds": time.monotonic() - start,
                "disjoint_author_sets": len(set(authors[source]) & set(authors[held]))
                == 0,
                "personality_prediction_claim": False,
                "larger_grid_started": False,
                "sources": {
                    "annotation": digest(annotated_path),
                    "contract": digest(out / "contract.json"),
                },
                "artifacts": {
                    p.name: digest(p)
                    for p in out.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                },
            }
            write(out / "summary.json", summary)
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "artifacts"}, indent=2
                ),
                flush=True,
            )
    except Exception as exc:
        (out / "error_log.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


def main():
    """Run the explicitly bounded cached interface tranche, not old queues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/semantic_link_r1"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
