#!/usr/bin/env python3
"""Same-material co-location versus marginal controls in two local contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.local_bilinear import (
    ARMS,
    ConstructionFeatures,
    author_blocks,
    chunk_spans,
    keyed_directions,
)
from grain_research.semantic_link import validate_reference_exposure
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, correlation_row, load_texts
from scripts.run_grain_native_reference import E1
from scripts.run_grain_paraphrase_reference import E5_REVISION, encode_batches
from scripts.run_grain_predecessor_reference import mean_pearson, paired_delta

PLAN = ROOT / "docs/GRAIN_LOCAL_BILINEAR_CONSTRUCTION_PLAN_V1.md"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def prepare(args):
    """Freeze source, units and pretrained references; resume only matching inputs."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "plan_sha256": digest(PLAN),
        "code_sha256": digest(Path(__file__)),
        "pool_code_sha256": digest(ROOT / "grain_research/local_bilinear.py"),
        "material_sha256": digest(DEFAULT / "material_manifest.json"),
        "essays_sha256": digest(ESSAYS),
        "domains": args.domains,
        "arms": ARMS,
        "alphas": ALPHAS,
        "chunk_tokens": 128,
        "bootstrap": 1000,
        "seeds": [924, 925, 926],
        "runtime_sha256": digest(
            E1 / "artifacts/common_source_comment_representation.joblib"
        ),
        "role": "source-selected Big5 heads; GRAIN unit interface, not support-gated G1",
    }
    contract = json.loads(json.dumps(contract))
    if (out / "contract.json").exists():
        if json.loads((out / "contract.json").read_text()) != contract:
            raise ValueError("Changed resume contract")
    else:
        write(out / "contract.json", contract)
    datasets = load_texts()
    previous = json.loads((BASE / "vm05_preflight_r1/manifest.json").read_text())
    if previous["runtime_sha256"] != contract["runtime_sha256"]:
        raise ValueError("Changed GRAIN runtime")
    runtime = joblib.load(E1 / "artifacts/common_source_comment_representation.joblib")
    validate_reference_exposure(
        pd.read_csv(BASE / "vm05_preflight_r1/budget_by_author.csv"),
        datasets["pandora"]["ids"],
    )
    sentence_contract = json.loads(
        (DEFAULT / "sentence_batches/contract.json").read_text()
    )
    if (
        sentence_contract["model"] != "intfloat/e5-large-v2"
        or sentence_contract["revision"] != E5_REVISION
    ):
        raise ValueError("Changed item encoder")
    item_vectors, next_start = [], 0
    for item_batch in sorted((DEFAULT / "sentence_batches").glob("batch_*.npz")):
        item_meta = json.loads(item_batch.with_suffix(".json").read_text())
        if (
            digest(item_batch) != item_meta["sha256"]
            or item_meta["start"] != next_start
        ):
            raise ValueError("Item vector cache mismatch")
        with np.load(item_batch, allow_pickle=False) as a:
            if len(a["vectors"]) != item_meta["stop"] - item_meta["start"]:
                raise ValueError("Item batch row mismatch")
            item_vectors.append(a["vectors"])
        next_start = item_meta["stop"]
        if next_start >= 300:
            break
    q = np.concatenate(item_vectors)[:300]
    items = json.loads((DEFAULT / "ipip_neo300.json").read_text())["items"]
    direction, names = keyed_directions(q, items)
    origin = json.loads((E1 / "run_manifest.json").read_text())
    if Path(origin["input"]).name != "tier_u_comments.parquet":
        raise ValueError("Unexpected discovery dataset; recheck Essays separation")
    for domain in args.domains:
        root = out / domain
        root.mkdir(exist_ok=True)
        summary_path = root / "material.json"
        if summary_path.exists():
            info = json.loads(summary_path.read_text())
            for name, sha in info["files"].items():
                if digest(root / name) != sha:
                    raise ValueError("Changed prepared features")
            continue
        d = datasets[domain]
        units, author, addresses = [], [], []
        for u, document in enumerate(d["documents"]):
            if domain == "pandora":
                start = 0
                for j, text in enumerate(d["units"][u]):
                    units.append(text)
                    author.append(u)
                    addresses.append(
                        {
                            "author_index": u,
                            "unit": j,
                            "start": start,
                            "end": start + len(text),
                        }
                    )
                    start += len(text) + 2
            else:
                for j, (a, b) in enumerate(chunk_spans(document)):
                    units.append(document[a:b])
                    author.append(u)
                    addresses.append(
                        {"author_index": u, "unit": j, "start": a, "end": b}
                    )
        g = runtime.transform(units)
        if g.shape != (len(units), 24):
            raise ValueError("Unexpected GRAIN shape")
        if domain == "pandora":
            with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as a:
                if a["author_ids"].tolist() != d["ids"]:
                    raise ValueError("PANDORA row mismatch")
                h = a["comments"].reshape(len(units), -1)
            encoder = {
                "mode": "verified_existing36comment_E5",
                "revision": E5_REVISION,
                "device": "historical_cuda",
                "cap": 512,
                "prefix": "query: ",
                "material_sha256": contract["material_sha256"],
            }
        else:
            h, encoder = encode_batches(
                units,
                root / "encoder_cache",
                "intfloat/e5-large-v2",
                E5_REVISION,
                prefix="query: ",
                cap=512,
                max_seconds=900,
            )
        norms = np.linalg.norm(h, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("Zero semantic vector")
        hn = h / norms[:, None]
        semantic = hn @ direction
        blocks, counts, energy = author_blocks(g, h, semantic, author)
        if domain == "pandora":
            with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as a:
                folds = a["folds"]
            val, held = np.flatnonzero(folds == 1), np.flatnonzero(folds == 0)
        else:
            table = pd.read_csv(ESSAYS, usecols=["user_id", "essay_fold"])
            if table.user_id.astype(str).tolist() != d["ids"]:
                raise ValueError("Essays row mismatch")
            folds = table.essay_fold.to_numpy()
            val, held = np.flatnonzero(folds == 8), np.flatnonzero(folds == 9)
        train = np.setdiff1d(np.arange(len(d["ids"])), np.r_[val, held])
        np.savez_compressed(
            root / "private_blocks.npz",
            **blocks,
            train=train,
            val=val,
            held=held,
            unit_count=counts,
            covariance_energy=energy,
        )
        np.savez_compressed(
            root / "private_units.npz", g=g, h=h, semantic=semantic, authors=author
        )
        pd.DataFrame(addresses).to_csv(root / "private_addresses.csv", index=False)
        write(
            root / "private_authors.json",
            {
                "ids": d["ids"],
                "document_hashes": [
                    hashlib.sha256(t.encode()).hexdigest() for t in d["documents"]
                ],
            },
        )
        write(
            root / "material.json",
            {
                "domain": domain,
                "authors": len(d["ids"]),
                "units": len(units),
                "training": len(train),
                "validation": len(val),
                "held": len(held),
                "facet_names": names,
                "encoder": encoder,
                "single_unit_authors": int(np.sum(counts == 1)),
                "files": {
                    p.name: digest(p)
                    for p in root.glob("private_*")
                    if not p.name.startswith("._")
                },
            },
        )
        print(
            json.dumps(
                {
                    "stage": "material",
                    "domain": domain,
                    "units": len(units),
                    "authors": len(d["ids"]),
                }
            ),
            flush=True,
        )


def evaluate(args):
    """Select only on source-validation; freeze all predictions before comparisons."""
    datasets = load_texts()
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    essay = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = essay.loc[datasets["essays"]["ids"], list(TRAITS)].to_numpy()
    for domain in args.domains:
        root = args.output / domain
        if (root / "summary.json").exists():
            saved = json.loads((root / "summary.json").read_text())
            for name, sha in saved["files"].items():
                if digest(root / name) != sha:
                    raise ValueError("Changed evaluated result")
            continue
        start = time.monotonic()
        with np.load(root / "private_blocks.npz", allow_pickle=False) as z:
            blocks = {k: z[k] for k in z.files}
        train, val, held = (blocks[k] for k in ("train", "val", "held"))
        source = np.sort(np.r_[train, val])
        if set(source) & set(held):
            raise ValueError("Source/held overlap")
        docs, y = datasets[domain]["documents"], ys[domain]
        predictions, tuning, selections, checks = {}, [], [], []
        for arm in ARMS:
            checkpoint = root / f"private_model_{arm}.joblib"
            meta = root / f"model_{arm}.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                if digest(checkpoint) != m["model_sha256"]:
                    raise ValueError("Changed saved model")
                obj = joblib.load(checkpoint)
                tuning.extend(m["tuning"])
            else:
                features = ConstructionFeatures(arm).fit(blocks, docs, train)
                x, xv = (features.transform(blocks, docs, ix) for ix in (train, val))
                trials = []
                for alpha in ALPHAS:
                    head = Ridge(alpha=alpha, solver="lsqr").fit(x, y[train])
                    r = mean_pearson(y[val], head.predict(xv))
                    trials.append(
                        {"arm": arm, "alpha": alpha, "validation_mean_r": float(r)}
                    )
                best = max(trials, key=lambda t: t["validation_mean_r"])
                features = ConstructionFeatures(arm).fit(blocks, docs, source)
                head = Ridge(alpha=best["alpha"], solver="lsqr").fit(
                    features.transform(blocks, docs, source), y[source]
                )
                obj = {
                    "features": features,
                    "head": head,
                    "source_indices": source,
                    "chosen": best,
                }
                joblib.dump(obj, checkpoint)
                m = {"arm": arm, "model_sha256": digest(checkpoint), "tuning": trials}
                write(meta, m)
                tuning.extend(trials)
            np.testing.assert_array_equal(obj["features"].fit_indices_, source)
            feature, head = obj["features"], obj["head"]
            xh = feature.transform(blocks, docs, held)
            predictions[arm] = head.predict(xh)
            direct = np.asarray(xh @ head.coef_.T) + head.intercept_
            np.testing.assert_allclose(direct, predictions[arm], atol=1e-12)
            single = np.vstack(
                [
                    head.predict(feature.transform(blocks, docs, [int(i)]))[0]
                    for i in held
                ]
            )
            np.testing.assert_allclose(single, predictions[arm], atol=1e-10)
            checks.append(
                {
                    "arm": arm,
                    "columns": xh.shape[1],
                    "linear_replay_error": float(
                        np.max(np.abs(direct - predictions[arm]))
                    ),
                    "single_batch_error": float(
                        np.max(np.abs(single - predictions[arm]))
                    ),
                }
            )
            selections.append(obj["chosen"])
            print(
                json.dumps({"stage": "fit", "domain": domain, **obj["chosen"]}),
                flush=True,
            )
        # Complete family frozen before held criterion comparisons or rankings.
        np.savez_compressed(
            root / "private_predictions.npz", **predictions, held_indices=held
        )
        rows = []
        for arm, p in predictions.items():
            for j, trait in enumerate(TRAITS):
                row = correlation_row(y[held, j], p[:, j])
                error = p[:, j] - y[held, j]
                rows.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "trait": trait,
                        **row,
                        "mae": float(np.mean(abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "auc": float(roc_auc_score(y[held, j], p[:, j]))
                        if domain == "essays"
                        else None,
                    }
                )
        metrics = pd.DataFrame(rows)
        metrics.to_csv(root / "metrics_by_trait.csv", index=False)
        metrics.groupby("arm", sort=False)[
            ["r", "rho", "mae", "rmse", "auc"]
        ].mean().to_csv(root / "metrics_by_arm.csv")
        pd.DataFrame(tuning).to_csv(root / "source_tuning.csv", index=False)
        pairs = [
            ("paired", arm)
            for arm in ("additive", "outer", "shuffle924", "shuffle925", "shuffle926")
        ] + [("additive", "semantic")]
        deltas = [
            {
                "arm": a,
                "reference": b,
                **paired_delta(y[held], predictions[a], predictions[b], draws=1000),
            }
            for a, b in pairs
        ]
        write(
            root / "summary.json",
            {
                "domain": domain,
                "scope": "same-unit co-location contrast, source-selected continuous heads",
                "source": len(source),
                "held": len(held),
                "selections": selections,
                "numeric_checks": checks,
                "deltas": deltas,
                "fitting_wall_seconds": time.monotonic() - start,
                "files": {
                    p.name: digest(p)
                    for p in root.glob("*")
                    if p.is_file() and not p.name.startswith("._")
                },
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/local_bilinear_r2"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["pandora", "essays"],
        default=["pandora", "essays"],
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        prepare(args)
        if not args.prepare_only:
            evaluate(args)
