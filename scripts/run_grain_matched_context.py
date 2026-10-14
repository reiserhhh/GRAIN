#!/usr/bin/env python3
"""Execute matched-token whole/piece semantic construction in two contexts."""

from __future__ import annotations

import argparse
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
from grain_research.local_bilinear import ConstructionFeatures, keyed_directions
from grain_research.matched_context import ARMS, id_hash, observations, pool_context
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, correlation_row, load_texts
from scripts.run_grain_paraphrase_reference import E5_REVISION
from scripts.run_grain_predecessor_reference import mean_pearson, paired_delta

PLAN = ROOT / "docs/GRAIN_MATCHED_CONTEXT_CONSTRUCTION_PLAN_V1.md"
PREVIOUS = BASE / "all_plan_closure_v1/local_bilinear_r2"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def local_tokenizer():
    """Load only cached pinned public weights/tokenizer; no download or remote code."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(
        "intfloat/e5-large-v2", revision=E5_REVISION, local_files_only=True
    )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False), path


def prepare(out):
    """Freeze source addresses and de-duplicated inputs before any new inference."""
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "plan": digest(PLAN),
        "script": digest(Path(__file__)),
        "module": digest(ROOT / "grain_research/matched_context.py"),
        "previous": digest(PREVIOUS / "contract.json"),
        "material": digest(DEFAULT / "material_manifest.json"),
        "essays": digest(ESSAYS),
        "budget": 500,
        "piece_size": 128,
        "arms": list(ARMS),
        "alpha": list(ALPHAS),
        "revision": E5_REVISION,
    }
    cp = out / "contract.json"
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError("Resume contract changed")
    write(cp, contract)
    if (out / "material.json").exists():
        meta = json.loads((out / "material.json").read_text())
        for name, sha in meta["files"].items():
            if digest(out / name) != sha:
                raise ValueError("Prepared inputs changed")
        return
    tok, _ = local_tokenizer()
    datasets = load_texts()
    unique, lookup, rows, coverage = [], {}, [], []
    for domain, d in datasets.items():
        old = json.loads((PREVIOUS / domain / "material.json").read_text())
        for name, sha in old["files"].items():
            if digest(PREVIOUS / domain / name) != sha:
                raise ValueError("Prior material changed")
        for u, text in enumerate(d["documents"]):
            packets, info = observations(text, tok)
            coverage.append({"domain": domain, "author_index": u, **info})
            for p in packets:
                ids = p.pop("ids")
                key = id_hash(ids)
                if key not in lookup:
                    lookup[key] = len(unique)
                    unique.append(ids)
                if unique[lookup[key]] != ids:
                    raise ValueError("Token commitment collision")
                rows.append(
                    {
                        "domain": domain,
                        "author_index": u,
                        "input_index": lookup[key],
                        "input_sha256": key,
                        **p,
                    }
                )
    write(out / "private_inputs.json", unique)
    pd.DataFrame(rows).to_csv(out / "private_addresses.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "private_coverage.csv", index=False)
    write(out / "private_authors.json", {k: v["ids"] for k, v in datasets.items()})
    write(
        out / "material.json",
        {
            "unique_inputs": len(unique),
            "logical_inputs": len(rows),
            "maximum_encoder_tokens": max(map(len, unique)),
            "domains": pd.DataFrame(coverage)
            .groupby("domain")
            .agg(
                authors=("author_index", "size"),
                complete=("complete_document", "sum"),
                median_coverage=("coverage", "median"),
                pieces=("pieces", "sum"),
            )
            .reset_index()
            .to_dict("records"),
            "files": {p.name: digest(p) for p in out.glob("private_*")},
        },
    )
    print((out / "material.json").read_text(), flush=True)


def encode(out, seconds):
    """Cache verified model-token batches; a terminal budget pause is resumable."""
    import torch
    import transformers
    from transformers import AutoModel

    torch.set_num_threads(4)
    tok, path = local_tokenizer()
    inputs = json.loads((out / "private_inputs.json").read_text())
    order = np.argsort([len(x) for x in inputs], kind="stable")
    inputs = [inputs[i] for i in order]
    cache = out / "encoder_cache"
    cache.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    contract = {
        "inputs": digest(out / "private_inputs.json"),
        "revision": E5_REVISION,
        "device": device,
        "dtype": "float32",
        "batch_size": 16,
        "pool": "masked_mean_all_nonpadding_then_L2",
        "cap": 512,
        "ordering": "stable_length_sort_then_restore_original_input_index",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    cp = cache / "contract.json"
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError("Encoding contract changed")
    write(cp, contract)
    began, chunks, model = time.monotonic(), [], None
    for start in range(0, len(inputs), 256):
        stop = min(start + 256, len(inputs))
        target = cache / f"batch_{start:06d}.npz"
        meta = target.with_suffix(".json")
        if target.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if (
                m["start"] != start
                or m["stop"] != stop
                or m["sha256"] != digest(target)
            ):
                raise ValueError("Changed encoder batch")
            with np.load(target, allow_pickle=False) as z:
                block = z["vectors"]
        else:
            if time.monotonic() - began > seconds:
                write(
                    cache / "status.json",
                    {
                        "status": "PAUSED_BUDGET",
                        "completed": start,
                        "total": len(inputs),
                    },
                )
                raise TimeoutError("Budget pause; verified completed batches retained")
            if model is None:
                model = (
                    AutoModel.from_pretrained(
                        path,
                        trust_remote_code=False,
                        use_safetensors=True,
                        dtype=torch.float32,
                    )
                    .eval()
                    .to(device)
                )
            blocks = []
            for i in range(start, stop, 16):
                native = inputs[i : min(i + 16, stop)]
                batch = tok.pad(
                    {"input_ids": native}, padding=True, return_tensors="pt"
                ).to(device)
                if batch["input_ids"].shape[1] > 512:
                    raise ValueError("No hidden truncation allowed")
                with torch.inference_mode():
                    h = model(**batch).last_hidden_state
                    mask = batch["attention_mask"][..., None]
                    v = torch.nn.functional.normalize(
                        (h * mask).sum(1) / mask.sum(1), dim=1
                    )
                blocks.append(v.cpu().numpy())
            block = np.concatenate(blocks)
            np.savez_compressed(target, vectors=block)
            write(meta, {"start": start, "stop": stop, "sha256": digest(target)})
        if block.shape != (stop - start, 1024) or not np.isfinite(block).all():
            raise ValueError("Malformed encoded vectors")
        chunks.append(block)
        print(
            json.dumps(
                {
                    "stage": "encode",
                    "completed": stop,
                    "total": len(inputs),
                    "invocation_seconds": time.monotonic() - began,
                }
            ),
            flush=True,
        )
    write(
        cache / "status.json",
        {"status": "COMPLETE", "completed": len(inputs), "total": len(inputs)},
    )
    restored = np.empty((len(inputs), 1024), dtype=np.float32)
    restored[order] = np.concatenate(chunks)
    return restored


def directions():
    """Reuse the complete pinned IPIP300 cache and fixed signed facet keys."""
    qs, end = [], 0
    for p in sorted((DEFAULT / "sentence_batches").glob("batch_*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        if meta["start"] != end or meta["sha256"] != digest(p):
            raise ValueError("Item cache provenance mismatch")
        with np.load(p, allow_pickle=False) as z:
            qs.append(z["vectors"])
        end = meta["stop"]
        if end >= 300:
            break
    items = json.loads((DEFAULT / "ipip_neo300.json").read_text())["items"]
    return keyed_directions(np.concatenate(qs)[:300], items)


def derive(out, vectors):
    """Compute every author's matched views and the exact score decomposition."""
    rows = pd.read_csv(out / "private_addresses.csv")
    d, names = directions()
    summary, diagnostics = [], []
    for domain, frame in rows.groupby("domain", sort=False):
        root = out / domain
        root.mkdir(exist_ok=True)
        features = {k: [] for k in (*ARMS, "direction", "radial")}
        for u, panel in frame.groupby("author_index", sort=True):
            whole = panel[panel.kind == "whole"]
            pieces = panel[panel.kind == "piece"].sort_values("lo")
            if len(whole) != 1:
                raise ValueError("One whole input per author required")
            f, radius = pool_context(
                vectors[whole.input_index.iloc[0]],
                vectors[pieces.input_index],
                (pieces.hi - pieces.lo).to_numpy(),
            )
            for k, values in features.items():
                values.append(f[k])
            a, b = f["direction"] @ d, f["radial"] @ d
            total = (f["whole"] - f["piece_mass"]) @ d
            diagnostics.append(
                {
                    "domain": domain,
                    "author_index": int(u),
                    "pieces": len(pieces),
                    "radius": radius,
                    "whole_unit_cosine": float(f["whole"] @ f["piece_mass_unit"]),
                    "affinity_total_energy": float(total @ total),
                    "affinity_direction_energy": float(a @ a),
                    "affinity_radial_energy": float(b @ b),
                    "affinity_cross_term": float(2 * a @ b),
                    "identity_max_error": float(np.max(abs(total - a - b))),
                    "equal_mass_distance": float(
                        np.linalg.norm(f["piece_equal"] - f["piece_mass"])
                    ),
                }
            )
        with np.load(PREVIOUS / domain / "private_blocks.npz", allow_pickle=False) as z:
            source = {k: z[k] for k in ("train", "val", "held")}
        np.savez_compressed(
            root / "private_features.npz", **features, **source, directions=d
        )
        summary.append({"domain": domain, "authors": len(features["whole"])})
    pd.DataFrame(diagnostics).to_csv(out / "private_geometry.csv", index=False)
    write(
        out / "derivation.json",
        {
            "facets": names,
            "domains": summary,
            "features": {
                k["domain"]: digest(out / k["domain"] / "private_features.npz")
                for k in summary
            },
        },
    )


def evaluate(out):
    """Freeze source-selected heads and all score routes, then report held criteria."""
    datasets = load_texts()
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    table = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = table.loc[datasets["essays"]["ids"], list(TRAITS)].to_numpy()
    facets = json.loads((out / "derivation.json").read_text())["facets"]
    mapping = dict(zip(TRAITS, ("E", "N", "A", "C", "O"), strict=True))
    for domain, ds in datasets.items():
        root = out / domain
        if (root / "summary.json").exists():
            prior = json.loads((root / "summary.json").read_text())
            for name, sha in prior["files"].items():
                if digest(root / name) != sha:
                    raise ValueError("Completed output changed")
            continue
        with np.load(root / "private_features.npz", allow_pickle=False) as z:
            f = {k: z[k] for k in z.files}
        train, val, held = (f[k] for k in ("train", "val", "held"))
        source = np.sort(np.r_[train, val])
        if len(np.unique(np.r_[source, held])) != len(ds["ids"]):
            raise ValueError("Source/validation/held must partition authors")
        y, docs, predictions, raw, tuning, cold = (
            ys[domain],
            ds["documents"],
            {},
            {},
            [],
            [],
        )
        for arm in ARMS:
            block = {"semantic": f[arm]}
            ft = ConstructionFeatures("semantic").fit(block, docs, train)
            x, xv = (ft.transform(block, docs, ix) for ix in (train, val))
            trials = []
            for alpha in ALPHAS:
                head = Ridge(alpha=alpha, solver="lsqr").fit(x, y[train])
                trials.append(
                    {
                        "arm": arm,
                        "alpha": alpha,
                        "validation_mean_r": float(
                            mean_pearson(y[val], head.predict(xv))
                        ),
                    }
                )
            best = max(trials, key=lambda t: t["validation_mean_r"])
            tuning.extend(trials)
            ft = ConstructionFeatures("semantic").fit(block, docs, source)
            head = Ridge(alpha=best["alpha"], solver="lsqr").fit(
                ft.transform(block, docs, source), y[source]
            )
            beta = ft.pca_.components_.T @ (head.coef_.T / ft.scaler_.scale_[:, None])
            intercept = (
                head.intercept_
                - ft.scaler_.mean_ @ (head.coef_.T / ft.scaler_.scale_[:, None])
                - ft.pca_.mean_ @ beta
            )
            joblib.dump(
                {
                    "features": ft,
                    "head": head,
                    "source_indices": source,
                    "chosen": best,
                },
                root / f"private_model_{arm}.joblib",
            )
            write(
                root / f"model_{arm}.json",
                {
                    "sha256": digest(root / f"private_model_{arm}.joblib"),
                    "chosen": best,
                },
            )
            predictions[arm] = head.predict(ft.transform(block, docs, held))
            direct = f[arm][held] @ beta + intercept
            np.testing.assert_allclose(direct, predictions[arm], atol=1e-10)
            single = np.stack(
                [head.predict(ft.transform(block, docs, [int(u)]))[0] for u in held]
            )
            np.testing.assert_allclose(single, predictions[arm], atol=1e-10)
            np.savez_compressed(
                root / f"private_affine_{arm}.npz", beta=beta, intercept=intercept
            )
            raw[arm] = np.column_stack(
                [
                    (f[arm] @ f["directions"])[
                        :, [i for i, n in enumerate(facets) if n.startswith(mapping[t])]
                    ].mean(1)
                    for t in TRAITS
                ]
            )
            cold.append(
                {
                    "arm": arm,
                    "affine_error": float(np.max(abs(direct - predictions[arm]))),
                    "single_error": float(np.max(abs(single - predictions[arm]))),
                }
            )
            print(json.dumps({"domain": domain, **best}), flush=True)
        # All prediction families are frozen before inspecting held contrasts.
        np.savez_compressed(root / "private_predictions.npz", **predictions, held=held)
        np.savez_compressed(root / "private_anchor_scores.npz", **raw)
        metric, link, agreements, delta = [], [], [], []
        for arm in ARMS:
            for j, t in enumerate(TRAITS):
                p = predictions[arm][:, j]
                metric.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "trait": t,
                        **correlation_row(y[held, j], p),
                        "mae": float(np.mean(abs(p - y[held, j]))),
                        "rmse": float(np.sqrt(np.mean((p - y[held, j]) ** 2))),
                        "auc": float(roc_auc_score(y[held, j], p))
                        if domain == "essays"
                        else None,
                    }
                )
                link.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "trait": t,
                        **correlation_row(y[held, j], raw[arm][held, j]),
                    }
                )
                agreements.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "trait": t,
                        **correlation_row(raw["whole"][held, j], raw[arm][held, j]),
                    }
                )
            if arm != "whole":
                delta.append(
                    {
                        "arm": arm,
                        "reference": "whole",
                        **paired_delta(
                            y[held], predictions[arm], predictions["whole"], draws=1000
                        ),
                    }
                )
        old = PREVIOUS / domain
        old_summary = json.loads((old / "summary.json").read_text())
        if (
            digest(old / "private_predictions.npz")
            != old_summary["files"]["private_predictions.npz"]
        ):
            raise ValueError("Changed full-material scores")
        with np.load(old / "private_predictions.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["held_indices"], held)
            delta.append(
                {
                    "arm": "full_material_semantic",
                    "reference": "whole",
                    "scope_note": "coverage AND boundary/weighting change, not isolated slicing",
                    **paired_delta(
                        y[held], z["semantic"], predictions["whole"], draws=1000
                    ),
                }
            )
        for name, rows in (
            ("metrics_by_trait", metric),
            ("anchor_links", link),
            ("anchor_agreement", agreements),
            ("source_tuning", tuning),
        ):
            pd.DataFrame(rows).to_csv(root / (name + ".csv"), index=False)
        write(
            root / "summary.json",
            {
                "domain": domain,
                "source": len(source),
                "held": len(held),
                "mean_r": pd.DataFrame(metric).groupby("arm").r.mean().to_dict(),
                "mean_anchor_r": pd.DataFrame(link).groupby("arm").r.mean().to_dict(),
                "deltas": delta,
                "cold_checks": cold,
                "files": {
                    p.name: digest(p)
                    for p in root.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                },
            },
        )


def main():
    """Prepare, encode with bounded resumability, derive and evaluate fixed arms."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/matched_context_r3"
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--max-seconds", type=float, default=900)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        prepare(args.output)
        if not args.prepare_only:
            vectors = encode(args.output, args.max_seconds)
            # Do not rewrite completed model input artifacts on resume.
            if not (args.output / "derivation.json").exists():
                derive(args.output, vectors)
            else:
                for domain, sha in json.loads(
                    (args.output / "derivation.json").read_text()
                )["features"].items():
                    if digest(args.output / domain / "private_features.npz") != sha:
                        raise ValueError("Derived features changed")
            evaluate(args.output)


if __name__ == "__main__":
    main()
