#!/usr/bin/env python3
"""Replay matched token inputs, source-only scores and geometric decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, load_texts
from scripts.run_grain_matched_context import ARMS, PREVIOUS, local_tokenizer
from scripts.run_grain_predecessor_reference import paired_delta


def run(out):
    """Use independent pooling and coefficient arithmetic; do not refit or infer."""
    material = json.loads((out / "material.json").read_text())
    for name, sha in material["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Input material hash mismatch")
    inputs = json.loads((out / "private_inputs.json").read_text())
    addresses = pd.read_csv(out / "private_addresses.csv")
    coverage = pd.read_csv(out / "private_coverage.csv")
    original_ids = json.loads((out / "private_authors.json").read_text())
    datasets = load_texts()
    tok, _ = local_tokenizer()
    prefix = tok.encode("query: ", add_special_tokens=False)
    token_checks = 0
    for domain, data in datasets.items():
        if data["ids"] != original_ids[domain]:
            raise ValueError("Author order mismatch")
        for u, doc in enumerate(data["documents"]):
            encoded = tok(doc, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded["input_ids"]
            panel = addresses[
                (addresses.domain == domain) & (addresses.author_index == u)
            ]
            c = coverage[
                (coverage.domain == domain) & (coverage.author_index == u)
            ].iloc[0]
            assert c.text_sha256 == hashlib.sha256(doc.encode()).hexdigest()
            assert c.original_tokens == len(ids)
            assert c.used_tokens == min(500, len(ids))
            assert c.omitted_tokens == len(ids) - c.used_tokens
            collected = []
            for row in panel.itertuples():
                body = ids[row.lo : row.hi]
                expected = [tok.cls_token_id, *prefix, *body, tok.sep_token_id]
                assert inputs[row.input_index] == expected
                assert row.char_lo == encoded["offset_mapping"][row.lo][0]
                assert row.char_hi == encoded["offset_mapping"][row.hi - 1][1]
                if row.kind == "piece":
                    collected.extend(body)
                token_checks += 1
            assert collected == ids[: int(c.used_tokens)]
    order = np.argsort([len(x) for x in inputs], kind="stable")
    batches, stop = [], 0
    for path in sorted((out / "encoder_cache").glob("batch_*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["start"] == stop and meta["sha256"] == digest(path)
        with np.load(path, allow_pickle=False) as z:
            assert len(z["vectors"]) == meta["stop"] - stop
            batches.append(z["vectors"])
        stop = meta["stop"]
    assert stop == len(inputs)
    v = np.empty((len(inputs), 1024))
    v[order] = np.concatenate(batches)
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    table = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = table.loc[datasets["essays"]["ids"], list(TRAITS)].to_numpy()
    diagnostics = pd.read_csv(out / "private_geometry.csv")
    checks, norms, summaries, anchor_deltas = [], [], [], []
    for domain, ds in datasets.items():
        root = out / domain
        report = json.loads((root / "summary.json").read_text())
        for name, sha in report["files"].items():
            assert digest(root / name) == sha
        with np.load(root / "private_features.npz", allow_pickle=False) as z:
            f = {k: z[k] for k in z.files}
        source = np.sort(np.r_[f["train"], f["val"]])
        np.testing.assert_array_equal(source, ds["train"])
        np.testing.assert_array_equal(f["held"], ds["held"])
        with np.load(PREVIOUS / domain / "private_blocks.npz", allow_pickle=False) as z:
            for k in ("train", "val", "held"):
                np.testing.assert_array_equal(f[k], z[k])
        with np.load(root / "private_predictions.npz", allow_pickle=False) as z:
            predictions = {k: z[k] for k in ARMS}
        d = f["directions"]
        maximum, attribution_error = 0.0, 0.0
        vector_geometry = []
        contributions = {k: [] for k in ARMS if k != "whole"}
        coefficients = {}
        for arm in ARMS:
            with np.load(root / f"private_affine_{arm}.npz", allow_pickle=False) as z:
                coefficients[arm] = (z["beta"], z["intercept"])
        for u, panel in addresses[addresses.domain == domain].groupby(
            "author_index", sort=True
        ):
            whole = v[panel[panel.kind == "whole"].input_index.iloc[0]]
            p = panel[panel.kind == "piece"].sort_values("lo")
            mass = (p.hi - p.lo).to_numpy(float)
            mass /= sum(mass)
            hs = v[p.input_index]
            weighted = sum(w * x for w, x in zip(mass, hs, strict=True))
            equal = sum(hs) / len(hs)
            radius = np.linalg.norm(weighted)
            unit = weighted / radius
            hnorm = np.linalg.norm(whole)
            cosine = float((whole @ unit) / hnorm)
            radial_energy = float((hnorm - radius) ** 2)
            angular_energy = float(2 * hnorm * radius * (1 - cosine))
            total_energy = float(np.sum((whole - weighted) ** 2))
            np.testing.assert_allclose(
                total_energy, radial_energy + angular_energy, atol=1e-12
            )
            vector_geometry.append(
                {
                    "author_index": int(u),
                    "vector_total_energy": total_energy,
                    "vector_radial_energy": radial_energy,
                    "vector_angular_energy": angular_energy,
                }
            )
            expected = {
                "whole": whole,
                "piece_equal": equal,
                "piece_mass": weighted,
                "piece_mass_unit": unit,
                "direction": whole - unit,
                "radial": unit - weighted,
            }
            for k, x in expected.items():
                maximum = max(maximum, float(np.max(abs(f[k][u] - x))))
            diag = diagnostics[
                (diagnostics.domain == domain) & (diagnostics.author_index == u)
            ].iloc[0]
            a, b = (whole - unit) @ d, (unit - weighted) @ d
            total = (whole - weighted) @ d
            np.testing.assert_allclose(
                diag.affinity_total_energy, total @ total, atol=1e-12
            )
            np.testing.assert_allclose(
                total @ total, a @ a + b @ b + 2 * a @ b, atol=1e-12
            )
            for arm, values in contributions.items():
                beta, intercept = coefficients[arm]
                weights = np.ones(len(hs)) / len(hs) if arm == "piece_equal" else mass
                if arm == "piece_mass_unit":
                    weights = weights / radius
                pieces = weights[:, None] * (hs @ beta)
                reconstructed = pieces.sum(0) + intercept
                cold = expected[arm] @ beta + intercept
                attribution_error = max(
                    attribution_error, float(np.max(abs(cold - reconstructed)))
                )
                values.extend(
                    {
                        "author_index": int(u),
                        "piece_input_index": int(ix),
                        **{t: float(s) for t, s in zip(TRAITS, score, strict=True)},
                    }
                    for ix, score in zip(p.input_index, pieces, strict=True)
                )
        assert maximum < 1e-10 and attribution_error < 1e-10
        for arm, records in contributions.items():
            pd.DataFrame(records).to_csv(
                root / f"private_piece_attribution_{arm}.csv", index=False
            )
        metrics = pd.read_csv(root / "metrics_by_trait.csv")
        with np.load(root / "private_anchor_scores.npz", allow_pickle=False) as z:
            raw = {k: z[k] for k in ARMS}
        link = pd.read_csv(root / "anchor_links.csv")
        for arm in ARMS:
            if arm != "whole":
                anchor_deltas.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "reference": "whole",
                        "interpretation": "secondary_descriptive_posthoc_fixed_anchor_comparison_no_selection",
                        **paired_delta(
                            ys[domain][f["held"]],
                            raw[arm][f["held"]],
                            raw["whole"][f["held"]],
                            draws=1000,
                        ),
                    }
                )
        cold_errors = []
        for arm in ARMS:
            model = joblib.load(root / f"private_model_{arm}.joblib")
            ft = model["features"]
            np.testing.assert_array_equal(model["source_indices"], source)
            np.testing.assert_array_equal(ft.fit_indices_, source)
            np.testing.assert_allclose(
                ft.pca_.mean_, f[arm][source].mean(0), atol=1e-12
            )
            np.testing.assert_allclose(
                ft.scaler_.mean_, ft.pca_.transform(f[arm][source]).mean(0), atol=1e-12
            )
            beta, intercept = coefficients[arm]
            p = f[arm][f["held"]] @ beta + intercept
            err = float(np.max(abs(p - predictions[arm])))
            assert err < 1e-10
            cold_errors.append(err)
            # Document source reference values, not held-population normalization.
            source_pred = f[arm][source] @ beta + intercept
            for j, trait in enumerate(TRAITS):
                y = ys[domain][f["held"], j]
                row = metrics[(metrics.arm == arm) & (metrics.trait == trait)].iloc[0]
                l = link[(link.arm == arm) & (link.trait == trait)].iloc[0]
                np.testing.assert_allclose(
                    row.r, np.corrcoef(y, p[:, j])[0, 1], atol=1e-12
                )
                np.testing.assert_allclose(
                    row.mae, np.mean(abs(y - p[:, j])), atol=1e-12
                )
                np.testing.assert_allclose(
                    row.rmse, np.linalg.norm(y - p[:, j]) / np.sqrt(len(y)), atol=1e-12
                )
                np.testing.assert_allclose(
                    l.r, np.corrcoef(y, raw[arm][f["held"], j])[0, 1], atol=1e-12
                )
                norms.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "trait": trait,
                        "source_n": len(source),
                        "anchor_mean": float(raw[arm][source, j].mean()),
                        "anchor_sd": float(raw[arm][source, j].std(ddof=1)),
                        "head_mean": float(source_pred[:, j].mean()),
                        "head_sd": float(source_pred[:, j].std(ddof=1)),
                    }
                )
        c = coverage[coverage.domain == domain]
        g = diagnostics[diagnostics.domain == domain]
        heldc, heldg = (
            c[c.author_index.isin(f["held"])],
            g[g.author_index.isin(f["held"])],
        )
        vg = pd.DataFrame(vector_geometry)
        for population, cc, gg in (("all", c, g), ("held", heldc, heldg)):
            gvec = vg[vg.author_index.isin(cc.author_index)]
            summaries.append(
                {
                    "domain": domain,
                    "population": population,
                    "authors": len(cc),
                    "complete_documents": int(cc.complete_document.sum()),
                    "single_piece": int(cc.pieces.eq(1).sum()),
                    "median_coverage": float(cc.coverage.median()),
                    "median_radius": float(gg.radius.median()),
                    "median_whole_unit_cosine": float(gg.whole_unit_cosine.median()),
                    "mean_affinity_total_energy": float(
                        gg.affinity_total_energy.mean()
                    ),
                    "mean_affinity_direction_energy": float(
                        gg.affinity_direction_energy.mean()
                    ),
                    "mean_affinity_radial_energy": float(
                        gg.affinity_radial_energy.mean()
                    ),
                    "mean_affinity_cross_term": float(gg.affinity_cross_term.mean()),
                    "median_equal_mass_distance": float(
                        gg.equal_mass_distance.median()
                    ),
                    "mean_vector_total_energy": float(gvec.vector_total_energy.mean()),
                    "mean_vector_radial_energy": float(
                        gvec.vector_radial_energy.mean()
                    ),
                    "mean_vector_angular_energy": float(
                        gvec.vector_angular_energy.mean()
                    ),
                }
            )
        checks.append(
            {
                "domain": domain,
                "authors": len(ds["ids"]),
                "held": len(f["held"]),
                "pool_max_error": maximum,
                "piece_attribution_max_error": attribution_error,
                "cold_head_max_error": max(cold_errors),
                "summary_sha256": digest(root / "summary.json"),
            }
        )
    pd.DataFrame(norms).to_csv(out / "source_norms.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "geometry_summary.csv", index=False)
    pd.DataFrame(anchor_deltas).to_csv(
        out / "anchor_deltas_descriptive.csv", index=False
    )
    write(
        out / "audit.json",
        {
            "status": "PASS",
            "input_packets_replayed": token_checks,
            "unique_vectors": len(inputs),
            "checks": checks,
            "new_neural_calls": 0,
            "new_fits": 0,
            "script_sha256": digest(Path(__file__)),
        },
    )
    print(json.dumps({"checks": checks, "geometry": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/matched_context_r3"
    )
    with threadpool_limits(limits=2):
        run(p.parse_args().output)
