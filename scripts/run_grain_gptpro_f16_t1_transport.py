#!/usr/bin/env python3
"""Run target-matched T1 transport with source by pilot uncertainty propagation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_public as public
from grain_research.gptpro_foundation_v3 import f16_t1_transport as model
from grain_research.gptpro_foundation_v3.f12_transport import (
    fit_transport,
    transport_score,
)
from scripts.run_grain_gptpro_f12_transport import transform

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
WORLDS = ("identity", "global", "conditional", "nonlinear")


def serial(value):
    """Turn numerical artifact arrays into strict JSON without losing nulls."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    """Write an immutable run artifact."""
    with Path(path).open("x") as f:
        f.write(json.dumps(serial(value), sort_keys=True, allow_nan=False) + "\n")


def known_inverse(new, z, world):
    """Use the declared forward map, never latent author truth."""
    if world == "identity":
        return new.copy()
    if world == "global":
        return (new - 1) / 1.5
    if world == "conditional":
        return (new - z) / (1 + 0.25 * z)
    if world == "nonlinear":
        return 2 * np.sqrt(10 / 3) * np.sinh(np.arcsinh(1.5 * np.sqrt(0.3) * new) / 3)
    raise ValueError("unknown reader world")


def corrected(cal, new, z):
    """Require full support rather than converting unsupported values to zeros."""
    value, support = transport_score(cal, new, np.broadcast_to(z, new.shape))
    return value, support.all(-1)


def fit_all(pilot, z, seed):
    """Use only the first60 pilot authors; retain all200 resample failures."""
    chosen = model.bootstrap_indices(seed)
    output = {}
    for world in WORLDS:
        new = transform(pilot, z, world)
        modes = (
            ("global", "conditional", "cubic")
            if world == "nonlinear"
            else ("global", "conditional")
        )
        for mode in modes:
            plugin = fit_transport(
                pilot[:60].ravel(), new[:60].ravel(), np.tile(z, 60), mode
            )
            draws = [
                fit_transport(pilot[ix].ravel(), new[ix].ravel(), np.tile(z, 60), mode)
                for ix in chosen
            ]
            output[(world, mode)] = {"plugin": plugin, "draws": draws}
    return chosen, output


def summarize(rows):
    """Report support and conditional-target metrics without dropping failed arms."""
    df = pd.DataFrame(rows)
    result = []
    for (seed, world, arm, estimator), g in df.groupby(
        ["seed", "world", "arm", "estimator"], sort=True
    ):
        good = g[g["score"].notna()]
        r = {
            "seed": int(seed),
            "world": world,
            "arm": arm,
            "estimator": estimator,
            "n": len(g),
            "supported": len(good),
        }
        if len(good) > 1:
            y = good.truth.to_numpy(float)
            pred = good.score.to_numpy(float)
            r.update(
                mse=float(np.mean((pred - y) ** 2)),
                bias=float(np.mean(pred - y)),
                r=float(np.corrcoef(y, pred)[0, 1]),
                coverage=float(np.mean((good.lower <= y) & (y <= good.upper))),
                width=float(np.mean(good.upper - good.lower)),
                pilot_variance=float(good.pilot.mean()),
            )
        else:
            r.update(
                {
                    k: None
                    for k in ("mse", "bias", "r", "coverage", "width", "pilot_variance")
                }
            )
        result.append(r)
    return result


def main():
    """Fit pilot objects before reading application outcomes; evaluate all arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = (20260915, 20260916, 20260917)
    old = BASE / "F12/transport_full_r1"
    sources = [old / f"observations_{s}_pilot.npz" for s in seeds] + [
        BASE / f"F16/t1_source_full_r1/{s}/bundle.json" for s in seeds
    ]
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_t1_transport.py",
        ROOT / "grain_research/gptpro_foundation_v3/f12_transport.py",
        ROOT / "tests/test_grain_gptpro_f16_t1_transport.py",
        ROOT / "docs/GRAIN_F16_T1_TRANSPORT_PROTOCOL_V1.md",
    ]
    save(
        args.output / "protocol_before_run.json",
        {
            "seeds": seeds,
            "target": "T1=2b in F12 linear world",
            "events": 32,
            "source_components": 200,
            "pilot_bootstraps": 200,
            "pilot_fit_validation": [60, 40],
            "worlds": WORLDS,
            "source_hashes": {str(p): public.file_hash(p) for p in sources},
            "code_hashes": {str(p): public.file_hash(p) for p in code},
            "new_source_fits": 0,
            "scope": "numeric T1 target-matched source/pilot product;not text/dynamic transport",
        },
    )
    snap = args.output / "source_snapshot"
    snap.mkdir()
    for p in code:
        (snap / p.name).write_bytes(p.read_bytes())
    ready = []
    for seed in seeds:
        source_path = BASE / f"F16/t1_source_full_r1/{seed}/bundle.json"
        bundle = json.loads(source_path.read_text())
        arr = np.load(old / f"observations_{seed}_pilot.npz")
        pilot, z = arr["old"], arr["conditions"]
        if pilot.shape != (100, 32):
            raise ValueError("expected all100 independentpilot authors")
        kernels = model.source_kernels(bundle, z)
        indices, calibrations = fit_all(pilot, z, seed)
        folder = args.output / str(seed)
        folder.mkdir()
        save(folder / "pilot_bootstrap_indices.json", indices)
        save(folder / "kernels.json", kernels)
        validation = []
        certificates = []
        for (world, mode), cal in calibrations.items():
            save(folder / f"{world}_{mode}.json", cal)
            new = transform(pilot[60:], z, world)
            fixed, support = corrected(cal["plugin"], new, z)
            record = {
                "world": world,
                "mode": mode,
                "status": cal["plugin"]["status"],
                "supported_authors": int(support.sum()),
                "paired_rmse": float(
                    np.sqrt(np.mean((fixed[support] - pilot[60:][support]) ** 2))
                )
                if support.any()
                else None,
                "condition_residuals": {
                    str(c): float(
                        (fixed[support] - pilot[60:][support])[:, z == c].mean()
                    )
                    for c in model.CONDITIONS
                }
                if support.any()
                else {},
            }
            validation.append(record)
        for world in WORLDS:
            new = transform(pilot[:60], z, world)
            mask = np.tile(z, 60) == -1
            certificate = fit_transport(
                pilot[:60].ravel()[mask],
                new.ravel()[mask],
                np.tile(z, 60)[mask],
                "conditional",
            )
            if certificate["status"] != "MISSING_ANCHOR":
                raise ValueError("missing anchor not rejected")
            certificates.append({"world": world, **serial(certificate)})
        save(folder / "pilot_validation.json", validation)
        save(folder / "missing_anchors.json", certificates)
        ready.append((seed, folder, bundle, kernels, z, calibrations))
    # No application truth has been opened before this point.
    flat = []
    checks = []
    for seed, folder, bundle, kernels, z, calibrations in ready:
        path = old / f"observations_{seed}_application.npz"
        values = np.load(path)["old"]
        if values.shape != (200, 32):
            raise ValueError("expected all200 newdomain authors")
        results = {}
        supports = {}
        statuses = []
        for world in WORLDS:
            new = transform(values, z, world)
            inverse = known_inverse(new, z, world)
            if np.max(abs(inverse - values)) > 1e-10:
                raise ValueError("knowninverse does not reproduce observations")
            for arm in ("uncalibrated", "known_inverse", "global", "conditional") + (
                ("cubic",) if world == "nonlinear" else ()
            ):
                if arm in ("uncalibrated", "known_inverse"):
                    obs = new if arm == "uncalibrated" else inverse
                    means = model.cell_means(obs, z)[None]
                    support = np.ones(200, dtype=bool)
                    plugin = model.integrate(kernels, means)
                    mixture = plugin
                    status = {
                        "world": world,
                        "arm": arm,
                        "plugin_status": "REFERENCE",
                        "bad_draws": 0,
                    }
                else:
                    cal = calibrations[(world, arm)]
                    corrected_plugin, support = corrected(cal["plugin"], new, z)
                    plugin = (
                        model.integrate(
                            kernels,
                            model.cell_means(corrected_plugin[support], z)[None],
                        )
                        if support.any()
                        else None
                    )
                    allmeans = []
                    common = support.copy()
                    bad = 0
                    for draw in cal["draws"]:
                        yc, ok = corrected(draw, new, z)
                        common &= ok
                        bad += int(draw["status"] != "READY")
                        allmeans.append(yc)
                    mixture = (
                        model.integrate(
                            kernels, model.cell_means(np.stack(allmeans)[:, common], z)
                        )
                        if common.any()
                        else None
                    )
                    status = {
                        "world": world,
                        "arm": arm,
                        "plugin_status": cal["plugin"]["status"],
                        "bad_draws": bad,
                    }
                    for key, result, mask in [
                        ("plugin", plugin, support),
                        ("mixture", mixture, common),
                    ]:
                        results[(world, arm, key)] = result
                        supports[(world, arm, key)] = mask
                    statuses.append(status)
                    continue
                for key, result in [("plugin", plugin), ("mixture", mixture)]:
                    results[(world, arm, key)] = result
                    supports[(world, arm, key)] = support
                statuses.append(status)
        save(folder / "application_status.json", statuses)
        saved = []
        for key, result in results.items():
            saved.append({"key": key, "support": supports[key], "moments": result})
        save(folder / "application_scores_before_truth.json", saved)
        truth = np.load(old / f"truth_{seed}_application.npz")["theta"][:, 1] * 2
        save(
            folder / "evaluation_hashes.json",
            {
                "observations": public.file_hash(path),
                "truth": public.file_hash(old / f"truth_{seed}_application.npz"),
            },
        )
        for (world, arm, key), result in results.items():
            mask = supports[(world, arm, key)]
            pos = 0
            for i in range(200):
                row = {
                    "seed": seed,
                    "world": world,
                    "arm": arm,
                    "estimator": key,
                    "author_index": i,
                    "truth": float(truth[i]),
                }
                for field in (
                    "score",
                    "variance",
                    "lower",
                    "upper",
                    "conditional",
                    "source",
                    "pilot",
                ):
                    row[field] = float(result[field][pos]) if mask[i] else None
                flat.append(row)
                pos += int(mask[i])
        # Exact affine inverses must match source-unit reference without normalization.
        for world, arm in [
            ("identity", "global"),
            ("identity", "conditional"),
            ("global", "global"),
            ("global", "conditional"),
            ("conditional", "conditional"),
        ]:
            left = results[(world, arm, "mixture")]
            right = results[(world, "known_inverse", "mixture")]
            err = max(
                float(np.max(abs(left[k] - right[k]))) for k in ("score", "variance")
            )
            if err > 1e-10:
                raise ValueError("matchedaffine changed target law")
            checks.append({"seed": seed, "world": world, "arm": arm, "max_error": err})
        print(
            json.dumps(
                {
                    "seed": seed,
                    "application_done": 200,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    pd.DataFrame(flat).to_csv(args.output / "predictions.csv", index=False)
    metrics = summarize(flat)
    pd.DataFrame(metrics).to_csv(args.output / "metrics_by_seed.csv", index=False)
    pd.DataFrame(metrics).groupby(["world", "arm", "estimator"], as_index=False)[
        ["mse", "bias", "r", "coverage", "width", "pilot_variance", "supported"]
    ].mean().to_csv(args.output / "metrics.csv", index=False)
    save(
        args.output / "summary.json",
        {
            "status": "T1_NUMERIC_TRANSPORT_EXECUTED_AUDIT_REQUIRED",
            "application_authors": 600,
            "prediction_rows": len(flat),
            "source_by_pilot_components": 40000,
            "affine_checks": checks,
            "seconds": time.monotonic() - started,
            "scope": "numeric target matched;not text/dynamic transport;working cut intervals;all rejected cubic fits retained",
        },
    )
    save(
        args.output / "output_manifest.json",
        {
            str(p.relative_to(args.output)): public.file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    main()
