#!/usr/bin/env python3
"""Audit dynamic transport by recursive filtering and saved-result reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from grain_research.gptpro_foundation_v3 import f16_public as public
from grain_research.gptpro_foundation_v3.f16_cutoff import (
    build_adapter,
)
from grain_research.gptpro_foundation_v3.f16_cutoff import (
    cold_score as old_clock,
)
from grain_research.gptpro_foundation_v3.f16_dynamic_gaussian_reference import (
    conditional_filter,
    joint_future,
)
from grain_research.gptpro_foundation_v3.f16_state_instrument import (
    cold_score as old_t2,
)
from scripts.audit_grain_gptpro_f16_public_transport import qr_fit
from scripts.run_grain_gptpro_f16_t1_transport import save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
TARGET = np.array([[1, 0, 0, 0, 0], [0, 0.25, 0.25, 0.25, 0.25]])


def shifted(y, z, world):
    """Replay channel laws independently of the scoring runner."""
    return {
        "identity": y,
        "global": 1 + 1.5 * y,
        "conditional": z + (1 + 0.25 * z) * y,
        "nonlinear": y + 0.1 * y**3,
    }[world]


def inverse(c, value, z):
    """Invert a saved affine calibrator directly, not via compiled basis code."""
    v = np.asarray(c["coefficients"])
    if c["mode"] == "global":
        return (value - v[0]) / v[1]
    return (value - v[0] - v[1] * z) / (v[2] + v[3] * z)


def replay_simulator(seed, role, observed, truth):
    """Check state truth using explicit causal convolution, not recursive generation."""
    n = len(observed["old"])
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 163, 5, 0 if role == "pilot" else 1])
    )
    a = rng.normal(1, np.sqrt(1.4), n)
    b = rng.normal(0.2, np.sqrt(0.7), n)
    rho = rng.uniform(0.2, 0.8, n)
    initial = rng.normal(size=n) * 0.5 / np.sqrt(1 - rho * rho)
    innovation = rng.normal(scale=0.5, size=(n, 52))
    noise = rng.normal(scale=0.5, size=(n, 52))
    t = np.arange(52)
    gaps = t[:, None] - t[None, :]
    causal = np.where(gaps[None] >= 0, rho[:, None, None] ** np.maximum(gaps, 0), 0)
    states = rho[:, None] ** (t + 1) * initial[:, None] + np.einsum(
        "nij,nj->ni", causal, innovation
    )
    z = np.r_[np.where((np.arange(48) // 4) % 2 == 0, -1, 1), [1, -1, 1, -1]]
    y = a[:, None] + b[:, None] * z + states + noise
    pairs = [
        (states, truth["states"]),
        (y[:, :48], observed["old"]),
        (y[:, 48:], truth["future"]),
        (a, truth["a"]),
        (b, truth["b"]),
        (rho, truth["rho"]),
        (states[:, 47], truth["T2"]),
        (y[:, 48:].mean(1), truth["T3"]),
    ]
    return max(float(np.max(abs(x - y))) for x, y in pairs)


def recursive_targets(models, values, z, nodes=64):
    """Separate augmented Kalman recursion and target projection for every source."""
    x, hw = hermgauss(nodes)
    shape = (len(values), 2)
    center = np.zeros(shape)
    within, parameter, second = (np.zeros((*shape, 2)) for _ in range(3))
    arrays = (
        np.broadcast_to(z, values.shape),
        values,
        np.ones_like(values),
        np.ones_like(values, bool),
    )
    for model in models:
        p = model["prior"]
        eta = p["mu"][2] + np.sqrt(2 * p["Sigma"][2][2]) * x
        m, c, ll = conditional_filter(model, eta, arrays)
        tm, state, personal = joint_future(m, c, eta, model["log_sigma"])
        tm = tm @ TARGET.T
        state = np.einsum("li,kaij,mj->kalm", TARGET, state, TARGET)
        personal = np.einsum("li,kaij,mj->kalm", TARGET, personal, TARGET)
        weights = np.exp(
            ll
            + np.log(hw / np.sqrt(np.pi))[:, None]
            - logsumexp(ll + np.log(hw / np.sqrt(np.pi))[:, None], axis=0)
        )
        mean = np.einsum("ka,kal->al", weights, tm)
        delta = tm - mean[None]
        within += np.einsum("ka,kalm->alm", weights, state) / len(models)
        parameter += (
            np.einsum("ka,kalm->alm", weights, personal)
            + np.einsum("ka,kal,kam->alm", weights, delta, delta)
        ) / len(models)
        center += mean / len(models)
        second += mean[..., :, None] * mean[..., None, :] / len(models)
    return {
        "mean": center,
        "state_future": within,
        "personal": parameter,
        "source": second - center[..., :, None] * center[..., None, :],
    }


def main():
    """Audit completed immutable outputs; retain different old-estimator readouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / "output_manifest.json").read_text())
    for name, h in manifest.items():
        if public.file_hash(args.run / name) != h:
            raise ValueError("changed completed artifact")
    maximum, numeric, qr_count, recursive_sources = 0.0, 0, 0, 0
    diagnostics, legacy = [], []
    for seed in [20260915, 20260916, 20260917]:
        folder = args.run / str(seed)
        source = json.loads((folder / "source_bundle.json").read_text())
        models = [m["model"] for m in source["source_models"]]
        if len(models) != 200:
            raise ValueError("full source ensemble required")
        for role in ("pilot", "application"):
            maximum = max(
                maximum,
                replay_simulator(
                    seed,
                    role,
                    np.load(folder / f"{role}_observations.npz"),
                    np.load(folder / f"{role}_truth.npz"),
                ),
            )
        p = np.load(folder / "pilot_observations.npz")
        old, z = p["old"], p["conditions"]
        indices = np.asarray(
            json.loads((folder / "pilot_bootstrap_indices.json").read_text())
        )
        if indices.shape != (200, 60) or indices.min() < 0 or indices.max() >= 60:
            raise ValueError("pilot role leakage")
        for world in ("identity", "global", "conditional", "nonlinear"):
            for mode in (
                "global",
                "conditional",
                *(["cubic"] if world == "nonlinear" else []),
            ):
                c = json.loads((folder / f"{world}_{mode}.json").read_text())
                new = shifted(old, z, world)
                for fit, ix in [
                    (c["plugin"], np.arange(60)),
                    *zip(c["draws"], indices, strict=True),
                ]:
                    coef = qr_fit(
                        old[ix].ravel(), new[ix].ravel(), np.tile(z, 60), mode
                    )
                    maximum = max(
                        maximum,
                        float(abs(coef - np.asarray(fit["coefficients"])).max()),
                    )
                    qr_count += 1
        app = np.load(folder / "application_observations.npz")["old"]
        todo, ranges = [], []
        for world, arm in [
            ("identity", "uncalibrated"),
            ("conditional", "global"),
            ("nonlinear", "global"),
            ("nonlinear", "conditional"),
        ]:
            new = shifted(app[[0, -1]], z, world)
            if arm == "uncalibrated":
                variants, js = [new], [0]
            else:
                c = json.loads((folder / f"{world}_{arm}.json").read_text())
                models_c = [c["plugin"], c["draws"][0], c["draws"][-1]]
                variants, js = [inverse(cal, new, z) for cal in models_c], [0, 1, 200]
            start = len(todo)
            todo.extend(np.concatenate(variants, axis=0))
            ranges.append((world, arm, js, start, len(todo)))
        expected = recursive_targets(models, np.asarray(todo), z)
        recursive_sources += len(models)
        for world, arm, js, first, last in ranges:
            saved = np.load(folder / f"{world}_{arm}_all_moments.npz")
            for key in ("mean", "state_future", "personal", "source"):
                actual = saved[key][js][:, [0, -1]].reshape(
                    (last - first, *saved[key].shape[2:])
                )
                diff = float(abs(actual - expected[key][first:last]).max())
                maximum = max(maximum, diff)
                numeric += actual.size
        for path in folder.glob("*_all_moments.npz"):
            if path.name.startswith("."):
                continue
            a = np.load(path)
            stem = path.name.removesuffix("_all_moments.npz")
            for label, select in [
                ("plugin", slice(0, 1)),
                ("mixture", slice(1, None) if len(a["mean"]) > 1 else slice(0, 1)),
            ]:
                target = folder / f"{stem}_{label}.npz"
                if not target.exists():
                    continue
                r = np.load(target)
                means = a["mean"][select]
                m = means.mean(0)
                total = sum(
                    a[key][select].mean(0)
                    for key in ("state_future", "personal", "source")
                )
                total += (
                    np.mean(means[..., :, None] * means[..., None, :], axis=0)
                    - m[..., :, None] * m[..., None, :]
                )
                maximum = max(
                    maximum,
                    float(abs(m - r["mean"]).max()),
                    float(abs(total - r["covariance"]).max()),
                )
                numeric += m.size + total.size
        for path in folder.glob("*_quadrature.json"):
            if not path.name.startswith("."):
                d = json.loads(path.read_text())
                values = [v for k, v in d.items() if k != "conditional_indices"]
                diagnostics.append(
                    {
                        "seed": seed,
                        "file": path.name,
                        "max_error": max(values),
                        "pass": max(values) <= 0.02,
                    }
                )
        # These are intentionally different integration estimators. Retain both,
        # never force equality or silently transport the old source norms.
        if not args.skip_legacy:
            y = app[0]
            records = [
                {
                    "author_id": "dynamic_transport_audit",
                    "role": "application",
                    "world": "core",
                    "episode_id": "matched_transport",
                    "event_id": f"e{t}",
                    "t": t,
                    "z": float(q),
                    "y": float(v),
                }
                for t, (q, v) in enumerate(zip(z, y, strict=True))
            ]
            r2 = old_t2(
                {
                    "world_contract": source["world_contract"],
                    "reader_hash": source["reader_hash"],
                    "records": records,
                },
                source,
            )
            t3path = BASE / f"F16/independent_t3_v1/core_{seed}/bundle.json"
            if not t3path.exists():
                raise ValueError(f"need actual independentT3 bundle:{t3path}")
            b3 = json.loads(t3path.read_text())
            r3 = old_clock(
                {
                    "world": "core",
                    "reader_hash": b3["reader_hash"],
                    "target_time": 47,
                    "records": records,
                },
                build_adapter(b3),
            )
            new = np.load(folder / "identity_uncalibrated_plugin.npz")
            legacy.append(
                {
                    "seed": seed,
                    "index": 0,
                    "new_scores": new["mean"][0].tolist(),
                    "old_T2": r2,
                    "old_T3": r3,
                    "scope": "different integration;not an equality gate",
                }
            )
            print(f"audit dynamic seed={seed} recursive and legacy done", flush=True)
        else:
            print(
                f"audit dynamic seed={seed} recursive done, legacy explicitly skipped",
                flush=True,
            )
    df = pd.read_csv(args.run / "predictions.csv")
    byseed = pd.read_csv(args.run / "metrics_by_seed.csv")
    for _, row in byseed.iterrows():
        g = df[
            (df.seed == row.seed)
            & (df.world == row.world)
            & (df.arm == row.arm)
            & (df.estimator == row.estimator)
            & (df.target == row.target)
        ].dropna(subset=["score"])
        if len(g) != row.supported:
            raise ValueError("missing application support")
        if len(g):
            maximum = max(
                maximum,
                abs(float(np.mean((g.score - g.truth) ** 2)) - row.mse),
                abs(
                    float(np.mean((g.lower <= g.truth) & (g.truth <= g.upper)))
                    - row.coverage
                ),
                abs(float(np.corrcoef(g.score, g.truth)[0, 1]) - row.r),
            )
    contrasts = []
    for target in ("T2", "T3"):
        for world, left, right in [
            ("global", "global", "uncalibrated"),
            ("conditional", "conditional", "global"),
            ("nonlinear", "global", "uncalibrated"),
        ]:
            g = df[
                (df.world == world)
                & (df.estimator == "mixture")
                & (df.target == target)
            ]
            a = g[g.arm == left].sort_values(["seed", "author_index"])
            b = g[g.arm == right].sort_values(["seed", "author_index"])
            if (
                not a.supported.all()
                or not b.supported.all()
                or len(a) != 600
                or len(b) != 600
            ):
                raise ValueError("paired contrast incomplete")
            delta = (a.score.to_numpy() - a.truth.to_numpy()) ** 2 - (
                b.score.to_numpy() - b.truth.to_numpy()
            ) ** 2
            rng = np.random.default_rng(163530)
            boot = np.concatenate(
                [
                    delta[j : j + 200][rng.integers(0, 200, (2000, 200))]
                    for j in (0, 200, 400)
                ],
                axis=1,
            ).mean(1)
            contrasts.append(
                {
                    "target": target,
                    "world": world,
                    "left": left,
                    "right": right,
                    "delta_mse": float(delta.mean()),
                    "ci": np.quantile(boot, [0.025, 0.975]).tolist(),
                }
            )
    save(args.output / "quadrature_diagnostics.json", diagnostics)
    save(args.output / "legacy_comparison.json", legacy)
    save(args.output / "paired_contrasts.json", contrasts)
    if maximum > 1e-9:
        raise ValueError(f"audit mathematical mismatch:{maximum}")
    save(
        args.output / "summary.json",
        {
            "status": "FINITE_DYNAMIC_TRANSPORT_AUDIT_PASS"
            if all(d["pass"] for d in diagnostics)
            else "ARITHMETIC_PASS_QUADRATURE_FOLLOWUP_REQUIRED",
            "artifact_hashes": len(manifest),
            "recursive_source_models": recursive_sources,
            "qr_pilot_fits": qr_count,
            "recomputed_numeric_values": numeric,
            "max_error": maximum,
            "legacy_cases": len(legacy),
            "scope": "same-agent recursive versus dense identities;not independent human validation or text transport",
        },
    )


if __name__ == "__main__":
    main()
