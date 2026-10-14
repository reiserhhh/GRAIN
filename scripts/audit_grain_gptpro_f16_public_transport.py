#!/usr/bin/env python3
"""Audit public objects and T1 pilot transport using alternate matrix identities."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_public as public
from scripts.run_grain_gptpro_f16_public import difference, lines
from scripts.run_grain_gptpro_f16_t1_transport import save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
Z = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
C = np.array([-1.0, 0, 0, 1.0])


def check_hashes(root):
    """Hash every saved artifact named by the completed run manifest."""
    entries = json.loads((root / "output_manifest.json").read_text())
    for name, expected in entries.items():
        if public.file_hash(root / name) != expected:
            raise ValueError(f"changed artifact:{root / name}")
    return len(entries)


def invert(cal, values, z):
    """Reconstruct old observations directly from saved calibration coefficients."""
    shape = values.shape
    bad = np.zeros(shape, dtype=bool)
    if cal["status"] != "READY":
        return np.full(shape, np.nan), bad
    coef = np.array(cal["coefficients"])
    valid = np.isin(np.broadcast_to(z, shape), cal["conditions"])
    if cal["mode"] == "cubic":
        valid &= (values >= cal["new_range"][0]) & (values <= cal["new_range"][1])
        old = coef[0] + coef[1] * values + coef[2] * values**3
    else:
        d = coef[0] if cal["mode"] == "global" else coef[0] + coef[1] * z
        c = coef[1] if cal["mode"] == "global" else coef[2] + coef[3] * z
        valid &= np.broadcast_to(abs(c) > 1e-12, shape)
        old = (values - d) / c
    return old, valid


def forward(old, z, world):
    """Replay the original four declared observation maps independently."""
    return {
        "identity": lambda: old.copy(),
        "global": lambda: 1 + 1.5 * old,
        "conditional": lambda: z + (1 + 0.25 * z) * old,
        "nonlinear": lambda: old + 0.1 * old**3,
    }[world]()


def qr_fit(old, new, z, mode):
    """Reconstruct least squares with a different QR solver, not the runner API."""
    if mode == "global":
        x = np.column_stack((np.ones(old.size), old))
        y = new
    elif mode == "conditional":
        x = np.column_stack((np.ones(old.size), z, old, z * old))
        y = new
    else:
        x = np.column_stack((np.ones(new.size), new, new**3))
        y = old
    q, r = np.linalg.qr(x, mode="reduced")
    return np.linalg.solve(r, q.T @ y)


def moment_formula(kernels, draws, z):
    """Use calibration covariance contractions instead of enumerating all pairs."""
    v = np.stack([draws[..., z == x].mean(-1) for x in Z], -1)
    mu = v.mean(0)
    center = v - mu
    cv = np.einsum("jac,jad->acd", center, center) / len(v)
    w = np.array(kernels["weights"])
    a = np.array(kernels["offsets"])
    wm = w.mean(0)
    wc = (w - wm).T @ (w - wm) / len(w)
    s = a[None] + mu @ w.T
    score = s.mean(-1)
    source = s.var(-1) + np.einsum("cd,adc->a", wc, cv)
    pilot = np.einsum("c,acd,d->a", wm, cv, wm)
    conditional = np.full(len(mu), np.mean(kernels["conditional"]))
    variance = source + pilot + conditional
    return {
        "score": score,
        "variance": variance,
        "source": source,
        "pilot": pilot,
        "conditional": conditional,
        "lower": score - 1.96 * np.sqrt(variance),
        "upper": score + 1.96 * np.sqrt(variance),
    }


def main():
    """Recompute required identities and write checks, never modify the runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-run", type=Path, required=True)
    parser.add_argument("--transport-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = check_hashes(args.public_run) + check_hashes(args.transport_run)
    error = 0.0
    public_objects = 0
    negative = 0
    for cell in sorted(args.public_run.glob("*_2026*")):
        if not cell.is_dir() or cell.name.startswith("."):
            continue
        b = json.loads((cell / "bundle.json").read_text())
        public.verify_bundle(b)
        warm = json.loads((cell / "warm.json").read_text())
        cold = lines(cell / "cold.jsonl")
        error = max(error, difference(warm, cold))
        public_objects += len(cold)
        for result in cold:
            if result["score"] is not None:
                error = max(
                    error,
                    abs(
                        result["reference_score"]
                        - (result["score"] - b["reference_mu"]) / b["reference_sigma"]
                    ),
                )
        invariance = json.loads((cell / "invariance.json").read_text())
        error = max(
            error,
            difference(invariance["single"], invariance["batch"][0]),
            difference(invariance["single"], invariance["batch"][2]),
        )
        rejections = json.loads((cell / "rejections.json").read_text())
        if len(rejections) != 13 or any(not r["message"] for r in rejections):
            raise ValueError("missing actual rejection")
        negative += len(rejections)
    # CLI byte authentication is checked in an actual separate process.
    cell = args.public_run / "text_T2_20260915"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "grain_research.gptpro_foundation_v3.f16_public",
            "--bundle",
            str((cell / "bundle.json").resolve()),
            "--sha256",
            "0" * 64,
            "--input",
            str((cell / "inputs.jsonl").resolve()),
            "--output",
            str((args.output / "forbidden.jsonl").resolve()),
        ],
        cwd=args.output,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    save(
        args.output / "wrong_external_hash.json",
        {"returncode": proc.returncode, "stderr": proc.stderr},
    )
    if proc.returncode == 0 or (args.output / "forbidden.jsonl").exists():
        raise ValueError("external authentication failed")
    refits = 0
    target_values = 0
    kernel_count = 0
    public_actual = []
    for seed in (20260915, 20260916, 20260917):
        folder = args.transport_run / str(seed)
        old = BASE / "F12/transport_full_r1"
        bpath = BASE / f"F16/t1_source_full_r1/{seed}/bundle.json"
        base = json.loads(bpath.read_text())
        k = json.loads((folder / "kernels.json").read_text())
        z = np.array(k["conditions"])
        x = (z[:, None] == Z).astype(float)
        for i, fit in enumerate(base["source_models"]):
            mu = np.array(fit["mu"])
            sigma = np.array(fit["Sigma"])
            g = x @ sigma @ C
            total = x @ sigma @ x.T + fit["sigma2"] * np.eye(32)
            weights = np.linalg.solve(total, g)
            alt = [float(C @ mu - weights @ x @ mu), float(C @ sigma @ C - weights @ g)]
            error = max(
                error,
                abs(alt[0] - k["offsets"][i]),
                abs(alt[1] - k["conditional"][i]),
                float(
                    np.max(
                        abs(
                            np.array([weights[z == q].sum() for q in Z])
                            - k["weights"][i]
                        )
                    )
                ),
            )
            kernel_count += 1
        pilot = np.load(old / f"observations_{seed}_pilot.npz")["old"]
        values = np.load(old / f"observations_{seed}_application.npz")["old"]
        indices = np.array(
            json.loads((folder / "pilot_bootstrap_indices.json").read_text())
        )
        if indices.shape != (200, 60) or indices.min() < 0 or indices.max() >= 60:
            raise ValueError("invalidfitpartition")
        calibrations = {}
        for world in ("identity", "global", "conditional", "nonlinear"):
            modes = (
                ("global", "conditional", "cubic")
                if world == "nonlinear"
                else ("global", "conditional")
            )
            new = forward(pilot, z, world)
            for mode in modes:
                entry = json.loads((folder / f"{world}_{mode}.json").read_text())
                calibrations[(world, mode)] = entry
                for draw, ix in [
                    (entry["plugin"], np.arange(60)),
                    *zip(entry["draws"], indices, strict=True),
                ]:
                    coef = qr_fit(
                        pilot[ix].ravel(), new[ix].ravel(), np.tile(z, 60), mode
                    )
                    error = max(
                        error, float(np.max(abs(coef - np.array(draw["coefficients"]))))
                    )
                    if mode == "cubic":
                        lo, hi = draw["new_range"]
                        pts = [lo, hi] + ([0] if lo <= 0 <= hi else [])
                        minimum = min(coef[1] + 3 * coef[2] * q * q for q in pts)
                        if (minimum <= 0) != (draw["status"] == "NONMONOTONE_INVERSE"):
                            raise ValueError("monotonicity statuswrong")
                    refits += 1
        for item in json.loads(
            (folder / "application_scores_before_truth.json").read_text()
        ):
            world, arm, kind = item["key"]
            new = forward(values, z, world)
            if arm in ("known_inverse", "uncalibrated"):
                obs = (values if arm == "known_inverse" else new)[None]
                support = np.ones(200, dtype=bool)
            else:
                c = calibrations[(world, arm)]
                models = [c["plugin"]] if kind == "plugin" else c["draws"]
                both = [invert(f, new, z) for f in models]
                support = np.logical_and.reduce([ok.all(-1) for _, ok in both])
                support &= invert(c["plugin"], new, z)[1].all(-1)
                obs = np.stack([val for val, _ in both])[:, support]
            np.testing.assert_array_equal(support, item["support"])
            if not support.any():
                if item["moments"] is not None:
                    raise ValueError("unsupported emittedscore")
                continue
            alt = moment_formula(k, obs, z)
            for field, v in alt.items():
                error = max(
                    error, float(np.max(abs(v - np.array(item["moments"][field]))))
                )
                target_values += len(v)
        # Corrected new-domain observations enter the actual public cold scorer.
        meta = public.describe(base, "T1")
        brief = {
            q: meta[q]
            for q in (
                "target_id",
                "object_class",
                "unit",
                "world_contract",
                "observed_role",
            )
        }
        pub = public.build_instrument(
            brief,
            {
                "path": str(bpath),
                "sha256": public.file_hash(bpath),
                "role": "source_fitted",
            },
        )
        for world, arm in [
            ("global", "global"),
            ("conditional", "conditional"),
            ("nonlinear", "global"),
        ]:
            val, ok = invert(
                calibrations[(world, arm)]["plugin"], forward(values, z, world), z
            )
            for index in (0, 199):
                if not ok[index].all():
                    raise ValueError("selectedpubliccase unsupported")
                records = [
                    {
                        "author_id": f"transport_app_{index}",
                        "role": "application",
                        "world": "L",
                        "event_id": f"e{t}",
                        "t": t,
                        "z": float(q),
                        "y": float(v),
                    }
                    for t, (q, v) in enumerate(zip(z, val[index], strict=True))
                ]
                request = {
                    "contract": public.request_contract(pub),
                    "data": {
                        "world_contract": pub["world_contract"],
                        "reader_hash": base["reader_hash"],
                        "records": records,
                    },
                }
                actual = public.cold_score(request, pub)
                expected = moment_formula(k, val[index : index + 1][None], z)
                diff = max(
                    abs(actual[q] - expected[q][0]) for q in ("score", "variance")
                )
                error = max(error, diff)
                public_actual.append(
                    {
                        "seed": seed,
                        "world": world,
                        "arm": arm,
                        "index": index,
                        "difference": diff,
                        "score": actual["score"],
                    }
                )
    df = pd.read_csv(args.transport_run / "predictions.csv")
    metrics = pd.read_csv(args.transport_run / "metrics_by_seed.csv")
    coverage = []
    for _, row in metrics.iterrows():
        g = df[
            (df.seed == row.seed)
            & (df.world == row.world)
            & (df.arm == row.arm)
            & (df.estimator == row.estimator)
        ].dropna(subset=["score"])
        if len(g) != row.supported:
            raise ValueError("wrong support count")
        if not len(g):
            continue
        mse = np.mean((g.score - g.truth) ** 2)
        cov = np.mean((g.lower <= g.truth) & (g.truth <= g.upper))
        error = max(
            error,
            abs(mse - row.mse),
            abs(cov - row.coverage),
            abs(np.corrcoef(g.score, g.truth)[0, 1] - row.r),
        )
        n = len(g)
        s = 1.96**2
        center = (cov + s / (2 * n)) / (1 + s / n)
        half = 1.96 * np.sqrt(cov * (1 - cov) / n + s / (4 * n * n)) / (1 + s / n)
        coverage.append(
            {
                "seed": int(row.seed),
                "world": row.world,
                "arm": row.arm,
                "estimator": row.estimator,
                "coverage": cov,
                "wilson_low": center - half,
                "wilson_high": center + half,
                "gate": bool(center - half <= 0.95 <= center + half),
            }
        )
    contrasts = []
    for world, a, b in [
        ("global", "global", "uncalibrated"),
        ("conditional", "conditional", "global"),
        ("nonlinear", "global", "uncalibrated"),
    ]:
        g = df[(df.world == world) & (df.estimator == "mixture")]
        left = g[g.arm == a].sort_values(["seed", "author_index"])
        right = g[g.arm == b].sort_values(["seed", "author_index"])
        delta = (left.score.to_numpy() - left.truth.to_numpy()) ** 2 - (
            right.score.to_numpy() - right.truth.to_numpy()
        ) ** 2
        rng = np.random.default_rng(163120)
        draws = np.concatenate(
            [
                delta[i : i + 200][rng.integers(0, 200, (2000, 200))]
                for i in (0, 200, 400)
            ],
            axis=1,
        ).mean(1)
        contrasts.append(
            {
                "world": world,
                "left": a,
                "right": b,
                "delta_mse": float(delta.mean()),
                "ci": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
        )
    if error > 1e-9:
        raise ValueError(f"audit mismatch {error}")
    save(args.output / "public_actual_comparisons.json", public_actual)
    save(args.output / "coverage_gates.json", coverage)
    save(args.output / "paired_contrasts.json", contrasts)
    save(
        args.output / "summary.json",
        {
            "status": "FINITE_PUBLIC_AND_T1_TRANSPORT_AUDIT_PASS",
            "hashes": hashes,
            "public_objects": public_objects,
            "recorded_negative_checks": negative,
            "dense_source_kernels": kernel_count,
            "qr_pilot_fits": refits,
            "alternate_target_values": target_values,
            "actual_public_transport_cases": len(public_actual),
            "max_error": error,
            "scope": "sameagent alternateformula/denseandQR audit;not independentbuilder or text/dynamictransport",
        },
    )


if __name__ == "__main__":
    main()
