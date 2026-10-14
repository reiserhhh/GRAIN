#!/usr/bin/env python3
"""Audit actual text calibration and target scores with alternate moment formulas."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import qmc

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260915, 20260916, 20260917)


def save(path, value):
    """Write strict audit JSON without modifying primary artifacts."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def priors(bundle, index, dynamic):
    """Reconstruct fixed quadrature addresses independently of scorer helpers."""
    seed = int(
        np.random.SeedSequence(
            [bundle["seed"], 164 if dynamic else 161, index, 0]
        ).generate_state(1)[0]
    )
    u = qmc.Sobol(3 if dynamic else 4, scramble=True, seed=seed).random_base2(
        bundle["qmc_power"]
    )
    n = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
    model = bundle["source_models"][index]
    prior = model["prior"] if dynamic else model
    return np.asarray(prior["mu"]) + n @ np.linalg.cholesky(prior["Sigma"]).T


def static_moments(raw, bundle):
    """Enumerate all four hidden label/eligibility states then raw second moments."""
    conditions = (-1.0, -1 / 3, 1 / 3, 1.0)
    counts = np.zeros((len(raw), 4, 4))
    for i, x in enumerate(raw):
        for op in x["history"]:
            r = op["observation"]
            m = bundle["materials"][r["material_id"]]
            c = 2 * int(m["pole_readout"] >= 0.5) + int(m["grain_qualification"] >= 0.5)
            k = int(np.argmin(np.abs(np.asarray(conditions) - r["z"])))
            if abs(conditions[k] - r["z"]) > 1e-10:
                raise ValueError("condition not on source design")
            counts[i, k, c] += 1
    centers, seconds = [], []
    for j, model in enumerate(bundle["source_models"]):
        h = priors(bundle, j, False)
        e = np.array(bundle["channel_draws"]["hybrid_joint"][j])
        p = ndtr(h / np.sqrt(model["sigma2"]))
        likelihood = np.zeros((*p.shape, 4))
        for b in (0, 1):
            for q in (0, 1):
                mass = (p if b else 1 - p) * 0.5 if q else np.full(p.shape, 0.25)
                likelihood += mass[..., None] * e[b, q]
        ll = (
            np.log(np.maximum(likelihood, np.finfo(float).tiny)).reshape(len(h), 16)
            @ counts.reshape(len(raw), 16).T
        )
        weights = np.exp(ll - logsumexp(ll, axis=0))
        d = h[:, -1] - h[:, 0]
        centers.append(d @ weights)
        seconds.append((d * d) @ weights)
    mean = np.mean(centers, axis=0)
    return mean, np.mean(seconds, axis=0) - mean**2


def dynamic_moments(raw, bundle):
    """Filter through explicit truncated-normal mixture components, not derivatives."""
    z = np.array([[op["observation"]["z"] for op in x["history"]] for x in raw])
    categories = np.array(
        [
            [
                2
                * int(
                    bundle["materials"][op["observation"]["material_id"]][
                        "pole_readout"
                    ]
                    >= 0.5
                )
                + int(
                    bundle["materials"][op["observation"]["material_id"]][
                        "grain_qualification"
                    ]
                    >= 0.5
                )
                for op in x["history"]
            ]
            for x in raw
        ]
    )
    means, seconds = [], []
    for j, model in enumerate(bundle["source_models"]):
        p = priors(bundle, j, True)
        rho = 0.99 * np.tanh(p[:, 2, None])
        q, r = np.exp(2 * np.asarray(model["log_sigma"]))
        e = np.array(bundle["channel_draws"]["hybrid_joint"][j])
        mu = np.zeros((len(p), len(raw)))
        var = np.broadcast_to(q / (1 - rho**2), mu.shape).copy()
        ll = np.zeros_like(mu)
        for t in range(48):
            if t:
                mu *= rho
                var = rho * rho * var + q
            k = (mu + p[:, 0, None] + p[:, 1, None] * z[:, t]) / np.sqrt(var + r)
            logphi = -0.5 * k * k - 0.5 * np.log(2 * np.pi)
            denominator = np.zeros_like(mu)
            first = np.zeros_like(mu)
            second = np.zeros_like(mu)
            unrelated = 0.25 * (e[0, 0, categories[:, t]] + e[1, 0, categories[:, t]])
            denominator += unrelated
            first += unrelated * mu
            second += unrelated * (var + mu * mu)
            for b, sign in ((0, -1), (1, 1)):
                logcdf = log_ndtr(sign * k)
                inverse_mills = np.exp(logphi - logcdf)
                weight = 0.5 * e[b, 1, categories[:, t]] * np.exp(logcdf)
                m = mu + sign * var / np.sqrt(var + r) * inverse_mills
                v = var - var * var / (var + r) * (
                    sign * k * inverse_mills + inverse_mills**2
                )
                denominator += weight
                first += weight * m
                second += weight * (v + m * m)
            mu = first / denominator
            var = second / denominator - mu * mu
            if np.any(var < -1e-10):
                raise ValueError("alternate negative filter variance")
            ll += np.log(denominator)
        weights = np.exp(ll - logsumexp(ll, axis=0))
        loading = sum(rho**k for k in range(1, 5)) / 4
        future_noise = (
            q * sum((sum(rho**k for k in range(4 - j)) / 4) ** 2 for j in range(4))
            + r / 4
        )
        m = np.stack([mu, p[:, 0, None] + mu * loading], axis=-1)
        covariance = np.zeros((*mu.shape, 2, 2))
        covariance[..., 0, 0] = var
        covariance[..., 0, 1] = covariance[..., 1, 0] = var * loading
        covariance[..., 1, 1] = var * loading**2 + future_noise
        means.append(np.sum(weights[..., None] * m, axis=0))
        seconds.append(
            np.sum(
                weights[..., None, None]
                * (covariance + m[..., :, None] * m[..., None, :]),
                axis=0,
            )
        )
    mean = np.mean(means, axis=0)
    return mean, np.mean(seconds, axis=0) - mean[:, :, None] * mean[:, None, :]


def main():
    """Verify saved evidence; cold-replay fixed cases; never select a better model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def compare(left, right, name, atol=1e-8):
        l, r = np.asarray(left, float), np.asarray(right, float)
        if l.shape != r.shape or not np.isfinite(l).all() or not np.isfinite(r).all():
            raise ValueError(f"invalid {name}")
        difference = float(np.max(abs(l - r))) if l.size else 0.0
        if difference > atol:
            raise ValueError(f"{name}: {difference}")
        checks.append({"name": name, "values": int(l.size), "max_error": difference})

    manifest = json.loads((args.run / "output_manifest.json").read_text())
    for p, h in manifest.items():
        if sha256_file(args.run / p) != h:
            raise ValueError(f"changed output {p}")
    bank = pd.read_csv(args.run / "reader_scores.csv")
    source = json.loads((args.run / "protocol_before_reading.json").read_text())
    for group in ("code", "inputs"):
        for p, h in source[group].items():
            if sha256_file(Path(p)) != h:
                raise ValueError(f"changed source {p}")
    if bank.text.nunique() != 380 or bank.material_id.nunique() != 380:
        raise ValueError("duplicate or missing material")
    raw = pd.read_csv(args.run / "nli_probabilities.csv").set_index("audit_id")
    rederived = []
    for row in bank.itertuples():
        positive = raw.loc[row.material_id + ":positive", "p_entailment"]
        negative = raw.loc[row.material_id + ":negative", "p_entailment"]
        rederived.append(
            [
                positive / (positive + negative),
                raw.loc[row.material_id + ":qualification", "p_entailment"],
            ]
        )
    compare(
        rederived,
        bank[["pole_readout", "nli_qualification"]],
        "all_actual_NLI_readouts",
    )
    features = np.load(args.run / "grain_features.npz")
    head = json.loads(
        (
            ROOT
            / "results/grain_gptpro_foundation_v3/F11/grain_comment_reader_r1/readout_bundle.json"
        ).read_text()
    )
    values = (features["features"] - head["center"]) / head["scale"] @ np.array(
        head["coef"]
    ).T + head["intercept"]
    compare(values, features["raw"], "all_GRAIN_linear_readouts")
    compare(np.clip(values[:, 1], 0, 1), bank.grain_qualification, "GRAIN_clip")
    laws = json.loads((args.run / "channels.json").read_text())
    for key, role, qkey in (
        ("old", "source", "grain_qualification"),
        ("pilot", "pilot", "nli_qualification"),
        ("known_pool", "application", "nli_qualification"),
    ):
        sel = bank[bank.role.eq(role)]
        if key == "pilot":
            sel = sel[sel.pilot_partition.eq("fit")]
        category = 2 * (sel.pole_readout >= 0.5).astype(int) + (
            sel[qkey] >= 0.5
        ).astype(int)
        table = pd.crosstab([sel.pole, sel.qualification], category).reindex(
            index=pd.MultiIndex.from_product([[0, 1], [0, 1]]),
            columns=range(4),
            fill_value=0,
        )
        compare(
            table.to_numpy().reshape(2, 2, 4), laws[key]["counts"], f"crosstab_{key}", 0
        )
    for entry in json.loads(
        (args.run / "missing_anchor_controls.json").read_text()
    ).values():
        if entry["status"] != "MISSING_ANCHOR":
            raise ValueError("anchor control not rejected")
    diagnostics = []
    for seed in SEEDS:
        folder = args.run / "scores" / str(seed)
        for is_dynamic, name in ((False, "T1"), (True, "dynamic")):
            bundlepath = folder / f"{name}_primary_bundle.json"
            bundle = json.loads(bundlepath.read_text())
            inputs = [
                json.loads(x)
                for x in (folder / f"{name}_inputs.jsonl").read_text().splitlines()
            ]
            results = json.loads(
                (folder / f"{name}_changed_pilot_mixture.json").read_text()
            )
            if len(inputs) != 200 or len(bundle["source_models"]) != 200:
                raise ValueError("abbreviated application or source")
            subset = [inputs[0], inputs[-1]]
            selected = [results[0], results[-1]]
            if is_dynamic:
                m, v = dynamic_moments(subset, bundle)
                compare(
                    m,
                    [
                        [r["targets"][t]["score"] for t in ("T2", "T3")]
                        for r in selected
                    ],
                    f"{seed}_dynamic_alternate_means",
                )
                compare(
                    np.diagonal(v, axis1=1, axis2=2),
                    [
                        [r["targets"][t]["variance"] for t in ("T2", "T3")]
                        for r in selected
                    ],
                    f"{seed}_dynamic_alternate_variance",
                )
            else:
                m, v = static_moments(inputs, bundle)
                compare(
                    m, [r["score"] for r in results], f"{seed}_T1_all_alternate_means"
                )
                compare(
                    v,
                    [r["variance"] for r in results],
                    f"{seed}_T1_all_raw_second_moments",
                )
            inputfile = args.output / f"{seed}_{name}_cold_inputs.jsonl"
            inputfile.write_text("".join(json.dumps(r) + "\n" for r in subset))
            outfile = args.output / f"{seed}_{name}_cold.json"
            cmd = [
                sys.executable,
                str(ROOT / "scripts/run_grain_gptpro_f16_text_transport.py"),
                "--stage",
                "cold",
                "--input",
                str(inputfile.resolve()),
                "--bundle",
                str(bundlepath.resolve()),
                "--output",
                str(outfile.resolve()),
            ]
            if is_dynamic:
                cmd.append("--dynamic")
            subprocess.run(
                cmd,
                cwd=args.output,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                check=True,
                capture_output=True,
            )
            cold = json.loads(outfile.read_text())
            if is_dynamic:
                compare(
                    [
                        [
                            r["targets"][t][k]
                            for t in ("T2", "T3")
                            for k in ("score", "variance")
                        ]
                        for r in cold
                    ],
                    [
                        [
                            r["targets"][t][k]
                            for t in ("T2", "T3")
                            for k in ("score", "variance")
                        ]
                        for r in selected
                    ],
                    f"{seed}_dynamic_cold",
                )
            else:
                compare(
                    [[r[k] for k in ("score", "variance")] for r in cold],
                    [[r[k] for k in ("score", "variance")] for r in selected],
                    f"{seed}_T1_cold",
                )
        for check in json.loads((folder / "diagnostics.json").read_text()):
            for mode in ("double_nodes", "alternate_channel_draws"):
                for index, (base, other) in enumerate(
                    zip(check["base"], check[mode], strict=True)
                ):
                    pairs = (
                        {"T1": (base, other)}
                        if check["target"] == "T1"
                        else {
                            t: (base["targets"][t], other["targets"][t])
                            for t in ("T2", "T3")
                        }
                    )
                    for t, (a, b) in pairs.items():
                        diagnostics.append(
                            {
                                "seed": seed,
                                "target": t,
                                "case": index,
                                "mode": mode,
                                "score_delta": b["score"] - a["score"],
                                "sd_delta": np.sqrt(b["variance"])
                                - np.sqrt(a["variance"]),
                            }
                        )
    frame = pd.read_csv(args.run / "predictions.csv")
    truth_hashes = {}
    for seed in SEEDS:
        for dynamic_target, targets, archive in (
            (
                False,
                ("T1",),
                ROOT
                / f"results/grain_gptpro_foundation_v3/F04/source_full_r1/L_{seed}",
            ),
            (
                True,
                ("T2", "T3"),
                ROOT
                / f"results/grain_gptpro_foundation_v3/F05/source_full_r2/core_{seed}",
            ),
        ):
            truth_path = archive / "truth.json"
            truth_hashes[str(truth_path)] = sha256_file(truth_path)
            latent = {
                r["author_id"]: r
                for r in json.loads(truth_path.read_text())
                if r["role"] == "application"
            }
            events = [
                json.loads(line)
                for line in (archive / "application.jsonl").read_text().splitlines()
            ]
            future = {}
            for row in events:
                if 48 <= row["t"] < 52:
                    future.setdefault(row["author_id"], []).append(row)
            for target in targets:
                g = frame[frame.seed.eq(seed) & frame.target.eq(target)]
                expected = []
                for a in g.author_id:
                    y = latent[a]
                    if not dynamic_target:
                        value = y["means"][-1] - y["means"][0]
                    elif target == "T2":
                        value = y["states"][47]
                    else:
                        ordered = sorted(future[a], key=lambda r: r["t"])
                        # Recover each innovation then reapply the declared future plan.
                        value = (
                            sum(
                                row["y"] - y["b"] * row["z"] + y["b"] * z
                                for row, z in zip(ordered, (1, -1, 1, -1), strict=True)
                            )
                            / 4
                        )
                    expected.append(value)
                compare(expected, g.truth, f"archive_truth_{seed}_{target}")
    summary = pd.read_csv(args.run / "metrics_by_seed.csv").set_index(
        ["seed", "target", "arm"]
    )
    for keys, g in frame.groupby(["seed", "target", "arm"]):
        p, y = g.score.to_numpy(), g.truth.to_numpy()
        values = [
            np.mean((p - y) ** 2),
            np.mean(p - y),
            np.corrcoef(p, y)[0, 1],
            np.mean((g.lower <= g.truth) & (g.truth <= g.upper)),
            np.mean(g.upper - g.lower),
        ]
        compare(
            values,
            summary.loc[keys, ["mse", "bias", "r", "coverage", "width"]],
            f"metric_{keys}",
        )
    for result in json.loads((args.run / "paired_contrasts.json").read_text()):
        g = frame[frame.target.eq(result["target"])].copy()
        g["loss"] = np.square(g.score - g.truth)
        wide = g.pivot(index=["seed", "author_id"], columns="arm", values="loss")
        comparator = result["contrast"].split("-minus-")[1]
        difference = wide.changed_pilot_mixture - wide[comparator]
        rng = np.random.default_rng(171)
        parts = []
        for seed in SEEDS:
            d = difference.loc[seed].to_numpy()
            indices = rng.integers(0, len(d), size=(2000, len(d)))
            parts.append(np.sum(d[indices], axis=1) / len(d))
        replicates = np.mean(parts, axis=0)
        compare(
            [difference.mean(), *np.quantile(replicates, [0.025, 0.975])],
            [result[k] for k in ("delta_mse", "lower", "upper")],
            f"paired_CI_{result['target']}_{comparator}",
        )
    save(args.output / "truth_artifact_hashes.json", truth_hashes)
    save(args.output / "checks.json", checks)
    save(args.output / "numerical_diagnostics.json", diagnostics)
    save(
        args.output / "summary.json",
        {
            "status": "FINITE_TEXT_TRANSPORT_NUMERICAL_AUDIT_PASS",
            "output_hashes": len(manifest),
            "checks": len(checks),
            "numeric_values": sum(r["values"] for r in checks),
            "max_error": max(r["max_error"] for r in checks),
            "cold_objects": 12,
            "alternate_T1_authors": 600,
            "alternate_dynamic_authors": 6,
            "scope": "same-agent alternate enumeration/truncatedGaussian/second-moment audit,not independent human replication",
        },
    )
    print((args.output / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
