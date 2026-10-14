#!/usr/bin/env python3
"""Alternate-formula verification of HM10B1 without new model calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/native_joint_r3"


def table(path):
    """Read an output table independently from the experiment's dataframe path."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def probabilities(logits):
    """Normalize directly, using max subtraction instead of logsumexp."""
    z = np.asarray(logits, float)
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def verify(source, out):
    """Replay all scores, conditional contrasts, cold predictions and intervals."""
    if out.exists():
        raise FileExistsError("use a fresh verification packet")
    m = read_json(source / "00/manifest.json")
    summary = read_json(source / "03/summary.json")
    checks, errors = defaultdict(int), defaultdict(float)

    def check(name, actual, expected):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"nonfinite or mismatched {name}")
        checks[name] += a.size
        errors[name] = max(errors[name], float(np.max(abs(a - b))))

    hashes = read_json(source / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(source / name) != expected:
            raise ValueError(f"output changed: {name}")
        checks["hashes"] += 1
    for name, expected in m["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"source changed: {name}")
        checks["source_hashes"] += 1
    values = np.asarray(m["alphabet"]["values"], float)
    reference = ROOT / m["spec"]["reference"]
    with np.load(source / "01/head.npz", allow_pickle=False) as a:
        head = a["head"].astype(float)
    functions = np.load(reference / "reference.npz", allow_pickle=False)
    units = {}
    for d in read_json(source / "00/design.json"):
        path = source / "01/units" / f"unit_{d['unit']:04d}.json"
        meta = read_json(path)
        if any(meta[k] != d[k] for k in ("unit", "text_id", "query")):
            raise ValueError("native unit identity mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            z = a["logits"].copy()
            p = probabilities(z)
            check("head_float32_replay", head @ a["final_normalized"].astype(float), z)
            check("native_probability", p, a["probabilities"])
            check("native_score", p @ values, meta["native"]["conditional_score"])
            q = d["query"]
            mean = functions[f"{q}__G"].sum(axis=0) @ a["mean_delta"]
            config = float(a["config_token_terms"].sum())
            final = functions[f"{q}__g"] @ (a["Q"] - functions[f"{q}__h"])
            affine = float(functions[f"{q}__b"] + mean + config + final)
            for k, v in (
                ("mean", mean),
                ("configuration", config),
                ("final", final),
                ("score", affine),
            ):
                check("affine_projection_replay", v, meta["affine"][k])
            check("affine_direct_pre_save", meta["affine_direct_error"], 0)
        units[d["text_id"], d["query"]] = {
            "logits": z,
            "p": p,
            "score": float(p @ values),
            "affine": affine,
        }
    functions.close()
    docs = read_json(source / "00/documents.json")
    cubes = defaultdict(dict)
    for d in docs:
        for query in m["spec"]["queries"]:
            cubes[d["cube_id"], query][d["mask"]] = units[d["text_id"], query]
    readouts = table(source / "02/readouts.csv")
    documents = {d["material_id"]: d for d in docs}
    for r in readouts:
        d = documents[r["material_id"]]
        if r["role"] != d["role"] or int(r["text_id"]) != d["text_id"]:
            raise ValueError("readout record mismatch")
        u = units[d["text_id"], r["query"]]
        check(
            "document_readouts",
            [float(r["native"]), float(r["affine"])],
            [u["score"], u["affine"]],
        )
    recomputed, raw_effect_rows = {}, table(source / "02/interactions.csv")
    for r in raw_effect_rows:
        cells = cubes[r["cube_id"], r["query"]]
        incidence = int(r["incidence"])
        sub = [cells[i + 4 * incidence] for i in range(4)]
        # Log probabilities differ from logits only by a common scalar gauge.
        centered = np.array([np.log(u["p"]) - np.log(u["p"]).mean() for u in sub])
        baseline = centered[1] + centered[2] - centered[0]
        delta = centered[3] - baseline
        scalar_add = sub[1]["score"] + sub[2]["score"] - sub[0]["score"]
        link_score = float(probabilities(baseline) @ values)
        total = sub[3]["score"] - scalar_add
        representation = sub[3]["score"] - link_score
        check("centered_logit_contrast", delta, json.loads(r["logit_interaction"]))
        check(
            "joint_scalar_split",
            [total, link_score - scalar_add, representation],
            [float(r[k]) for k in ("total", "link_only", "representation_action")],
        )

        def integrand(t, origin=baseline, direction=delta):
            p = probabilities(origin + t * direction)
            return float(p @ (values * direction) - (p @ values) * (p @ direction))

        integral = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
        check("adaptive_integral", integral, float(r["gradient_integral"]))
        p, q = sub[3]["p"], probabilities(baseline)
        kl = float(np.sum(p * np.log(p / q)))
        check("kl_bregman", [kl, kl], [float(r["kl"]), float(r["bregman"])])
        fisher = np.diag(q) - np.outer(q, q)
        check("fisher_quadratic", delta @ fisher @ delta, float(r["fisher_energy"]))
        if kl < -1e-12 or np.linalg.eigvalsh(fisher).min() < -1e-12:
            raise ValueError("invalid categorical geometry")
        recomputed[r["cube_id"], r["query"], incidence] = {
            "baseline": baseline,
            "delta": delta,
            "total": total,
            "actual": sub[3]["score"],
            "scalar_additive": scalar_add,
            "frame": int(r["frame"]),
            "actor": r["actor"],
            "incidence": incidence,
            "family": int(r["family"]),
            "role": r["role"],
            "query": r["query"],
        }
    for r in table(source / "02/content_incidence_pairs.csv"):
        cells = cubes[r["cube_id"], r["query"]]
        i = int(r["mask"])
        check(
            "incidence_pairs",
            [float(r["native_delta"]), float(r["affine_delta"])],
            [
                cells[i + 4]["score"] - cells[i]["score"],
                cells[i + 4]["affine"] - cells[i]["affine"],
            ],
        )
        if r["identical_raw"] == "True":
            check(
                "identical_raw_null",
                [float(r["native_delta"]), float(r["affine_delta"])],
                [0, 0],
            )
    source_groups = defaultdict(list)
    for r in recomputed.values():
        if r["role"] == "source":
            source_groups[r["query"], r["frame"], r["actor"], r["incidence"]].append(r)
    prediction_rows = table(source / "03/predictions.csv")
    for p in prediction_rows:
        r = recomputed[p["cube_id"], p["query"], int(p["incidence"])]
        key = (r["query"], r["frame"], r["actor"], r["incidence"])
        group = source_groups[key]
        if len({s["family"] for s in group}) != 8:
            raise ValueError("source family support changed")
        global_group = [
            s for g, rs in source_groups.items() if g[0] == r["query"] for s in rs
        ]
        arm = p["arm"]
        if arm == "scalar_additive":
            value = r["scalar_additive"]
        elif arm == "link_only":
            value = float(probabilities(r["baseline"]) @ values)
        elif arm == "source_conditional_scalar":
            value = r["scalar_additive"] + np.mean([s["total"] for s in group])
        else:
            if arm == "source_global_logit":
                chosen = global_group
            elif arm == "wrong_key_logit":
                chosen = source_groups[
                    r["query"], (r["frame"] + 1) % 4, r["actor"], 1 - r["incidence"]
                ]
            elif arm == "source_conditional_logit":
                chosen = group
            else:
                raise ValueError("unknown completion arm")
            correction = np.sum([s["delta"] for s in chosen], axis=0) / len(chosen)
            value = float(probabilities(r["baseline"] + correction) @ values)
        check(
            "cold_predictions",
            [value, (value - r["actual"]) ** 2],
            [float(p["prediction"]), float(p["squared_error"])],
        )
    for r in table(source / "03/completion_summary.csv"):
        selected = [
            p
            for p in prediction_rows
            if (p["query"], p["role"], p["arm"]) == (r["query"], r["role"], r["arm"])
        ]
        baseline = [
            p
            for p in prediction_rows
            if (p["query"], p["role"], p["arm"]) == (r["query"], r["role"], "link_only")
        ]
        mse = np.mean([float(p["squared_error"]) for p in selected])
        gain = np.mean([float(p["squared_error"]) for p in baseline]) - mse
        check(
            "summary_metrics",
            [np.sqrt(mse), gain],
            [float(r["rmse"]), float(r["mse_gain_over_link"])],
        )
    for qi, query in enumerate(m["spec"]["queries"]):
        keys = sorted(k[1:] for k in source_groups if k[0] == query)
        src = np.array(
            [
                [
                    next(
                        s["delta"]
                        for s in source_groups[(query, *k)]
                        if s["family"] == f
                    )
                    for k in keys
                ]
                for f in sorted(
                    {s["family"] for rs in source_groups.values() for s in rs}
                )
            ]
        )
        cold = [
            r
            for r in recomputed.values()
            if r["query"] == query and r["role"] == "application"
        ]
        families = sorted({r["family"] for r in cold})
        lookup = {
            (r["family"], (r["frame"], r["actor"], r["incidence"])): r for r in cold
        }
        z = np.array([[lookup[f, k]["baseline"] for k in keys] for f in families])
        y = np.array([[lookup[f, k]["actual"] for k in keys] for f in families])
        base_errors = (probabilities(z) @ values - y) ** 2
        rng = np.random.default_rng(m["spec"]["seed"] + qi)
        samples = []
        for _ in range(m["spec"]["bootstrap_draws"]):
            sw = np.bincount(
                rng.integers(0, len(src), len(src)), minlength=len(src)
            ) / len(src)
            aw = np.bincount(
                rng.integers(0, len(families), len(families)), minlength=len(families)
            ) / len(families)
            theta = np.einsum("f,fka->ka", sw, src)
            err = (probabilities(z + theta) @ values - y) ** 2
            samples.append(float(aw @ (base_errors - err).mean(axis=1)))
        ci = np.quantile(samples, [0.025, 0.975])
        reported = next(
            r for r in summary["source_application_uncertainty"] if r["query"] == query
        )
        check(
            "uncertainty_interval_endpoints",
            ci,
            [reported["ci_low"], reported["ci_high"]],
        )
    algebra_max = max(v for k, v in errors.items() if k != "head_float32_replay")
    passed = algebra_max < 1e-10 and errors["head_float32_replay"] < 3e-4
    out.mkdir(parents=True)
    report = {
        "packet": "HM-10B1",
        "status": "PASS" if passed else "FAIL",
        "checks": dict(checks),
        "max_errors": dict(errors),
        "max_algebra_error": algebra_max,
        "head_float32_tolerance": 3e-4,
        "projection_replay_does_not_reconstruct_full_P": True,
        "new_model_calls": 0,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
    }
    write_json(out / "verification.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise ValueError("native-response alternate verification failed")


def main():
    """Verify a frozen packet once without changing its primary artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_joint_verification_r1",
    )
    args = parser.parse_args()
    verify(args.source, args.out)


if __name__ == "__main__":
    main()
