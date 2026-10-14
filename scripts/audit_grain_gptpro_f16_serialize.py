#!/usr/bin/env python3
"""Independent checks of the public calibrated F16 bundles and sealed cold outputs.

Numeric T1 is recomputed by event-level Gaussian conditioning rather than the
cell-mean kernel. Numeric T2/T3 are recomputed by direct joint-covariance
conditioning on each eta node rather than the compiled affine basis. Text is
replayed against archived pilot-calibrated outputs of the original runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from grain_research.gptpro_foundation_v3 import f16_serialize as pub
from grain_research.gptpro_foundation_v3.f12_transport import transport_score
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file

ROOT = Path(__file__).resolve().parents[1]
F16 = ROOT / "results/grain_gptpro_foundation_v3/F16"
CONTRAST = np.array([-1.0, 0.0, 0.0, 1.0])
CONDITIONS = (-1.0, -1 / 3, 1 / 3, 1.0)
PLAN = np.array([1.0, -1.0, 1.0, -1.0])


def calibrated(fits, new, z):
    """Apply each serialized calibrator directly: (fit, author, event)."""
    out = []
    for fit in fits:
        value, ok = transport_score(fit, new, np.broadcast_to(z, new.shape))
        if not ok.all():
            raise ValueError("unsupported observation in independent check")
        out.append(value)
    return np.stack(out)


def independent_t1(bundle, new, z):
    """Event-level conditioning per source/calibrator, then total variance."""
    fits = (
        [bundle["calibration"]["plugin"]]
        if bundle["estimator"] == "plugin"
        else bundle["calibration"]["draws"]
    )
    x = calibrated(fits, new, z)
    h = np.array([[float(v == c) for c in CONDITIONS] for v in z])
    means, conditional = [], []
    for model in bundle["engine"]["source_models"]:
        mu = np.asarray(model["mu"])
        sigma = np.asarray(model["Sigma"])
        k = h @ sigma @ h.T + model["sigma2"] * np.eye(len(z))
        g = np.linalg.solve(k, h @ sigma @ CONTRAST)
        means.append(CONTRAST @ mu + (x - h @ mu) @ g)
        conditional.append(CONTRAST @ sigma @ CONTRAST - g @ h @ sigma @ CONTRAST)
    m = np.stack(means)  # source, calibrator, author
    score = m.mean((0, 1))
    variance = (
        np.mean(conditional)
        + m.var(axis=0, ddof=0).mean(0)
        + m.mean(0).var(axis=0, ddof=0)
    )
    return score, variance


def ar_cov(rho, q, n):
    """Stationary AR(1) state covariance."""
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return q / (1 - rho * rho) * rho**lag


def independent_dynamic(bundle, new, z, nodes=64):
    """Direct joint conditioning of (y_obs, s_47, mean future y) per eta node."""
    fits = [bundle["calibration"]["plugin"]]
    if bundle["estimator"] == "mixture":
        fits = bundle["calibration"]["draws"]
    x = calibrated(fits, new, z)  # calibrator, author, time
    zz = np.r_[z, PLAN]
    design = np.column_stack([np.ones(52), zz])
    g = np.zeros(52)
    g[48:] = 0.25
    hx, hw = hermgauss(nodes)
    parts_mean, parts_cov = [], []
    for model in bundle["engine"]["source_models"]:
        mu = np.asarray(model["prior"]["mu"])
        sigma = np.asarray(model["prior"]["Sigma"])
        q, r = np.exp(2 * np.asarray(model["log_sigma"]))
        eta = mu[2] + np.sqrt(2 * sigma[2, 2]) * hx
        logw = np.log(hw / np.sqrt(np.pi))
        cab = sigma[:2, :2] - np.outer(sigma[:2, 2], sigma[2, :2]) / sigma[2, 2]
        node_mean, node_cov, node_ll = [], [], []
        for k, e in enumerate(eta):
            rho = 0.99 * np.tanh(e)
            mab = mu[:2] + (e - mu[2]) * sigma[:2, 2] / sigma[2, 2]
            s = ar_cov(rho, q, 52)
            cy = design @ cab @ design.T + s + r * np.eye(52)
            obs = slice(0, 48)
            c_obs = cy[obs, obs]
            t2_cross = s[47, obs]
            t3_cross = g @ cy[:, obs]
            cross = np.stack([t2_cross, t3_cross])
            prior = np.array([[s[47, 47], g @ s[:, 47]], [g @ s[:, 47], g @ cy @ g]])
            mean_obs = design[obs] @ mab
            mean_target = np.array([0.0, mab[0] + mab[1] * PLAN.mean()])
            chol = np.linalg.cholesky(c_obs)
            resid = x - mean_obs  # calibrator, author, time
            white = np.linalg.solve(chol, resid.reshape(-1, 48).T)  # time, J*A
            solved = np.linalg.solve(chol.T, white)
            cond_mean = mean_target[:, None] + cross @ solved
            cond_cov = prior - cross @ np.linalg.solve(c_obs, cross.T)
            ll = (
                -0.5 * np.sum(white * white, axis=0)
                - np.log(np.diag(chol)).sum()
                - 24 * np.log(2 * np.pi)
            )
            node_mean.append(cond_mean.T.reshape(*x.shape[:2], 2))
            node_cov.append(cond_cov)
            node_ll.append(ll.reshape(x.shape[:2]) + logw[k])
        node_mean = np.stack(node_mean)  # node, J, A, 2
        node_ll = np.stack(node_ll)
        w = np.exp(node_ll - logsumexp(node_ll, axis=0))
        m = np.einsum("kja,kjal->jal", w, node_mean)
        d = node_mean - m[None]
        c = np.einsum("kja,klm->jalm", w, np.stack(node_cov)) + np.einsum(
            "kja,kjal,kjam->jalm", w, d, d
        )
        parts_mean.append(m)
        parts_cov.append(c)
    m = np.stack(parts_mean)  # source, J, A, 2
    c = np.stack(parts_cov)
    mean = m.mean((0, 1))
    within = c.mean((0, 1))
    per_j = m.mean(0)
    source = np.einsum("sjal,sjam->jalm", m - per_j[None], m - per_j[None]).mean(
        0
    ) / len(m)
    pilot = np.einsum("jal,jam->alm", per_j - mean[None], per_j - mean[None]) / len(
        per_j
    )
    return mean, within + source + pilot


def read_jsonl(path):
    """Read JSON lines."""
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def leaves(a, b, skip=("reference_score", "reference_variance")):
    """Max absolute numeric difference over full engine objects; exact non-numerics."""
    if isinstance(a, dict):
        if not set(a) <= set(b):
            raise ValueError("archived engine field missing from cold output")
        return max([leaves(a[k], b[k], skip) for k in a if k not in skip] or [0.0])
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("engine list lengths differ")
        return max([leaves(x, y, skip) for x, y in zip(a, b, strict=True)] or [0.0])
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    if a != b:
        raise ValueError(f"non-numeric engine field differs: {a} vs {b}")
    return 0.0


def replay(run, entry, rows):
    """Compare sealed cold outputs with canonical archived application results."""
    seed, route = entry["seed"], entry["route"]
    if route == "numeric_t1":
        canon = json.loads(
            (
                F16
                / f"t1_transport_full_r1/{seed}/application_scores_before_truth.json"
            ).read_text()
        )
        c = next(
            x
            for x in canon
            if x["key"] == [entry["world"], entry["mode"], entry["estimator"]]
        )
        m = c["moments"]
        if not all(c["support"]):
            raise ValueError("canonical support incomplete")
        return max(
            max(
                abs(r["score"] - m["score"][i]),
                abs(r["variance"] - m["variance"][i]),
                *(
                    abs(r["components"][k] - m[k][i])
                    for k in ("conditional", "source", "pilot")
                ),
            )
            for i, r in enumerate(rows)
        )
    if route == "numeric_dynamic":
        arr = np.load(
            F16
            / f"dynamic_transport_full_r1/{seed}/{entry['world']}_{entry['mode']}_{entry['estimator']}.npz"
        )
        t = 0 if entry["target"] == "T2" else 1
        return max(
            max(
                abs(r["score"] - arr["mean"][i, t]),
                abs(r["variance"] - arr["covariance"][i, t, t]),
                *(
                    abs(r["components"][k] - arr[k][i, t, t])
                    for k in ("state_future", "personal", "source", "pilot")
                ),
            )
            for i, r in enumerate(rows)
        )
    name = (
        "T1" if route == "text_t1" else "dynamic"
    ) + f"_changed_pilot_{entry['estimator']}.json"
    canon = json.loads(
        (F16 / f"text_transport_full_r1/scores/{seed}/{name}").read_text()
    )
    extras = set()
    for c, r in zip(canon, rows, strict=True):
        extras |= set(r["engine_output"]) - set(c)
    declared = {
        "active_materials",
        "stale_annotations",
        "reference_score",
        "reference_variance",
    }
    if not extras <= declared:
        raise ValueError(f"undeclared cold-only engine fields {extras}")
    return max(leaves(c, r["engine_output"]) for c, r in zip(canon, rows, strict=True))


def main():
    """Audit a completed serialize run into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.run / "bundle_index.json").read_text())
    replays, sealing, explanations, references, independent = [], [], [], [], []
    for entry in index:
        path = args.run / "bundles" / f"{entry['name']}.json"
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("bundle changed")
        bundle = json.loads(path.read_text())
        pub.verify(bundle)
        cold = args.run / "cold" / f"{entry['name']}.scores.jsonl"
        rows = read_jsonl(cold)
        manifest = json.loads(Path(str(cold) + ".manifest.json").read_text())
        if manifest["bundle_sha256"] != entry["sha256"]:
            raise ValueError("cold output bound to another bundle")
        if any(r["status"] != "SCORED" for r in rows):
            raise ValueError(
                f"unexpected rejection in canonical replay {entry['name']}"
            )
        replays.append(
            {
                "name": entry["name"],
                "route": entry["route"],
                "authors": len(rows),
                "max_abs_diff": replay(args.run, entry, rows),
            }
        )
        sealing.append(
            {
                "name": entry["name"],
                "refused_opens": manifest["refused_opens"],
                "undeclared_results_reads": [
                    p
                    for p in manifest["files_opened"]
                    if "/results/" in p
                    and p
                    not in (
                        str(path.resolve()),
                        str(
                            (
                                args.run / "cold" / f"{entry['name']}.requests.jsonl"
                            ).resolve()
                        ),
                        str(cold.resolve()),
                        str(cold.resolve()) + ".manifest.json",
                    )
                ],
            }
        )
        if entry["route"].startswith("numeric"):
            explanations.append(
                {
                    "name": entry["name"],
                    "max_identity_error": max(
                        r["explanation"]["identity_error"] for r in rows
                    ),
                }
            )
        if entry["route"].startswith("text"):
            ref_rows = read_jsonl(
                args.run / "cold" / f"{entry['name']}.reference_scores.jsonl"
            )
            values = np.array([r["score"] for r in ref_rows])
            references.append(
                {
                    "name": entry["name"],
                    "authors": len(values),
                    "mu_error": abs(values.mean() - bundle["reference_policy"]["mu"]),
                    "sigma_error": abs(
                        values.std(ddof=1) - bundle["reference_policy"]["sigma"]
                    ),
                    "fresh_mu": bundle["reference_policy"]["mu"],
                    "fresh_sigma": bundle["reference_policy"]["sigma"],
                    "historical_mu": bundle["reference_policy"]["historical_positions"][
                        0
                    ]["mu"],
                    "historical_sigma": bundle["reference_policy"][
                        "historical_positions"
                    ][0]["sigma"],
                }
            )
        if entry["route"] == "numeric_t1":
            requests = read_jsonl(args.run / "cold" / f"{entry['name']}.requests.jsonl")
            new = np.array([[x["y"] for x in r["data"]["records"]] for r in requests])
            z = np.array([x["z"] for x in requests[0]["data"]["records"]])
            score, variance = independent_t1(bundle, new, z)
            independent.append(
                {
                    "name": entry["name"],
                    "method": "event-level conditioning,all authors",
                    "authors": len(rows),
                    "score_max": float(
                        np.max(np.abs(score - [r["score"] for r in rows]))
                    ),
                    "variance_max": float(
                        np.max(np.abs(variance - [r["variance"] for r in rows]))
                    ),
                }
            )
        elif entry["route"] == "numeric_dynamic":
            requests = read_jsonl(args.run / "cold" / f"{entry['name']}.requests.jsonl")
            chosen = [0, len(requests) - 1]
            new = np.array(
                [[x["y"] for x in requests[i]["data"]["records"]] for i in chosen]
            )
            z = np.array([x["z"] for x in requests[0]["data"]["records"]])
            mean, cov = independent_dynamic(bundle, new, z)
            t = 0 if entry["target"] == "T2" else 1
            independent.append(
                {
                    "name": entry["name"],
                    "method": "direct joint conditioning per eta node,first/last author",
                    "authors": 2,
                    "score_max": float(
                        max(
                            abs(mean[a, t] - rows[i]["score"])
                            for a, i in enumerate(chosen)
                        )
                    ),
                    "variance_max": float(
                        max(
                            abs(cov[a, t, t] - rows[i]["variance"])
                            for a, i in enumerate(chosen)
                        )
                    ),
                }
            )
        print(json.dumps({"audited": entry["name"]}), flush=True)
    tolerance = {"numeric_t1": 1e-9, "numeric_dynamic": 1e-6}
    route_of = {e["name"]: e["route"] for e in index}
    summary = {
        "bundles": len(index),
        "replay_max_abs_diff": max(r["max_abs_diff"] for r in replays),
        "replay_pass": all(r["max_abs_diff"] <= 1e-10 for r in replays),
        "sealed_refused_opens": sum(len(s["refused_opens"]) for s in sealing),
        "sealed_undeclared_results_reads": sum(
            len(s["undeclared_results_reads"]) for s in sealing
        ),
        "explanation_identity_max": max(e["max_identity_error"] for e in explanations),
        "fresh_reference_replay_max": max(
            max(r["mu_error"], r["sigma_error"]) for r in references
        ),
        "independent_max": {
            route: {
                "score": max(
                    i["score_max"] for i in independent if route_of[i["name"]] == route
                ),
                "variance": max(
                    i["variance_max"]
                    for i in independent
                    if route_of[i["name"]] == route
                ),
            }
            for route in tolerance
        },
    }
    summary["independent_pass"] = bool(
        all(
            v["score"] <= tolerance[k] and v["variance"] <= tolerance[k]
            for k, v in summary["independent_max"].items()
        )
    )

    summary["reference_pass"] = bool(summary["fresh_reference_replay_max"] <= 1e-10)
    summary["sealing_pass"] = (
        summary["sealed_refused_opens"] == 0
        and summary["sealed_undeclared_results_reads"] == 0
    )
    for name, value in (
        ("replays", replays),
        ("sealing", sealing),
        ("explanations", explanations),
        ("references", references),
        ("independent", independent),
        ("summary", summary),
    ):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, sort_keys=True, indent=1) + "\n"
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
