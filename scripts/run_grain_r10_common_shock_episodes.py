#!/usr/bin/env python
"""R10/R29 common-shock episodes (docs/GRAIN_R10_COMMON_SHOCK_EPISODES_PROTOCOL_V1.md). CPU only, 0 neural calls.

Stages: checks (archived replay + equivalence/no-future/independence/dense checks) | smoke (timing only) | full.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped  # noqa: E402
from grain_research.gptpro_foundation_v3.f05_joint import joint_filter, learn_common_variance  # noqa: E402
from grain_research.gptpro_foundation_v3.f05_recovery import (  # noqa: E402
    filter_arrays,
    fit_person,
    fit_source_dynamics,
    generate_dynamics,
)
from scripts.run_grain_gptpro_f05_joint import panel_predictions  # noqa: E402  (original, unchanged)

PROTOCOL = ROOT / "docs/GRAIN_R10_COMMON_SHOCK_EPISODES_PROTOCOL_V1.md"
PROTOCOL_SHA = "5179228b862609d398037348b528997651812749efac5851a018905d3a10ad4e"
R = ROOT / "results/grain_gptpro_foundation_v3"
OUT = R / "F05/common_shock_episodes_r1"
WORLDS = ("core", "shared_shock")
EPISODE_SEEDS = tuple(range(20261001, 20261033))
COUNTS = {"source": 300, "validation": 0, "application": 200}
BUDGETS = (16, 32, 48)
Z = 1.959963984540054
TRUE_COMMON = {"core": 0.0, "shared_shock": 0.125}


def sha_rows(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def block_of(budget: int) -> str:
    return f"n{budget}"


def score_rows(preds, people, truth, rho_of, q, r):
    """Evaluation-side metrics for panel_predictions rows (truth enters only here)."""
    out = []
    for row in preds:
        a, t = row["author_id"], row["t"]
        tr = truth[a]
        x = tr["states"][t]
        m, P = row["state_mean"], row["state_variance"]
        rec = {"author_id": a, "t": t, "budget": row["parameter_prefix"], "method": row["method"], "arm": row["arm"],
               "state_mean": m, "state_variance": P, "state_truth": x, "state_err2": (m - x) ** 2,
               "state_cover": float(abs(m - x) <= Z * np.sqrt(P)), "state_width": 2 * Z * np.sqrt(P)}
        rho = rho_of[a]
        for k in (1, 4):
            key = f"forecast{k}"
            if key not in row:
                continue
            fut = people[a][t + k]
            var = rho ** (2 * k) * P + q * sum(rho ** (2 * j) for j in range(k)) + r
            cond = tr["a"] + tr["b"] * fut["z"] + tr["rho"] ** k * x
            yhat = row[key]
            rec.update({f"f{k}_mean": yhat, f"f{k}_var": var, f"f{k}_err2": (yhat - fut["y"]) ** 2,
                        f"f{k}_cover": float(abs(yhat - fut["y"]) <= Z * np.sqrt(var)), f"f{k}_width": 2 * Z * np.sqrt(var),
                        f"f{k}_cond_err2": (yhat - cond) ** 2})
        out.append(rec)
    return out


def run_episode(world: str, seed: int, out_dir: Path | None, budgets=BUDGETS):
    t0 = time.time()
    rows, truths = generate_dynamics(world, seed, counts=COUNTS)
    source_rows = [r for r in rows if r["role"] == "source"]
    app_rows = [r for r in rows if r["role"] == "application"]
    assert source_rows and app_rows and not ({r["author_id"] for r in source_rows} & {r["author_id"] for r in app_rows})
    # ---- source side: source observations only
    model = fit_source_dynamics(source_rows)
    moment = learn_common_variance(source_rows, model)
    logs = np.asarray(model["log_sigma"])
    r_hat = float(np.exp(2 * logs[1]))
    t_src = time.time() - t0
    # ---- application personal parameters from prefixes only
    people = grouped(app_rows)
    psi, fails = {n: {} for n in budgets}, {n: 0 for n in budgets}
    for a, events in people.items():
        for n in budgets:
            prefix = events[:n]
            assert max(e["t"] for e in prefix) < n
            fit = fit_person(prefix, logs, model["prior"])
            psi[n][a] = fit["psi"]
            fails[n] += int(not fit["success"])
    truth = {tr["author_id"]: tr for tr in truths if tr["role"] == "application"}
    events_out = []
    for n in budgets:
        preds = panel_predictions(people, psi[n], moment["total_variance"], r_hat, moment["common_variance"], n, "personal_MAP")
        rho_of = {a: 0.99 * np.tanh(psi[n][a][2]) for a in people}
        events_out += score_rows(preds, people, truth, rho_of, moment["total_variance"], r_hat)
        oracle_psi = {a: [truth[a]["a"], truth[a]["b"], eta_from_rho(truth[a]["rho"])] for a in people}
        preds = panel_predictions(people, oracle_psi, 0.25, 0.25, TRUE_COMMON[world], n, "oracle")
        rho_of = {a: 0.99 * np.tanh(oracle_psi[a][2]) for a in people}
        events_out += score_rows(preds, people, truth, rho_of, 0.25, 0.25)
    ev = pd.DataFrame(events_out)
    ev.insert(0, "seed", seed)
    ev.insert(0, "world", world)
    metric_cols = [c for c in ev.columns if c.endswith(("_err2", "_cover", "_width"))]
    auth = ev.groupby(["world", "seed", "author_id", "budget", "method", "arm"])[metric_cols].mean().reset_index()
    app_truth = [tr for tr in truths if tr["role"] == "application"]
    common_app = np.asarray(app_truth[0]["common_innovations"])
    common_src = np.asarray(next(tr for tr in truths if tr["role"] == "source")["common_innovations"])
    info = {"world": world, "seed": seed, "source_rows_sha256": sha_rows(source_rows), "application_rows_sha256": sha_rows(app_rows),
            "source_authors": len({r["author_id"] for r in source_rows}), "application_authors": len(people),
            "log_sigma": model["log_sigma"], "prior_mu": model["prior"]["mu"],
            "source_successful_authors_by_round": [h["successful_authors"] for h in model["rounds"]],
            "source_noise_success_by_round": [h["noise_success"] for h in model["rounds"]],
            **{f"moment_{k}": v for k, v in moment.items() if k != "method"},
            "personal_fit_failures": {str(n): v for n, v in fails.items()},
            "truth_side_realized_common_msq_application": float(np.mean(common_app ** 2)),
            "truth_side_common_app_src_corr": float(np.corrcoef(common_app, common_src)[0, 1]) if world == "shared_shock" else None,
            "seconds_source": t_src, "seconds_total": time.time() - t0}
    if out_dir is not None:
        d = out_dir / "episodes" / f"{world}_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        ev.to_csv(d / "events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        auth.to_csv(d / "author_metrics.csv", index=False)
        (d / "episode.json").write_text(json.dumps(info, indent=1) + "\n")
    return info, auth


# ---------------------------------------------------------------- checks
def dense_check():
    rng = np.random.default_rng(7)
    n, T = 3, 5
    psi = np.array([[0.3, 0.4, eta_from_rho(0.5)], [-0.2, 0.7, eta_from_rho(0.3)], [0.1, 0.2, eta_from_rho(0.7)]])
    rho = 0.99 * np.tanh(psi[:, 2])
    q_total, r, common = 0.25, 0.25, 0.1
    z = np.array([[-1.0 if (t // 4) % 2 == 0 else 1.0] * n for t in range(T)])
    y = rng.normal(size=(T, n))
    Q = np.full((n, n), common); np.fill_diagonal(Q, q_total)
    S0 = Q / (1 - rho[:, None] * rho[None, :])
    K = np.zeros((T * n, T * n))
    for s in range(T):
        for t in range(T):
            h = abs(t - s)
            blk = np.diag(rho ** h) @ S0 if t >= s else S0 @ np.diag(rho ** h)
            K[t * n:(t + 1) * n, s * n:(s + 1) * n] = blk
    resid = (y - psi[:, 0] - psi[:, 1] * z).ravel()
    f = joint_filter(y, z, psi, q_total, r, common)
    err_m, err_v = 0.0, 0.0
    for t in range(T):
        idx = np.arange((t + 1) * n)
        Kyy = K[np.ix_(idx, idx)] + r * np.eye(len(idx))
        Kxy = K[t * n:(t + 1) * n][:, idx]
        mean = Kxy @ np.linalg.solve(Kyy, resid[idx])
        cov = K[t * n:(t + 1) * n, t * n:(t + 1) * n] - Kxy @ np.linalg.solve(Kyy, Kxy.T)
        err_m = max(err_m, float(np.abs(mean - f["state_mean"][t]).max()))
        err_v = max(err_v, float(np.abs(np.diag(cov) - f["state_variance"][t]).max()))
    return {"max_mean_err": err_m, "max_var_err": err_v, "status": "PASS" if max(err_m, err_v) < 1e-10 else "FAIL"}


def checks():
    out = {}
    arch = R / "F05/source_full_r2/shared_shock_20260915"
    rows, _truths = generate_dynamics("shared_shock", 20260915)
    src = [r for r in rows if r["role"] == "source"]
    app = [r for r in rows if r["role"] == "application"]
    a_src = [json.loads(line) for line in (arch / "source.jsonl").read_text().splitlines()]
    a_app = [json.loads(line) for line in (arch / "application.jsonl").read_text().splitlines()]
    out["C1a_generator_rows_equal"] = src == a_src and app == a_app
    model = fit_source_dynamics(src)
    amodel = json.loads((arch / "source_model.json").read_text())
    ls_err = float(np.abs(np.asarray(model["log_sigma"]) - amodel["log_sigma"]).max())
    mu_err = float(np.abs(np.asarray(model["prior"]["mu"]) - amodel["prior"]["mu"]).max())
    sg_err = float(np.abs(np.asarray(model["prior"]["Sigma"]) - np.asarray(amodel["prior"]["Sigma"])).max())
    out["C1b_source_refit_max_err"] = {"log_sigma": ls_err, "prior_mu": mu_err, "prior_Sigma": sg_err}
    moment = learn_common_variance(src, model)
    amoment = json.loads((R / "F05/joint_full_r1/shared_shock_20260915/source_common_moment.json").read_text())
    out["C1c_common_moment_err"] = abs(moment["common_variance"] - amoment["common_variance"])
    prev = json.loads((R / "F05/application_full_r1/shared_shock_20260915/observed_predictions.json").read_text())
    people = grouped(app)
    apsi = {p["author_id"]: p["psi"] for p in prev if p["method"] == "personal_MAP" and p["n"] == 48}
    logs = np.asarray(amodel["log_sigma"])
    new_psi = {a: fit_person(e[:48], logs, amodel["prior"])["psi"] for a, e in people.items()}
    out["C1d_personal_psi_max_err"] = float(max(np.abs(np.asarray(new_psi[a]) - apsi[a]).max() for a in people))
    preds = pd.DataFrame(panel_predictions(people, apsi, amoment["total_variance"], float(np.exp(2 * logs[1])), amoment["common_variance"], 48, "personal_MAP"))
    ref = pd.read_csv(R / "F05/joint_full_r1/joint_state_metrics.csv", float_precision="round_trip")
    ref = ref[(ref.world == "shared_shock") & (ref.seed == 20260915) & (ref.method == "personal_MAP") & (ref.parameter_prefix == 48)]
    m = preds.merge(ref, on=["author_id", "t", "arm"], suffixes=("", "_a"), validate="one_to_one")
    out["C1e_panel_rows"] = len(m)
    out["C1e_panel_max_err"] = {k: float(np.nanmax(np.abs(m[k] - m[f"{k}_a"]))) for k in ("state_mean", "state_variance", "forecast1", "forecast4")}
    # C2 zero-shared-variance marginal equivalence on the archived application panel
    ids = sorted(people)
    y = np.array([[people[a][t]["y"] for a in ids] for t in range(64)])
    z = np.array([[people[a][t]["z"] for a in ids] for t in range(64)])
    P = np.array([apsi[a] for a in ids])
    f0 = joint_filter(y, z, P, amoment["total_variance"], float(np.exp(2 * logs[1])), 0.0)
    ls_marg = np.array([0.5 * np.log(amoment["total_variance"]), logs[1]])
    e_m, e_v = 0.0, 0.0
    for i, a in enumerate(ids):
        u = filter_arrays(P[i], y[:, i], z[:, i], np.arange(64), ls_marg)
        e_m = max(e_m, float(np.abs(np.asarray(u["state_mean"]) - f0["state_mean"][:, i]).max()))
        e_v = max(e_v, float(np.abs(np.asarray(u["state_variance"]) - f0["state_variance"][:, i]).max()))
    out["C2_zero_common_marginal_max_err"] = {"mean": e_m, "variance": e_v}
    # C3 no future access: perturb rows after t0
    t0 = 40
    y2 = y.copy(); y2[t0 + 1:] += 5.0
    fj = joint_filter(y, z, P, amoment["total_variance"], float(np.exp(2 * logs[1])), amoment["common_variance"])
    fj2 = joint_filter(y2, z, P, amoment["total_variance"], float(np.exp(2 * logs[1])), amoment["common_variance"])
    out["C3_no_future_equal_through_t0"] = bool(np.array_equal(fj["state_mean"][: t0 + 1], fj2["state_mean"][: t0 + 1])
                                                and np.array_equal(fj["state_variance"][: t0 + 1], fj2["state_variance"][: t0 + 1]))
    out["C3_future_changes_after_t0"] = bool(not np.allclose(fj["state_mean"][t0 + 1:], fj2["state_mean"][t0 + 1:]))
    # C4 independent episodes: keys, author disjointness, distinct common arrays (generation only)
    keys, authors, commons = set(), set(), []
    for w in WORLDS:
        for s in EPISODE_SEEDS:
            rws, trs = generate_dynamics(w, s, counts=COUNTS)
            for role_id in (0, 2):
                keys.add((s, 5, ("core", "near_root", "shared_shock").index(w), role_id, 1))
            ids_ep = {r["author_id"] for r in rws}
            assert not (ids_ep & authors)
            authors |= ids_ep
            if w == "shared_shock":
                for role in ("source", "application"):
                    commons.append(np.asarray(next(tr for tr in trs if tr["role"] == role)["common_innovations"]))
    C = np.corrcoef(np.asarray(commons))
    off = C[~np.eye(len(C), dtype=bool)]
    out["C4_independence"] = {"episode_role_keys": len(keys), "distinct_authors": len(authors), "common_arrays": len(commons),
                              "common_arrays_all_distinct": len({c.tobytes() for c in commons}) == len(commons),
                              "mean_abs_offdiag_corr": float(np.abs(off).mean()), "max_abs_offdiag_corr": float(np.abs(off).max())}
    out["C5_dense_vs_efficient"] = dense_check()
    tol = 1e-10
    ok = (out["C1a_generator_rows_equal"] and max(out["C1b_source_refit_max_err"].values()) <= tol and out["C1c_common_moment_err"] <= tol
          and out["C1d_personal_psi_max_err"] <= 1e-8 and out["C1e_panel_rows"] == 32 * 200 and max(out["C1e_panel_max_err"].values()) <= tol
          and max(out["C2_zero_common_marginal_max_err"].values()) <= tol and out["C3_no_future_equal_through_t0"] and out["C3_future_changes_after_t0"]
          and out["C4_independence"]["common_arrays_all_distinct"] and out["C4_independence"]["episode_role_keys"] == 2 * 32 * 2
          and out["C5_dense_vs_efficient"]["status"] == "PASS")
    out["status"] = "PASS" if ok else "STOP_CHECKS"
    return out


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["checks", "smoke", "full"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.stage == "checks":
        out = checks()
        out["seconds"] = time.time() - t0
        (OUT / "checks.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
        return 0 if out["status"] == "PASS" else 2
    if a.stage == "smoke":
        tm = {}
        for w in WORLDS:
            t1 = time.time()
            run_episode(w, EPISODE_SEEDS[0], None)
            tm[w] = time.time() - t1
        est = {"smoke_seconds": tm, "note": "timing only; outputs discarded, no scientific output inspected",
               "projected_full_minutes_2_workers": sum(tm.values()) * len(EPISODE_SEEDS) / 2 / 60}
        (OUT / "smoke_timing.json").write_text(json.dumps(est, indent=1) + "\n")
        print(json.dumps(est))
        return 0
    if json.loads((OUT / "checks.json").read_text())["status"] != "PASS":
        raise RuntimeError("checks gate not passed")
    from joblib import Parallel, delayed
    jobs = [(w, s) for s in EPISODE_SEEDS for w in WORLDS]
    done = {p.parent.name for p in (OUT / "episodes").glob("*/episode.json") if not p.name.startswith("._")} if (OUT / "episodes").exists() else set()
    todo = [(w, s) for w, s in jobs if f"{w}_{s}" not in done]
    Parallel(n_jobs=a.workers)(delayed(run_episode)(w, s, OUT) for w, s in todo)
    completed = sorted(p.parent.name for p in (OUT / "episodes").glob("*/episode.json") if not p.parent.name.startswith("._"))
    (OUT / "runtime_full.json").write_text(json.dumps({"seconds_this_invocation": time.time() - t0, "neural_calls": 0, "episodes_declared": len(jobs),
                                                       "episodes_completed": len(completed), "resumed_from": len(done), "workers": a.workers,
                                                       "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"completed": len(completed), "seconds": time.time() - t0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
