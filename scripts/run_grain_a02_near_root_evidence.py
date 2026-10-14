#!/usr/bin/env python
"""A02 conditional evidence repair (packet 19 Part C): independent check of F05/invariants_r1 certificates and an
independent per-cell re-derivation of the near-root vs core application summary (registers the previously
unregistered F05/application_audit_r1 state summary). CPU only, no refit, no new data."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

ROOT = Path(__file__).resolve().parents[1]
F05 = ROOT / "results/grain_gptpro_foundation_v3/F05"
OUT = F05 / "near_root_identification_r1"
KEYS = ["world", "n", "phase", "method"]
METRICS = ["a_squared_error", "b_squared_error", "rho_squared_error", "state_squared_error", "state_covered"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def autocov(A, Q, C, R, lags=16):
    P = solve_discrete_lyapunov(A, Q)
    out, Ak = [], np.eye(len(A))
    for k in range(lags):
        out.append((C @ Ak @ P @ C.T).item() + (R if k == 0 else 0.0)); Ak = Ak @ A
    return np.array(out), P


def jac_rank(A, Q, C, R, free, h=1e-6):
    def pack():
        return np.r_[A.ravel(), Q[0, 0], Q[0, 1], Q[1, 1], C.ravel(), R]

    def unpack(p):
        a = p[:4].reshape(2, 2); q = np.array([[p[4], p[5]], [p[5], p[6]]]); c = p[7:9].reshape(1, 2); return a, q, c, p[9]
    idx = {"all": list(range(10)), "fixed_CQ": [0, 1, 2, 3, 9], "fixed_CQR": [0, 1, 2, 3]}[free]
    p0 = pack(); cols = []
    for i in idx:
        e = np.zeros(10); e[i] = h
        cols.append((autocov(*unpack(p0 + e))[0] - autocov(*unpack(p0 - e))[0]) / (2 * h))
    J = np.array(cols).T
    s = np.linalg.svd(J, compute_uv=False)
    return int((s > s.max() * 1e-7).sum()), s.tolist()


def main():
    t0 = time.time()
    if OUT.exists():
        raise FileExistsError("near_root_identification_r1 exists")
    OUT.mkdir(parents=True)
    # ---- 1. certificates (independent implementation)
    cert = json.loads((F05 / "invariants_r1/similarity_invariants.json").read_text())
    A, Q, C, R = np.array(cert["A"]), np.array(cert["Q"]), np.array(cert["C"]), float(cert["R"])
    g0, P = autocov(A, Q, C, R)
    cert_out = {"stationary_P_max_abs_diff_vs_recorded": float(np.abs(P - np.array(cert["P0"])).max()), "transforms": [], "ranks": {}}
    for tr in cert["transforms"]:
        S = np.array(tr["S"]); Si = np.linalg.inv(S)
        A2, Q2, C2 = np.array(tr["A"]), np.array(tr["Q"]), np.array(tr["C"])
        rel = {"A=SAS^-1": float(np.abs(A2 - S @ A @ Si).max()), "Q=SQS^T": float(np.abs(Q2 - S @ Q @ S.T).max()), "C=CS^-1": float(np.abs(C2 - C @ Si).max())}
        g2, _ = autocov(A2, Q2, C2, R)
        cert_out["transforms"].append({"name": tr["name"], "relation_errors": rel, "autocov_max_abs_diff": float(np.abs(g2 - g0).max())})
    for free in ("all", "fixed_CQ", "fixed_CQR"):
        rk, sv = jac_rank(A, Q, C, R, free)
        rec = cert["local_jacobian_ranks"][free]
        rec_rank = rec.get("rank") if isinstance(rec, dict) else rec
        cert_out["ranks"][free] = {"independent_rank": rk, "recorded_rank": rec_rank, "match": rk == rec_rank, "singular_values": sv}
    cert_out["pass"] = bool(cert_out["stationary_P_max_abs_diff_vs_recorded"] < 1e-10 and all(t["autocov_max_abs_diff"] < 1e-10 for t in cert_out["transforms"])
                            and all(v["match"] for v in cert_out["ranks"].values()))
    # ---- 2. near-root re-derivation from application rows
    man = json.loads((F05 / "application_full_r1/output_manifest.json").read_text())
    pin = {f["path"]: f["sha256"] for f in man["files"]}["state_parameter_metrics.csv"]
    if sha(F05 / "application_full_r1/state_parameter_metrics.csv") != pin:
        raise RuntimeError("application rows changed")
    st = pd.read_csv(F05 / "application_full_r1/state_parameter_metrics.csv")
    st["state_covered"] = st.state_covered.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    agg = st.groupby(KEYS)[METRICS].mean().reset_index()
    old = pd.read_csv(F05 / "application_audit_r1/state_summary.csv")
    mg = agg.merge(old, on=KEYS, suffixes=("", "_audit"))
    diffs = {m: float(np.nanmax(np.abs(mg[m] - mg[m + "_audit"]))) for m in METRICS}
    rng = np.random.default_rng(19201)
    rows = []
    for (n, phase, method), g in st.groupby(["n", "phase", "method"]):
        core = g[g.world == "core"]; near = g[g.world == "near_root"]
        if core.empty or near.empty:
            continue
        rec = {"n": n, "phase": phase, "method": method, "core_authors": len(core), "near_root_authors": len(near)}
        for m in METRICS:
            a, b = near[m].dropna().to_numpy(), core[m].dropna().to_numpy()
            bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(1000)]
            rec[f"{m}_core"] = float(b.mean()); rec[f"{m}_near"] = float(a.mean()); rec[f"{m}_near_minus_core"] = float(a.mean() - b.mean())
            rec[f"{m}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        rows.append(rec)
    C_ = pd.DataFrame(rows)
    agg.to_csv(OUT / "state_summary_rederived.csv", index=False); C_.to_csv(OUT / "near_root_vs_core_by_cell.csv", index=False)
    res = {"certificates": cert_out, "rederivation_max_abs_diff_vs_audit": diffs, "rederivation_pass": bool(max(diffs.values()) < 1e-12),
           "cells": len(C_), "near_root_worse_state_error_cells": int((C_.state_squared_error_near_minus_core > 0).sum()),
           "near_root_worse_state_error_ci_excludes0": int((np.array([c[0] for c in C_.state_squared_error_ci]) > 0).sum()),
           "near_root_worse_a_error_ci_excludes0": int((np.array([c[0] for c in C_.a_squared_error_ci]) > 0).sum()),
           "inputs": {"state_parameter_metrics.csv": pin, "application_audit_r1/state_summary.csv": sha(F05 / "application_audit_r1/state_summary.csv"),
                      "invariants_r1/similarity_invariants.json": sha(F05 / "invariants_r1/similarity_invariants.json")},
           "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat(), "neural_calls": 0}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("rederivation_max_abs_diff_vs_audit", "rederivation_pass", "cells", "near_root_worse_state_error_cells",
                                            "near_root_worse_state_error_ci_excludes0", "near_root_worse_a_error_ci_excludes0")}, indent=1, default=float))
    print(json.dumps({"cert_pass": cert_out["pass"], "P_diff": cert_out["stationary_P_max_abs_diff_vs_recorded"], "transforms": cert_out["transforms"],
                      "ranks": {k: (v["independent_rank"], v["recorded_rank"]) for k, v in cert_out["ranks"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
