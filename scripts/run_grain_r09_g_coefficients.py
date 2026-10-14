#!/usr/bin/env python
"""R09 design-specific G coefficients from saved reliability-spine REML moments (docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1.md).
Closed-form; no fits, no data, 0 neural calls. Half = module rows_half(0, 0) (1-based occasion 1, wording 1)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import reliability_spine as rs  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1.md"
PROTOCOL_SHA = "77e37e56f8681b77b3908720a902fd1281cde8a8675ad88bebcd731c944739ac"
SPINE = ROOT / "results/grain_gptpro_foundation_v3/F04/reliability_spine_r1/results.json"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/generalizability_coefficients_r1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")


def composites(rows):
    """Weight vectors over 96 rows restricted to `rows`; each reproduces its universe-score part exactly."""
    out = {}
    mask = np.zeros(rs.NROW, bool); mask[rows] = True
    for k in range(2):
        rk = mask & (rs.R_K == k)
        n = rk.sum()
        c1, c2 = rs.OBS[rs.R_CI, 0], rs.OBS[rs.R_CI, 1]
        out[f"a{k + 1}"] = np.where(rk, 1.0 / n, 0.0)
        out[f"B{k + 1}1"] = np.where(rk, c1 / n, 0.0)
        out[f"B{k + 1}2"] = np.where(rk, c2 / n, 0.0)
        for ci in range(4):
            rc = rk & (rs.R_CI == ci)
            out[f"m{k + 1}_c{ci}"] = np.where(rc, 1.0 / rc.sum(), 0.0)
    return out


def target_loading(name):
    """Loading of the universe score on theta = (a1, a2, B11, B12, B21, B22)."""
    L = np.zeros(6)
    k = int(name[1]) - 1
    if name.startswith("a"):
        L[k] = 1
    elif name.startswith("B"):
        L[2 + 2 * k + int(name[2]) - 1] = 1
    else:
        c = rs.OBS[int(name.split("_c")[1])]
        L[k] = 1; L[2 + 2 * k: 4 + 2 * k] = c
    return L


def parts(w, Sig, sd2, rho, sw2, se2):
    vt = float(w @ rs.ZT @ Sig @ rs.ZT.T @ w)
    vd = float(sd2 * np.sum((rs.ZD.T @ w) ** 2))
    vs = float(np.mean([w @ rs.state_cov(rho, sw2 / (1 - rho ** 2), rs.pair_of(u)) @ w for u in range(4)]))
    ve = float(se2 * w @ w)
    return vt, vd, vs, ve


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("generalizability_coefficients_r1 exists; preserve earlier runs")
    spine = json.loads(SPINE.read_text())
    designs = {"full48": np.arange(rs.NROW), "half8_o1_f1": rs.rows_half(0, 0)}
    checks = {}
    for dn, rows in designs.items():
        for name, w in composites(rows).items():
            # universe-score reproduction: w' ZT == target loading; fixed facet contrasts cancel: w' X[:,6:] == 0
            facet = w @ rs.X[:, 6:]
            checks[f"{dn}|{name}"] = {"loading_err": float(np.max(np.abs(w @ rs.ZT - target_loading(name)))),
                                      "facet_cancel_err" if dn == "full48" else "facet_constant_offset_loading_AMENDMENT_1": (float(np.max(np.abs(facet))) if dn == "full48" else facet.tolist())}
    bad = [k for k, v in checks.items() if v["loading_err"] > 1e-12 or v.get("facet_cancel_err", 0.0) > 1e-12]
    if bad:
        raise RuntimeError(f"composite does not reproduce its universe score: {bad}")
    rows_out = []
    params = {f"{k}|plugin": (np.array(v["Sig_hat"]), v["sd2_hat"], v["rho_hat"], v["sw2_hat"], v["se2_hat"]) for k, v in spine["fits"].items()}
    for world, cfg in rs.WORLDS.items():
        params[f"{world}|oracle"] = (rs.SIG, cfg["sd2"], rs.RHO, rs.SW2, rs.SE2)
    for key, (Sig, sd2, rho, sw2, se2) in params.items():
        world, seed = key.split("|")[0], key.split("|")[1]
        for dn, rows in designs.items():
            for name, w in composites(rows).items():
                vt, vd, vs, ve = parts(w, Sig, sd2, rho, sw2, se2)
                tot = vt + vd + vs + ve
                rows_out.append({"world": world, "source": "oracle_generating" if seed == "oracle" else "REML_plugin", "seed": seed, "design": dn, "target": name,
                                 "var_universe_theta": vt, "var_crossed_d": vd, "var_state": vs, "var_resid": ve, "total": tot,
                                 "G_rel_random_wording": vt / tot, "G_rel_fixed_wording": (vt + vd) / tot, "Phi_fixed_wording": (vt + vd) / tot, "Phi_random_wording": "NOT_ESTIMATED"})
    df = pd.DataFrame(rows_out)
    OUT.mkdir(parents=True)
    df.to_csv(OUT / "g_coefficients.csv", index=False)
    pl = df[df.source == "REML_plugin"]
    summ = {}
    for (world, dn, tgt), g in pl.groupby(["world", "design", "target"]):
        o = df[(df.source == "oracle_generating") & (df.world == world) & (df.design == dn) & (df.target == tgt)].iloc[0]
        summ[f"{world}|{dn}|{tgt}"] = {"G_random_seed_range": [float(g.G_rel_random_wording.min()), float(g.G_rel_random_wording.max())],
                                       "G_fixed_seed_range": [float(g.G_rel_fixed_wording.min()), float(g.G_rel_fixed_wording.max())],
                                       "G_random_oracle": float(o.G_rel_random_wording), "G_fixed_oracle": float(o.G_rel_fixed_wording)}
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "fits": 0, "composite_checks_max_err": float(max(max(v["loading_err"], v.get("facet_cancel_err", 0.0)) for v in checks.values())), "amendment": "docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1_AMENDMENT_1.md sha " + sha(ROOT / "docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1_AMENDMENT_1.md"),
           "rows": int(len(df)), "summary": summ,
           "labels": {"Phi_random_wording": "NOT_ESTIMATED: only 3 fixed wording effects estimated; no population variance",
                      "source_uncertainty": "NOT propagated (plug-in REML moments; 3-seed spread shown)", "not_EB": "G of unweighted declared composites, not EB posterior reliability",
                      "half": "rows_half(0,0) = occasion 1, wording 1 (8 events per coordinate: 4 conditions x 2 repetitions)"},
           "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "composite_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print("done", round(res["seconds"], 2), "rows", len(df), "check_err", res["composite_checks_max_err"])


if __name__ == "__main__":
    main()
