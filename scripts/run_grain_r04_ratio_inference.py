#!/usr/bin/env python
"""Packet 33 runner (docs/GRAIN_R04_RATIO_INFERENCE_PROTOCOL_V1.md): Fieller set-valued inference on the frozen packet-32
observations (via the regenerated provenance archive). 0 neural calls; no new world/seed/grid."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import ratio_sets as R  # noqa: E402
from grain_research import r04_generation_query as m  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R04_RATIO_INFERENCE_PROTOCOL_V1.md"
PROTOCOL_SHA = "44e9bb7ce113873d0539ad20cbcd5fddeead753d24e9c310bf48e5c227301fa6"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
PROV = V3 / "F01/generation_query_provenance_s1"
OUT = V3 / "F01/ratio_inference_r1"
DF, SEEDS, NAPP = 29, (0, 1, 2), 400
T05, TJ = stats.t.ppf(0.975, DF), stats.t.ppf(1 - 0.025 / 2, DF)
CALS = {"distinct": ("oracle", "anchor", "noanchor"), "replica": ("oracle", "anchor")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def indep_set(a, b, Vaa, Vab, Vbb, t):
    """Independent audit calculation (numpy roots + sign sampling)."""
    f = lambda r: (a - r * b) ** 2 - t * t * (Vaa - 2 * r * Vab + r * r * Vbb)
    coef = [b * b - t * t * Vbb, -2 * (a * b - t * t * Vab), a * a - t * t * Vaa]
    rts = sorted(float(x.real) for x in np.roots(coef) if abs(x.imag) < 1e-9 * max(1, abs(x.real)))
    pts = [-math.inf] + rts + [math.inf]; out = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else (hi - 1e6 * max(1, abs(hi)) if math.isfinite(hi) else (lo + 1e6 * max(1, abs(lo)) if math.isfinite(lo) else 0.0))
        if f(mid) <= 0:
            out.append((lo, hi))
    return R._norm(out)


def same(U, W, tol=1e-7):
    return len(U) == len(W) and all((x1 == x2 or abs(x1 - x2) <= tol * max(1, abs(x1))) for (lo1, hi1), (lo2, hi2) in zip(U, W) for x1, x2 in ((lo1, lo2), (hi1, hi2)))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("ratio_inference_r1 exists")
    OUT.mkdir(parents=True)
    S = np.load(PROV / "sufficient_statistics.npz"); TR = np.load(PROV / "SIMULATOR_ONLY_truth.npz"); CB = json.loads((PROV / "calibration_bundles.json").read_text())
    prov = json.loads((PROV / "results.json").read_text())
    rows, disagree_set, disagree_mem, analytic = [], 0, 0, []
    for seed in SEEDS:
        th, K, L = TR[f"s{seed}_app_theta"], TR[f"s{seed}_app_K"], TR[f"s{seed}_app_L"].astype(int)
        for pair in ("distinct", "replica"):
            al, bg, Sg, Xi = (S[f"s{seed}_{pair}_app_{k}"] for k in ("alpha", "bG", "S", "XtXi"))
            st = {"alpha": al, "bG": bg, "bA": S[f"s{seed}_{pair}_app_bA"], "S": Sg, "XtXi": Xi}
            # analytic unbounded probability (simulator quantities; per-query alpha = .05)
            for j, q in enumerate(m.PAIRS[pair]):
                r_q = m.SENS[q][0]; sres = math.sqrt(m.SIG_X ** 2 * (r_q @ r_q) + m.SIG_Q ** 2)
                for lang in (0, 1):
                    sel = L == lang; bG_true = r_q @ m.M[lang] @ m.W_G
                    delta = K[sel] * bG_true / (sres * np.sqrt(Xi[sel, 1, 1]))
                    p_unb = stats.nct.cdf(T05, DF, delta) - stats.nct.cdf(-T05, DF, delta)
                    obs = [R.classify(R.fieller(al[u, j], bg[u, j], Sg[u, j, j] * Xi[u, 0, 0], Sg[u, j, j] * Xi[u, 0, 1], Sg[u, j, j] * Xi[u, 1, 1], T05)) in ("two_rays", "whole_line", "ray") for u in np.flatnonzero(sel)]
                    analytic.append({"seed": seed, "pair": pair, "query": q, "language": lang, "expected_unbounded_fraction": float(p_unb.mean()), "observed_unbounded_fraction": float(np.mean(obs)), "n": int(sel.sum())})
            for cname in CALS[pair]:
                cb = CB[f"s{seed}|{pair}|{cname}"]; off = {int(k): np.array(v) for k, v in cb["off"].items()}; sc = {int(k): np.array(v) for k, v in cb["scale"].items()}
                cal = {"off": off, "scale": sc, "method_L1": cb["method_L1"]}
                th_hat, se = m.score(st, L, cal)
                for u in range(NAPP):
                    lang = L[u]; sets = {}
                    for level, tcrit in (("05", T05), ("J", TJ)):
                        for j in range(2):
                            Vaa, Vab, Vbb = Sg[u, j, j] * Xi[u, 0, 0], Sg[u, j, j] * Xi[u, 0, 1], Sg[u, j, j] * Xi[u, 1, 1]
                            Ur = R.fieller(al[u, j], bg[u, j], Vaa, Vab, Vbb, tcrit)
                            if not same(Ur, indep_set(al[u, j], bg[u, j], Vaa, Vab, Vbb, tcrit)):
                                disagree_set += 1
                            r_true = sc[lang][j] * th[u] + off[lang][j]
                            if R.contains(Ur, r_true) != R.in_fieller(r_true, al[u, j], bg[u, j], Vaa, Vab, Vbb, tcrit):
                                disagree_mem += 1
                            sets[(level, j)] = R.map_affine(Ur, float(sc[lang][j]), float(off[lang][j]))
                    EJ = R.intersect(sets[("J", 0)][0], sets[("J", 1)][0])
                    for ep, (U, flag) in (("E1_q1", sets[("05", 0)]), ("E2_q2", sets[("05", 1)]), ("EJ_joint_bonferroni", (EJ, sets[("J", 0)][1] + "|" + sets[("J", 1)][1]))):
                        rows.append({"seed": seed, "pair": pair, "calibration": cname, "calibration_status": "fixed-known reference (oracle)" if cname == "oracle" else "plug-in fitted (conditional diagnostic)",
                                     "author_id": f"g32-app-s{seed}-{u:03d}", "language": int(lang), "endpoint": ep, "set_type": R.classify(U), "set_json": json.dumps(R.to_json(U)),
                                     "map_flag": flag, "covers_truth_audit": bool(R.contains(U, th[u])), "width": R.width(U),
                                     "old_delta_covers": bool(abs(th_hat[u] - th[u]) <= 1.96 * se[u])})
    P = pd.DataFrame(rows); P.to_csv(OUT / "per_author_sets.csv", index=False)
    A = pd.DataFrame(analytic); A.to_csv(OUT / "analytic_unbounded_check.csv", index=False)
    met = []
    for keys, g in P.groupby(["pair", "calibration", "endpoint", "language"]):
        k = int(g.covers_truth_audit.sum()); n = len(g); fin = g.width[np.isfinite(g.width)]
        met.append({"pair": keys[0], "calibration": keys[1], "endpoint": keys[2], "language": keys[3], "n_pooled_3_seeds": n, "coverage": k / n, "coverage_wilson95": wilson(k, n),
                    **{f"frac_{t}": float((g.set_type == t).mean()) for t in ("bounded", "point", "ray", "two_rays", "whole_line", "empty", "bounded_union", "mixed_union")},
                    "median_finite_width": float(fin.median()) if len(fin) else None, "old_delta_coverage": float(g.old_delta_covers.mean()),
                    "per_seed_coverage": [float(x) for x in g.groupby("seed").covers_truth_audit.mean()], "negative_c_rows": int(g.map_flag.str.contains("c_negative").sum())})
    MT = pd.DataFrame(met); MT.to_csv(OUT / "set_domain_metrics.csv", index=False)
    checks = {"independent_set_agreement_failures": disagree_set, "membership_disagreements": disagree_mem,
              "all_authors_retained": bool(P.groupby(["seed", "pair", "calibration", "endpoint"]).size().eq(NAPP).all()),
              "provenance_metric_max_abs": prov["metric_max_abs_diff"], "t_crit_per_query": T05, "t_crit_joint_component": TJ, "sets_computed": len(P)}
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "checks": checks, "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat(),
           "labels": {"oracle": "fixed-known calibration reference (simulator loadings)", "anchor/noanchor": "plug-in fitted calibration; coverage conditional, not unconditional",
                      "coverage": "audit only; theta truth used after set construction", "unbounded": "finite-sample weak information, not structural non-identification",
                      "empty": "incompatibility at the chosen level, not proof of misspecification"}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/ratio_sets.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
