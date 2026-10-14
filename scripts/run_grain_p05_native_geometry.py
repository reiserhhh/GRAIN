#!/usr/bin/env python3
"""Seminar-06 item 1: cached native-geometry diagnostic (no fits, no calls)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_P05_NATIVE_GEOMETRY_DIAGNOSTIC_V1.md"
PROTOCOL_SHA = "82a206a36ec7744bc8000451524e6cfdb725312eeac9d029faa140f61241153e"
NA = ROOT / "results/grain_gptpro_foundation_v3/P05/native_action_r1"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/native_geometry_r1"
QUERIES, OFF = ("extraversion", "neuroticism"), (0, 7552, 15232)
SEED, BOOT = 20260926, 2000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def basis(M):
    u, s, _ = np.linalg.svd(M.T, full_matrices=False)
    keep = s > s.max() * 1e-10
    return u[:, keep]


def share(Q, Y):
    Y = np.atleast_2d(Y)
    return ((Y @ Q) ** 2).sum(1) / np.maximum((Y ** 2).sum(1), 1e-300)


def summ(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    return {"n": len(x), "mean": float(x.mean()), "ci": [float(v) for v in np.percentile(x[idx].mean(1), [2.5, 97.5])],
            "q10_q50_q90": [float(v) for v in np.percentile(x, [10, 50, 90])]}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.load(OP / "action_arrays.npz", allow_pickle=False)
    src = arr["source"]
    si = np.flatnonzero(src)
    prep = np.load(NA / "private_prepared.npz")
    rng = np.random.default_rng(SEED)
    res = {"protocol_sha256": PROTOCOL_SHA, "queries": {}}
    for qi, q in enumerate(QUERIES):
        sl = slice(OFF[qi], OFF[qi + 1])
        D = OFF[qi + 1] - OFF[qi]
        J = np.stack([np.load(NA / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
        Q = basis(J)
        loo = np.array([share(basis(np.delete(J, k, 0)), J[k])[0] for k in range(len(J))])
        dh = arr["dh"].astype(float)
        out = {"D": D, "span_rank": int(Q.shape[1]), "random_reference": Q.shape[1] / D,
               "Q1_loo_gradient_coverage": summ(loo, rng),
               "Q2_source_donor_energy_share": summ(share(Q, dh[src, sl]), rng),
               "Q2_application_donor_energy_share": summ(share(Q, dh[~src, sl]), rng),
               "donor_norm_median": {"source": float(np.median(np.linalg.norm(dh[src, sl], axis=1))),
                                     "application": float(np.median(np.linalg.norm(dh[~src, sl], axis=1)))},
               "gradient_norm_median": float(np.median(np.linalg.norm(J, axis=1))),
               "Q3_compiled_application_share": {}}
        for cell in ("E4", "J4", "E24", "J24"):
            a = prep[cell][~src, sl]
            out["Q3_compiled_application_share"][cell] = {**summ(share(Q, a), rng), "median_norm": float(np.median(np.linalg.norm(a, axis=1))),
                                                          "note": "in span by construction" if cell.startswith("J") else "empirical"}
        res["queries"][q] = out
    res["input_support_zmax_source_app"] = json.loads((NA / "prepare.json").read_text())["zmax_source_app"]
    res.update({"code_sha256": digest(Path(__file__)), "neural_calls": 0, "fits": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    (OUT / "native_geometry.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
