#!/usr/bin/env python
"""Label-free diagnosis of the C2 stage-1 replay gate failure (mcd_tau rel diff .0093 > 1e-3).

Frozen questions (written before computation):
  D1 Local determinism: does rerunning the declared fold-0 MCD fit reproduce tau and the
     features bit-for-bit?
  D2 Locus: do the text-SVD and context spaces already match the archive, placing the drift
     at or after k-means?
  D3 k-means seed variability: what spread of tau and cluster sizes do 20 alternative
     MiniBatchKMeans seeds give on identical inputs, and where does the archived tau
     (48.5824) fall?
No labels, no predictions and no test outcomes are used. Arm A, np_mcd author set, fold 0 only.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2", REPO / "scripts/run_grain_c2_mcd_venue_dependence.py")
c2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2)
orig = c2.orig

ARCH_TAU, ARCH_MIN, ARCH_MED = 48.58244512529325, 196, 654.5


def main() -> int:
    t0 = time.time()
    ctx = c2.Context()
    args = ctx.args
    frame, _v, _p = ctx.frame("np_mcd", "agreeableness")
    tr_idx, va_idx, te_idx = orig.split_indices(frame, 0)
    authors = frame["author"].astype(str).to_numpy()
    train_u, target_u = list(authors[tr_idx]), list(authors[va_idx]) + list(authors[te_idx])
    comments = ctx.arm_comments["A"]
    train_c = comments.loc[comments["user_id"].isin(set(train_u))].reset_index(drop=True)
    target_c = comments.loc[comments["user_id"].isin(set(target_u))].reset_index(drop=True)
    seed = args.random_seed
    h_tr, h_ta, text_diag = orig.fit_comment_text_space(train_c, target_c, max_features=args.comment_max_features,
                                                         svd_dims=args.comment_svd_dims, min_df=args.min_df, seed=seed)
    c_tr, c_ta, ctx_diag = orig.fit_context_metadata_space(train_c, target_c, max_dims=min(32, args.comment_svd_dims), seed=seed)
    x = StandardScaler().fit_transform(np.hstack([h_tr, c_tr]))

    def cluster(s: int):
        labels, _t, model = orig.fit_clusters(x, x[:10], args.mcd_k, s)
        q, tau = orig.soft_membership(x, model.cluster_centers_, tau_scale=args.mcd_tau_scale)
        counts = np.bincount(labels, minlength=model.n_clusters)
        return float(tau), int(counts.min()), float(np.median(counts)), q

    declared = seed + 31
    tau1, mn1, md1, q1 = cluster(declared)
    tau2, mn2, md2, q2 = cluster(declared)
    alt = [cluster(declared + 1000 * (i + 1)) for i in range(20)]
    taus = np.array([a[0] for a in alt])
    meds = np.array([a[2] for a in alt])
    mins = np.array([a[1] for a in alt])
    cached = json.loads(next((c2.OUT / "mcd_cache").glob("*.json")).read_text())["diag"]
    out = {
        "D1_rerun_identical": {"tau_equal": tau1 == tau2, "q_bit_equal": bool(np.array_equal(q1, q2)),
                               "tau_matches_stage1_cache": tau1 == cached["mcd_tau"]},
        "D2_locus": {"svd_ev_local": text_diag["comment_svd_explained_variance"], "svd_ev_archived": 0.1023811769098331,
                     "context_ev_local": ctx_diag["context_metadata_explained_variance"], "context_ev_archived": 0.8657977618476849,
                     "context_feature_count_local": ctx_diag["context_metadata_feature_count"], "context_feature_count_archived": 4181},
        "D3_seed_spread": {
            "declared_seed_tau": tau1, "declared_min": mn1, "declared_median": md1,
            "archived_tau": ARCH_TAU, "archived_min": ARCH_MIN, "archived_median": ARCH_MED,
            "alt_tau_min": float(taus.min()), "alt_tau_max": float(taus.max()), "alt_tau_sd": float(taus.std(ddof=1)),
            "alt_tau_rel_range": float((taus.max() - taus.min()) / np.median(taus)),
            "alt_tau_rel_diff_to_declared": [float(abs(t - tau1) / tau1) for t in taus],
            "share_alt_seeds_rel_diff_gt_1e-3": float(np.mean(np.abs(taus - tau1) / tau1 > 1e-3)),
            "archived_tau_within_alt_range": bool(taus.min() <= ARCH_TAU <= taus.max()),
            "alt_median_cluster_range": [float(meds.min()), float(meds.max())],
            "alt_min_cluster_range": [int(mins.min()), int(mins.max())],
            "alt_taus": taus.tolist(),
        },
        "seconds": time.time() - t0,
        "labels_used": False,
    }
    (c2.OUT / "replay_diagnosis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
