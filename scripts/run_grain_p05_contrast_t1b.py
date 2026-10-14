#!/usr/bin/env python3
"""P05 C1 amendment A1: T1 with the 24-coordinate GRAIN lexical runtime (zero neural)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.run_grain_p05_contrast_tests import (
    N_PERM,
    OP,
    OUT,
    PROTOCOL,
    QUERIES,
    SEED,
    digest,
    loco,
    r,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA_A1 = "5208397f6f50f3c7928134cc03dded42004c19bcdc0731b7d060b1799cab0bb3"
GEN = ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed after A1")
    texts = {}
    for line in GEN.read_text().splitlines():
        rec = json.loads(line)
        texts[rec["text_id"]] = rec["text"]
    order = json.loads((OP / "pair_order.json").read_text())
    pairs = pd.read_csv(OP / "pairs.csv").set_index("pair").loc[order].reset_index()
    arrays = np.load(OP / "action_arrays.npz", allow_pickle=False)
    src, ds = arrays["source"], arrays["ds"]
    runtime = joblib.load(RUNTIME)
    gh = runtime.transform([texts[t] for t in pairs.high_id])
    gl = runtime.transform([texts[t] for t in pairs.low_id])
    dx24 = np.asarray(gh - gl, float)
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    app = ~src
    ap = pairs[app].reset_index(drop=True)
    cells = (ap.axis + "|" + ap.family + "|" + ap.situation).to_numpy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(ap)) for _ in range(N_PERM)]
    X = dx24[app]
    res = {"protocol_sha256": PROTOCOL_SHA_A1, "dim": int(X.shape[1]),
           "rank_application": int(np.linalg.matrix_rank(X)), "queries": {}}
    for qi, q in enumerate(QUERIES):
        e = eff[eff["query"] == q].set_index(["pair", "kind"])["effect"]
        D = np.array([e[(p, "donor")] for p in ap.pair])
        out_q = {}
        for name, y in (("donor_action", D), ("full_contrast", ds[app, qi])):
            pr = loco(X, y, cells)
            null = [r(loco(X[pi], y, cells), y) for pi in perms]
            rr = r(pr, y)
            out_q[f"T1b_{name}"] = {
                "loco_r": rr, "loco_rmse": float(np.sqrt(np.mean((pr - y) ** 2))),
                "zero_rmse": float(np.sqrt(np.mean(y ** 2))),
                "null_q95": float(np.nanpercentile(null, 95)),
                "p_perm": float((1 + np.sum(np.asarray(null) >= rr)) / (1 + N_PERM)),
                "rival_I_linear": "SUPPORTED" if not rr > np.nanpercentile(null, 95) else "NOT_SUPPORTED",
            }
        res["queries"][q] = out_q
    res["code_sha256"] = digest(Path(__file__))
    res["neural_forwards"] = 0
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (OUT / "c1_t1b_grain24.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
