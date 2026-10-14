#!/usr/bin/env python
"""Fresh-process cold scorer for F14/raw_grain_history_r1 (packet 24 A repair).
Uses ONLY the adjunct bundle plus declared observed numeric inputs (raw comment vectors, VM05 units, event-address
timestamps, held role list). A sealing audit hook refuses every other file open (no source rows, fit files or scores)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_EXTRA = []
OPENED = []


def seal(allowed):
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
    files = {os.path.realpath(p) for p in allowed}

    def hook(event, args):
        if event == "open":
            p = args[0]
            if isinstance(p, int) or p is None:
                return
            real = os.path.realpath(os.fsdecode(p))
            ok = real in files or any(real.startswith(r + os.sep) for r in roots) or real.startswith("/dev/") or "__pycache__" in real
            OPENED.append([real, ok])
            if not ok:
                raise PermissionError(f"sealed cold scorer refused {real}")
    sys.addaudithook(hook)


def main():
    bundle, comments_npz, grain_npz, addresses, roles, out = map(Path, sys.argv[1:7])
    seal([bundle, comments_npz, grain_npz, addresses, roles, out])
    b = np.load(bundle, allow_pickle=False)
    names = json.loads(str(b["names_json"]))
    with np.load(comments_npz, allow_pickle=False) as z:
        comments = np.asarray(z["comments"]); ids = z["author_ids"].astype(str)
    with np.load(grain_npz, allow_pickle=False) as z:
        gids = z["author_ids"].astype(str); ix = {u: i for i, u in enumerate(gids)}
        gunits = np.asarray(z["grain_units_raw"][[ix[u] for u in ids]], float)
    ea = pd.read_csv(addresses, usecols=["author_index", "unit", "timestamp_unix_seconds"]).sort_values(["author_index", "timestamp_unix_seconds"])
    held = json.loads(roles.read_text())["held"]
    direction, kmu, ksd = b["direction"], b["keyed_mu"], b["keyed_sd"]
    rows = []
    for a in held:
        un = ea[ea.author_index == a].unit.to_numpy()
        h = np.asarray(comments[a, un], float); hn = h / np.linalg.norm(h, axis=1, keepdims=True)
        Zk = (hn @ direction - kmu) / ksd
        P = (hn - b["pca_mu"]) @ b["pca_W"]
        G = gunits[a, un]
        for t in range(25, 36):
            r = {"author": int(a), "t": t}
            for arm in ("KEYED_STATIC", "RAW_STATIC", "GRAIN_STATIC", "RAW_ORDERED"):
                if arm == "KEYED_STATIC":
                    x = Zk[:t].mean(0)
                elif arm == "RAW_STATIC":
                    x = np.r_[P[:t].mean(0), P[:t].std(0, ddof=0)]
                elif arm == "GRAIN_STATIC":
                    mean, std = G[:t].mean(0), G[:t].std(0, ddof=0)
                    x = np.array([(mean if n.split("::svd_")[0] == "mean" else std)[int(n.split("::svd_")[1])] for n in names])
                else:
                    m_ = P[:t].mean(0)
                    x = np.r_[m_, P[:t].std(0, ddof=0), P[t - 1] - m_, P[t - 4:t].mean(0) - m_]
                xs = (x - b[f"{arm}_sc_mu"]) / b[f"{arm}_sc_sd"]; xs[b[f"{arm}_sc_zero"]] = 0.0
                r[f"mse_{arm}"] = float(((xs @ b[f"{arm}_B"] + b[f"{arm}_b"] - Zk[t]) ** 2).mean())
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False)
    refused = [p for p, ok in OPENED if not ok]
    sys.stdout.write(json.dumps({"rows": len(rows), "opened_non_stdlib": sorted({p for p, ok in OPENED if ok and "site-packages" not in p and "/lib/python" not in p}), "refused": refused}) + "\n")


if __name__ == "__main__":
    main()
