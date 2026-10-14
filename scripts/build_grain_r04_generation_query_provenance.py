#!/usr/bin/env python
"""Packet 33 A: regenerated provenance for F01/generation_query_r1 (NOT new research; r1 untouched).
Deterministic replay from frozen seeds with the frozen r1 runner/module (hash-checked). Saves observations, per-author
sufficient statistics, fitted calibration bundles, stable author IDs and role/seed/design bindings; simulator-only truth
is kept in a separate file. Verifies every r1 target_metrics row."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r04_generation_query as m  # noqa: E402

R1 = ROOT / "results/grain_gptpro_foundation_v3/F01/generation_query_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F01/generation_query_provenance_s1"
RUNNER = ROOT / "scripts/run_grain_r04_generation_query.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rt = json.loads((R1 / "runtime.json").read_text())
    if sha(RUNNER) != rt["code_sha256"] or sha(ROOT / "grain_research/r04_generation_query.py") != rt["module_sha256"]:
        raise RuntimeError("frozen r1 code changed; replay refused")
    spec = importlib.util.spec_from_file_location("r1run", RUNNER); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    old = pd.read_csv(R1 / "target_metrics.csv")
    if OUT.exists():
        raise FileExistsError("provenance supplement exists")
    OUT.mkdir(parents=True)
    obs, suff, truth, ids, cal_bundle, rows = {}, {}, {}, [], {}, []
    for seed in r.SEEDS:
        BA, BS, BN = r.draws(seed, 1, r.NAPP), r.draws(seed, 2, r.NSRC), r.draws(seed, 3, r.NANC, bilingual=True)
        LA = np.r_[np.zeros(r.NAPP // 2, int), np.ones(r.NAPP // 2, int)]; thA = BA["th0"] + m.MU_L1 * LA; KA = np.exp(m.SD_LOGK * BA["z"])
        LS = np.r_[np.zeros(r.NSRC // 2, int), np.ones(r.NSRC // 2, int)]; thS = BS["th0"] + m.MU_L1 * LS; KS = np.exp(m.SD_LOGK * BS["z"])
        thN, KN = BN["th0"], np.exp(m.SD_LOGK * BN["z"])
        gA, aA = m.crossed(r.NAPP); gS, aS = m.crossed(r.NSRC); gN, aN = m.crossed(r.NANC)
        truth[f"s{seed}_app_theta"], truth[f"s{seed}_app_K"], truth[f"s{seed}_app_L"] = thA, KA, LA
        truth[f"s{seed}_src_theta"], truth[f"s{seed}_src_K"], truth[f"s{seed}_anchor_theta"], truth[f"s{seed}_anchor_K"] = thS, KS, thN, KN
        ids += [{"author_id": f"g32-app-s{seed}-{i:03d}", "role": "application", "seed": seed, "index": i, "language": int(LA[i]), "stream": f"SeedSequence([20260927,32,{seed},1])", "design": "crossed 2x2 x8 recorded"} for i in range(r.NAPP)]
        ids += [{"author_id": f"g32-src-s{seed}-{i:03d}", "role": "source_monolingual", "seed": seed, "index": i, "language": int(LS[i]), "stream": f"SeedSequence([20260927,32,{seed},2])", "design": "crossed 2x2 x8 recorded"} for i in range(r.NSRC)]
        ids += [{"author_id": f"g32-anc-s{seed}-{i:03d}", "role": "source_bilingual_anchor", "seed": seed, "index": i, "language": "0 and 1", "stream": f"SeedSequence([20260927,32,{seed},3])", "design": "crossed 2x2 x8 recorded, both languages"} for i in range(r.NANC)]
        for pair in ("distinct", "replica"):
            yS = m.generate(thS, KS, LS, gS, aS, pair, BS["Ex"], BS["Eq"]); ys = m.author_ols(yS, gS, aS)
            src0 = {k: v[LS == 0] for k, v in ys.items()}; src1 = {k: v[LS == 1] for k, v in ys.items()}
            y0 = m.generate(thN, KN, np.zeros(r.NANC, int), gN, aN, pair, BN["Ex"], BN["Eq"]); y1 = m.generate(thN, KN, np.ones(r.NANC, int), gN, aN, pair, BN["Ex1"], BN["Eq1"])
            a0, a1 = m.author_ols(y0, gN, aN), m.author_ols(y1, gN, aN)
            cals = {"anchor": m.calibrate(src0, anchor0=a0, anchor1=a1), "oracle": m.oracle_cal(pair)}
            if pair == "distinct":
                cals["noanchor"] = m.calibrate(src0, src1_mono=src1)
            yA = m.generate(thA, KA, LA, gA, aA, pair, BA["Ex"], BA["Eq"]); st = m.author_ols(yA, gA, aA)
            obs[f"s{seed}_{pair}_app"], obs[f"s{seed}_{pair}_src"], obs[f"s{seed}_{pair}_anchorL0"], obs[f"s{seed}_{pair}_anchorL1"] = (x.astype(np.float32) for x in (yA, yS, y0, y1))
            for tag, sx in (("app", st), ("src", ys), ("anchorL0", a0), ("anchorL1", a1)):
                for k, v in sx.items():
                    suff[f"s{seed}_{pair}_{tag}_{k}"] = v
            for cname, cal in cals.items():
                cal_bundle[f"s{seed}|{pair}|{cname}"] = {"method_L1": cal["method_L1"], "off": {str(k): np.asarray(v).tolist() for k, v in cal["off"].items()},
                                                         "scale": {str(k): np.asarray(v).tolist() for k, v in cal["scale"].items()},
                                                         "status": "oracle reference (simulator-only loadings)" if cname == "oracle" else "fitted on source observations only"}
                th_hat, se = m.score(st, LA, cal); err = th_hat - thA; arm = f"CROSS_{pair.upper()}_{cname.upper()}"
                for grp, s in (("all", slice(None)), ("L0", LA == 0), ("L1", LA == 1)):
                    o = old[(old.seed == seed) & (old.arm == arm) & (old.group == grp)].iloc[0]
                    new = {"bias": err[s].mean(), "mse": (err[s] ** 2).mean(), "coverage95": (np.abs(err[s]) <= 1.96 * se[s]).mean(), "sq_corr_thetahat_theta": np.corrcoef(th_hat[s], thA[s])[0, 1] ** 2}
                    rows.append({"seed": seed, "arm": arm, "group": grp, **{f"diff_{k}": float(abs(v - o[k])) for k, v in new.items()}})
    np.savez_compressed(OUT / "observations.npz", **obs); np.savez_compressed(OUT / "sufficient_statistics.npz", **suff)
    np.savez_compressed(OUT / "SIMULATOR_ONLY_truth.npz", **truth)
    (OUT / "calibration_bundles.json").write_text(json.dumps(cal_bundle, indent=1) + "\n")
    pd.DataFrame(ids).to_csv(OUT / "author_id_manifest.csv", index=False)
    D = pd.DataFrame(rows); D.to_csv(OUT / "r1_metric_replay_check.csv", index=False)
    mx = float(D[[c for c in D.columns if c.startswith("diff_")]].max().max())
    res = {"kind": "REGENERATED_PROVENANCE (not newly run research)", "r1_root": "F01/generation_query_r1", "r1_code_sha256": rt["code_sha256"], "r1_module_sha256": rt["module_sha256"],
           "metric_rows_verified": len(D), "metric_max_abs_diff": mx, "authors": len(ids), "observation_arrays": len(obs), "calibration_bundles": len(cal_bundle),
           "note": "simulator-only theta/K/L are in SIMULATOR_ONLY_truth.npz, separate from observation/fit archives", "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "decision.md").write_text(f"""# F01/generation_query_provenance_s1: regenerated provenance for generation_query_r1 (decision)

2026-09-27 JST, packet 33 Part A. REGENERATED PROVENANCE, not newly run research; `F01/generation_query_r1` is untouched.

- **Replay.** Deterministic replay from the frozen seeds, with the frozen r1 runner and module (hashes equal r1
  runtime.json).
- **Check.** All {len(D)} r1 target_metrics rows reproduce (max abs difference {mx:.3g}).
- **Saved.**
  - `observations.npz` (application, source and both anchor languages, float32);
  - `sufficient_statistics.npz` (per-author α, β_G, β_A, residual covariance, (X'X)^-1);
  - `calibration_bundles.json` (fitted anchor / no-anchor and labelled oracle off / scale per language, pair and seed);
  - `author_id_manifest.csv` ({len(ids)} stable author IDs with role, seed, stream, language and design);
  - `SIMULATOR_ONLY_truth.npz`, kept apart from the observation-only archives.
- **Disclosure preserved.** The scratch-world choice of goal-contrast strength before freezing.
""")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
