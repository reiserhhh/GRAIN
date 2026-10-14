#!/usr/bin/env python
"""Packet 32 A: regenerated-provenance supplement for F04/informative_contexts_r1 (NOT new research; r1 untouched).
Replays source observations from the frozen streams, verifies all calibration rows and archived application readings,
and saves source observations, stable IDs and role/seed bindings."""
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
from grain_research import r12_informative_contexts as m  # noqa: E402

R1 = ROOT / "results/grain_gptpro_foundation_v3/F04/informative_contexts_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/informative_contexts_provenance_s1"
RUNNER = ROOT / "scripts/run_grain_r12_informative_contexts.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rt = json.loads((R1 / "runtime.json").read_text())
    if sha(RUNNER) != rt["code_sha256"] or sha(ROOT / "grain_research/r12_informative_contexts.py") != rt["module_sha256"]:
        raise RuntimeError("frozen r1 code changed; replay refused")
    spec = importlib.util.spec_from_file_location("r1run", RUNNER); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    cal = pd.read_csv(R1 / "source_calibration.csv"); arch = np.load(R1 / "observation_archive.npz")
    if OUT.exists():
        raise FileExistsError("provenance supplement exists")
    OUT.mkdir(parents=True)
    src_store, ids, cal_max, app_max = {}, [], 0.0, 0.0
    for seed in r.SEEDS:
        BS, BA = r.draws(seed, 2, r.NSRC), r.draws(seed, 1, r.NAPP)
        for rg, t2 in r.REGIMES.items():
            Ys = r.generate(BS, m.design("D3"), t2); f = m.source_fit(Ys, m.design("D3"))
            row = cal[(cal.seed == seed) & (cal.regime == rg)].iloc[0]
            cal_max = max(cal_max, max(abs(f[k] - row[k]) for k in ("sigma2", "tau2_raw", "tau2_used", "ms_context_residual")))
            src_store[f"source_s{seed}_{rg}_D3layout"] = Ys
            for d in r.DESIGNS:
                app_max = max(app_max, float(np.abs(r.generate(BA, m.design(d), t2).astype(np.float32) - arch[f"s{seed}_{rg}_{d}"]).max()))
        ids += [{"role": "source", "seed": seed, "stream": f"SeedSequence([20260927,31,{seed},2])", "author_index": i, "author_id": f"src-s{seed}-{i:03d}"} for i in range(r.NSRC)]
        ids += [{"role": "application", "seed": seed, "stream": f"SeedSequence([20260927,31,{seed},1])", "author_index": i, "author_id": f"app-s{seed}-{i:03d}"} for i in range(r.NAPP)]
    np.savez_compressed(OUT / "source_observations_regenerated.npz", **src_store)
    pd.DataFrame(ids).to_csv(OUT / "author_role_manifest.csv", index=False)
    res = {"kind": "REGENERATED_PROVENANCE (not newly run research)", "r1_root": "F04/informative_contexts_r1", "calibration_rows_verified": len(cal), "calibration_max_abs": cal_max,
           "application_arrays_verified": len(arch.files), "application_max_abs_vs_float32_archive": app_max, "source_arrays_saved": len(src_store),
           "r1_code_sha256": rt["code_sha256"], "r1_module_sha256": rt["module_sha256"], "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "decision.md").write_text(f"""# F04/informative_contexts_provenance_s1: regenerated provenance supplement (decision)

2026-09-27 JST, packet 32 Part A (supervisor 0955 carry-forward). REGENERATED PROVENANCE, not newly run research;
`F04/informative_contexts_r1` is untouched.

- **Replay.** Source panels were replayed from the frozen r1 streams with the frozen r1 runner and module (hashes
  checked against r1 runtime.json).
- **Checks.**
  - All {len(cal)} source calibration rows reproduce exactly (max abs {cal_max:.3g}).
  - All {len(arch.files)} archived application arrays reproduce within float32 precision (max abs {app_max:.3g}).
- **Saved.**
  - `source_observations_regenerated.npz` (6 source arrays, D3 layout);
  - `author_role_manifest.csv` (stable IDs: 600 source and 1,200 application authors, with role and seed / stream
    bindings).
""")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
