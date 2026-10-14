#!/usr/bin/env python
"""Post-hoc descriptive analysis of the frozen R15-M1 model (not declared in protocol V1; changes no result).
Per-observation Fisher information at the reference point, and the per-context n needed for SD <= .5 in each design."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r15_speaker_identifiability as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/speaker_identifiability_r1"
res = {}
for dname, cs in m.designs().items():
    for joint in (False, True):
        F1 = m.fim(m.REF, cs, joint)
        r, s = m.rank_info(F1)
        key = f"{dname}|{'joint' if joint else 'text_only'}"
        if r < 8:
            res[key] = {"rank": r, "note": "rank deficient: some parameters have no finite n"}
            continue
        cov1 = np.linalg.inv(F1)
        sd1 = np.sqrt(np.diag(cov1))
        res[key] = {"rank": r, "sd_per_context_obs_at_n1": dict(zip(m.NAMES, sd1.round(3).tolist())),
                    "n_per_context_for_sd_0.5": dict(zip(m.NAMES, np.ceil((sd1 / 0.5) ** 2).astype(int).tolist())),
                    "total_n_for_all_sd_0.5": int(np.ceil((sd1.max() / 0.5) ** 2) * len(cs))}
(OUT / "sample_size_posthoc.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if "C0+all" in k}, indent=1))
