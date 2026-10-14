#!/usr/bin/env python3
"""R16 cross-reader intervention-effect concordance and single-law test (cached, zero calls)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R16_READER_CONCORDANCE_PROTOCOL_V1.md"
PROTOCOL_SHA = "6b95a797fddf0177a97d80aff558bbce04e221949aaba456a797324eead4ba80"
TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F11/reader_concordance_r1"
READERS = {"pole": ("pole_readout", "grain_pole"), "qualification": ("nli_qualification", "grain_qualification")}
SEED, BOOT = 20260926, 2000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pairs(m):
    rows = []
    for t, g in m.groupby("template"):
        g = g.set_index(["pole", "qualification"])
        for q in (0, 1):
            if (0, q) in g.index and (1, q) in g.index:
                a, b = g.loc[(0, q)], g.loc[(1, q)]
                rows.append({"template": t, "intervention": "pole_flip", **{c: b[c] - a[c] for r in READERS.values() for c in r}})
        for p in (0, 1):
            if (p, 0) in g.index and (p, 1) in g.index:
                a, b = g.loc[(p, 0)], g.loc[(p, 1)]
                rows.append({"template": t, "intervention": "qualification_flip", **{c: b[c] - a[c] for r in READERS.values() for c in r}})
    return pd.DataFrame(rows)


def det_stat(df, target):
    nli, gr = READERS[target]
    e = df.groupby("intervention")[[nli, gr]].mean().loc[["pole_flip", "qualification_flip"]].to_numpy()
    return float(np.linalg.det(e) / max((e ** 2).sum(), 1e-300)), e


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(TT / "reader_scores.csv")
    m["template"] = m.material_id.str.rsplit("_", n=2).str[0]
    df = pairs(m)
    df.to_csv(OUT / "intervention_effects.csv", index=False)
    conc = {}
    for inter, g in df.groupby("intervention"):
        for target, (nli, gr) in READERS.items():
            a, b = g[nli].to_numpy(), g[gr].to_numpy()
            key = f"{inter}|{target}"
            if np.std(b) < 1e-12 or np.std(a) < 1e-12:
                conc[key] = {"status": "ABSENT_CHANNEL", "nli_mean": float(a.mean()), "grain_mean": float(b.mean()),
                             "nli_sd": float(np.std(a)), "grain_sd": float(np.std(b)), "n": len(g)}
            else:
                conc[key] = {"status": "COMPUTED", "pearson": float(np.corrcoef(a, b)[0, 1]),
                             "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                             "nli_mean": float(a.mean()), "grain_mean": float(b.mean()), "n": len(g)}
    rng = np.random.default_rng(SEED)
    temps = df.template.unique()
    single = {}
    for target, cols in READERS.items():
        d0, e = det_stat(df, target)
        boots = []
        for _ in range(BOOT):
            pick = rng.choice(temps, len(temps))
            bd = pd.concat([df[df.template == t] for t in pick])
            boots.append(det_stat(bd, target)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        single[target] = {"D": d0, "ci": [float(lo), float(hi)], "mean_effect_matrix": e.tolist(),
                          "rows": ["pole_flip", "qualification_flip"], "cols": list(cols),
                          "single_law_compatible": bool(lo <= 0 <= hi)}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "pairs": len(df),
           "concordance": conc, "single_law": single, "neural_calls": 0,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "reader_concordance.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"concordance": conc, "single_law": single}, indent=1))


if __name__ == "__main__":
    main()
