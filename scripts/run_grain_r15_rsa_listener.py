#!/usr/bin/env python3
"""R15: Rational Speech Acts listeners vs the cached NLI reader stack on F16 materials."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R15_RSA_LISTENER_PROTOCOL_V1.md"
PROTOCOL_SHA = "437c3122e75900605806e1eadd201f38a14c13a0648677be0a5218c6108f0dae"
TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F08/rsa_listener_r1"
STATES = ("supports", "rejects", "none")
STATUSES = ("actual", "negated", "quoted", "hypothetical")
COST = {"actual": 0.0, "negated": 1.0, "quoted": 1.0, "hypothetical": 1.0}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def truth(status, pole):
    p = "supports" if pole == 1 else "rejects"
    if status in ("actual", "negated"):
        return np.array([s == p for s in STATES], float)
    if status == "quoted":
        return np.ones(3)
    return np.array([s == "none" for s in STATES], float)


UTTS = [(s, p) for s in STATUSES for p in (0, 1)]


def l0():
    m = np.array([truth(s, p) for s, p in UTTS])
    return m / m.sum(1, keepdims=True)


def l1(alpha):
    L0 = l0()
    with np.errstate(divide="ignore"):
        util = alpha * (np.log(L0) - np.array([COST[s] for s, _ in UTTS])[:, None])
    S1 = np.exp(util - util.max(0, keepdims=True))
    S1 = S1 / S1.sum(0, keepdims=True)
    return S1 / S1.sum(1, keepdims=True)


def surface():
    return np.array([[1.0, 0, 0] if p == 1 else [0, 1.0, 0] for _, p in UTTS])


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(TT / "reader_scores.csv")
    m["P_supports"] = m.pole_readout * m.nli_qualification
    m["P_rejects"] = (1 - m.pole_readout) * m.nli_qualification
    m["P_none"] = 1 - m.nli_qualification
    g = m.groupby(["subject_status", "pole"])[["P_supports", "P_rejects", "P_none"]].mean()
    reader = np.array([g.loc[(s, p)].to_numpy() for s, p in UTTS])
    listeners = {"L0_literal": l0(), "L1_alpha1": l1(1.0), "L1_alpha4": l1(4.0), "SURFACE": surface()}
    res = {}
    for name, L in listeners.items():
        tv = 0.5 * np.abs(L - reader).sum(1)
        res[name] = {"mean_tv": float(tv.mean()),
                     "tv_by_status": {s: float(tv[[i for i, (ss, _) in enumerate(UTTS) if ss == s]].mean()) for s in STATUSES}}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
           "reader_stack_by_cell": {f"{s}|{p}": [round(x, 4) for x in reader[i]] for i, (s, p) in enumerate(UTTS)},
           "listeners_by_cell": {n: {f"{s}|{p}": [round(x, 4) for x in L[i]] for i, (s, p) in enumerate(UTTS)} for n, L in listeners.items()},
           "fit": res, "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "rsa_listener.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"reader_stack_by_cell": out["reader_stack_by_cell"], "fit": res}, indent=1))


if __name__ == "__main__":
    main()
