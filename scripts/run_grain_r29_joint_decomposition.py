#!/usr/bin/env python3
"""R29 joint multi-facet variance decomposition (declared crossed simulation, exact ANOVA EMS)."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R29_JOINT_DECOMPOSITION_PROTOCOL_V1.md"
PROTOCOL_SHA = "58c705b17a54782e360a78c96ca5c351b4345167c9d7c1ff52eec809a8fa5d1f"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/joint_decomposition_r1"
FACTORS = ("P", "C", "L", "T", "M", "O", "S")
LEVELS = {"P": 200, "C": 4, "L": 2, "T": 3, "M": 2, "O": 2, "S": 2}
REPS, SEED, BOOT = 2, 20260926, 500
TRUE = {("P",): 1.0, ("C",): 0.5, ("P", "C"): 0.3, ("P", "S"): 0.4, ("O",): 0.2, ("P", "O"): 0.2,
        ("L",): 0.1, ("P", "L"): 0.15, ("T",): 0.3, ("P", "T"): 0.1, ("M",): 0.2, ("P", "M"): 0.1,
        ("C", "M"): 0.05}
RESID = 1.0


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def subsets():
    return [s for k in range(1, len(FACTORS) + 1) for s in itertools.combinations(FACTORS, k)]


def simulate(rng):
    shape = tuple(LEVELS[f] for f in FACTORS)
    y = np.zeros((*shape, REPS))
    for eff, var in TRUE.items():
        sh = [LEVELS[f] if f in eff else 1 for f in FACTORS]
        y += np.sqrt(var) * rng.standard_normal(sh)[..., None]
    return y + np.sqrt(RESID) * rng.standard_normal(y.shape)


def anova(y, factors=FACTORS):
    """Exact balanced random-effects ANOVA over all subsets of the given axes."""
    n = dict(zip(factors, y.shape[:-1], strict=True))
    r = y.shape[-1]
    cell = y.mean(-1)
    ms_e = float(((y - cell[..., None]) ** 2).sum() / (cell.size * (r - 1)))
    axes = {f: i for i, f in enumerate(factors)}
    marg = {}
    for k in range(len(factors) + 1):
        for g in itertools.combinations(factors, k):
            other = tuple(axes[f] for f in factors if f not in g)
            marg[g] = cell.mean(axis=other, keepdims=True) if other else cell
    subs = [s for k in range(1, len(factors) + 1) for s in itertools.combinations(factors, k)]
    ms = {}
    for e in subs:
        eff = np.zeros_like(marg[e])
        for k in range(len(e) + 1):
            for g in itertools.combinations(e, k):
                eff = eff + (-1) ** (len(e) - k) * marg[g]
        rest = int(np.prod([n[f] for f in factors if f not in e]))
        df = int(np.prod([n[f] - 1 for f in e]))
        ms[e] = float(r * rest * (eff ** 2).sum() / df)
    sig = {}
    for e in sorted(subs, key=len, reverse=True):
        acc = ms[e] - ms_e
        for f in subs:
            if f != e and set(e) < set(f):
                acc -= r * np.prod([n[k] for k in factors if k not in f]) * sig[f]
        sig[e] = acc / (r * np.prod([n[k] for k in factors if k not in e]))
    sig[("resid",)] = ms_e
    return sig


def collapsed(y):
    """Separate-world style: average L,T,M,O; keep P x C x S with pseudo-replicates."""
    ax = {f: i for i, f in enumerate(FACTORS)}
    z = y.mean(axis=(ax["L"], ax["T"], ax["M"], ax["O"]))
    return anova(z, ("P", "C", "S"))


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    y = simulate(rng)
    est = anova(y)
    col = collapsed(y)
    brng = np.random.default_rng(SEED + 1)
    boots, cboots = [], []
    for _ in range(BOOT):
        idx = brng.integers(0, LEVELS["P"], LEVELS["P"])
        yb = y[idx]
        boots.append(anova(yb))
        cboots.append(collapsed(yb))
    rows = []
    for e in [*subsets(), ("resid",)]:
        vals = np.array([b[e] for b in boots])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        truth = RESID if e == ("resid",) else TRUE.get(e, 0.0)
        rows.append({"effect": "x".join(e), "true": truth, "estimate": est[e], "ci_lo": lo, "ci_hi": hi,
                     "covers": bool(lo <= truth <= hi), "declared_nonzero": truth > 0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "joint_components.csv", index=False)
    crow = []
    for e in [*(s for k in (1, 2, 3) for s in itertools.combinations(("P", "C", "S"), k)), ("resid",)]:
        vals = np.array([b[e] for b in cboots])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        crow.append({"effect": "x".join(e), "collapsed_estimate": col[e], "ci_lo": lo, "ci_hi": hi,
                     "joint_true": RESID if e == ("resid",) else TRUE.get(e, 0.0)})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(OUT / "collapsed_components.csv", index=False)
    nz = df[df.declared_nonzero]
    zero = df[~df.declared_nonzero]
    summary = {
        "protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
        "observations": int(y.size), "effects": len(df),
        "nonzero_covered": f"{int(nz.covers.sum())}/{len(nz)}",
        "zero_covered": f"{int(zero.covers.sum())}/{len(zero)}",
        "max_abs_true_zero_estimate": float(zero.estimate.abs().max()),
        "max_abs_bias_nonzero": float((nz.estimate - nz.true).abs().max()),
        "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(nz.round(4).to_string())
    print(cdf.round(4).to_string())


if __name__ == "__main__":
    main()
