#!/usr/bin/env python3
"""R20: F11 protocol-form conditional-loading world (new protocol; frozen F11 r1 untouched)."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R20_PROTOCOL_FORM_LOADING_PROTOCOL_V1.md"
PROTOCOL_SHA = "b9c5c2d34bed06120c7fa501f97855d7e62bd7c4bc4127c8b8dbc15e9046e7ba"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F11/protocol_form_loading_r1"
CELLS = list(itertools.product(range(3), (-1, 0, 1), (0, 1), (-1, 1)))  # k, r, t, z
SIGMA = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
METHOD, EVENT, ERROR = 0.25, 0.10, 0.25
SEED, NS, NA = 20260926, 2000, 1000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design(kind):
    f, m, c = np.zeros((36, 3)), np.zeros((36, 3)), np.zeros((36, 2))
    for i, (k, r, t, z) in enumerate(CELLS):
        lam = {"M1": 1 + 0.3 * r * z, "M2": 1 + 0.3 * r * (2 * t - 1), "M3": 1.0}[kind]
        f[i, k] = lam
        m[i, r + 1] = 1
        c[i, t] = 1
    return f, m, c


def cov(kind, sigma, method, event, error):
    f, m, c = design(kind)
    return f @ sigma @ f.T + method * m @ m.T + event * c @ c.T + error * np.eye(36)


def simulate(rng, n):
    f, m, c = design("M1")
    traits = rng.multivariate_normal(np.zeros(3), SIGMA, n)
    x = traits @ f.T + np.sqrt(METHOD) * rng.standard_normal((n, 3)) @ m.T         + np.sqrt(EVENT) * rng.standard_normal((n, 2)) @ c.T + np.sqrt(ERROR) * rng.standard_normal((n, 36))
    return traits, x


def fit(kind, x):
    f, m, c = design(kind)
    basis, pairs = [], []
    for i in range(3):
        for j in range(i + 1):
            s = np.zeros((3, 3))
            s[i, j] = s[j, i] = 1
            basis.append(f @ s @ f.T)
            pairs.append((i, j))
    basis += [m @ m.T, c @ c.T, np.eye(36)]
    emp = np.cov(x, rowvar=False)
    coef = np.linalg.lstsq(np.column_stack([b.ravel() for b in basis]), emp.ravel(), rcond=None)[0]
    sigma = np.zeros((3, 3))
    for v, (i, j) in zip(coef[:6], pairs, strict=True):
        sigma[i, j] = sigma[j, i] = v
    w, v = np.linalg.eigh(sigma)
    sigma = (v * np.maximum(w, 1e-8)) @ v.T
    return {"sigma": sigma, "method": max(coef[6], 1e-8), "event": max(coef[7], 1e-8),
            "error": max(coef[8], 1e-8), "mean": x.mean(0), "raw": coef.tolist()}


def blup(kind, model, x, cells):
    f, _, _ = design(kind)
    v = cov(kind, model["sigma"], model["method"], model["event"], model["error"])[np.ix_(cells, cells)]
    cross = model["sigma"] @ f[cells].T
    gain = np.linalg.solve(v, cross.T).T
    scores = (x[:, cells] - model["mean"][cells]) @ gain.T
    var = model["sigma"] - gain @ cross.T
    return scores, var


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _, xs = simulate(rng, NS)
    ta, xa = simulate(rng, NA)
    all_cells = np.arange(36)
    zpos = np.array([i for i, cell in enumerate(CELLS) if cell[3] == 1])
    res = {}
    for kind in ("M1", "M2", "M3"):
        model = fit(kind, xs)
        entry = {"sigma_hat": model["sigma"].round(4).tolist(), "method": float(model["method"]),
                 "event": float(model["event"]), "error": float(model["error"])}
        for label, cells in (("all_cells", all_cells), ("z_plus_only", zpos)):
            sc, var = blup(kind, model, xa, cells)
            sd = np.sqrt(np.maximum(np.diag(var), 1e-12))
            cover = [float(np.mean(np.abs(sc[:, k] - ta[:, k]) <= 1.959964 * sd[k])) for k in range(3)]
            entry[label] = {
                "rmse": [float(np.sqrt(np.mean((sc[:, k] - ta[:, k]) ** 2))) for k in range(3)],
                "corr": [float(np.corrcoef(sc[:, k], ta[:, k])[0, 1]) for k in range(3)],
                "coverage": cover, "calibrated": bool(all(0.93 <= c <= 0.97 for c in cover)),
                "mean_bias": [float(np.mean(sc[:, k] - ta[:, k])) for k in range(3)],
            }
        res[kind] = entry
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
           "source_authors": NS, "application_authors": NA, "models": res, "neural_calls": 0,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "protocol_form_loading.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in res.items():
        print(k, "method/event/error", round(v["method"], 3), round(v["event"], 3), round(v["error"], 3))
        for lab in ("all_cells", "z_plus_only"):
            print("   ", lab, {a: [round(x, 3) for x in b] if isinstance(b, list) else b for a, b in v[lab].items()})


if __name__ == "__main__":
    main()
