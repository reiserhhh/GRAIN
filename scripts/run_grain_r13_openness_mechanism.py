#!/usr/bin/env python3
"""R13: response-freedom information optimum derived from a declared mechanism (exact Gaussian)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R13_OPENNESS_MECHANISM_PROTOCOL_V1.md"
PROTOCOL_SHA = "9191cd8fac3d6af896163b67bd1eff4e0c13d970aff679389b399dfa6d616a9c"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F07/openness_mechanism_r1"
R = 8
SETS = {"A": {"e0": 0.3, "e1": 0.7, "s1": 0.8, "p1": 1.5}, "B": {"e0": 0.3, "e1": 0.7, "s1": 0.8, "p1": 0.5},
        "C": {"e0": 0.3, "e1": 0.7, "s1": 1.5, "p1": 1.5}}
CONSTRUCTS = {"C1_selection_sensitive": (1.0, 0.5), "C2_selection_free": (1.0, 0.0)}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def components(o, e0, e1, s1, p1):
    e, s, p = e0 + e1 * o, s1 * o, p1 * o ** 2
    return {"target_theta": e ** 2, "selection": s ** 2, "pragmatic": p ** 2 / R, "error": 1.0 / R}, e, s


def info(o, pars, construct):
    comp, e, s = components(o, **pars)
    var_x = sum(comp.values())
    a, b = CONSTRUCTS[construct]
    cov = a * e + b * s
    var_t = a ** 2 + b ** 2
    rho2 = cov ** 2 / (var_x * var_t)
    return float(-0.5 * np.log(1 - min(rho2, 1 - 1e-15))), comp


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 1001)
    res = {}
    for name, pars in SETS.items():
        for cons in CONSTRUCTS:
            curve = [info(o, pars, cons)[0] for o in grid]
            k = int(np.argmax(curve))
            levels = {f"{o:.2f}": {"info_nats": info(o, pars, cons)[0],
                                   "components": {kk: round(v, 4) for kk, v in info(o, pars, cons)[1].items()}}
                      for o in (0, 0.25, 0.5, 0.75, 1.0)}
            shape = "interior_peak" if 0 < k < len(grid) - 1 else ("increasing" if k == len(grid) - 1 else "decreasing")
            res[f"{name}|{cons}"] = {"argmax_openness": float(grid[k]), "max_info": float(curve[k]),
                                     "shape": shape, "levels": levels}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "responses_per_person": R,
           "results": res, "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "openness_mechanism.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in res.items():
        print(k, v["shape"], "argmax", v["argmax_openness"], "max", round(v["max_info"], 4),
              {o: round(x["info_nats"], 3) for o, x in v["levels"].items()})


if __name__ == "__main__":
    main()
