#!/usr/bin/env python
"""R08 conditional information bottleneck witness (docs/GRAIN_R08_CONDITIONAL_IB_PROTOCOL_V1.md). Exact enumeration."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_text_behaviour_quotient as r1  # noqa: E402  (frozen r1 runner, imported only)

PROTOCOL = ROOT / "docs/GRAIN_R08_CONDITIONAL_IB_PROTOCOL_V1.md"
PROTOCOL_SHA = "e932797bd58e5a83dff90373e93e0f0af96b22b51f34e62fb05ec6d11a5aca01"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/conditional_ib_r1"
HSYM = list(itertools.product((0, 1), repeat=3))          # (c_{t-2}, c_{t-1}, B_{t-1})
FSYM = list(itertools.product((0, 1), repeat=2))          # (c_t, B_t)


def arrays(name):
    w = r1.world_W(name)
    S = len(w["states"])
    K = np.array([[[[float(v) for v in o] for o in row] for row in w["K"][a]] for a in (0, 1)])   # a,s,o,s'
    obs = w["obs"]                                       # (c, f, b)
    T0 = K[0].sum(axis=1)
    pi = np.full(S, 1 / S) @ np.linalg.matrix_power(T0, 500)
    return w, K, obs, pi


def joint(name):
    """P(H, C, F) and P(S_t, C, F) exactly."""
    w, K, obs, pi = arrays(name)
    S = len(pi)
    PH = np.zeros((8, 2, 4))
    PS = np.zeros((S, 2, 4))
    for s2 in range(S):
        for oi2, (c2, f2, b2) in enumerate(obs):
            for s1 in range(S):
                p2 = pi[s2] * K[0, s2, oi2, s1]
                if p2 == 0:
                    continue
                for oi1, (c1, f1, b1) in enumerate(obs):
                    for s0 in range(S):
                        p1 = p2 * K[0, s1, oi1, s0]
                        if p1 == 0:
                            continue
                        h = HSYM.index((c2, c1, b1))
                        for a in (0, 1):
                            for oi0, (c0, f0, b0) in enumerate(obs):
                                pf = 0.5 * p1 * K[a, s0, oi0].sum()
                                PH[h, a, FSYM.index((c0, b0))] += pf
                                PS[s0, a, FSYM.index((c0, b0))] += pf
    return PH, PS


def H_(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_cond_entropy(PZCF):
    """I(Z;F|C) = H(F|C) - H(F|Z,C)."""
    out = 0.0
    for a in range(PZCF.shape[1]):
        pc = PZCF[:, a, :].sum()
        cond = PZCF[:, a, :] / pc
        out += pc * (H_(cond.sum(0)) - sum(cond[z].sum() * H_(cond[z] / cond[z].sum()) for z in range(cond.shape[0]) if cond[z].sum() > 0))
    return out


def mi_cond_kl(PZCF):
    """Independent: sum p(z,c,f) log p(z,f|c) / (p(z|c) p(f|c))."""
    out = 0.0
    for a in range(PZCF.shape[1]):
        pc = PZCF[:, a, :].sum()
        J = PZCF[:, a, :] / pc
        pz, pf = J.sum(1), J.sum(0)
        for z in range(J.shape[0]):
            for f in range(J.shape[1]):
                if J[z, f] > 0:
                    out += pc * J[z, f] * np.log2(J[z, f] / (pz[z] * pf[f]))
    return out


def set_partitions(n):
    def rec(i, rgs, m):
        if i == n:
            yield tuple(rgs)
            return
        for k in range(m + 1):
            rgs.append(k)
            yield from rec(i + 1, rgs, max(m, k + 1))
            rgs.pop()
    yield from rec(0, [], 0)


def code_joint(PH, labels):
    nb = max(labels) + 1
    PZ = np.zeros((nb, PH.shape[1], PH.shape[2]))
    for h, z in enumerate(labels):
        PZ[z] += PH[h]
    return PZ


def analyse(name):
    PH, PS = joint(name)
    ph = PH.sum(axis=(1, 2))
    total = float(PH.sum())
    I_full = mi_cond_entropy(PH)
    I_oracle = mi_cond_entropy(PS)
    rows = []
    for labels in set_partitions(8):
        PZ = code_joint(PH, labels)
        pz = PZ.sum(axis=(1, 2))
        rows.append((labels, H_(pz), mi_cond_entropy(PZ), max(labels) + 1))
    frontier, best_by_k = [], {}
    for lab, ih, ifc, k in sorted(rows, key=lambda r: (r[1], -r[2])):
        if not frontier or ifc > frontier[-1][2] + 1e-12:
            frontier.append((lab, ih, ifc, k))
        if k not in best_by_k or ifc > best_by_k[k][2] + 1e-15:
            best_by_k[k] = (lab, ih, ifc, k)
    summaries = {"mean_content": [c2 + c1 for (c2, c1, b1) in HSYM], "last_content": [c1 for (c2, c1, b1) in HSYM],
                 "last_behaviour": [b1 for (c2, c1, b1) in HSYM], "last_construct": [2 * c1 + b1 for (c2, c1, b1) in HSYM]}
    summ = {}
    for sname, raw in summaries.items():
        uniq = sorted(set(raw))
        labels = [uniq.index(v) for v in raw]
        PZ = code_joint(PH, labels)
        k = len(uniq)
        collisions = []
        for i, j in itertools.combinations(range(8), 2):
            if labels[i] == labels[j] and ph[i] > 0 and ph[j] > 0:
                tvs = [float(.5 * np.abs(PH[i, a] / PH[i, a].sum() - PH[j, a] / PH[j, a].sum()).sum()) for a in (0, 1)]
                if max(tvs) > 1e-12:
                    collisions.append({"h1": HSYM[i], "h2": HSYM[j], "TV_a0": tvs[0], "TV_a1": tvs[1]})
        summ[sname] = {"blocks": k, "I_ZH": H_(PZ.sum(axis=(1, 2))), "I_ZF_given_C": mi_cond_entropy(PZ),
                       "best_same_k_I_ZF_given_C": best_by_k[k][2], "gap_to_best_same_k": best_by_k[k][2] - mi_cond_entropy(PZ),
                       "future_collisions": sorted(collisions, key=lambda x: -max(x["TV_a0"], x["TV_a1"]))[:6], "n_collisions": len(collisions)}
    # checks
    PZ_rand = code_joint(PH, rows[1234][0])
    checks = {"joint_mass": total, "entropy_vs_kl_full": abs(mi_cond_entropy(PH) - mi_cond_kl(PH)), "entropy_vs_kl_sample_code": abs(mi_cond_entropy(PZ_rand) - mi_cond_kl(PZ_rand)),
              "data_processing": bool(max(r[2] for r in rows) <= I_full + 1e-12 and I_full <= I_oracle + 1e-12),
              "constant_code_zero": abs(rows[0][2]) < 1e-15 if max(rows[0][0]) == 0 else None, "partitions": len(rows)}
    mono = True
    lab_to = {r[0]: r[2] for r in rows}
    for lab, ih, ifc, k in rows:                       # merging two blocks never increases I(Z;F|C)
        if k >= 2:
            merged = tuple(0 if x == 1 else (x - 1 if x > 1 else x) for x in lab)
            rgs, seen = [], {}
            for x in merged:
                seen.setdefault(x, len(seen)); rgs.append(seen[x])
            if lab_to[tuple(rgs)] > ifc + 1e-12:
                mono = False
                break
    checks["refinement_monotone"] = mono
    return {"world": name, "H_of_H": H_(ph), "p_history": dict(zip(map(str, HSYM), ph.round(6).tolist())),
            "I_full_code": I_full, "I_oracle_latent_state": I_oracle, "frontier": [{"labels": list(l), "blocks": k, "I_ZH": ih, "I_ZF_given_C": ifc} for l, ih, ifc, k in frontier],
            "best_by_blocks": {str(k): {"labels": list(v[0]), "I_ZH": v[1], "I_ZF_given_C": v[2]} for k, v in sorted(best_by_k.items())},
            "fixed_summaries": summ, "checks": checks}


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    if OUT.exists():
        raise FileExistsError("conditional_ib_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    t0 = time.time()
    res = {w: analyse(w) for w in ("W1", "W6")}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"seconds": time.time() - t0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
