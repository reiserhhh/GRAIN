#!/usr/bin/env python3
"""GRAIN-C4-02 runner: occasion-state brief on existing sequences (prequential, permutation control)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import declared_target as DT
from grain_research.construction import occasion_state as OS

PROTOCOL = ROOT / "docs/GRAIN_C4_02_STATE_PROTOCOL_V1.json"
OUT = ROOT / "results/grain_construction_v1/C4/state_r1"
ABUNDLE = ROOT / "results/grain_construction_v1/C4/transport_r1/anonymous_bundle.json"
KBUNDLE = ROOT / "results/grain_construction_v1/C3/declared_target_r1/source_fitted_bundle.json"
CR = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
ADDR = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
ROLES = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json"
MAT = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material"
H, NH, B = list(range(25)), 25, 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def residuals(fits, ev, c, j):
    """History-window BLUP class means (frozen) and residuals for all 36 events of one author."""
    s = DT.score_author(fits, ev[:NH], c[:NH])
    mp, mm = s["m_plus"]["mean"][j], s["m_minus"]["mean"][j]
    vp, vm = s["m_plus"]["var"][j], s["m_minus"]["var"][j]
    mean = np.where(c > 0, mp, mm)
    lvar = np.where(c > 0, vp, vm)
    return ev[:, j] - mean, lvar


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))
    roles = json.loads(ROLES.read_text())
    src, held = np.array(roles["source_train"]), np.array(roles["held"])
    addr = pd.read_csv(ADDR, usecols=["author_index", "unit", "condition", "timestamp_unix_seconds"]).sort_values(["author_index", "unit"])
    n_auth = int(addr.author_index.max()) + 1
    venue = addr.condition.to_numpy().reshape(n_auth, 36)
    days = addr.timestamp_unix_seconds.to_numpy().reshape(n_auth, 36) / 86400.0
    gaps = np.diff(days, axis=1)
    ab = json.loads(ABUNDLE.read_text())
    AB.verify_bundle(ab)
    kb = json.loads(KBUNDLE.read_text())
    DT.verify_bundle(kb)
    nrm = AB.normalize(np.load(MAT / "private_arrays.npz")["comments"].astype(np.float64))
    raw = nrm @ np.asarray(ab["directions"])
    anon = np.stack([AB.apply_mode(raw[a], venue[a], ab["venue_offsets"], ab["pooled_offset"], "within_venue")[0] for a in range(n_auth)])
    keyed = np.load(CR / "pandora/private_units.npz")["scores"].reshape(n_auth, 36, 30)
    cmat = np.where(np.isin(venue, ab["context_rule"]["top"]), 1.0, -1.0)
    families = {"anon_within_venue": (anon, ab["fits"]["within_venue"], 2.94), "keyed": (keyed, kb["fits"], 3.14)}
    rng_b = np.random.default_rng(20261032)
    rng_p = np.random.default_rng(20261033)
    perms = {a: rng_p.permutation(36) for a in held}
    idx = rng_b.integers(0, len(held), (B, len(held)))
    params_rows, held_rows = [], []
    unsupported = int(np.sum(gaps[held][:, NH - 1:] > 365))
    for fam, (ev, fits, z) in families.items():
        k = ev.shape[2]
        for j in range(k):
            R = np.stack([residuals(fits, ev[a], cmat[a], j)[0][:NH] for a in src])
            p = OS.fit(R, gaps[src][:, :NH - 1])
            params_rows.append({"family": fam, "coordinate": j, **p})
            se_l, se_s, se_lp, se_sp, cov, n_ev = (np.zeros(len(held)) for _ in range(6))
            for i, a in enumerate(held):
                for perm, store in ((None, (se_l, se_s)), (perms[a], (se_lp, se_sp))):
                    e_a = ev[a] if perm is None else ev[a][perm]
                    c_a = cmat[a] if perm is None else cmat[a][perm]
                    r, lvar = residuals(fits, e_a, c_a, j)
                    pm, pv = OS.prequential(p, r, gaps[a], NH)
                    fut = r[NH:]
                    store[0][i] += np.sum(fut ** 2)
                    store[1][i] += np.sum((fut - pm) ** 2)
                    if perm is None:
                        cov[i] += np.sum(np.abs(fut - pm) <= 1.959964 * np.sqrt(pv + lvar[NH:]))
                        n_ev[i] += len(fut)
            rel = lambda ix, l_, s_: (l_[ix].sum() - s_[ix].sum()) / l_[ix].sum()  # noqa: E731
            full = np.arange(len(held))
            rb = np.array([rel(ix, se_l, se_s) for ix in idx])
            db = np.array([rel(ix, se_l, se_s) - rel(ix, se_lp, se_sp) for ix in idx])
            val, se = rel(full, se_l, se_s), float(np.std(rb, ddof=1))
            diff = rel(full, se_l, se_s) - rel(full, se_lp, se_sp)
            ntot = int(n_ev.sum())
            covr = float(cov.sum() / ntot)
            lb = float(binom.ppf(.025, ntot, .95) / ntot)
            supported = bool(val - z * se > 0 and np.percentile(db, 2.5) > 0 and covr >= lb)
            held_rows.append({"family": fam, "coordinate": j, "rel_improvement_vs_level": val, "se": se, "fw_lower": val - z * se,
                              "perm_rel_improvement": rel(full, se_lp, se_sp), "orig_minus_perm": diff,
                              "orig_minus_perm_lo": float(np.percentile(db, 2.5)), "orig_minus_perm_hi": float(np.percentile(db, 97.5)),
                              "coverage": covr, "coverage_lower_bound": lb, "events": ntot, "state_supported": supported,
                              "half_life_days": p["half_life_days"], "state_share": p["state_share"]})
    write_csv(OUT / "state_params.csv", params_rows)
    write_csv(OUT / "state_held.csv", held_rows)
    hr = pd.DataFrame(held_rows)
    acc = {"supported": {f: hr[(hr.family == f) & hr.state_supported].coordinate.tolist() for f in families},
           "positive_state_construction": bool(hr.state_supported.any()),
           "held_future_events_per_coordinate": int(hr.events.iloc[0]), "unsupported_horizon_events_gap_gt_365d": unsupported,
           "reading": "positive if any coordinate supported; otherwise a null answer that does not demonstrate state measurement"}
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=float))
    run = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version, "numpy": np.__version__,
           "platform": platform.platform(), "code_sha256": {p: sha(ROOT / p) for p in ("grain_research/construction/occasion_state.py", "scripts/run_grain_c4_state.py")},
           "inputs": {"anonymous_bundle": sha(ABUNDLE), "keyed_bundle": sha(KBUNDLE)}}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(acc, indent=1))
    print(hr.round(4).to_string(index=False))
    print("cpu", run["total_cpu_s"])


if __name__ == "__main__":
    main()
