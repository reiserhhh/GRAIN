#!/usr/bin/env python3
"""Privileged evaluator-side witness (C1): is the W2 style leak removable by an
invariance constraint? Projects the oracle least-squares readout orthogonal to the
TRUE style pattern u_s and reports analytic leak / gain / wording noise. Not an
estimator arm and not a capability claim."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import hidden_evidence_map as H

rows = []
for run, seeds, writers in (("hidden_map_r1", (20261002, 20261003, 20261004), 12),
                            ("hidden_map_r2", (20261005, 20261006, 20261007), 12),
                            ("hidden_map_r3", (20261008, 20261009, 20261010), 36)):
    H.PILOT_WRITERS, H.PILOT_TRAIN_WRITERS = writers, writers * 3 // 4
    for s in seeds:
        view, truth = H.build_world(s, "W2_conditional")
        orc = H.oracle_readouts(view, truth)
        us = truth.maps["u_s"]
        e = truth.source_latent["e"].reshape(-1)
        perp = orc.direction - us * (us @ orc.direction)
        a, g = H.calibrate(view.source_x @ perp, e)
        inv = H.Readout("oracle_perp_style", perp, a, g, "PRIVILEGED")
        for r in (orc, inv):
            m = H.readout_mechanism(r, truth.maps)
            rows.append({"run": run, "seed": s, "readout": r.arm, "cos_us_ue": float(us @ truth.maps["u_e"]),
                         "gain": m["gain_expression"], "R_F_style_analytic": 2 * abs(m["style_response_per_sd"]) / m["gain_expression"],
                         "fingerprint_per_sd": m["fingerprint_per_sd"], "wording_sd": m["wording_sd"]})
out = ROOT / "results/grain_construction_v1/C1/invariance_witness_r1"
out.mkdir(parents=True, exist_ok=False)
(out / "invariance_witness.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
