#!/usr/bin/env python3
"""Apply the R32 admission rule, triangulation model and dynamic-parameter check."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.r32_admission import (
    Candidate,
    admit,
    check_dynamic_parameter,
    triangulate,
    two_indicator_counterexample,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/admission_r1"
V3 = "results/grain_gptpro_foundation_v3"

CANDIDATES = [
    Candidate("frozen GRAIN 24-d default dimensions", "latent_cause", False,
              {}, {}, {"structural_reliability": f"{V3}/F13/structural_multiverse_r1 (frozen lexical/SVD default misses the declared .80 conditional-basis subspace rule: .755/.689; not a verdict on every GRAIN representation)"}),
    Candidate("MCD condition-choice composition", "dispositional", True,
              {"incremental_cross_situational_prediction": True},
              {"topic": False, "demographic": False, "language_proficiency": False, "social_desirability": False},
              {"increment": "NP+MCD +.0236 [.013,.034]; choice composition is venue/topic by construction (C2)"}),
    Candidate("first_person usage -> Neuroticism (lockbox H2)", "dispositional", True,
              {"novel_falsifiable_prediction": False},
              {}, {"association": "r .111 confirmatory; an association of a known lexical marker with an existing construct, not a novel construct"}),
    Candidate("reader pole cross-talk (NLI qualification)", "formative_index", True,
              {"unique_experimental_signature": True},
              {}, {"signature": f"{V3}/F11/reader_concordance_r1 (template-reliable ICC .41)",
                   "note": "a reader property, not a psychological construct"}, domain="instrument"),
]

DYNAMIC = {
    "state persistence rho (F05/F13)": {"operation": "provocation then neutral turns", "rival": "topic persistence",
                                        "prediction": "decay with half-life log(.5)/log(rho)", "boundary": "within session",
                                        "criterion": ""},
    "X->s feedback xi (R04)": {"operation": "vary own-output feedback exposure", "rival": "assessor-action carry-over",
                               "prediction": "scale inflation 1/(1-rho-xi) without rank change", "boundary": "stable regime rho+xi<1",
                               "criterion": ""},
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = [admit(c) | {"evidence": c.evidence} for c in CANDIDATES]
    rng = np.random.default_rng(20260926)
    lam = np.array([0.8, 0.6, 0.4])
    theta = rng.standard_normal(20000)
    y = theta[:, None] * lam + rng.standard_normal((20000, 3)) * np.sqrt(1 - lam ** 2)
    tri = triangulate(np.cov(y, rowvar=False))
    a, b = two_indicator_counterexample(0.8, 0.5, 1.5)
    dyn = {k: check_dynamic_parameter(v) for k, v in DYNAMIC.items()}
    out = {
        "admission": decisions,
        "triangulation_three_indicators": {"true_loadings": lam.tolist(), "estimated": tri["loadings"].round(4).tolist(),
                                           "max_error": float(np.abs(tri["loadings"] - lam).max()),
                                           "reliability": tri["reliability"].round(4).tolist()},
        "two_indicator_nonidentification": {"covariance_equal": bool(np.allclose(a, b)),
                                            "loadings_A": [0.8, 0.5], "loadings_B": [1.2, 0.3333]},
        "dynamic_parameter_missing_fields": dyn,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (OUT / "admission.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
