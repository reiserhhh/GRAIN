#!/usr/bin/env python3
"""Post-hoc fixed-bias diagnostic on the already audited complete semantic panel."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_encoding_effects import analyze
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def run(a):
    """Bind original artifacts/audit, then save all diagnostic values, not corrected scores."""
    m, s = verified_packet(a.packet)
    audit = json.loads((a.audit / "summary.json").read_text(encoding="utf-8"))
    if (
        material_hash({k: v for k, v in audit.items() if k != "identity"})
        != audit["identity"]
        or audit["packet_identity"] != m["identity"]
        or audit["status"] != "COMPLETE"
        or audit["cases"] != 90
        or audit["max_arithmetic_error"] > 1e-10
        or m["preflight_only"]
        or not s["real_neural_collection"]
        or s["forwards"] != 180
        or any(m[k] or s[k] for k in ("labels_read", "bridge_outcomes_read"))
    ):
        raise ValueError("Require complete audited real semantic panel")
    for name, h in audit["outputs_sha256"].items():
        if digest(a.audit / name) != h:
            raise ValueError("Independent semantic audit changed")
    rows = json.loads((a.packet / "observations.json").read_text(encoding="utf-8"))
    tasks, cases, pairs = analyze(rows)
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Output must be under private results")
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for name, table in (("by_task", tasks), ("by_case", cases), ("by_pair", pairs)):
        write_json(out / (name + ".json"), table)
        scalar = [
            {k: v for k, v in r.items() if not isinstance(v, list)} for r in table
        ]
        pd.DataFrame(scalar).to_csv(out / (name + ".csv"), index=False)
        files += [name + ".json", name + ".csv"]
    summary = {
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC",
        "packet_identity": m["identity"],
        "audit_identity": audit["identity"],
        "source_observations_sha256": digest(a.packet / "observations.json"),
        "protocol_sha256": digest(
            ROOT / "docs/GRAIN_VM05_ENCODING_INTERACTION_DIAGNOSTIC_V1.md"
        ),
        "code_sha256": {
            n: digest(ROOT / n)
            for n in (
                "grain_research/vector_mechanisms/direct_encoding_effects.py",
                "scripts/analyze_grain_direct_encoding_effects.py",
            )
        },
        "tasks": len(tasks),
        "behavior_cases": len(cases),
        "pairs": len(pairs),
        "new_forwards": 0,
        "source_labels_read": False,
        "bridge_values_read": False,
        "calibration_fitted": False,
        "predictions_replaced": False,
        "population_p_values_computed": False,
        "scope": "Necessary constant-bias conditions on authored examples; not natural accuracy or native circuit localization",
        "outputs_sha256": {n: digest(out / n) for n in files},
    }
    summary["identity"] = material_hash(summary)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("packet", "audit", "output-dir"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
