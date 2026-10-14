#!/usr/bin/env python3
"""Cold numerical relation scoring with two inputs, no reader or target table."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_current_transport import probability_map, pullback


def main():
    """Audit every file opened after imports; application outcomes are forbidden."""
    if len(sys.argv) != 5:
        raise ValueError("Expected bundle,input,prediction,audit paths")
    bundle_path, input_path, prediction_path, audit_path = [
        Path(p).resolve() for p in sys.argv[1:]
    ]
    if prediction_path.exists() or audit_path.exists():
        raise FileExistsError("Do not overwrite cold scoring results")
    # Finish NumPy's lazy archive/header imports using only an in-memory fixture.
    fixture = BytesIO()
    np.savez_compressed(fixture, fixture=np.zeros(1))
    fixture.seek(0)
    with np.load(fixture, allow_pickle=False) as archive:
        archive["fixture"]
    allowed = {str(p) for p in (bundle_path, input_path, prediction_path, audit_path)}
    observed = set()

    def file_guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = str(Path(args[0]).resolve())
        if path not in allowed:
            raise PermissionError("Cold instrument attempted undeclared file: " + path)
        observed.add(path)

    sys.addaudithook(file_guard)
    source = np.load(bundle_path, allow_pickle=False)
    application = np.load(input_path, allow_pickle=False)
    arrays, source_errors = {}, []
    for context in range(2):
        for bank in ("factor4", "svd24"):
            x = source[f"points_{context}_{bank}"]
            y = application[f"points_{context}_{bank}"]
            t = probability_map(x, y)
            arrays[f"map_{context}_{bank}"] = t
            arrays[f"kernel_{context}_{bank}"] = pullback(source["kernel"][context], t)
            source_errors.append(
                float(np.max(np.abs(probability_map(x, x) - np.eye(18))))
            )
    np.savez_compressed(prediction_path, **arrays)
    observed.add(str(audit_path))
    audit = {
        "status": "COLD_NUMERICAL_SCORING",
        "file_access_scope": "after imports;explicit data IO,not full OS sandbox",
        "opened_paths": sorted(observed),
        "allowed_paths": sorted(allowed),
        "source_cardinality_error": max(source_errors),
        "application_target_tables_queries_truth_or_text": False,
        "application_encoder_calls": 0,
        "new_predictor_fit": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print("COLD HM98", len(arrays), "arrays;source interpolation", max(source_errors))


if __name__ == "__main__":
    main()
