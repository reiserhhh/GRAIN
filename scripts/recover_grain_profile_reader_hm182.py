#!/usr/bin/env python3
"""Continue existing HM182 coefficients after a binary-label reporting dtype error."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_profile_reader as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_profile_reader_hm182 as S

PLAN = "docs/GRAIN_PROFILE_READER_RECOVERY_HM182.md"
SCRIPT = "scripts/recover_grain_profile_reader_hm182.py"
TEST = "tests/test_grain_profile_reader_recovery.py"
DEFAULT = S.DEFAULT / "engineering_recovery_r1"


def canonical_binary(y):
    """Change binary storage dtype only;never round new label values."""
    y = np.asarray(y)
    if (
        y.ndim != 2
        or y.shape[1] != 5
        or not np.isfinite(y).all()
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Inherited labels must be exactly binary")
    cast = y.astype(np.uint8)
    if not np.array_equal(cast, y):
        raise ValueError("Binary values changed")
    return cast


def main():
    """Bind original partial artifacts and finish diagnostics without reconstruction."""
    old = S.DEFAULT
    out = DEFAULT
    m = R.check(old)
    spec = m["config"]
    if (old / "compiled_seal.json").exists():
        raise ValueError("Not an incomplete original compiler")
    inputs = (
        "manifest.json",
        "prepared_seal.json",
        "observations.npz",
        "cold_input.jsonl",
        "source_fields.npz",
        "reader_bundle.json",
    )
    z = R.prepared(old)
    prior_dtype = str(z["targets"].dtype)
    original_labels = z["targets"].copy()
    z["targets"] = canonical_binary(original_labels)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": m["code"] | {p: R.digest(ROOT / p) for p in (PLAN, SCRIPT, TEST)},
            "inputs": {
                str((old / p).relative_to(ROOT)): R.digest(old / p) for p in inputs
            },
            "original_scope_complete": False,
            "classification": "ENGINEERING_CONTINUATION_WITH_EXISTING_COEFFICIENTS_NOT_CONSTRUCTION_RETRY",
        },
    )
    R.archive(out / "observations.npz", **z)
    for name in ("cold_input.jsonl", "source_fields.npz", "reader_bundle.json"):
        with (out / name).open("xb") as sink:
            sink.write((old / name).read_bytes())
    R.write(
        out / "failure_record.json",
        {
            "original_phase": "compile",
            "terminal_command": "f2109c exit1",
            "error": "TypeError:Cannot cast float64 array to int64 in reporting np.bincount",
            "source_coefficients_already_saved": True,
            "original_files_changed": False,
            "constructor_rerun": False,
            "label_old_dtype": prior_dtype,
            "label_new_dtype": str(z["targets"].dtype),
            "label_value_difference": float(
                np.max(np.abs(z["targets"].astype(float) - original_labels))
            ),
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_input.jsonl", "failure_record.json"),
    )
    with np.load(out / "source_fields.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    M.validate_bundle(R.read(out / "reader_bundle.json"))
    rows = []
    balance_error = 0.0
    for v, q in np.ndindex(2, 5):
        fields = a["fields"][v, q]
        n = a["counts"][v, q]
        w = a["weights"][v, q]
        key = M.profile_codes(z["targets"], q)
        ess = []
        for g in (0, 1):
            mask = z["targets"][:, q] == g
            ww = np.abs(w[mask])
            if not np.array_equal(np.bincount(key[mask], minlength=16), n[g]):
                raise ValueError("Source group identity changed")
            balance_error = max(
                balance_error,
                float(
                    np.max(
                        np.abs(
                            np.bincount(key[mask], weights=ww, minlength=16)
                            - a["omega"][v, q]
                        )
                    )
                ),
            )
            ess.append(float(ww.sum() ** 2 / (ww @ ww)))
        cross = float(
            np.linalg.norm(fields[0]) ** 2
            - sum(np.linalg.norm(zj) ** 2 for zj in fields[1:])
        )
        rows.append(
            {
                "source_view": v,
                "target": M.TARGETS[q],
                "marginal_matched_cosine": S.cosine(fields[0], fields[1]),
                "marginal_norm": float(np.linalg.norm(fields[0])),
                "matched_norm": float(np.linalg.norm(fields[1])),
                "reference_shift_norm": float(np.linalg.norm(fields[2])),
                "composition_norm": float(np.linalg.norm(fields[3])),
                "signed_three_term_cross": cross,
                "low_ESS": ess[0],
                "high_ESS": ess[1],
                "min_group_count": int(n.min()),
            }
        )
    R.table(out / "field_diagnostics.csv", rows)
    erows = []
    for v, q, bindex in np.ndindex(2, 5, 16):
        erows.append(
            {
                "source_view": v,
                "target": M.TARGETS[q],
                "other_profile": bindex,
                "count0": int(a["counts"][v, q, 0, bindex]),
                "count1": int(a["counts"][v, q, 1, bindex]),
                "omega": float(a["omega"][v, q, bindex]),
                "edge_matched_cosine": S.cosine(
                    a["edges"][v, q, bindex], a["fields"][v, q, 1]
                ),
                "cross_view_edge_cosine": S.cosine(
                    a["edges"][0, q, bindex], a["edges"][1, q, bindex]
                ),
            }
        )
    R.table(out / "all_edge_diagnostics.csv", erows)
    gram = []
    for q, r, j, k in np.ndindex(5, 5, 4, 4):
        gram.append(
            {
                "target_view0": M.TARGETS[q],
                "target_view1": M.TARGETS[r],
                "component_view0": M.COMPONENTS[j],
                "component_view1": M.COMPONENTS[k],
                "cosine": S.cosine(a["fields"][0, q, j], a["fields"][1, r, k]),
            }
        )
    R.table(out / "all_cross_view_fields.csv", gram)
    error = float(np.max(np.abs(a["fields"][:, :, 0] - a["fields"][:, :, 1:].sum(2))))
    if max(error, balance_error) > spec["comparison_tolerance"]:
        raise ValueError("Existing coefficient balance/decomposition failed")
    receipt = {
        "source_authors": 386,
        "conditional_edges": 160,
        "marginal_and_component_fields": 40,
        "profile_balance_error": balance_error,
        "decomposition_error": error,
        "all_source_weights_nonzero": bool(np.all(a["weights"] != 0)),
        "coefficient_byte_hash_unchanged": all(
            R.digest(out / p) == R.digest(old / p)
            for p in ("source_fields.npz", "reader_bundle.json")
        ),
        "constructor_rerun": False,
        "no_fit_accuracy_or_new_neural": True,
        "original_scope_complete": False,
    }
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_fields.npz",
            "reader_bundle.json",
            "field_diagnostics.csv",
            "all_edge_diagnostics.csv",
            "all_cross_view_fields.csv",
            "construction_receipt.json",
        ),
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
