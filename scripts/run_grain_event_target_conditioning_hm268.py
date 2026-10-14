#!/usr/bin/env python3
"""Finite co-target routing versus target-conditional SAME-event correspondence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_event_behavior as B
from grain_research.construction import math_event_target_conditioning as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_event_anchor_field_hm267 import selected

CONFIG = "configs/grain_event_target_conditioning_hm268.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_target_conditioning_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_TARGET_CONDITIONING_PLAN_HM268.md",
    "grain_research/construction/math_event_target_conditioning.py",
    "grain_research/construction/math_event_anchor_field.py",
    "grain_research/construction/math_profile_reader.py",
    "grain_research/construction/math_target_evidence_routing.py",
    "grain_research/construction/math_event_behavior.py",
    "scripts/run_grain_event_target_conditioning_hm268.py",
    "scripts/score_grain_event_target_conditioning_cold.py",
    "scripts/run_grain_event_anchor_field_hm267.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_event_target_conditioning.py",
)


def prepare(out):
    """Reuse complete sealed observations without regenerating scores or behavior."""
    if out.exists():
        raise FileExistsError("A frozen packet/budget cannot be replaced")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    codes = selected(parent / "source_bundle.npz", ("codes",))["codes"]
    obs = selected(parent / "observations.npz", ("behavior", "points"))
    if len(codes) != 386 or obs["behavior"].shape != (48, 5):
        raise ValueError("Changed complete source/panel scope")
    ref = M.reference(codes)
    names = [
        f"{assignment}/{space}/{family}/{kind}"
        for assignment in ("actual", "fixed_null")
        for space in A.SPACES
        for family in A.FAMILIES
        for kind in ("profile_mass", "probability")
    ]
    scored = selected(parent / "scores.npz", names)
    for name, values in scored.items():
        shape = (48, 2, 32) if name.endswith("profile_mass") else (48, 2, 5)
        if values.shape != shape:
            raise ValueError("Changed SAME-event observation count")
    out.mkdir(parents=True)
    inputs = [
        parent / p
        for p in (
            "scores.npz",
            "observations.npz",
            "source_bundle.npz",
            "computed_seal.json",
            "verification_seal.json",
        )
    ]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **scored, **obs, **ref)
    R.write(out / "source_reference.json", {k: v.tolist() for k, v in ref.items()})
    R.seal(out, "prepared_seal.json", ("observations.npz", "source_reference.json"))
    print("PREPARED HM268:SAME48 uses/386 SOURCE/32 profiles/no new model", flush=True)


def compute(out):
    """Compute every fixed signed account, not a selected positive predictor."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z = R.prepared(out)
        arrays, rows, diagonal, cells = {}, [], {}, []
        reconstruction, readback = 0.0, 0.0
        for assignment in ("actual", "fixed_null"):
            for space in A.SPACES:
                for family in A.FAMILIES:
                    key = f"{assignment}/{space}/{family}"
                    d = M.decompose(z[key + "/profile_mass"], z["omega"])
                    for kind, values in d.items():
                        arrays[key + "/" + kind] = values
                    v = d["values"]
                    old = z[key + "/probability"][:, 0] - z[key + "/probability"][:, 1]
                    reconstruction = max(
                        reconstruction,
                        float(np.max(abs(v[..., 0] - v[..., 1:].sum(-1)))),
                    )
                    readback = max(readback, float(np.max(abs(v[..., 0] - old))))
                    for use, ix in (
                        ("pooled", np.arange(48)),
                        ("window0", np.arange(24)),
                        ("window1", np.arange(24, 48)),
                    ):
                        diag = {}
                        for j, part in enumerate(M.PARTS):
                            contact = B.contacts(v[ix, :, j], z["behavior"][ix])[
                                "action"
                            ]
                            diag[part] = np.diag(contact).tolist()
                            for p in range(5):
                                for q in range(5):
                                    rows.append(
                                        {
                                            "assignment": assignment,
                                            "space": space,
                                            "family": family,
                                            "use": use,
                                            "part": part,
                                            "SOURCE_target": A.TARGETS[p],
                                            "behavior_target": A.TARGETS[q],
                                            "contact": float(contact[p, q]),
                                        }
                                    )
                        diagonal[key + "/" + use] = diag
                    for q in range(5):
                        for b in range(16):
                            for use, ix in (
                                ("window0", np.arange(24)),
                                ("window1", np.arange(24, 48)),
                            ):
                                cell = (
                                    d["matched_cells"][ix, q, b] * z["behavior"][ix, q]
                                )
                                cells.append(
                                    {
                                        "assignment": assignment,
                                        "space": space,
                                        "family": family,
                                        "use": use,
                                        "target": A.TARGETS[q],
                                        "hypothetical_other_profile": b,
                                        "omega": float(z["omega"][q, b]),
                                        "weighted_same_target_contact": float(
                                            cell.mean()
                                        ),
                                    }
                                )
        elapsed = time.perf_counter() - start
        if elapsed > cfg["compute_seconds_cap"]:
            raise TimeoutError("Nonrenewable CPU cap exceeded")
        R.archive(out / "accounts.npz", **arrays)
        R.table(out / "contacts.csv", rows)
        R.table(out / "conditional_profiles.csv", cells)
        summary = {
            "state": "COMPUTED_NOJOB",
            "scientific_CPU_seconds": elapsed,
            "diagonals": diagonal,
            "reconstruction_error": reconstruction,
            "parent_probability_readback_error": readback,
            "all_negative_parts_and_cross_contacts_retained": True,
            "no_fit_or_parameter_search": True,
            "not_prediction_accuracy_or_causal_trait_isolation": True,
            "original_scope_complete": False,
        }
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "computed_seal.json",
            (
                "accounts.npz",
                "contacts.csv",
                "conditional_profiles.csv",
                "summary.json",
            ),
        )
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        R.write(
            out / "failure.json",
            {"error": repr(exc), "restart_or_rescue_allowed": False},
        )
        raise


def verify(out):
    """Check bounded scalar profile accounts and independently callable cold scoring."""
    R.check(out, ("computed_seal.json",))
    z = R.prepared(out)
    saved = selected(out / "accounts.npz", ("actual/E5/comment_mixture/values",))
    p, omega = z["actual/E5/comment_mixture/profile_mass"], z["omega"]
    error = 0.0
    for i in (0, 17, 47):
        for q in range(5):
            pieces = np.zeros(4)
            low = [g for g in range(32) if not ((g >> q) & 1)]
            for b, g in enumerate(low):
                mass = [float(p[i, a, g] + p[i, a, g | (1 << q)]) for a in (0, 1)]
                rate = [float(p[i, a, g | (1 << q)] / mass[a]) for a in (0, 1)]
                dr = rate[0] - rate[1]
                pieces[0] += mass[0] * rate[0] - mass[1] * rate[1]
                pieces[1] += omega[q, b] * dr
                pieces[2] += ((mass[0] + mass[1]) / 2 - omega[q, b]) * dr
                pieces[3] += (mass[0] - mass[1]) * (rate[0] + rate[1]) / 2
            error = max(
                error,
                float(
                    np.max(
                        abs(pieces - saved["actual/E5/comment_mixture/values"][i, q])
                    )
                ),
            )
    bank = selected(
        ROOT / R.read(out / "manifest.json")["config"]["parent"] / "source_bundle.npz",
        ("source_comments", "codes", "anchors", "variance", "directions"),
    )
    record = {
        "frame_key": "HM268_use0",
        "after": {"event_id": "own0", "vector": z["points"][0, 0].tolist()},
        "before": {"event_id": "foreign0", "vector": z["points"][0, 1].tolist()},
    }
    cold = M.cold_record(bank, record)
    for space in A.SPACES:
        for family in A.FAMILIES:
            key = f"actual/{space}/{family}/values"
            expected = selected(out / "accounts.npz", (key,))[key][0]
            error = max(
                error,
                float(
                    np.max(
                        abs(
                            np.asarray(cold["readings"][f"{space}/{family}"]["values"])
                            - expected
                        )
                    )
                ),
            )
    if error > 1e-10:
        raise ValueError("Alternate/cold account failure; do not rescore")
    R.write(
        out / "verification.json",
        {
            "state": "VERIFIED_NOJOB",
            "alternate_max_error": error,
            "fixed_units": [0, 17, 47],
            "audit": "same_agent_not_blind",
            "scientific_rerun": False,
        },
    )
    R.write(out / "cold_input.json", record)
    R.write(out / "cold_readback.json", cold)
    R.seal(
        out,
        "verification_seal.json",
        ("verification.json", "cold_input.json", "cold_readback.json"),
    )
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Run one explicit finite stage with sealed inputs and retained failures."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        R.check(a.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL_SEALS_PASS HM268")
    else:
        globals()[a.stage](a.out)


if __name__ == "__main__":
    main()
