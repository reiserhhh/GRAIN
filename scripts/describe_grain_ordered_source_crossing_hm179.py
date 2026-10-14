#!/usr/bin/env python3
"""Post-outcome source-function/native-view crossing,without fitting either."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_relation as C
from grain_research.construction import math_ordered_valuation as O
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_ordered_valuation_hm179 as H


def main():
    """Retain all four corners and independently verify actual event contacts."""
    manifest = R.check(H.DEFAULT, ("analysis_seal.json", "audit_seal.json"))
    parent = ROOT / manifest["config"]["relation_parent"]
    R.check(parent, ("scored_seal.json", "analysis_seal.json"))
    z = R.prepared(parent)
    fields = np.asarray(R.read(parent / "source_bundle.json")["pair_fields"], float)
    with np.load(parent / "mechanism_accounts.npz", allow_pickle=False) as a:
        diagonal = a["actions"]
    with np.load(H.DEFAULT / "order_accounts.npz", allow_pickle=False) as a:
        native = a["native_measure"]
    out = H.DEFAULT / "source_crossing_followup_r1"
    out.mkdir(exist_ok=False)
    plan = "docs/GRAIN_ORDERED_VALUATION_SOURCE_CROSSING_HM179.md"
    R.write(
        out / "manifest.json",
        {
            "classification": "POST_OUTCOME_NUMERIC_SOURCE_NATIVE_VIEW_CROSSING",
            "code_hash": R.digest(Path(__file__)),
            "plan_hash": R.digest(ROOT / plan),
            "primary_analysis_seal": R.digest(H.DEFAULT / "analysis_seal.json"),
            "primary_audit_seal": R.digest(H.DEFAULT / "audit_seal.json"),
            "parent_analysis_seal": R.digest(parent / "analysis_seal.json"),
            "new_neural_calls": 0,
            "source_or_scale_fit": False,
        },
    )
    actions = np.empty((2, 2, 24, 2, 3, 5))
    direct = np.zeros_like(actions)
    pairs = 0
    for arm, source_view, u, native_view in np.ndindex(2, 2, 24, 2):
        cells = z["panel_cells"][arm, u, native_view]
        selected = set(z["selected"][u, native_view])
        control = set(z["control"][u, native_view])
        rest = set(range(12)) - selected - control
        actions[arm, source_view, u, native_view] = C.contact_action(
            fields[source_view], cells, selected, control
        )
        for residual in sorted(rest):
            for sign, group in ((1, selected), (-1, control)):
                for other in sorted(group):
                    if (residual < 6) == (other < 6):
                        continue
                    i, j = sorted((residual, other))
                    direct[arm, source_view, u, native_view] += (
                        sign * fields[source_view, :, :, cells[i], cells[j]] / 36
                    )
                    pairs += 1
    finite = np.isfinite(actions) & np.isfinite(direct)
    if not np.array_equal(np.isfinite(actions), np.isfinite(direct)):
        raise ValueError("Crossed support changed in independent enumeration")
    direct_error = float(np.max(np.abs(actions[finite] - direct[finite])))
    diagonal_error = max(
        float(np.nanmax(np.abs(actions[:, v, :, v] - diagonal[:, :, v])))
        for v in range(2)
    )
    if max(direct_error, diagonal_error) > O.RESOLUTION:
        raise ValueError("Crossed SAME-event accounts disagree")
    components = np.concatenate(
        [
            actions,
            (actions[..., 1, :] - actions[..., 0, :])[..., None, :],
            (actions[..., 2, :] - actions[..., 1, :])[..., None, :],
        ],
        axis=-2,
    )
    rows = []
    for arm, source_view, native_view, component, target in np.ndindex(2, 2, 2, 5, 5):
        x = components[arm, source_view, :, native_view, component, target]
        base = {
            "arm": manifest["config"]["arms"][arm],
            "source_view": source_view,
            "native_view": native_view,
            "component": manifest["config"]["components"][component],
            "target": manifest["config"]["targets"][target],
        }
        if not np.isfinite(x).all():
            cert, tails = {"status": "CROSSED_SUPPORT_MISSING"}, None
        else:
            b, cert = O.source_alignment(x, native[:, native_view])
            tails = None if b is None else O.tail_actions(b)
        rows.append(H.row_for(base, cert, tails))
    primary = [
        r
        for r in rows
        if (r["arm"], r["component"], r["target"])
        == ("cue_mask", "specific_change", "N")
    ]
    states = {(r["source_view"], r["native_view"]): r["status"] for r in primary}
    by_native = all(
        states[s, 0] == "ORDINALLY_NONNEGATIVE"
        and states[s, 1] == "ORDINALLY_NONPOSITIVE"
        for s in range(2)
    )
    summary = {
        "primary_crossing": primary,
        "primary_sign_follows_native_view_in_all_four_corners": by_native,
        "source_change_alone_explanation_sufficient_on_this_support": False
        if by_native
        else None,
        "mechanism_attribution": "NATIVE_VIEW_SIGNATURE_SURVIVES_FIXED_SOURCE_FIELDS"
        if by_native
        else "MIXED_CROSSING_DO_NOT_ISOLATE_A_CAUSE",
        "certificates": len(rows),
        "missing_crossed_support": sum(
            r["status"] == "CROSSED_SUPPORT_MISSING" for r in rows
        ),
        "enumerated_contact_pairs": pairs,
        "alternate_contact_error": direct_error,
        "diagonal_replay_error": diagonal_error,
        "same_agent_not_blind": True,
        "psychological_calibration_transport_claimed": False,
        "new_neural_calls": 0,
        "original_scope_complete": False,
    }
    R.table(out / "source_native_crossing.csv", rows)
    R.archive(out / "crossing_accounts.npz", actions=actions, components=components)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "description_seal.json",
        ("source_native_crossing.csv", "crossing_accounts.npz", "summary.json"),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
