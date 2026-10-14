#!/usr/bin/env python3
"""Result-guided, frozen-program diagnosis of evidence unit composition.

The old natural-anchor result remains frozen. Fresh public non-target controls
are declared before encoding and reported separately, without refitting models.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_grain_math_evidence import DEFAULT_OUT, OLD, ROOT, load_bundle
from run_grain_math_response import encode, sha, write_csv, write_json

from grain_research.construction import math_response as M
from grain_research.construction.math_evidence_diagnostics import coordinate_attribution

CONTROLS = (
    (
        "target",
        "I took the first step and arranged a meeting for the team.",
        "I did not take the first step or arrange a meeting for the team.",
    ),
    (
        "target",
        "I contacted everyone and started coordinating our work before anyone asked.",
        "I did not contact everyone or start coordinating our work before anyone asked.",
    ),
    (
        "target",
        "I initiated the group plan and gathered people to decide the next steps.",
        "I did not initiate the group plan or gather people to decide the next steps.",
    ),
    (
        "target",
        "I volunteered to organize our meeting while the others waited.",
        "I did not volunteer to organize our meeting while the others waited.",
    ),
    (
        "target",
        "I was the first to invite people to discuss our project.",
        "I was not the first to invite people to discuss our project.",
    ),
    (
        "target",
        "I got our team together and proposed how we could divide the work.",
        "I did not get our team together or propose how we could divide the work.",
    ),
    (
        "target",
        "I started the discussion about our shared task without being prompted.",
        "I did not start the discussion about our shared task without being prompted.",
    ),
    (
        "target",
        "I offered the first concrete plan for how our group should proceed.",
        "I did not offer the first concrete plan for how our group should proceed.",
    ),
    (
        "non_target",
        "I slept for eight hours last night.",
        "I did not sleep for eight hours last night.",
    ),
    (
        "non_target",
        "I read a newspaper while eating breakfast.",
        "I did not read a newspaper while eating breakfast.",
    ),
    (
        "non_target",
        "I bought a new kettle for the kitchen.",
        "I did not buy a new kettle for the kitchen.",
    ),
    (
        "non_target",
        "I walked across the park yesterday.",
        "I did not walk across the park yesterday.",
    ),
    (
        "non_target",
        "I watched a film at home on Saturday.",
        "I did not watch a film at home on Saturday.",
    ),
    (
        "non_target",
        "I remembered the color of the door.",
        "I did not remember the color of the door.",
    ),
    (
        "non_target",
        "I listened to music on the train.",
        "I did not listen to music on the train.",
    ),
    (
        "non_target",
        "I ate a sandwich for lunch.",
        "I did not eat a sandwich for lunch.",
    ),
)


def prepare_controls(dest: Path, source: Path):
    """Freeze fresh, authored controls and their diagnostic purpose before readout."""
    contract = {
        "result_guided": True,
        "purpose": "test whether assertion-denial action is target-specific or generic negation",
        "source": "researcher-authored public controls, not natural text or a new personality validation",
        "controls": CONTROLS,
        "bundle_hashes": {
            r: sha(source / f"06/reader_bundles/{r}_joint_binding.npz")
            for r in ("e5", "bge")
        },
        "code_sha256": sha(Path(__file__)),
        "attribution_code_sha256": sha(
            ROOT / "grain_research/construction/math_evidence_diagnostics.py"
        ),
        "no_fitting": True,
    }
    path = dest / "00/protocol_before_run.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(
        json.dumps(contract)
    ):
        raise ValueError("Control protocol or code changed")
    write_json(path, contract)
    bank = [
        {
            "material_id": f"control{i}:{pole}",
            "family": i,
            "category": kind,
            "variant": pole,
            "text": text,
        }
        for i, (kind, yes, no) in enumerate(CONTROLS)
        for pole, text in (("asserted", yes), ("denied", no))
    ]
    bankpath = dest / "00/material_bank.jsonl"
    payload = "".join(json.dumps(r) + "\n" for r in bank)
    if bankpath.exists() and bankpath.read_text() != payload:
        raise ValueError("Changed control bank")
    bankpath.write_text(payload)
    write_json(
        dest / "00/reader_registry.json",
        json.loads((source / "00/reader_registry.json").read_text()),
    )
    return bank


def diagnose(out: Path, source: Path, dest: Path, controls: list[dict]):
    """Save exact composition witnesses, atomic routing and lexical controls."""
    summaries, attribution_rows, controls_rows = {}, [], []
    for reader in ("e5", "bge"):
        bundle = load_bundle(source, reader)
        table = pd.read_csv(out / f"private_scores_{reader}.csv")
        table["anchor"] = table.material_id.str.split(":").str[0]
        columns = ["pole_coordinate", "ownership_coordinate", "condition_coordinate"]
        wait = table[table.variant.eq("wait_only")].sort_values("anchor")
        mixed = table[table.variant.eq("mixed_other")].sort_values("anchor")
        start, end = wait[columns].to_numpy(), mixed[columns].to_numpy()

        def score(z, model=bundle):
            """Evaluate this reader's frozen scalar program in action coordinates."""
            return (
                (M.binding_features(z, True) - model["mu"]) @ model["w"]
                + model["intercept"]
            )[:, 0]

        phi = coordinate_attribution(start, end, score)
        delta = score(end) - score(start)
        for i, aid in enumerate(wait.anchor):
            attribution_rows.append(
                {
                    "reader": reader,
                    "anchor": aid,
                    "full_score_change": delta[i],
                    "pole_contribution": phi[i, 0],
                    "ownership_contribution": phi[i, 1],
                    "condition_contribution": phi[i, 2],
                    "declared_unit_routed_change": 0.0,
                    "route_reason": "self wait span identical, explicitly other report retained outside self actual channel",
                }
            )
        x = np.load(dest / f"01B/embeddings/{reader}.npy").astype(float)
        control_scores = M.score_bundle(bundle, x)
        for i, (kind, _, _) in enumerate(CONTROLS):
            controls_rows.append(
                {
                    "reader": reader,
                    "family": i,
                    "category": kind,
                    "assertion_denial_effect": float(
                        control_scores[2 * i, 0] - control_scores[2 * i + 1, 0]
                    ),
                    "asserted_score": float(control_scores[2 * i, 0]),
                    "denied_score": float(control_scores[2 * i + 1, 0]),
                }
            )
        cc = pd.DataFrame([r for r in controls_rows if r["reader"] == reader])
        summaries[reader] = {
            "mixed_other_full_change": float(delta.mean()),
            "coordinate_attribution_means": dict(
                zip(
                    ("pole", "ownership", "condition"),
                    phi.mean(axis=0).tolist(),
                    strict=True,
                )
            ),
            "attribution_sum_error": float(np.max(abs(phi.sum(axis=1) - delta))),
            "atomic_route_change": 0.0,
            "atomic_route_boundary": "algebraic property with explicitly declared unit ownership; not a learned natural-role classification score",
            "controls": {
                kind: {
                    "n": len(group),
                    "mean_action": float(group.assertion_denial_effect.mean()),
                    "ci": M.bootstrap_interval(
                        group.assertion_denial_effect.to_numpy()
                    ),
                    "positive": int((group.assertion_denial_effect > 0).sum()),
                }
                for kind, group in cc.groupby("category")
            },
        }
    write_csv(dest / "private_coordinate_attributions.csv", attribution_rows)
    write_csv(dest / "public_negation_controls.csv", controls_rows)
    write_json(dest / "composition_diagnosis.json", summaries)
    snapshots = dest / "code_snapshot"
    snapshots.mkdir(exist_ok=True)
    sources = (
        "grain_research/construction/math_evidence.py",
        "grain_research/construction/math_evidence_diagnostics.py",
        "scripts/run_grain_math_evidence.py",
        "scripts/analyze_grain_math_evidence.py",
        "grain_research/evidence_history.py",
    )
    for name in sources:
        (snapshots / Path(name).name).write_bytes((ROOT / name).read_bytes())
    write_json(
        dest / "code_manifest.json", {name: sha(ROOT / name) for name in sources}
    )


def main():
    """Run one small fresh control set; no tuning or re-encoding old natural text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    started = time.monotonic()
    dest = args.out / "followup_r1"
    controls = prepare_controls(dest, args.old)
    for reader in ("e5", "bge"):
        encode(dest, reader, args.device)
    diagnose(args.out, args.old, dest, controls)
    print(
        f"composition diagnostic finished in {time.monotonic() - started:.1f}s; no refit",
        flush=True,
    )


if __name__ == "__main__":
    main()
