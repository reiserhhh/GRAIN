#!/usr/bin/env python3
"""Cold arithmetic replay of narrow actions and their mixed-state interaction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from scripts.run_grain_narrow_actions import DEFAULT, REFERENCE, verify
from scripts.run_grain_navigation_baseline import checked_arrays


def run(out):
    """Audit saved logits, direct dot products and all four-corner interactions."""
    manifest = verify(out)
    summary = read_json(out / "summary.json")
    for name, expected in summary["outputs"].items():
        if digest(out / name) != expected:
            raise ValueError("Analysis output changed before audit")
    worlds = {w["world_id"]: w for w in read_json(out / "private_worlds.json")}
    design = read_json(out / "private_design.json")
    table = pd.read_csv(out / "scores.csv").set_index(["index", "readout"])
    contrasts = pd.read_csv(out / "attribution_contrasts.csv").set_index(
        ["target", "query", "reader_role", "pair"]
    )
    errors = {
        k: 0.0
        for k in ("native", "probabilities", "candidate_mass", "affine", "mixed_action")
    }
    metadata = []
    with np.load(REFERENCE / "reference.npz", allow_pickle=False) as z:
        coefficients = dict(z)
    for d in design:
        w = worlds[d["world_id"]]
        token_rows = np.asarray(d["input_ids"])[np.asarray(d["mask"], bool)].tolist()
        if token_rows != manifest["question_tokens"][d["query"]]:
            raise ValueError("Question token coordinate mismatch")
        if w["evidence_body"] not in w["texts"][0] or not w["synthetic"]:
            raise ValueError("Evidence provenance mismatch")
        counterpart = worlds[
            w["world_id"].rsplit("_", 1)[0]
            + ("_other" if w["owner"] == "self" else "_self")
        ]
        if counterpart["evidence_body"] != w["evidence_body"]:
            raise ValueError("Ownership wrapper changed the evidence body")
        meta, states = checked_arrays(
            out / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        q = d["query"]
        native = meta["native"]
        p = softmax(np.array(native["candidate_logits"], float))
        score = float(p @ np.array(d["alphabet"]["values"], float))
        mass = float(
            np.exp(
                logsumexp(native["candidate_logits"])
                - native["full_vocab_log_normalizer"]
            )
        )
        point = float(
            coefficients[q + "__b"]
            + np.sum(
                coefficients[q + "__G"]
                * (states["P"].astype(float) - coefficients[q + "__Z"])
            )
            + coefficients[q + "__g"]
            @ (states["Q"].astype(float) - coefficients[q + "__h"])
        )
        errors["native"] = max(
            errors["native"], abs(score - table.loc[(d["index"], "native"), "score"])
        )
        errors["probabilities"] = max(
            errors["probabilities"],
            float(np.max(np.abs(p - native["conditional_probabilities"]))),
        )
        errors["candidate_mass"] = max(
            errors["candidate_mass"], abs(mass - native["candidate_mass"])
        )
        errors["affine"] = max(
            errors["affine"], abs(point - table.loc[(d["index"], "affine"), "score"])
        )
        metadata.append(
            {k: d[k] for k in ("index", "target", "query", "reader_role")}
            | {k: w[k] for k in ("pair", "polarity", "owner")}
        )
    interactions = []
    for identity, group in pd.DataFrame(metadata).groupby(
        ["target", "query", "reader_role", "pair"], sort=True
    ):
        corners = {}
        for _, row in group.iterrows():
            _, states = checked_arrays(
                out / "private_packets" / f"unit_{row['index']:04d}.json"
            )
            corners[(row.polarity, row.owner)] = {
                k: states[k].astype(float) for k in ("P", "Q")
            }
        if len(corners) != 4:
            raise ValueError("Four distinct corners required")
        mixed = {
            k: corners[("high", "self")][k]
            - corners[("low", "self")][k]
            - corners[("high", "other")][k]
            + corners[("low", "other")][k]
            for k in ("P", "Q")
        }
        q = identity[1]
        p_action = float(np.sum(coefficients[q + "__G"] * mixed["P"]))
        q_action = float(coefficients[q + "__g"] @ mixed["Q"])
        expected = contrasts.loc[identity]
        errors["mixed_action"] = max(
            errors["mixed_action"], abs(p_action + q_action - expected.affine)
        )
        interactions.append(
            dict(zip(("target", "query", "reader_role", "pair"), identity, strict=True))
            | {
                "mixed_P_norm": float(np.linalg.norm(mixed["P"])),
                "mixed_Q_norm": float(np.linalg.norm(mixed["Q"])),
                "P_action": p_action,
                "Q_action": q_action,
                "affine_interaction": p_action + q_action,
                "native_interaction": float(expected.native),
                "remainder_interaction": float(expected.native - p_action - q_action),
            }
        )
    frame = pd.DataFrame(interactions)
    frame.to_csv(out / "mixed_state_interactions.csv", index=False)
    audit = {
        "status": "PASS" if max(errors.values()) < 1e-10 else "FAIL",
        "errors": errors,
        "logit_and_point_replays": len(design),
        "query_coordinate_checks": len(design),
        "ownership_body_checks": len(design),
        "mixed_interactions": len(frame),
        "mixed_states_nonzero": int(
            ((frame.mixed_P_norm + frame.mixed_Q_norm) > 1e-12).sum()
        ),
        "auditor_sha256": digest(Path(__file__)),
        "interaction_sha256": digest(out / "mixed_state_interactions.csv"),
        "independent_executor": False,
        "scope": "independent arithmetic formulas, same executor and packets",
    }
    write_json(out / "cold_replay_audit.json", audit)
    print(audit)
    if audit["status"] != "PASS":
        raise ValueError("Cold replay failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    run(parser.parse_args().out)
