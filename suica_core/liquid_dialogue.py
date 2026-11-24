#!/usr/bin/env python3
"""Attach speaker provenance to frozen liquid trajectories, without pooling people."""

import argparse

import numpy as np

from . import liquid_instrument as li


def prepare(doc):
    if doc.get("language") != "en" or doc.get("clock") != "opportunity":
        raise ValueError("English opportunity-clock input required")
    packed = {"language": "en", "clock": "opportunity", "episodes": []}
    layout = {}
    for i, d in enumerate(doc["dialogues"]):
        if not isinstance(d["id"], str) or not d["id"] or d["id"] in layout:
            raise ValueError("blank or duplicate dialogue id")
        speakers = [t["speaker"] for t in d["turns"]]
        if (
            not all(isinstance(s, str) and s.strip() for s in speakers)
            or len(set(speakers)) != 2
        ):
            raise ValueError("supply exactly two named speakers")
        events = [t["text"] for t in d["turns"]]
        shared = f"d{i}:joint"
        own = {}
        for j, who in enumerate([None, *dict.fromkeys(speakers)]):
            key = shared if who is None else f"d{i}:speaker{j}"
            turns = [k for k, s in enumerate(speakers) if who is None or who == s]
            if who is not None:
                own[who] = (key, turns)
            packed["episodes"].append(
                {
                    "id": key,
                    "author": key,
                    "condition": d["condition"],
                    "carrier": d["carrier"],
                    "view": d["view"],
                    "events": [events[k] for k in turns],
                }
            )
        layout[d["id"]] = (shared, speakers, own)
    if not layout:
        raise ValueError("empty dialogue set")
    return packed, layout


def attribute(doc, layout, result):
    out = {}
    for did, (shared, speakers, own) in layout.items():
        joint = result["episodes"][shared]
        origin = [speakers[e] for e, j in joint["slots"]]
        delta = np.asarray(joint["missing_effect"])
        field = joint.get("nine_fields")
        field_delta = (
            np.asarray(field["source_deletion_delta"]) if field is not None else None
        )
        paths = {}
        for s, (key, turns) in own.items():
            separate = result["episodes"][key]
            ids = [k for k, name in enumerate(origin) if name == s]
            aligned = [[turns[e], j] for e, j in separate["slots"]]
            if [joint["slots"][k] for k in ids] != aligned or [
                joint["unit_digests"][k] for k in ids
            ] != separate["unit_digests"]:
                raise ValueError(
                    "cut or fingerprint changed between joint and own turns"
                )
            prior_own, prior_other = [], []
            f_own, f_other = [], []
            for k in ids:
                # A separate bucket for the current turn prevents same-turn slices
                # from masquerading as reception of a partner's earlier speech.
                earlier = np.array(
                    [e < joint["slots"][k][0] for e, _ in joint["slots"]]
                )
                mine = np.array([t == s for t in origin])
                prior_own.append(delta[earlier & mine, k].sum(axis=0).tolist())
                prior_other.append(delta[earlier & ~mine, k].sum(axis=0).tolist())
                if field_delta is not None:
                    f_own.append(field_delta[earlier & mine, k].sum(axis=0).tolist())
                    f_other.append(field_delta[earlier & ~mine, k].sum(axis=0).tolist())
            paths[s] = {
                "joint_slice_indices": ids,
                "joint_slots": aligned,
                "own_only_readout": separate["liquid"],
                "joint_minus_own_only": (
                    np.asarray(joint["liquid"])[ids] - separate["liquid"]
                ).tolist(),
                "prior_own_turn_single_omission_sum": prior_own,
                "prior_partner_turn_single_omission_sum": prior_other,
            }
            if field_delta is not None:
                paths[s]["nine_fields"] = {
                    "names": field["names"],
                    "prior_own_turn_single_omission_sum": f_own,
                    "prior_partner_turn_single_omission_sum": f_other,
                }
        out[did] = {
            "condition": joint["condition"],
            "carrier": joint["carrier"],
            "view": joint["view"],
            "speaker_at_slice": origin,
            "joint_trajectory": {
                k: v for k, v in joint.items() if k not in ("author", "id")
            },
            "speaker_paths": paths,
        }
    return {
        "kind": "dyad_reader_not_personal_factor_scores",
        "input_digest": li.digest(doc),
        "reference_digest": result["reference_digest"],
        "node_names": result["node_names"],
        "clock": "joint runs advance once per shared slice; own-only runs advance only on own slices",
        "dialogues": out,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "reference", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--transport", choices=["local", "ssh"], default="local")
    a = p.parse_args()
    doc, ref = li.load(a.input), li.load(a.reference)
    packed, layout = prepare(doc)
    li.save(a.out, attribute(doc, layout, li.run("score", packed, ref, a.transport)))


if __name__ == "__main__":
    main()
