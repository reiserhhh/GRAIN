#!/usr/bin/env python3
"""Report-rooted state exposures with preserved condition incidence.

The coverage query asks whether a state report contains a target-qualified
condition. It does not estimate emotion intensity, verify an action, or assume
independent conditions. Predicate count and report count are different units.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from . import math_observation_dag as D
from . import math_purpose_accounts as P

TARGETS = ("coordination_action", "verification_action")


def observe_reports(sentences, text):
    """Extract report roots and temporal conditions without target/gold input.

    Basic UD coordination can omit a shared subject. A conjunct may use the
    first conjunct's explicit subject, recorded as a candidate inheritance;
    this never inherits the experiencer into a condition actor. Other modal
    ancestors, relative modifiers and unknown participants remain unresolved.
    """
    graph = D.observe_dag(sentences, text)
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    reports = []
    for si, words in enumerate(sentences):
        by = {w["id"]: w for w in words}
        children = {i: [] for i in by}
        for w in words:
            if w["head"] in children:
                children[w["head"]].append(w["id"])

        def edge(i, by=by):
            return by[i]["deprel"].split(":")[0]

        def negative(i, by=by, children=children):
            return any(by[j]["lemma"] in {"not", "never"} for j in children[i])

        for root in [
            n
            for n in graph["nodes"]
            if n["sentence_index"] == si and n["action"] == "feel"
        ]:
            rid = root["ud_path"][0]
            flags = []
            if edge(rid) not in {"root", "conj"} or any(
                by[i]["lemma"] in set(P.GOVERNORS) | P.UNSUPPORTED
                for i in root["ud_path"][1:]
            ):
                flags.append("nested_or_reported_state")
            if root["actor"] not in {"self", "other"}:
                flags.append("unknown_experiencer")
            triggers = []
            for nid, node in nodes.items():
                if node["sentence_index"] != si or nid == root["node_id"]:
                    continue
                path = node["ud_path"]
                if rid not in path[1:]:
                    continue
                below = path[: path.index(rid)]
                if any(edge(i) in {"acl", "amod"} for i in below):
                    continue
                if any(
                    by[i]["lemma"] in set(P.GOVERNORS) | P.UNSUPPORTED
                    for i in below[1:]
                ):
                    continue
                temporal = [
                    i
                    for i in below
                    if edge(i) == "advcl"
                    and any(
                        by[j]["lemma"] in {"when", "while", "whenever"}
                        and edge(j) in {"mark", "advmod"}
                        for j in children[i]
                    )
                ]
                if not temporal:
                    continue
                trigger_flags = []
                if node["actor"] not in {"self", "other"}:
                    trigger_flags.append("unknown_trigger_actor")
                if not node["raw_object_spans"]:
                    trigger_flags.append("no_direct_object")
                for span in node["raw_object_spans"] or [None]:
                    triggers.append(
                        {
                            "node_id": nid,
                            "report_id": root["node_id"],
                            "action": node["action"],
                            "action_span": node["action_span"],
                            "object": text[slice(*span)] if span else None,
                            "object_span": span,
                            "actor": node["actor"],
                            "actor_spans": node["actor_spans"],
                            "subject_origin": node["subject_origin"],
                            "temporal_heads": [f"s{si}:p{i}" for i in temporal],
                            "polarity": -1 if negative(path[0]) else 1,
                            "status": "unresolved" if trigger_flags else "ok",
                            "flags": trigger_flags,
                            "is_verified_actual_action": False,
                            "experiencer_inherited_as_actor": False,
                        }
                    )
            if not triggers:
                flags.append("no_observed_temporal_condition")
            if any(t["status"] != "ok" for t in triggers):
                flags.append("unresolved_condition")
            reports.append(
                {
                    "report_id": root["node_id"],
                    "root_span": root["action_span"],
                    "experiencer": root["actor"],
                    "experiencer_spans": root["actor_spans"],
                    "state_polarity": -1 if negative(rid) else 1,
                    "projection_text": root["projection_text"],
                    "status": "unresolved" if flags else "ok",
                    "flags": flags,
                    "triggers": triggers,
                    "report_mass": 1,
                    "unit": "one state expression report, not one trigger predicate",
                }
            )
    return {"graph": graph, "reports": reports, "raw_text_retained": True}


def report_capacity(scores, report, *, actor_filter=False):
    """Return bounded coverage, retaining every condition score and root sign.

    The clipped map is a declared coverage index, not a calibrated probability.
    Max is order/duplication invariant. The reference denominator is reports,
    including reports without a qualified trigger, not qualified predicates.
    """
    x = np.asarray(scores, float)
    if x.shape != (len(report["triggers"]), 2) or not np.isfinite(x).all():
        raise ValueError("aligned finite two-target condition scores required")
    gates = np.clip((x + 1) / 2, 0, 1)
    hard = (x > 0).astype(float)
    if actor_filter:
        mask = np.array([t["actor"] == "self" for t in report["triggers"]])
        gates, hard = gates * mask[:, None], hard * mask[:, None]
    capacity = gates.max(axis=0) if len(gates) else np.zeros(2)
    qualified = hard.max(axis=0) if len(hard) else np.zeros(2)
    eligible = report["status"] == "ok" and report["experiencer"] == "self"
    return {
        "eligible": eligible,
        "report_mass": int(eligible),
        "coverage": capacity.tolist() if eligible else None,
        "hard_coverage": qualified.tolist() if eligible else None,
        "signed_expression": (report["state_polarity"] * capacity).tolist()
        if eligible
        else None,
        "coexposure": float(min(capacity)) if eligible else None,
        "hard_coexposure": float(min(qualified)) if eligible else None,
        "same_trigger_coexposure": float(np.min(gates, axis=1).max())
        if eligible and len(gates)
        else None,
        "flat_sum": gates.sum(axis=0).tolist() if eligible else None,
        "flat_mean": gates.mean(axis=0).tolist() if eligible and len(gates) else None,
        "noisy_or_control": (1 - np.prod(1 - gates, axis=0)).tolist()
        if eligible
        else None,
    }


def incidence_witness():
    """Same trigger multiset, different report incidence and joint exposure."""
    joint = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]], float)
    separate = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], float)
    return {
        "joint": joint.tolist(),
        "distributed": separate.tolist(),
        "flat_sum_error": float(
            np.max(np.abs(joint.sum((0, 1)) - separate.sum((0, 1))))
        ),
        "joint_report_coexposure": float(np.min(joint.max(1), axis=1).mean()),
        "distributed_report_coexposure": float(np.min(separate.max(1), axis=1).mean()),
        "question": "do the two target conditions occur in the same report?",
    }


def replay_reports(records):
    """Replay explicit versions before counting roots or retaining conditions.

    A supplied alias adds no observation; a revision replaces the whole report
    graph, not just its scalar score. Record identity is not proof that two
    reports describe the same physical event.
    """
    active = P.replay_versions(records)
    reports = []
    for record in active:
        for report in record["reports"]:
            rid = record["record_id"] + "/" + report["report_id"]
            reports.append(
                {
                    **report,
                    "report_id": rid,
                    "record_id": record["record_id"],
                    "author_id": record["author_id"],
                    "triggers": [
                        {
                            **t,
                            "report_id": rid,
                            "node_id": record["record_id"] + "/" + t["node_id"],
                        }
                        for t in report["triggers"]
                    ],
                }
            )
    return reports


def joint_extension_witness():
    """Singleton source actions do not uniquely identify a joint response law.

    Two bounded monotone and duplication-invariant set functions agree on
    every singleton yet differ on a pair. Psychological intensity cannot be
    inferred by choosing one of them without a joint source observation law.
    """

    def alternative(values):
        distinct = sorted(set(values), reverse=True)
        largest = distinct[0] if distinct else 0.0
        return largest + (1 - largest) * (distinct[1] if len(distinct) > 1 else 0)

    single = [0.2, 0.6, 0.9]
    return {
        "singleton_error": max(abs(v - alternative([v])) for v in single),
        "pair_max": 0.6,
        "pair_alternative": alternative([0.2, 0.6]),
        "duplicate_pair_alternative": alternative([0.2, 0.6, 0.6]),
        "unit": "declared response-law witness, not emotion intensity",
    }


def material_bank(source):
    """Freeze new report/condition combinations using only source-covered slots."""
    pairs = {}
    for target, ai in product(TARGETS, range(8)):
        candidates = [
            r
            for r in source
            if r["target"] == target
            and r["action_index"] == ai
            and r["role"] == "source"
            and r["object_type"] == 1
            and r["bits"][1:] == [1, 1, 1]
        ]
        pairs[target, ai] = min(candidates, key=lambda r: r["object_index"])
    rows = []

    def sentence(
        conditions, own, join="and", root_negative=False, condition_negative=False
    ):
        who = "I" if own == 1 else "My colleague"
        predicate = (
            ("do" if own == 1 else "does") + " not feel"
            if root_negative
            else "feel" + ("s" if own == -1 else "")
        )
        text = f"{who} {predicate} uneasy when "
        root_at = text.index("feel")
        trigger_gold = []
        for j, (pair, actor) in enumerate(conditions):
            if j:
                text += {
                    "and": " and ",
                    "while": " while ",
                    "when": " and when ",
                    "implicit": " and ",
                }[join]
            subject = "I" if actor == 1 else "my colleague"
            if j == 0 or join != "implicit":
                text += subject + " "
            text += "must " + ("not " if condition_negative else "")
            at = len(text)
            text += pair["action"] + " "
            start = len(text)
            text += pair["object"]
            trigger_gold.append(
                {
                    "action": pair["action"],
                    "action_span": [at, at + len(pair["action"])],
                    "object": pair["object"],
                    "object_span": [start, len(text)],
                    "actor": "self" if actor == 1 else "other",
                    "polarity": -1 if condition_negative else 1,
                    "domain": pair["target"],
                    "relevant": pair["bits"][0] == 1,
                }
            )
        text += "."
        return text, {
            "root_span": [root_at, root_at + (4 if own == 1 or root_negative else 5)],
            "experiencer": "self" if own == 1 else "other",
            "state_polarity": -1 if root_negative else 1,
            "triggers": trigger_gold,
        }

    def add(mid, kind, parts, **meta):
        text, reports = "", []
        for part, report in parts:
            offset = len(text) + (1 if text else 0)
            if text:
                text += " "
            text += part
            report = {
                **report,
                "projection_text": part,
                "root_span": [v + offset for v in report["root_span"]],
                "triggers": [
                    {
                        **t,
                        "action_span": [v + offset for v in t["action_span"]],
                        "object_span": [v + offset for v in t["object_span"]],
                    }
                    for t in report["triggers"]
                ],
            }
            reports.append(report)
        rows.append(
            {
                "material_id": mid,
                "kind": kind,
                "text": text,
                "expected_reports": reports,
                **meta,
            }
        )

    for family, pc, pv, own, ac, av in product(
        range(4), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)
    ):
        c, v = (
            pairs[TARGETS[0], family + (4 if pc == -1 else 0)],
            pairs[TARGETS[1], family + (4 if pv == -1 else 0)],
        )
        key = f"{family}_{pc}_{pv}_{own}_{ac}_{av}"
        for join in ("and", "while", "when", "implicit"):
            if join == "implicit" and ac != av:
                continue
            add(
                f"pair_{join}_{key}",
                join,
                [sentence([(c, ac), (v, av)], own, join)],
                family=family,
            )
        add(
            f"two_{key}",
            "two_reports",
            [sentence([(c, ac)], own), sentence([(v, av)], own)],
            family=family,
        )
    for target, ai, own, actor in product(TARGETS, range(8), (-1, 1), (-1, 1)):
        pair = pairs[target, ai]
        key = f"{target}_{ai}_{own}_{actor}"
        for duplicate in (False, True):
            conditions = [(pair, actor)] * (2 if duplicate else 1)
            add(
                f"{'duplicate' if duplicate else 'single'}_{key}",
                "duplicate" if duplicate else "single",
                [sentence(conditions, own)],
                pair_key=key,
            )
        if ai < 4:
            for root_negative, condition_negative in ((True, False), (False, True)):
                add(
                    f"neg_{key}_{int(root_negative)}",
                    "root_negation" if root_negative else "condition_negation",
                    [
                        sentence(
                            [(pair, actor)],
                            own,
                            root_negative=root_negative,
                            condition_negative=condition_negative,
                        )
                    ],
                )
    for family, own, ac, av in product(range(4), (-1, 1), (-1, 1), (-1, 1)):
        c, v, nc, nv = [
            pairs[target, family + negative]
            for target, negative in (
                (TARGETS[0], 0),
                (TARGETS[1], 0),
                (TARGETS[0], 4),
                (TARGETS[1], 4),
            )
        ]
        key = f"{family}_{own}_{ac}_{av}"
        for bundled in (False, True):
            first, second = (
                ([(c, ac), (v, av)], [(nc, ac), (nv, av)])
                if bundled
                else ([(c, ac), (nv, av)], [(nc, ac), (v, av)])
            )
            add(
                f"bundle_{key}_{int(bundled)}",
                "bundle",
                [sentence(first, own), sentence(second, own)],
                bundle_key=key,
                bundled=bundled,
            )
    if len(rows) != 832 or len({r["material_id"] for r in rows}) != len(rows):
        raise ValueError("HM-09B factorial material count changed")
    return rows
