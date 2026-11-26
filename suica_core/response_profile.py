#!/usr/bin/env python3
"""Opportunity-standardized, addressed binary behavior profiles (opt-in)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "grain-response-profile-1"
REPLIES = {"support", "refute", "unresolved"}
ALL = REPLIES | {"no_reply", "reply_unknown", "opportunity_unconfirmed"}


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def _reference(ref):
    if any(
        not _name(ref.get(k)) for k in ("behavior", "reader", "condition_rule", "unit")
    ):
        raise ValueError("name the behavior, reader, condition rule, and unit")
    frame = ref.get("frame")
    if frame not in ("presented_opportunities", "observed_replies"):
        raise ValueError("declare the sampling frame")
    weights = ref["weights"]
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            not _name(c) or type(w) not in (int, float) or not math.isfinite(w) or w < 0
            for c, w in weights.items()
        )
        or not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0)
    ):
        raise ValueError("finite nonnegative reference weights must sum to one")
    pair = ref.get("contrast")
    if pair is not None and (
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or any(not _name(c) or c not in weights for c in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError("contrast is [higher-condition, lower-condition]")
    return frame, weights, pair


def _rate(k, unknown, n):
    return [k / n, (k + unknown) / n] if n else [0.0, 1.0]


def _linear(coeff, cells):
    """Sharp bounds for a linear functional of the same independent cell intervals."""
    terms = [sorted((w * cells[c][0], w * cells[c][1])) for c, w in coeff.items()]
    return [math.fsum(t[j] for t in terms) for j in (0, 1)]


def _card(focal, by_condition, weights, pair, frame):
    name = "per_opportunity" if frame == "presented_opportunities" else "among_replies"
    rows, rates, ns = {}, {}, {}
    for c in sorted(set(weights) | set(by_condition)):
        a = by_condition.get(c, Counter())
        yes, ambiguous, unseen = a["support"], a["unresolved"], a["reply_unknown"]
        spoken = yes + a["refute"] + ambiguous
        n = spoken + a["no_reply"] + unseen
        rate = _rate(yes, ambiguous + unseen, n)
        # Unknown replies may all be speech. Taking them all as negatives
        # minimizes the conditional rate; taking them all as positives maximizes it.
        conditional = (
            _rate(yes, ambiguous + unseen, spoken + unseen) if spoken + unseen else None
        )
        rows[c] = {
            "counts": dict(sorted(a.items())),
            "n": n,
            "confirmed_replies": spoken,
            "reply_rate": _rate(spoken, unseen, n)
            if name == "per_opportunity" and n
            else None,
            "behavior_given_reply": conditional,
            "given_reply_denominator_may_be_zero": spoken == 0,
            "reading_bounds": rate,
            "reference_weight": weights.get(c, 0.0),
        }
        rates[c], ns[c] = rate, n
    N = sum(ns.values())
    q = {c: n / N for c, n in ns.items()} if N else None
    for c, row in rows.items():
        row["observed_share"] = q[c] if q is not None else None
    return {
        "focal": focal,
        "denominator": name,
        "n": N,
        "conditions": rows,
        "natural": _linear(q, rates) if q is not None else None,
        "standardized": _linear(weights, rates),
        "unsupported_weight": math.fsum(w for c, w in weights.items() if not ns[c]),
        "mix_gap_natural_minus_standardized": _linear(
            {c: q[c] - weights.get(c, 0.0) for c in q}, rates
        )
        if q is not None
        else None,
        "response_contrast": _linear({pair[0]: 1, pair[1]: -1}, rates)
        if pair
        else None,
        "unconfirmed_excluded": sum(
            a["opportunity_unconfirmed"] for a in by_condition.values()
        ),
    }


def _cluster_precision(card, totals, parts, unit, weights, pair, frame):
    keys = ("natural", "standardized", "response_contrast")
    k = len(parts)
    rep = {key: [] for key in keys}
    if k >= 3:
        for part in parts.values():
            remainder = {c: cnt - part.get(c, Counter()) for c, cnt in totals.items()}
            other = _card(card["focal"], remainder, weights, pair, frame)
            for key in keys:
                v = other[key]
                if v is not None and v[0] == v[1]:
                    rep[key].append(v[0])

    def se(key):
        v = rep[key]
        if k < 3 or len(v) != k or card[key] is None or card[key][0] != card[key][1]:
            return None
        avg = math.fsum(v) / k
        return math.sqrt((k - 1) / k * math.fsum((x - avg) ** 2 for x in v))

    return {
        "unit": unit,
        "clusters": k,
        "delete_one_cluster_se": {key: se(key) for key in keys},
    }


def measure(reference, events, *, sampling_cluster=None):
    """One call = one frozen behavioral definition, one row per addressed opportunity.

    Bounds describe the recorded finite panel, not sampling confidence intervals.
    Multiple readers of the same action must be reconciled before calling.
    'no_reply' requires independent confirmation that a response window closed.
    """
    frame, weights, pair = _reference(reference)
    if sampling_cluster not in (None, "partner", "dialogue"):
        raise ValueError("use partner or dialogue as sampling unit")
    seen, responses = set(), set()
    people = defaultdict(lambda: defaultdict(Counter))
    clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for e in events:
        person, session, eid, c, s = (
            e[k] for k in ("focal", "dialogue", "id", "condition", "status")
        )
        if not all(_name(x) for x in (person, session, eid, c)) or s not in ALL:
            raise ValueError("invalid opportunity address, condition or status")
        if frame == "observed_replies" and s not in REPLIES:
            raise ValueError(
                "a replies-only frame cannot supply nonresponse or unknown exposure"
            )
        key = (person, session, eid)
        if key in seen:
            raise ValueError("duplicate opportunity in a single behavior profile")
        seen.add(key)
        rid = e.get("response_id")
        if s in REPLIES:
            if not _name(rid) or (person, session, rid) in responses:
                raise ValueError("observed replies require distinct addressed actions")
            responses.add((person, session, rid))
        elif rid is not None:
            raise ValueError("no observed action should be bound here")
        people[person][c][s] += 1
        if sampling_cluster and s != "opportunity_unconfirmed":
            cluster = e.get(sampling_cluster)
            if not _name(cluster):
                raise ValueError("missing sampling cluster")
            clusters[person][cluster][c][s] += 1
    encoded = json.dumps(
        reference,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    cards = []
    for who, data in sorted(people.items()):
        card = _card(who, data, weights, pair, frame)
        if sampling_cluster:
            card["sampling_precision"] = _cluster_precision(
                card, data, clusters[who], sampling_cluster, weights, pair, frame
            )
        cards.append(card)
    return {
        "schema": VERSION,
        "reference": json.loads(encoded),
        "reference_digest": hashlib.sha256(encoded.encode()).hexdigest(),
        "cards": cards,
    }


def compare(left, right, focal_left, focal_right):
    """Account for a difference between two panels/people using symmetric weights.

    Outside common support, the observed natural difference can remain known
    while its allocation to mechanisms depends on an unobserved response.
    """
    if (
        left["schema"] != VERSION
        or right["schema"] != VERSION
        or left["reference"] != right["reference"]
    ):
        raise ValueError("same frozen reference required for comparison")

    def card(result, who):
        matches = [c for c in result["cards"] if c["focal"] == who]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous focal unit")
        return matches[0]

    a, b = card(left, focal_left), card(right, focal_right)

    def diff(k):
        x, y = a[k], b[k]
        if a is b and x is not None:
            return [0.0, 0.0]
        return None if x is None or y is None else [y[0] - x[1], y[1] - x[0]]

    out = {
        "from": focal_left,
        "to": focal_right,
        "natural_change": diff("natural"),
        "standardized_change": diff("standardized"),
        "contrast_change": diff("response_contrast"),
        "decomposition": None,
    }
    if not a["n"] or not b["n"]:
        return {**out, "status": "empty_panel"}
    mix = within = participation = expression = 0.0
    opp = a["denominator"] == "per_opportunity"
    split = opp
    for c in sorted(a["conditions"].keys() | b["conditions"].keys()):
        x, y = (z["conditions"].get(c) for z in (a, b))
        if not (x and x["n"] or y and y["n"]):
            continue
        if not (x and x["n"] and y and y["n"]):
            return {**out, "status": "unobserved_condition_response"}
        if any(z["reading_bounds"][0] != z["reading_bounds"][1] for z in (x, y)):
            return {**out, "status": "unresolved_behavior"}
        q0, q1 = x["observed_share"], y["observed_share"]
        b0, b1 = x["reading_bounds"][0], y["reading_bounds"][0]
        qm = (q0 + q1) / 2
        mix += (q1 - q0) * (b1 + b0) / 2
        within += qm * (b1 - b0)
        if opp:
            m0, m1 = x["behavior_given_reply"], y["behavior_given_reply"]
            if m0 is None or m1 is None:
                split = False
                continue
            p0, p1 = x["reply_rate"][0], y["reply_rate"][0]
            participation += qm * (p1 - p0) * (m0[0] + m1[0]) / 2
            expression += qm * (m1[0] - m0[0]) * (p0 + p1) / 2
    return {
        **out,
        "status": "observed_accounting",
        "decomposition": {
            "recorded_condition_mix": mix,
            "within_condition_response": within,
            "speaking_given_opportunity": participation if split else None,
            "expression_given_speaking": expression if split else None,
            "closure_error": out["natural_change"][0] - mix - within,
        },
    }


def correction_profiles(events, weights, contrast=None):
    """Bridge from event_response: freeze one old/new proposition per opportunity, not per document."""
    ref = {
        "behavior": "explicit adoption of the corrected premise",
        "reader": "event_response:lexical-or-bound-coded-v1",
        "unit": "presented partner correction; first addressable focal reply",
        "condition_rule": "dialogue.condition",
        "weights": weights,
        "contrast": contrast,
        "frame": "presented_opportunities",
    }
    rows = []
    versions = set()
    triggers = set()
    for e in events:
        if e.get("component_events"):
            raise ValueError("compound propositions need their own reference")
        s = {
            "reply_unobserved": "reply_unknown",
            "trigger_unconfirmed": "opportunity_unconfirmed",
            "observed": {
                "support": "support",
                "refute": "refute",
                "insufficient": "unresolved",
            }[e["status"]],
        }[e["opportunity"]]
        semantic = e.get("semantic_reader") or {}
        if semantic.get("kind") == "coded_response":
            versions.add(semantic["version"])
        if e.get("correction_reader"):
            triggers.add(e["correction_reader"]["version"])
        rows.append(
            {
                "focal": e["focal"],
                "dialogue": str(e["dialogue"]),
                "id": str(e["trigger"]["turn_id"]),
                "condition": e["condition"],
                "status": s,
                "response_id": str(e["response"]["turn_id"]) if e["response"] else None,
            }
        )
    ref["external_coding_versions"] = sorted(versions)
    ref["external_trigger_versions"] = sorted(triggers)
    return measure(ref, rows)


def measure_calibrated(
    reference,
    events,
    reader_calibration,
    *,
    calibration_denominator,
    sampling_cluster=None,
):
    """Keep the original D89 card beside its R24 reader correction."""
    from .psychometric_open_stream import calibrate_reader_misclassification

    result = measure(reference, events, sampling_cluster=sampling_cluster)
    for card in result["cards"]:
        card["reader_correction"] = calibrate_reader_misclassification(
            card,
            reader_calibration,
            contrast=reference.get("contrast"),
            calibration_denominator=calibration_denominator,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    doc = json.loads(a.input.read_text(encoding="utf-8"))
    if "reader_calibration" in doc:
        result = measure_calibrated(
            doc["reference"],
            doc["events"],
            doc["reader_calibration"],
            calibration_denominator=doc["calibration_denominator"],
            sampling_cluster=doc.get("sampling_cluster"),
        )
    else:
        result = measure(
            doc["reference"],
            doc["events"],
            sampling_cluster=doc.get("sampling_cluster"),
        )
    if "comparisons" in doc:
        result["comparisons"] = [
            compare(result, result, *pair) for pair in doc["comparisons"]
        ]
    out = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False)
    if a.out:
        with a.out.open("x", encoding="utf-8") as f:
            print(out, file=f)
    else:
        print(out)


if __name__ == "__main__":
    main()
