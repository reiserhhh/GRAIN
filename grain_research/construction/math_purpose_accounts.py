#!/usr/bin/env python3
"""Purpose-indexed event observations, after version replay rather than filtering.

The scoped modal interface is linguistic observation, not verification of acts
or personality. Existing vector maps retain their source units. A state account
records exposure only: action polarity is not an emotional intensity scale.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from grain_research.source_event_graph import validate_words

ACCOUNTS = ("actual", "hypothetical", "desire", "plan", "state")
GOVERNORS = {
    "want": "desire",
    "wish": "desire",
    "plan": "plan",
    "intend": "plan",
    "imagine": "hypothetical",
    "feel": "state",
}
UNSUPPORTED = {"believe", "say", "think", "ask", "tell", "deny"}


def material_bank(source):
    """Author balanced new mode constructions on cold action/object pairs."""
    lookup = {
        (r["target"], r["action_index"], r["object_index"]): i
        for i, r in enumerate(source)
        if r["bits"][1:] == [1, 1, 1]
    }
    rows = []
    for target in sorted({r["target"] for r in source}):
        for family, mode, style, p, o in product(
            range(8), ACCOUNTS, range(2), (-1, 1), (-1, 1)
        ):
            index = lookup[target, family, 8 + family % 2]
            original = source[index]
            if original["role"] != "unseen_object":
                raise ValueError("HM-06 must not use source-fitted object pairs")
            subject = "I" if o == 1 else "my colleague"
            do = "do" if o == 1 else "does"
            action, obj = original["action"], original["object"]
            neg = " not" if p == -1 else ""
            if mode == "actual":
                prefix = "On this occasion, " if style == 0 else "Yesterday, "
                text = f"{prefix}{subject} did{neg} {action} {obj}."
            elif mode == "hypothetical":
                text = (
                    f"In an imagined event, {subject} would{neg} {action} {obj}."
                    if style == 0
                    else f"{subject.capitalize()} imagine{'s' if o == -1 else ''} that {subject} would{neg} {action} {obj}."
                )
            elif mode in {"desire", "plan"}:
                verb = (
                    ("want", "wish")[style]
                    if mode == "desire"
                    else ("plan", "intend")[style]
                )
                predicate = (
                    f"{do} not {verb}"
                    if p == -1
                    else verb
                    + ("es" if verb == "wish" and o == -1 else "s" if o == -1 else "")
                )
                text = f"{subject.capitalize()} {predicate} to {action} {obj}."
            else:
                state = "anxious" if style == 0 else "uneasy"
                predicate = (
                    f"{do} not feel" if p == -1 else "feel" + ("s" if o == -1 else "")
                )
                text = f"{subject.capitalize()} {predicate} {state} when {subject} must {action} {obj}."
            start = text.index(action + " " + obj)
            rows.append(
                {
                    "material_id": f"{target}_{family}_{mode}_{style}_{p}_{o}",
                    "target": target,
                    "family": family,
                    "style": style,
                    "kind": "single",
                    "text": text,
                    "events": [
                        {
                            "event_id": "e",
                            "span": [0, len(text)],
                            "action_span": [start, start + len(action)],
                            "source_index": index,
                            "mode": mode,
                            "bits": [original["bits"][0], p, o],
                        }
                    ],
                }
            )
    singles = list(rows)
    by_key = {
        (
            r["target"],
            r["family"],
            r["events"][0]["mode"],
            r["style"],
            *r["events"][0]["bits"][1:],
        ): r
        for r in singles
    }
    for target, family, left, right, swapped in product(
        sorted({r["target"] for r in source}),
        range(4),
        ("actual", "desire", "plan", "state"),
        ("hypothetical",),
        (False, True),
    ):
        a = by_key[target, family, left, 0, 1, -1 if swapped else 1]
        b = by_key[target, (family + 1) % 4, right, 1, -1, 1 if swapped else -1]
        text = a["text"] + " " + b["text"]
        offset = len(a["text"]) + 1
        ea, eb = dict(a["events"][0]), dict(b["events"][0])
        ea["event_id"], eb["event_id"] = "a", "b"
        eb["span"] = [v + offset for v in eb["span"]]
        eb["action_span"] = [v + offset for v in eb["action_span"]]
        rows.append(
            {
                "material_id": f"mix_{target}_{family}_{left}_{int(swapped)}",
                "target": target,
                "family": family,
                "style": 0,
                "kind": "mixture",
                "text": text,
                "events": [ea, eb],
            }
        )
    for target, family in product(sorted({r["target"] for r in source}), range(4)):
        i = lookup[target, family, 8 + family % 2]
        s = source[i]
        challenge = (
            f"I believe that my colleague did {s['action']} {s['object']}.",
            f"I want my colleague to {s['action']} {s['object']}.",
            f"I imagine that I want to {s['action']} {s['object']}.",
            f"I want not to {s['action']} {s['object']}.",
        )
        for j, text in enumerate(challenge):
            at = text.index(s["action"] + " " + s["object"])
            rows.append(
                {
                    "material_id": f"scope_{target}_{family}_{j}",
                    "target": target,
                    "family": family,
                    "style": j,
                    "kind": "unsupported_scope",
                    "text": text,
                    "events": [
                        {
                            "event_id": "e",
                            "span": [0, len(text)],
                            "action_span": [at, at + len(s["action"])],
                            "source_index": i,
                            "mode": "unresolved",
                            "bits": [1, 1, 1],
                        }
                    ],
                }
            )
    return rows


def observe_modes(sentences, text):
    """Observe scoped mode governors and action/object spans from real UD words.

    Object-control, nested modalities and lower-clause negation remain open.
    This inventory cannot infer wishes merely from 'would', or prove an action.
    No material IDs, expected roles, source verbs or author traits are accepted.
    """
    events = []
    for words in sentences:
        validate_words(words, text)
        by = {w["id"]: w for w in words}
        children = {i: [] for i in by}
        for w in words:
            if w["head"] in children:
                children[w["head"]].append(w["id"])

        def edge(i, by=by):
            return by[i]["deprel"].split(":")[0]

        def tree(i, children=children):
            todo, ids = [i], set()
            while todo:
                j = todo.pop()
                if j not in ids:
                    ids.add(j)
                    todo.extend(children[j])
            return ids

        for w in words:
            if w["upos"] != "VERB" or w["lemma"] in set(GOVERNORS) | UNSUPPORTED:
                continue
            objects = [i for i in children[w["id"]] if edge(i) == "obj"]
            if not objects:
                continue
            path, node = [w["id"]], w
            while node["head"]:
                node = by[node["head"]]
                path.append(node["id"])
            if any(edge(i) in {"acl", "amod"} for i in path):
                continue
            governors = [i for i in path[1:] if by[i]["lemma"] in GOVERNORS]
            unsupported = [i for i in path[1:] if by[i]["lemma"] in UNSUPPORTED]
            flags = []
            head = governors[0] if governors else w["id"]
            mode = GOVERNORS[by[head]["lemma"]] if governors else "actual"
            if len(governors) > 1 or unsupported:
                flags.append("unsupported_nested_or_belief_scope")
            if not governors and edge(w["id"]) != "root":
                flags.append("unsupported_main_clause")
            if not governors and any(
                by[i]["lemma"] == "would" for i in children[w["id"]]
            ):
                mode = "hypothetical"
            subject = [i for i in children[w["id"]] if edge(i) == "nsubj"]
            origin = "direct"
            # State accounts belong to the experiencer, not an embedded actor.
            if mode == "state" or not subject:
                subject = [i for i in children[head] if edge(i) == "nsubj"]
                origin = "mode_governor"
            if not subject:
                flags.append("missing_subject")
            if governors and mode in {"desire", "plan"}:
                if any(edge(i) in {"obj", "iobj"} for i in children[head]):
                    flags.append("unsupported_object_control")
                if any(by[i]["lemma"] in {"not", "never"} for i in children[w["id"]]):
                    flags.append("lower_clause_negation_not_governor_denial")
            for obj in objects:
                ids = tree(obj)
                start, end = (
                    min(by[i]["start"] for i in ids),
                    max(by[i]["end"] for i in ids),
                )
                raw_start, raw_end = (
                    min(x["start"] for x in words),
                    max(x["end"] for x in words),
                )
                events.append(
                    {
                        "mode": mode if not flags else "unresolved",
                        "candidate_mode": mode,
                        "status": "unresolved" if flags else "ok",
                        "flags": flags,
                        "subject_origin": origin,
                        "subject_spans": [
                            [by[i]["start"], by[i]["end"]] for i in subject
                        ],
                        "governor_span": [by[head]["start"], by[head]["end"]],
                        "action": w["lemma"],
                        "action_span": [w["start"], w["end"]],
                        "object": text[start:end],
                        "object_span": [start, end],
                        "projection_text": text[raw_start:raw_end],
                    }
                )
    return events


def purpose_values(z, modes, account, arm="typed"):
    """Return contribution, soft mass, and hard observability for one purpose.

    Soft masses are bounded gates, not calibrated posterior probabilities.
    Modes supply a typed coordinate; k/p/o remain frozen vector observations.
    An absent account is missing empirical evidence, not a neutral author.
    """
    z, modes = np.asarray(z, float), np.asarray(modes)
    if (
        z.ndim != 2
        or z.shape[1] != 4
        or len(modes) != len(z)
        or account not in ACCOUNTS
    ):
        raise ValueError("aligned four coordinates and a declared purpose required")
    if not np.isfinite(z).all():
        raise ValueError("nonfinite observation")
    gate = np.clip((z[:, [0, 2]] + 1) / 2, 0, 1).prod(axis=1)
    hard = np.all(z[:, [0, 2]] > 0, axis=1)
    if arm == "universal_actual":
        mass = np.clip((z[:, 3] + 1) / 2, 0, 1)
        eligible = z[:, 3] > 0
    elif arm == "binary_mirror":
        sign = 1 if account == "actual" else -1
        mass = (
            np.clip((sign * z[:, 3] + 1) / 2, 0, 1)
            if account in ACCOUNTS[:2]
            else np.zeros(len(z))
        )
        eligible = (
            sign * z[:, 3] > 0 if account in ACCOUNTS[:2] else np.zeros(len(z), bool)
        )
    elif arm == "typed":
        mass = (modes == account).astype(float)
        eligible = modes == account
    else:
        raise ValueError("unknown purpose arm")
    value = np.ones(len(z)) if account == "state" else z[:, 1]
    return gate * mass * value, gate * mass, hard & eligible


def replay_versions(records):
    """Replay declared same-event relations before any purpose/owner filtering."""
    active, seen, events, authors, contexts = {}, set(), set(), set(), {}
    for row in records:
        rid, event, op = row["record_id"], row["event_id"], row["operation"]
        authors.add(row["author_id"])
        if len(authors) > 1 or rid in seen:
            raise ValueError("unique record IDs and one author required")
        seen.add(rid)
        if op == "new":
            if event in events:
                raise ValueError("new report is not a new event")
            active[event], contexts[event] = row, row["condition"]
            events.add(event)
        elif op in {"alias", "revise", "retract"}:
            parent = active.get(event)
            if parent is None or parent["record_id"] != row["parent_id"]:
                raise ValueError("operation requires the active same-event parent")
            if row["condition"] != contexts[event]:
                raise ValueError("event condition changed without a new event")
            if op == "revise":
                active[event] = row
            elif op == "retract":
                del active[event]
        else:
            raise ValueError("unsupported version operation")
    return list(active.values())


def account_profile(records, mass_positive=0.5):
    """Compute fixed-reference condition endpoints with missingness preserved.

    Input rows are already active unique events, qualified for THIS purpose.
    Context choice is kept alongside scores; it is not centered out as noise.
    """
    if not 0 <= mass_positive <= 1:
        raise ValueError("reference mass required")
    if len({r["event_id"] for r in records}) != len(records):
        raise ValueError("replay event identities before scoring")
    endpoints, counts = {}, {}
    for c in (-1, 1):
        rows = [r for r in records if r["condition"] == c]
        counts[str(c)] = len(rows)
        endpoints[str(c)] = float(np.mean([r["value"] for r in rows])) if rows else None
    if any(r["condition"] not in (-1, 1) for r in records):
        raise ValueError("unknown conditions cannot enter this endpoint score")
    complete = all(v is not None for v in endpoints.values())
    return {
        "endpoints": endpoints,
        "event_counts": counts,
        "observed_events": len(records),
        "reference_mass_positive": mass_positive,
        "level": (1 - mass_positive) * endpoints["-1"] + mass_positive * endpoints["1"]
        if complete
        else None,
        "response": (endpoints["1"] - endpoints["-1"]) / 2 if complete else None,
        "condition_choice_positive": counts["1"] / len(records) if records else None,
    }


def finite_witnesses():
    """Show mode refinement is necessary and replay/routing cannot commute."""
    z = np.array([[1, 1, 1, -1], [1, -1, 1, -1]], float)
    modes = ["desire", "hypothetical"]
    typed = {a: purpose_values(z, modes, a)[0].tolist() for a in ACCOUNTS}
    actual = purpose_values(z, modes, "actual", "universal_actual")[0].tolist()
    old = {
        "record_id": "old",
        "event_id": "e",
        "author_id": "u",
        "operation": "new",
        "condition": -1,
        "mode": "actual",
    }
    revision = {
        **old,
        "record_id": "new",
        "parent_id": "old",
        "operation": "revise",
        "mode": "hypothetical",
    }
    active = replay_versions([old, revision])
    return {
        "same_binary_actuality_distinct_modes": {
            "coordinates": z.tolist(),
            "modes": modes,
            "typed_contributions": typed,
            "universal_contributions": actual,
        },
        "replay_before_route": {
            "actual": [r["record_id"] for r in active if r["mode"] == "actual"],
            "hypothetical": [
                r["record_id"] for r in active if r["mode"] == "hypothetical"
            ],
        },
        "incorrect_filter_before_replay_retains": "old",
    }
