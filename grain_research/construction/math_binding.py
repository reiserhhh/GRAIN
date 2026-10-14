#!/usr/bin/env python3
"""Typed action/object kernels and finite compositional holdouts.

The observer uses frozen vectors, not truth codes at scoring. The tensor kernel
is a standard product-kernel implementation, not a full categorical grammar.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score

from . import math_local as L


def pair_role(action: int, obj: int) -> str:
    """Assign entire combinations, including all their polarity/role variants."""
    if obj >= 8:
        return "unseen_object"
    cell = (action + obj) % 4
    return "source" if cell < 2 else "validation" if cell == 2 else "unseen_pair"


def material_bank(spec: dict) -> list[dict]:
    """Build public factorial cases with objects and actions crossed independently."""
    rows = []
    for target, design in spec.items():
        verbs = design["positive_actions"] + design["negative_actions"]
        for ai, oi, p, o, a in product(range(8), range(12), (-1, 1), (-1, 1), (-1, 1)):
            verb = verbs[ai]
            obj, object_type = design["objects"][oi]
            action_type = 1 if ai < 4 else -1
            k = 1 if action_type == object_type == 1 else -1
            subject = "I" if o == 1 else "my colleague"
            auxiliary = "did" if a == 1 else "would"
            prefix = "On this occasion, " if a == 1 else "In an imagined event, "
            text = f"{prefix}{subject} {auxiliary}{' not' if p == -1 else ''} {verb} {obj}."
            rows.append(
                {
                    "material_id": f"{target}_{ai}_{oi}_{p}_{o}_{a}",
                    "target": target,
                    "action_index": ai,
                    "object_index": oi,
                    "pair_id": f"{target}_{ai}_{oi}",
                    "role": pair_role(ai, oi),
                    "action": verb,
                    "object": obj,
                    "action_type": action_type,
                    "object_type": object_type,
                    "bits": [k, p, o, a],
                    "text": text,
                    "kind": "single",
                }
            )
    return rows


def relation_controls(spec: dict, bank: list[dict]) -> list[dict]:
    """Swap two predicates while preserving lexical content and event roles."""
    lookup = {
        (r["target"], r["action_index"], r["object_index"]): i
        for i, r in enumerate(bank)
        if r["bits"][1:] == [1, 1, 1]
    }
    rows = []
    for target, design in spec.items():
        verbs = design["positive_actions"] + design["negative_actions"]
        for family in range(8):
            vp, vn, op, on = (
                family % 4,
                4 + (family // 2) % 4,
                family % 4,
                4 + family % 4,
            )
            for swapped in (False, True):
                first, second = (vn, vp) if swapped else (vp, vn)
                phrase1, phrase2 = design["objects"][op][0], design["objects"][on][0]
                rows.append(
                    {
                        "material_id": f"{target}_relations_{family}_{int(swapped)}",
                        "target": target,
                        "family": family,
                        "swapped": swapped,
                        "expected_present": not swapped,
                        "event_indices": [
                            lookup[target, first, op],
                            lookup[target, second, on],
                        ],
                        "text": f"On this occasion, I did {verbs[first]} {phrase1} and {verbs[second]} {phrase2}.",
                        "kind": "relation",
                    }
                )
    return rows


def pair_kernel(a, b, ar, br, kind):
    """Contract augmented typed features without allocating a million columns."""
    ka, kb = np.asarray(a) @ np.asarray(ar).T, np.asarray(b) @ np.asarray(br).T
    if kind == "action":
        return 1 + ka
    if kind == "object":
        return 1 + kb
    if kind == "additive":
        return 1 + ka + kb
    if kind == "tensor":
        return (1 + ka) * (1 + kb)
    raise ValueError(f"unknown typed kernel {kind}")


def fit_kernel(a, b, y, kind="tensor", alpha=0.1):
    """Fit a centered source-only product/additive kernel ridge map."""
    a, b, y = (np.asarray(v, float) for v in (a, b, y))
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b) or len(y) != len(a):
        raise ValueError("aligned source observations required")
    k = pair_kernel(a, b, a, b, kind)
    mean = k.mean(axis=0)
    center = float(k.mean())
    kc = k - mean[None, :] - mean[:, None] + center
    bias = float(y.mean())
    dual = np.linalg.solve(kc + alpha * np.eye(len(k)), y - bias)
    return {
        "a": a,
        "b": b,
        "dual": dual,
        "column_mean": mean,
        "grand_mean": np.array(center),
        "bias": np.array(bias),
        "kind": np.array(kind),
        "alpha": np.array(alpha),
    }


def score_kernel(model, a, b):
    """Cold numeric action/object scoring; no labels, identifiers or role input."""
    k = pair_kernel(a, b, model["a"], model["b"], str(model["kind"]))
    centered = (
        k
        - k.mean(axis=1)[:, None]
        - model["column_mean"][None, :]
        + model["grand_mean"]
    )
    return centered @ model["dual"] + model["bias"]


def select_kernel(a, b, y, role, kind):
    """Choose three fixed ridge values using validation combinations only."""
    src, val = np.asarray(role) == "source", np.asarray(role) == "validation"
    if not src.any() or not val.any():
        raise ValueError("source and validation combinations required")
    candidates = []
    for alpha in (0.01, 0.1, 1.0):
        model = fit_kernel(a[src], b[src], y[src], kind, alpha)
        loss = float(np.mean((score_kernel(model, a[val], b[val]) - y[val]) ** 2))
        candidates.append((loss, alpha, model))
    loss, alpha, model = min(candidates, key=lambda r: r[0])
    return model, {
        "alpha": alpha,
        "validation_mse": loss,
        "source_pairs": int(src.sum()),
        "validation_pairs": int(val.sum()),
    }


def product_scores(z, relevance=None):
    """Combine observed qualification and polarity without filling absent authors."""
    z = np.asarray(z, float).copy()
    if relevance is not None:
        z[:, 0] = np.asarray(relevance)
    gates = np.prod(np.clip((z[:, [0, 2, 3]] + 1) / 2, 0, 1), axis=1)
    return gates * z[:, 1], z


def binding_metrics(score, z, bits):
    """Separate relevance ranking, qualification and contribution calibration."""
    bits = np.asarray(bits)
    expected = L.ideal_contribution(bits)[:, 0]
    q = np.all(bits[:, [0, 2, 3]] == 1, axis=1)
    pred = L.qualify(z)
    result = {
        "n": len(score),
        "unit_rmse": float(np.sqrt(np.mean((score - expected) ** 2))),
        "relevance_rmse": float(np.sqrt(np.mean((z[:, 0] - bits[:, 0]) ** 2))),
        "relevance_auc": float(roc_auc_score(bits[:, 0] == 1, z[:, 0])),
        "tp": int(np.sum(pred & q)),
        "fp": int(np.sum(pred & ~q)),
        "fn": int(np.sum(~pred & q)),
        "tn": int(np.sum(~pred & ~q)),
        "unqualified_rms": float(np.sqrt(np.mean(score[~q] ** 2))),
    }
    m = (bits[:, 0] == -1) & np.all(bits[:, [2, 3]] == 1, axis=1)
    result["irrelevant_polarity_gain"] = float(2 * np.mean(score[m] * bits[m, 1]))
    return result


def dependency_slots(sentences: list[list[dict]], text: str) -> list[dict]:
    """Extract direct-object argument subtrees, never search a target verb list.

    Main and conjoined predicates remain separate; reduced object modifiers are
    retained inside the object, not promoted to an independent finite event.
    Argument text is a contiguous raw span. This is a scoped extraction rule.
    No evidence polarity/ownership truth is injected by this parser.
    """
    events = []
    for words in sentences:
        children = {w["id"]: [] for w in words}
        by_id = {w["id"]: w for w in words}
        for w in words:
            if w["head"] in children:
                children[w["head"]].append(w["id"])
        main = {w["id"] for w in words if w["upos"] == "VERB" and w["deprel"] == "root"}
        while True:
            expanded = main | {
                w["id"]
                for w in words
                if w["upos"] == "VERB" and w["deprel"] == "conj" and w["head"] in main
            }
            if expanded == main:
                break
            main = expanded
        for verb in words:
            if verb["id"] not in main:
                continue
            objects = [
                by_id[i] for i in children[verb["id"]] if by_id[i]["deprel"] == "obj"
            ]
            for obj in objects:
                ids, stack = set(), [obj["id"]]
                while stack:
                    current = stack.pop()
                    ids.add(current)
                    stack.extend(children[current])
                start = min(by_id[i]["start"] for i in ids)
                end = max(by_id[i]["end"] for i in ids)
                events.append(
                    {
                        "action": verb["lemma"],
                        "object": text[start:end],
                        "action_span": [verb["start"], verb["end"]],
                        "object_span": [start, end],
                    }
                )
    return events


def exact_binding_witness():
    """Show an additive readout cannot realize an AND interaction exactly."""
    a, b = np.array(list(product((-1, 1), repeat=2))).T
    y = np.where((a == 1) & (b == 1), 1.0, -1.0)
    add = np.c_[np.ones(4), a, b]
    tensor = np.c_[add, a * b]
    ap = add @ np.linalg.lstsq(add, y, rcond=None)[0]
    tp = tensor @ np.linalg.solve(tensor, y)
    return {
        "additive_rmse": float(np.sqrt(np.mean((ap - y) ** 2))),
        "tensor_rmse": float(np.sqrt(np.mean((tp - y) ** 2))),
        "interaction_contrast": float(y[3] - y[2] - y[1] + y[0]),
        "scope": "known conjunction witness, not a learned psychological truth",
    }
