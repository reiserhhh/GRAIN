#!/usr/bin/env python3
"""Aggregate conditional pair updates and crossed author/event transport."""

import numpy as np

VERSION = "HM192_aggregate_pair_update_v1"
MODELS = ("profile_product", "same_author")


def distributions(cells):
    """Convert prescribed distinct events into probabilities, not iid samples."""
    x = np.asarray(cells)
    if x.ndim != 3 or x.shape[1] != 2 or not np.isin(x, np.arange(8)).all():
        raise ValueError("Two windows of eight-cell observations required")
    if x.shape[2] == 0:
        raise ValueError("Empty event subset")
    return np.eye(8)[x.astype(int)].mean(2)


def source_laws(p, codes, prior):
    """Keep the full profile prior and singleton laws in both mechanisms."""
    p, c, pi = np.asarray(p, float), np.asarray(codes), np.asarray(prior, float)
    if (
        p.shape != (len(c), 2, 8)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(2), 1, atol=2e-12, rtol=0)
        or not np.isin(c, np.arange(32)).all()
        or pi.shape != (32,)
        or np.any(pi <= 0)
        or not np.isfinite(pi).all()
        or not np.isclose(pi.sum(), 1)
    ):
        raise ValueError("Invalid source profile/event law")
    law = np.zeros((2, 32, 8, 8))
    for g in range(32):
        x = p[c == g]
        if not len(x):
            raise ValueError("Missing full profile; no smoothing")
        a, b = x.mean(0)
        law[0, g] = pi[g] * np.outer(a, b)
        law[1, g] = pi[g] * (x[:, 0].T @ x[:, 1] / len(x))
    return law


def update_actions(laws):
    """Disintegrate a two-event law without assuming recursive history closure."""
    law = np.asarray(laws, float)
    if (
        law.shape != (2, 32, 8, 8)
        or not np.isfinite(law).all()
        or np.any(law < 0)
        or not np.allclose(law.sum((1, 2, 3)), 1, atol=2e-12, rtol=0)
    ):
        raise ValueError("Two normalized full-profile/event laws required")
    labels = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
    joint = np.einsum("cg,mcab->mgab", labels, law)
    total = law.sum(1)
    first = total.sum(-1)
    high_first = joint.sum(-1)
    before = np.divide(
        high_first,
        first[:, None],
        out=np.full_like(high_first, np.nan),
        where=first[:, None] > 0,
    )
    after = np.divide(
        joint,
        total[:, None],
        out=np.full_like(joint, np.nan),
        where=total[:, None] > 0,
    )
    increment = after - before[..., None]
    gamma_numerator = joint - before[..., None] * total[:, None]
    return {
        "before": before,
        "after": after,
        "increment": increment,
        "gamma_numerator": gamma_numerator,
        "pair_mass": total,
        "relation": after[1] - after[0],
        "supported": (total > 0).all(0),
    }


def compile_fields(p, codes, folds, prior):
    """Construct four separately estimated actions, not a tuned common field."""
    p, c, f = np.asarray(p), np.asarray(codes), np.asarray(folds)
    if p.shape != (len(c), 2, 2, 8) or f.shape != c.shape:
        raise ValueError("Author x disjoint-experiment x window x cell required")
    if not np.isin(f, (0, 1)).all():
        raise ValueError("Exactly two inherited author banks required")
    laws = np.array(
        [[source_laws(p[f == s, v], c[f == s], prior) for v in (0, 1)] for s in (0, 1)]
    )
    actions = [[update_actions(laws[s, v]) for v in (0, 1)] for s in (0, 1)]
    fields = np.array([[a["relation"] for a in bank] for bank in actions])
    supported = np.array([[a["supported"] for a in bank] for bank in actions])
    return laws, fields, supported


def transport(fields, supported, reference):
    """Keep signed cross-person action accounts and unsupported reference mass."""
    h, support, rho = (
        np.asarray(fields, float),
        np.asarray(supported, bool),
        np.asarray(reference, float),
    )
    if h.shape != (2, 2, 5, 8, 8) or support.shape != (2, 2, 8, 8):
        raise ValueError("Four target-action fields and support required")
    if rho.shape != (8, 8) or np.any(rho < 0) or not np.isclose(rho.sum(), 1):
        raise ValueError("Fixed normalized observation reference required")
    common = support.all((0, 1))
    if not np.isfinite(h[..., common]).all():
        raise ValueError("Invalid action on supported pairs")
    x = np.where(common[None, None, None], h, 0)
    w = rho * common
    cross = 0.5 * (x[0, 0] * x[1, 1] + x[0, 1] * x[1, 0])
    same_text = 0.5 * (x[0, 0] * x[1, 0] + x[0, 1] * x[1, 1])
    same_bank = 0.5 * (x[0, 0] * x[0, 1] + x[1, 0] * x[1, 1])
    target = (cross * w).sum((1, 2))
    by_condition = np.array(
        [
            (
                cross[:, 4 * a : 4 * a + 4, 4 * b : 4 * b + 4]
                * w[4 * a : 4 * a + 4, 4 * b : 4 * b + 4]
            ).sum((1, 2))
            for a in (0, 1)
            for b in (0, 1)
        ]
    )
    gram = 0.5 * (
        np.einsum("qab,rab,ab->qr", x[0, 0], x[1, 1], w)
        + np.einsum("qab,rab,ab->qr", x[0, 1], x[1, 0], w)
    )
    gram = (gram + gram.T) / 2
    scale = np.sqrt(
        (0.5 * (x[0, 0] ** 2 + x[0, 1] ** 2) * w).sum((1, 2))
        * (0.5 * (x[1, 0] ** 2 + x[1, 1] ** 2) * w).sum((1, 2))
    )
    return {
        "primary": float(target.mean()),
        "by_target": target,
        "within_bank_disjoint_text": (same_bank * w).sum((1, 2)),
        "cross_bank_same_text_pattern": (same_text * w).sum((1, 2)),
        "coherence": np.divide(target, scale, out=np.full(5, np.nan), where=scale > 0),
        "by_condition": by_condition,
        "signed_cross_gram": gram,
        "supported": common,
        "reference_mass": float(w.sum()),
        "supported_pair_count": int(common.sum()),
    }


def permute_late(p, codes, folds, rng):
    """Break cross-window author identity, preserving both late-quarter links."""
    x, c, f = np.asarray(p), np.asarray(codes), np.asarray(folds)
    perm = np.arange(len(c))
    for bank in (0, 1):
        for g in range(32):
            ids = np.flatnonzero((f == bank) & (c == g))
            perm[ids] = rng.permutation(ids)
    mixed = x.copy()
    mixed[:, :, 1] = x[perm, :, 1]
    return mixed, perm


def make_bundle(laws):
    """Export only aggregate observation laws, never source author fingerprints."""
    update_actions(laws)
    return {
        "version": VERSION,
        "models": list(MODELS),
        "aggregate_profile_event_laws": np.asarray(laws).tolist(),
        "interpretation": "qualified_two_event_update_NOT_recursive_history_or_personality_change",
        "smoothing": 0,
    }


def query_pair(bundle, events):
    """Evaluate one qualified pair; copied observations cannot be new evidence."""
    if bundle.get("version") != VERSION or set(bundle) != {
        "version",
        "models",
        "aggregate_profile_event_laws",
        "interpretation",
        "smoothing",
    }:
        raise ValueError("Unknown or author-catalogue bundle")
    if bundle["models"] != list(MODELS) or bundle["smoothing"] != 0:
        raise ValueError("Changed observation models")
    if len(events) != 2:
        raise ValueError("This law supports exactly two qualified events")
    for window, event in enumerate(events):
        if (
            set(event) != {"event_id", "window", "cell"}
            or not isinstance(event["event_id"], str)
            or not event["event_id"]
            or type(event["window"]) is not int
            or event["window"] != window
            or type(event["cell"]) is not int
            or not 0 <= event["cell"] < 8
        ):
            raise ValueError("Distinct window/event/cell addresses required")
    if events[0]["event_id"] == events[1]["event_id"]:
        raise ValueError("The same event cannot occupy both evidence roles")
    a, b = (e["cell"] for e in events)
    v = update_actions(bundle["aggregate_profile_event_laws"])
    return {
        "supported": bool(v["supported"][a, b]),
        "before": v["before"][:, :, a],
        "after": v["after"][:, :, a, b],
        "increment": v["increment"][:, :, a, b],
        "relation_increment": v["relation"][:, a, b],
    }
