#!/usr/bin/env python3
"""Disintegrate pair target action without deleting condition-selection signals."""

import numpy as np

from grain_research.construction import math_transfer_update as T

VERSION = "HM193_route_preserving_pair_law_v1"
MODELS = ("profile_product", "route_preserving", "same_author")
COMPONENTS = ("routing_marginals", "within_route_relation", "total")


def route_product(joint):
    """Keep profile-route masses and both conditional expression marginals."""
    j = np.asarray(joint, float)
    if (
        j.shape != (32, 8, 8)
        or not np.isfinite(j).all()
        or np.any(j < 0)
        or not np.isclose(j.sum(), 1, atol=2e-12, rtol=0)
    ):
        raise ValueError("Normalized full-profile joint law required")
    r = np.zeros_like(j)
    for a in (0, 1):
        for b in (0, 1):
            sa, sb = slice(4 * a, 4 * a + 4), slice(4 * b, 4 * b + 4)
            block = j[:, sa, sb]
            mass = block.sum((1, 2))
            numerator = block.sum(2)[:, :, None] * block.sum(1)[:, None, :]
            r[:, sa, sb] = np.divide(
                numerator,
                mass[:, None, None],
                out=np.zeros_like(numerator),
                where=mass[:, None, None] > 0,
            )
    return r


def three_laws(pair_laws):
    """Add a constrained middle law without changing either old mechanism."""
    x = np.asarray(pair_laws, float)
    T.update_actions(x)
    return np.stack((x[0], route_product(x[1]), x[1]))


def actions(laws):
    """Keep target-reading changes from routing and conditional relations separate."""
    p, r, j = np.asarray(laws, float)
    routed = T.update_actions(np.stack((p, r)))
    within = T.update_actions(np.stack((r, j)))
    total = T.update_actions(np.stack((p, j)))
    supported = routed["supported"] & within["supported"]
    return {
        "fields": np.stack((routed["relation"], within["relation"], total["relation"])),
        "supported": supported,
        "before": np.stack(
            (routed["before"][0], routed["before"][1], within["before"][1])
        ),
        "after": np.stack((routed["after"][0], routed["after"][1], within["after"][1])),
        "pair_mass": np.stack((p.sum(0), r.sum(0), j.sum(0))),
    }


def divergences(laws):
    """Finite KL accounts; zeros are structural and never smoothed or clipped."""
    p, r, j = np.asarray(laws, float)

    def kl(a, b):
        take = a > 0
        if np.any(b[take] == 0):
            raise ValueError("Reference omits observed mass")
        return float(np.sum(a[take] * np.log(a[take] / b[take])))

    total, within, routing = kl(j, p), kl(j, r), kl(r, p)
    return {"total": total, "within": within, "routing": routing}


def compile_fields(p, codes, folds, prior):
    """Use inherited author/event banks, not a new optimized conditional grid."""
    old, _, _ = T.compile_fields(p, codes, folds, prior)
    laws = np.array([[three_laws(x) for x in bank] for bank in old])
    values = [[actions(x) for x in bank] for bank in laws]
    fields = np.array([[x["fields"] for x in bank] for bank in values])
    support = np.array([[x["supported"] for x in bank] for bank in values])
    return laws, fields, support


def transport(fields, supported, reference):
    """Compare component actions and retain signed, nonorthogonal cross terms."""
    h = np.asarray(fields, float)
    if h.shape != (2, 2, 3, 5, 8, 8):
        raise ValueError("Four banks of three five-target action fields required")
    results = [T.transport(h[:, :, i], supported, reference) for i in range(3)]
    common = np.asarray(supported).all((0, 1))
    w = np.asarray(reference) * common
    x = np.where(common[None, None, None, None], h[:, :, :2], 0)
    crossed = 0.5 * (
        np.einsum("iqab,jqab,ab->qij", x[0, 0], x[1, 1], w)
        + np.einsum("iqab,jqab,ab->qij", x[0, 1], x[1, 0], w)
    )
    return {"components": results, "crossed_accounts": crossed}


def make_bundle(laws):
    """Export only aggregate conditional observation laws and their fixed scope."""
    actions(laws)
    return {
        "version": VERSION,
        "models": list(MODELS),
        "aggregate_profile_event_laws": np.asarray(laws).tolist(),
        "condition": "cell//4_venue_bit_NOT_observed_volition_or_actor",
        "scope": "two_qualified_events_NOT_recursive_history_or_causal_response",
        "smoothing": 0,
    }


def query_pair(bundle, events):
    """Read every law on one event pair; provenance is caller-owned, not inferred."""
    if bundle.get("version") != VERSION or set(bundle) != {
        "version",
        "models",
        "aggregate_profile_event_laws",
        "condition",
        "scope",
        "smoothing",
    }:
        raise ValueError("Unknown or author-catalogue bundle")
    if bundle["models"] != list(MODELS) or bundle["smoothing"] != 0:
        raise ValueError("Changed models")
    laws = np.asarray(bundle["aggregate_profile_event_laws"])
    T.query_pair(T.make_bundle(laws[[0, 2]]), events)
    a, b = (e["cell"] for e in events)
    v = actions(laws)
    return {
        "supported": bool(v["supported"][a, b]),
        "route": [a // 4, b // 4],
        "before": v["before"][:, :, a],
        "after": v["after"][:, :, a, b],
        "components": v["fields"][:, :, a, b],
    }
