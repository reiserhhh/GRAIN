#!/usr/bin/env python3
"""Targeted behavioral contrasts for old query functions, without human labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reader_links(definitions, specification):
    """Fix target, related and distant readers independently of model output."""
    if set(definitions) != set(specification["related"]):
        raise ValueError("Related reader coverage must equal the full target set")
    links = []
    for target, definition in definitions.items():
        family = definition.get("family", target)
        distant = "adapted_na8" if family.startswith("ai_mind") else "pre_anthro_agency"
        readers = {
            "target": target,
            "related": specification["related"][target],
            "distant": distant,
        }
        if len(set(readers.values())) != 3 or not set(readers.values()) <= set(
            definitions
        ):
            raise ValueError("Each target needs three distinct existing queries")
        links.extend(
            {"target": target, "query": q, "reader_role": role}
            for role, q in readers.items()
        )
    return links


def build_worlds(definitions, specification):
    """Construct paired authored posts; ownership changes only the outer wrapper."""
    worlds = []
    for target, definition in definitions.items():
        family = definition.get("family", target)
        pairs = specification["pairs"].get(family)
        if not isinstance(pairs, list) or len(pairs) != 2:
            raise ValueError("Each family needs exactly two evidence pairs")
        for pair, bodies in enumerate(pairs):
            if (
                len(bodies) != 2
                or any(not isinstance(t, str) or not t.strip() for t in bodies)
                or bodies[0] == bodies[1]
            ):
                raise ValueError("Distinct nonempty high/low bodies required")
            for polarity, body in zip(("high", "low"), bodies, strict=True):
                for owner in ("self", "other"):
                    wrapper = (
                        "This is my own account, not a friend's account: "
                        if owner == "self"
                        else "This is a friend's account, not my own account: "
                    )
                    worlds.append(
                        {
                            "world_id": f"{target}_p{pair}_{polarity}_{owner}",
                            "target": target,
                            "family": family,
                            "pair": pair,
                            "polarity": polarity,
                            "owner": owner,
                            "evidence_body": body,
                            "texts": [wrapper + '"' + body + '"'],
                            "synthetic": True,
                        }
                    )
    return worlds


def paired_response_tables(scores):
    """Form exact high-low and ownership contrasts, rejecting unmatched packets."""
    keys = ["target", "query", "reader_role", "pair", "owner", "readout"]
    required = {*keys, "polarity", "score", "mean", "configuration", "final"}
    if not required <= set(scores):
        raise ValueError("Missing response columns")
    if scores.duplicated(keys + ["polarity"]).any():
        raise ValueError("Duplicate paired score identity")
    if set(scores["readout"]) != {"native", "affine"}:
        raise ValueError("Both native and affine readouts required")
    if not np.isfinite(scores["score"].to_numpy(float)).all():
        raise ValueError("Nonfinite response score")
    rows = []
    for identity, group in scores.groupby(keys, sort=True):
        if set(group["polarity"]) != {"high", "low"} or len(group) != 2:
            raise ValueError("Incomplete polarity pair")
        indexed = group.set_index("polarity")
        row = dict(zip(keys, identity, strict=True))
        row["delta"] = float(indexed.loc["high", "score"] - indexed.loc["low", "score"])
        for part in ("mean", "configuration", "final"):
            row[part] = (
                float(indexed.loc["high", part] - indexed.loc["low", part])
                if row["readout"] == "affine"
                else np.nan
            )
        if row["readout"] == "affine" and (
            not np.isfinite([row[p] for p in ("mean", "configuration", "final")]).all()
            or abs(
                row["delta"] - sum(row[p] for p in ("mean", "configuration", "final"))
            )
            > 1e-10
        ):
            raise ValueError("Affine components do not sum to endpoint response")
        rows.append(row)
    responses = pd.DataFrame(rows)
    pair_keys = keys[:-1]
    native = responses[responses.readout == "native"].set_index(pair_keys)
    affine = responses[responses.readout == "affine"].set_index(pair_keys)
    if set(native.index) != set(affine.index):
        raise ValueError("Unmatched native and affine pairs")
    paired = (
        native[["delta"]]
        .rename(columns={"delta": "native"})
        .join(
            affine[["delta", "mean", "configuration", "final"]].rename(
                columns={"delta": "affine"}
            )
        )
    )
    paired["error"] = paired.affine - paired.native
    paired = paired.reset_index()
    attribution = []
    for identity, group in paired.groupby(pair_keys[:-1], sort=True):
        if set(group.owner) != {"self", "other"} or len(group) != 2:
            raise ValueError("Incomplete ownership contrast")
        indexed = group.set_index("owner")
        row = dict(zip(pair_keys[:-1], identity, strict=True))
        for name in ("native", "affine", "mean", "configuration", "final"):
            row[name] = float(indexed.loc["self", name] - indexed.loc["other", name])
        attribution.append(row)
    return paired, pd.DataFrame(attribution)


def response_quality(native, affine):
    """Describe fidelity in model units, with no population inference from templates."""
    native, affine = np.asarray(native, float), np.asarray(affine, float)
    if (
        native.ndim != 1
        or native.shape != affine.shape
        or not len(native)
        or not np.isfinite([native, affine]).all()
    ):
        raise ValueError("Finite matched response vectors required")
    informative = np.abs(native) > 0.01
    energy = float(native @ native)
    return {
        "pairs": len(native),
        "native_mean": float(native.mean()),
        "affine_mean": float(affine.mean()),
        "native_positive": int((native > 0).sum()),
        "affine_positive": int((affine > 0).sum()),
        "native_resolution_count": int(informative.sum()),
        "sign_agreement_over_resolution": (
            float((np.sign(native[informative]) == np.sign(affine[informative])).mean())
            if informative.any()
            else None
        ),
        "response_r": (
            float(np.corrcoef(native, affine)[0, 1])
            if len(native) >= 3 and native.std() > 1e-12 and affine.std() > 1e-12
            else None
        ),
        "mae": float(np.abs(affine - native).mean()),
        "rmse": float(np.sqrt(np.mean((affine - native) ** 2))),
        "bias": float((affine - native).mean()),
        "amplitude_slope": float(native @ affine / energy) if energy > 1e-20 else None,
    }
