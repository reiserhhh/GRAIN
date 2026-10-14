#!/usr/bin/env python3
"""Versioned natural-language evidence anchors and explicit review-based feedback.

SIMPA motivates anchors closer to natural expressions. This implementation keeps
episodic observations distinct from generalized self-descriptions and requires
independent source-side review, not outcome improvement, before promotion.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from grain_research.predecessors import FACETS, LETTERS, TRAITS


def fingerprint(value):
    """Hash canonical JSON for immutable bank and proposal lineage."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def expand_bank(config):
    """Expand a public development specification without accessing text or outcomes."""
    scopes = config["scopes"]
    if scopes != ["habitual_plain", "habitual_colloquial", "episode"]:
        raise ValueError("Unknown anchor scope specification")
    output = []
    for entry in config["facets"]:
        facet = entry["facet"]
        if facet not in FACETS:
            raise ValueError("Unknown facet")
        for direction, key in (("positive", 1), ("negative", -1)):
            if len(entry[direction]) != len(scopes):
                raise ValueError("Missing anchor scope")
            for scope, text in zip(scopes, entry[direction], strict=True):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty anchor")
                output.append(
                    {
                        "id": f"LOCAL1_{facet}_{direction}_{scope}",
                        "text": text,
                        "facet": facet,
                        "key": key,
                        "trait": TRAITS[LETTERS.index(facet[0])],
                        "scope": "episode" if scope == "episode" else "habitual",
                        "wording": scope,
                        "provenance": config["provenance"],
                        "expert_validated": False,
                    }
                )
    if len({r["id"] for r in output}) != len(output) or len(
        {r["text"] for r in output}
    ) != len(output):
        raise ValueError("Duplicate anchor")
    return output


def retrieve(vectors, anchor_vectors, anchors, threshold=0.6):
    """Return one top match per candidate, never call its cosine a probability."""
    h, q = np.asarray(vectors, float), np.asarray(anchor_vectors, float)
    if (
        h.ndim != 2
        or q.ndim != 2
        or h.shape[1] != q.shape[1]
        or len(q) != len(anchors)
        or not len(q)
    ):
        raise ValueError("Invalid retrieval dimensions")
    if not -1 <= threshold <= 1 or not np.isfinite(h).all() or not np.isfinite(q).all():
        raise ValueError("Invalid vectors or threshold")
    hn, qn = np.linalg.norm(h, axis=1), np.linalg.norm(q, axis=1)
    if (hn < 1e-12).any() or (qn < 1e-12).any():
        raise ValueError("Zero vector")
    score = (h / hn[:, None]) @ (q / qn[:, None]).T
    ix = score.argmax(axis=1)
    cosine = score[np.arange(len(h)), ix]
    return ix, cosine, cosine >= threshold


def promote_feedback(bank, proposals, reviews, *, require_human=True, min_raters=2):
    """Append only reviewed source proposals, with model-only promotion provisional.

    Review votes refer to a content hash, so editing a proposal invalidates old votes.
    Any negative vote prevents promotion; held-out proposals are rejected outright.
    """
    if min_raters < 2:
        raise ValueError("At least two independent review identities required")
    ids = {r["id"] for r in bank}
    proposal_ids = [p["id"] for p in proposals]
    if len(set(proposal_ids)) != len(proposal_ids) or ids.intersection(proposal_ids):
        raise ValueError("Duplicate proposal identity")
    texts = [p["text"].strip().casefold() for p in proposals]
    if len(set(texts)) != len(texts) or {
        r["text"].strip().casefold() for r in bank
    }.intersection(texts):
        raise ValueError("Duplicate proposal text")
    known = {p["id"]: p for p in proposals}
    seen = set()
    for r in reviews:
        pair = (r["proposal_id"], r["rater_id"])
        if pair in seen or not r["rater_id"] or r["proposal_id"] not in known:
            raise ValueError("Unknown proposal or duplicate rater")
        seen.add(pair)
        if r["proposal_sha256"] != fingerprint(known[r["proposal_id"]]):
            raise ValueError("Changed proposal under existing reviews")
        if r["rater_kind"] not in {"human", "model"} or type(r["supports"]) is not bool:
            raise ValueError("Invalid review provenance or verdict")
    result, decisions = list(bank), []
    for p in proposals:
        if p["source_partition"] != "development":
            raise ValueError("Feedback from held-out/validation material is forbidden")
        if (
            p["facet"] not in FACETS
            or type(p["key"]) is not int
            or p["key"] not in {-1, 1}
            or p["scope"] not in {"habitual", "episode"}
            or not p["text"].strip()
        ):
            raise ValueError("Invalid proposed anchor")
        if p.get("criterion_selected", False):
            raise ValueError("Criterion-selected evidence is not semantic feedback")
        votes = [r for r in reviews if r["proposal_id"] == p["id"]]
        eligible = [r for r in votes if not require_human or r["rater_kind"] == "human"]
        accepted = len(eligible) >= min_raters and all(r["supports"] for r in votes)
        decisions.append(
            {"id": p["id"], "promoted": accepted, "eligible_raters": len(eligible)}
        )
        if accepted:
            result.append(
                {
                    **p,
                    "trait": TRAITS[LETTERS.index(p["facet"][0])],
                    "provenance": "reviewed_local_feedback_not_original_eTRS",
                    "expert_validated": False,
                    "human_reviewed": all(r["rater_kind"] == "human" for r in eligible),
                    "review_sha256": fingerprint(votes),
                }
            )
    return result, decisions
