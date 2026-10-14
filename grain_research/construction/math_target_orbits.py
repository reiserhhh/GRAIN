#!/usr/bin/env python3
"""Target-content/ownership orbits and exact character response components.

The group acts on deliberately constructed text addresses. A frozen observer
need not be equivariant: its lifted response on the complete orbit is decomposed
without discarding any component or identifying a psychological parameter.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


def wordbag(text):
    """Return a case-folded word multiset for same-word relation controls."""
    return Counter(re.findall(r"\b\w+\b", text.casefold()))


def character_basis(dimension=5):
    """Return the real characters of (Z/2Z)^dimension in mask order."""
    if not isinstance(dimension, int) or not 1 <= dimension <= 8:
        raise ValueError("bounded positive finite-product dimension required")
    n = 1 << dimension
    return np.array(
        [
            [(-1.0) ** ((cell & term).bit_count()) for term in range(n)]
            for cell in range(n)
        ]
    )


def decompose(values):
    """Keep every orthogonal character coefficient of a complete response law."""
    f = np.asarray(values, float)
    n = len(f) if f.ndim else 0
    if n < 2 or n & (n - 1) or not np.isfinite(f).all():
        raise ValueError("finite mask-indexed complete orbit required")
    basis = character_basis(n.bit_length() - 1)
    flat = f.reshape(n, -1)
    coefficients = basis.T @ flat / n
    reconstruction = basis @ coefficients
    centered = flat - flat.mean(axis=0)
    energies = np.sum(coefficients**2, axis=1)
    return {
        "coefficients": coefficients.reshape(f.shape),
        "energies": energies,
        "reconstruction_error": float(np.max(abs(reconstruction - flat))),
        "energy_closure_error": abs(
            float(energies[1:].sum() - np.mean(np.sum(centered**2, axis=1)))
        ),
    }


def source_coefficients(tables):
    """Fit shared conditional response components from source families only."""
    source = [r for r in tables if r["role"] == "source"]
    if not source or len({r["family"] for r in source}) != len(source):
        raise ValueError("distinct source families required")
    return np.mean([decompose(r["values"])["coefficients"] for r in source], axis=0)


def anchored_completion(anchor, coefficients, terms, wrong_binding=False):
    """Transport selected source components around one actual cold anchor.

    The method requires cold cell zero. It is not a single-text psychological
    prediction or a model fit using the remaining application outcomes.
    """
    c = np.asarray(coefficients, float)
    basis = character_basis(len(c).bit_length() - 1)
    a = np.asarray(anchor, float)
    if c.shape[1:] != a.shape or not np.isfinite(c).all() or not np.isfinite(a).all():
        raise ValueError("matched finite anchor and source response required")
    terms = list(terms)
    if len(set(terms)) != len(terms) or any(t <= 0 or t >= len(c) for t in terms):
        raise ValueError("distinct nonconstant character addresses required")
    if not terms:
        return np.broadcast_to(a, c.shape).copy()
    weights = basis[:, terms] - basis[0, terms]
    if wrong_binding:
        weights[:, [i for i, term in enumerate(terms) if term in (5, 10)]] *= -1
    return a + (weights @ c[terms].reshape(len(terms), -1)).reshape(c.shape)


def material_bank(spec):
    """Construct nonaliased five-factor texts plus missing-content controls."""
    families = spec["families"]
    if len(families) != 16 or spec["source_validation_application_families"] != [
        8,
        4,
        4,
    ]:
        raise ValueError("frozen 8/4/4 family design required")
    rows = []

    def sentence(owner, action, filler):
        actor, other = ("I", "my colleague") if owner == 0 else ("my colleague", "I")
        raw = f"{actor} {action} while {other} {filler}."
        return raw[0].upper() + raw[1:]

    for family, (
        context,
        high_a,
        low_a,
        high_t,
        low_t,
        filler_a,
        filler_t,
    ) in enumerate(families):
        role = (
            "source" if family < 8 else "validation" if family < 12 else "application"
        )

        def append(
            kind,
            a,
            t,
            u,
            v,
            order,
            mask=None,
            context=context,
            high_a=high_a,
            low_a=low_a,
            high_t=high_t,
            low_t=low_t,
            filler_a=filler_a,
            filler_t=filler_t,
            family=family,
            role=role,
        ):
            sentences = []
            if a is not None:
                sentences.append(
                    sentence(u, f"felt {high_a if a else low_a}", filler_a)
                )
            if t is not None:
                sentences.append(
                    sentence(
                        v,
                        f"judged AI systems generally {high_t if t else low_t}",
                        filler_t,
                    )
                )
            if order:
                sentences.reverse()
            body = (
                " ".join(sentences)
                if sentences
                else "I noted the time while my colleague checked the door."
            )
            text = f"{context}, {body}"
            # Strip only the constructed sentence's initial capitalization after a comma.
            # Pronoun I remains uppercase; no natural data are normalized by this generator.
            text = text.replace(", My colleague", ", my colleague")
            evidence = {
                "adapted_na8": "direct_expression_subset"
                if a is not None and u == 0
                else "unobserved_direct_self_evidence",
                "adapted_dejected5": "unobserved_direct_self_evidence",
                "pre_trust_positive": "direct_general_view_subset"
                if t is not None and v == 0
                else "unobserved_direct_self_evidence",
            }
            rows.append(
                {
                    "material_id": f"target_orbit_{len(rows):04d}",
                    "family": family,
                    "role": role,
                    "kind": kind,
                    "mask": mask,
                    "anxiety_polarity": a,
                    "trust_polarity": t,
                    "anxiety_owner": u,
                    "trust_owner": v,
                    "sentence_order": order,
                    "text": text,
                    "evidence_status": evidence,
                    "human_author_id": None,
                    "psychological_truth_supplied": False,
                }
            )

        for mask in range(32):
            append(
                "full",
                mask & 1,
                (mask >> 1) & 1,
                (mask >> 2) & 1,
                (mask >> 3) & 1,
                (mask >> 4) & 1,
                mask,
            )
        for order in range(2):
            for polarity in range(2):
                for owner in range(2):
                    append("anxiety_only", polarity, None, owner, None, order)
                    append("trust_only", None, polarity, None, owner, order)
            append("neither", None, None, None, None, order)
    for family in range(16):
        full = [r for r in rows if r["family"] == family and r["kind"] == "full"]
        if len({r["text"] for r in full}) != 32:
            raise ValueError("five-factor design aliases distinct cells")
        for polarity_mask in range(4):
            orbit = [r for r in full if r["mask"] & 3 == polarity_mask]
            if len(orbit) != 8 or any(
                wordbag(r["text"]) != wordbag(orbit[0]["text"]) for r in orbit
            ):
                raise ValueError("ownership/order orbit changes words")
    return rows
