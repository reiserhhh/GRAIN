"""R32: construct admission rule, triangulation model and dynamic-parameter declarations.

Sources: P04 pp.16-17 (admission: at least one of four conditions, after a replicated
phenomenon), P03 pp.15-16 and 25-29 (ontology; Y_q, Y_b, Y_t as imperfect indicators),
P03 p.28 (operation, rival, prediction, boundary, criterion per dynamic parameter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONDITIONS = ("explains_multiple_phenomena", "novel_falsifiable_prediction",
              "unique_experimental_signature", "incremental_cross_situational_prediction")
SHORTCUTS = ("topic", "demographic", "language_proficiency", "social_desirability")
ONTOLOGIES = ("latent_cause", "dispositional", "dynamic_state", "network_emergent",
              "predictive_state", "formative_index")


@dataclass
class Candidate:
    name: str
    ontology: str
    replicated_phenomenon: bool
    conditions: dict = field(default_factory=dict)
    shortcut_controls: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    domain: str = "psychological"


def admit(c: Candidate) -> dict:
    """Admit a psychological candidate only after replication and at least one source condition."""
    if c.domain != "psychological":
        return {"name": c.name, "admitted": False, "conditions_met": [],
                "reasons": ["instrument phenomenon: outside construct admission"],
                "status": "INSTRUMENT_PHENOMENON_NOT_A_CONSTRUCT"}
    reasons = []
    if c.ontology not in ONTOLOGIES:
        reasons.append("ontology not declared")
    if not c.replicated_phenomenon:
        reasons.append("regularity not replicated as a phenomenon")
    met = [k for k in CONDITIONS if c.conditions.get(k) is True]
    if "incremental_cross_situational_prediction" in met:
        missing = [s for s in SHORTCUTS if c.shortcut_controls.get(s) is not True]
        if missing:
            met.remove("incremental_cross_situational_prediction")
            reasons.append("increment not controlled for " + ", ".join(missing))
    if not met:
        reasons.append("no admission condition met")
    admitted = c.ontology in ONTOLOGIES and c.replicated_phenomenon and bool(met)
    return {"name": c.name, "admitted": admitted, "conditions_met": met, "reasons": reasons,
            "status": "CANDIDATE_CONSTRUCT" if admitted else "NOT_ADMITTED"}


def triangulate(cov: np.ndarray) -> dict:
    """One-factor triad: loadings from three indicators' covariance (identified up to sign)."""
    cov = np.asarray(cov, float)
    if cov.shape != (3, 3):
        raise ValueError("three indicators required")
    c = cov
    lam = np.array([np.sqrt(max(c[0, 1] * c[0, 2] / c[1, 2], 0.0)),
                    np.sqrt(max(c[0, 1] * c[1, 2] / c[0, 2], 0.0)),
                    np.sqrt(max(c[0, 2] * c[1, 2] / c[0, 1], 0.0))])
    unique = np.diag(c) - lam ** 2
    return {"loadings": lam, "unique_variance": unique,
            "reliability": lam ** 2 / np.diag(c), "heywood": bool(np.any(unique < 0))}


def two_indicator_counterexample(l1: float, l2: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Two loading pairs with identical observable covariance: (l1, l2) and (l1*s, l2/s)."""
    def cov(a, b):
        return np.array([[a * a + 1.0, a * b], [a * b, b * b + 1.0]])
    a, b = cov(l1, l2), cov(l1 * scale, l2 / scale)
    # adjust unique variances so the diagonals also match
    b[0, 0], b[1, 1] = a[0, 0], a[1, 1]
    return a, b


DYNAMIC_FIELDS = ("operation", "rival", "prediction", "boundary", "criterion")


def check_dynamic_parameter(decl: dict) -> list[str]:
    """Return the missing P03 p.28 fields of a dynamic-parameter declaration."""
    return [f for f in DYNAMIC_FIELDS if not decl.get(f)]
