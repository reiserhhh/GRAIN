#!/usr/bin/env python3
"""Phase-aware factor registry and strict, local-only questionnaire score audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from suica_core.meps_layout import normalize_participant_code

from .direct_traits import AXES, BIG5, question


@dataclass(frozen=True)
class Scale:
    """An administered score, with reverse keys relative to selected item order."""

    name: str
    query: str
    items: tuple[str, ...]
    low: int = 1
    high: int = 7
    reverse_positions: tuple[int, ...] = ()
    kind: str = "relational_self_report"
    reliability: bool = True

    @property
    def phase(self):
        """Return the measurement occasion, not an inferred temporal trait."""
        return self.name.split("_", 1)[0]


def scales():
    """Enumerate actual Study 2 scored composites and single indicators."""
    out = []
    for phase in ("pre", "post"):
        for suffix, indices, kind, reliability in (
            ("total", (1, 2, 3, 4, 5), "adapted_relational_composite", True),
            ("agency", (1, 4), "exploratory_item_split", False),
            ("experience", (2, 3, 5), "exploratory_item_split", False),
        ):
            out.append(
                Scale(
                    f"{phase}_anthro_{suffix}",
                    f"ai_mind_{suffix}",
                    tuple(f"{phase}_anthro_{i}" for i in indices),
                    0,
                    10,
                    kind=kind,
                    reliability=reliability,
                )
            )
        for suffix, indices in (("anxiety", range(1, 5)), ("avoidance", range(5, 8))):
            out.append(
                Scale(
                    f"{phase}_attachment_{suffix}",
                    f"ai_attachment_{suffix}",
                    tuple(f"{phase}_attachment_{i}" for i in indices),
                )
            )
    for suffix, indices in (
        ("trust_positive", range(1, 6)),
        ("distrust", range(6, 11)),
    ):
        out.append(
            Scale(
                f"pre_{suffix}",
                f"ai_{suffix}",
                tuple(f"pre_trust_{i}" for i in indices),
            )
        )
    out.extend(
        [
            Scale(
                "pre_leak_concern",
                "ai_privacy_concern",
                ("pre_trust_11",),
                kind="single_indicator",
                reliability=False,
            ),
            Scale(
                "post_perceived_empathy",
                "perceived_ai_empathy",
                tuple(f"post_empathy_{i}" for i in range(1, 14)),
                reverse_positions=(9, 11, 12, 13),
                kind="session_evaluation",
            ),
            Scale(
                "post_session_depth",
                "session_depth",
                tuple(f"post_session_eval_{i}" for i in (3, 4, 7, 8, 9)),
                reverse_positions=(1, 3, 5),
                kind="session_evaluation",
            ),
            Scale(
                "post_session_smoothness",
                "session_smoothness",
                tuple(f"post_session_eval_{i}" for i in (2, 5, 6, 10, 11)),
                reverse_positions=(2, 5),
                kind="session_evaluation",
            ),
            Scale(
                "post_session_global_evaluation",
                "session_global_evaluation",
                ("post_session_eval_1",),
                kind="single_session_indicator",
                reliability=False,
            ),
        ]
    )
    for level in range(1, 5):
        out.append(
            Scale(
                f"post_disclosability_lv{level}",
                f"disclosure_depth_{level}",
                (f"post_disclosability_{level}",),
                kind="ordered_single_indicator",
                reliability=False,
            )
        )
    out.append(
        Scale(
            "post_disclosability_mean_exploratory",
            "disclosure_composite",
            tuple(f"post_disclosability_{i}" for i in range(1, 5)),
            kind="exploratory_nonparallel_mean",
            reliability=False,
        )
    )
    return tuple(out)


def catalog():
    """Describe all available criteria without inventing unmeasured facets."""
    core = []
    for q in (*BIG5, *AXES):
        core.append(
            {
                "id": q,
                "dataset": "pandora_source",
                "phase": "source_self_report",
                "query": q,
                "question": question(q),
                "kind": "big5_domain" if q in BIG5 else "mbti_axis",
                "positive": "high" if q in BIG5 else AXES[q][1],
                "function": "existing"
                if q in {"neuroticism", "TF_cont"}
                else "to_construct",
                "native_and_external_baseline": "D1_complete",
            }
        )
    return {
        "version": "GRAIN_ALL_FACTOR_CATALOG_V1",
        "core": core,
        "counseling": [
            {
                **asdict(s),
                "phase": s.phase,
                "id": s.name,
                "dataset": "meps_study2",
                "function": "to_construct",
            }
            for s in scales()
        ],
        "unlabeled_existing_query": "confirmation_seeking",
        "deferred": {
            "mood": "verify adapted items and event timing first",
            "big5_facets": "not present in available PANDORA domain labels",
        },
    }


def recompute(frame, spec):
    """Strict complete-item means, with range checking and explicit reverse keys."""
    if len(set(spec.items)) != len(spec.items) or not spec.items:
        raise ValueError("Invalid item definition")
    if any(p < 1 or p > len(spec.items) for p in spec.reverse_positions):
        raise ValueError("Reverse key is not a selected-item position")
    raw = frame.loc[:, list(spec.items)]
    values = raw.apply(pd.to_numeric, errors="raise").astype(float)
    finite = values.to_numpy()[values.notna().to_numpy()]
    if not np.isfinite(finite).all() or np.any(
        (finite < spec.low) | (finite > spec.high)
    ):
        raise ValueError(f"Out-of-range items for {spec.name}")
    for pos in spec.reverse_positions:
        col = spec.items[pos - 1]
        values[col] = spec.low + spec.high - values[col]
    return values.mean(axis=1, skipna=False), values


def coefficient_alpha(items):
    """Descriptive complete-case alpha, never a test of construct validity."""
    x = items.dropna().to_numpy(float)
    n, k = x.shape
    if n < 3 or k < 2:
        return None
    total = x.sum(axis=1).var(ddof=1)
    return (
        float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / total))
        if total > 0
        else None
    )


def audit_scored_meps(root: Path, *, authorized=False):
    """Read only allowlisted numeric scored columns under explicit branch consent.

    Returns private pseudonymous scores plus aggregate audits. No raw survey,
    comments, coding, downstream analysis master, or Study 1 data are opened.
    """
    if not authorized:
        raise PermissionError(
            "MEPS scored external linkage requires explicit branch authorization"
        )
    root = Path(root)
    folders = sorted(
        p
        for p in (root / "decrypted_output").iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    ids = [normalize_participant_code(p.name) for p in folders]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Ambiguous text-export identities")
    pseudonyms = dict(
        zip(ids, (f"P{i:03d}" for i in range(1, len(ids) + 1)), strict=True)
    )
    scores, audits, paths = [], [], []
    for phase in ("pre", "post"):
        specs = [s for s in scales() if s.phase == phase]
        cols = {"participant_id"}
        for spec in specs:
            cols.update(spec.items)
            cols.update((spec.name, spec.name + "_complete"))
        path = root / "research2_anal/processed" / f"{phase}_questionnaire_scored.csv"
        paths.append(path)
        frame = pd.read_csv(path, usecols=sorted(cols), dtype={"participant_id": str})
        codes = frame.participant_id.map(normalize_participant_code)
        if not codes.is_unique or not codes.map(bool).all():
            raise ValueError("Duplicate or empty questionnaire IDs")
        if set(codes) != set(ids):
            raise ValueError(
                "Questionnaire/text membership mismatch; inspect counts locally"
            )
        frame.index = codes.map(pseudonyms)
        for spec in specs:
            value, items = recompute(frame, spec)
            stored = pd.to_numeric(frame[spec.name], errors="raise")
            error = float((value - stored).abs().max()) if value.notna().any() else 0.0
            complete = (
                frame[spec.name + "_complete"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
            )
            if (
                not value.isna().equals(stored.isna())
                or error > 1e-10
                or complete.isna().any()
                or not complete.equals(value.notna())
            ):
                raise ValueError(f"Stored scoring/completeness mismatch: {spec.name}")
            audits.append(
                {
                    "scale": spec.name,
                    "query": spec.query,
                    "phase": phase,
                    "kind": spec.kind,
                    "participants": len(frame),
                    "valid": int(value.notna().sum()),
                    "missing": int(value.isna().sum()),
                    "minimum": value.min(),
                    "maximum": value.max(),
                    "mean": value.mean(),
                    "sd": value.std(ddof=1),
                    "items": len(spec.items),
                    "stored_max_error": error,
                    "alpha": coefficient_alpha(items) if spec.reliability else None,
                }
            )
            scores.extend(
                {"participant_code": uid, "scale": spec.name, "score": y}
                for uid, y in value.items()
            )
    return pd.DataFrame(scores), pd.DataFrame(audits), paths


def association(x, y, *, binary=False):
    """Pairwise-complete association with constant flags and Fisher z intervals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Paired one-dimensional observations required")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if binary and not np.isin(y, (0, 1)).all():
        raise ValueError("Binary criterion is not 0/1")
    result = {
        "n": len(x),
        "pearson_r": None,
        "p": None,
        "r_lower": None,
        "r_upper": None,
        "spearman_rho": None,
        "auc": None,
        "status": "constant_or_small",
    }
    if len(x) < 4 or np.ptp(x) < 1e-14 or np.ptp(y) < 1e-14:
        return result
    r, p = pearsonr(x, y)
    z = np.arctanh(np.clip(r, -1 + 1e-14, 1 - 1e-14))
    lo, hi = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) / np.sqrt(len(x) - 3))
    result.update(
        pearson_r=float(r),
        p=float(p),
        r_lower=float(lo),
        r_upper=float(hi),
        spearman_rho=float(spearmanr(x, y).statistic),
        status="ok",
    )
    if binary:
        result["auc"] = float(roc_auc_score(y, x))
    return result


def bh_adjust(values):
    """BH adjustment retaining missing tests in their original positions."""
    values = np.asarray(values, float)
    valid = np.flatnonzero(np.isfinite(values))
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("Invalid p value")
    order = valid[np.argsort(values[valid])]
    out = np.full(values.shape, np.nan)
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return out
