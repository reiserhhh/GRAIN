#!/usr/bin/env python3
"""One same-event contact test with exact pair-deletion accounting, never answer fitting."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_distinct_event_action as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_distinct_event_target_contact_hm190.json"
DEFAULT = ROOT / "results/grain_math_response_v1/distinct_event_target_contact_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_DISTINCT_EVENT_TARGET_CONTACT_PLAN_HM190.md",
    "grain_research/construction/math_distinct_event_action.py",
    "scripts/run_grain_distinct_event_target_contact_hm190.py",
    "tests/test_grain_distinct_event_action.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Read the registered numeric derivative, not raw text or questionnaires."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def number(value):
    """Serialize missing observations as null rather than nonstandard NaN or zero."""
    return float(value) if np.isfinite(value) else None


def prepare(out):
    """Bind actual events and freeze source readers before any new native contact."""
    cfg = R.read(ROOT / CONFIG)
    stages = {
        "source_parent": ("computed_seal.json", "verification_seal.json"),
        "vector_parent": (
            "compiled_seal.json",
            "scored_seal.json",
            "analysis_seal.json",
            "verification_seal.json",
        ),
        "event_parent": ("compiled_seal.json", "analysis_seal.json"),
        "native_parent": ("scored_seal.json", "analysis_seal.json"),
    }
    code, inputs, parents = {}, {}, {}
    for name, receipts in stages.items():
        parent = ROOT / cfg[name]
        manifest = R.check(parent, receipts)
        code.update(manifest["code"])
        parents[name] = R.prepared(parent)
        for name2 in ("manifest.json", "prepared_seal.json", *receipts):
            files = (
                (name2,)
                if name2 == "manifest.json"
                else (name2, *R.read(parent / name2))
            )
            for f in files:
                p = str((parent / f).relative_to(ROOT))
                inputs[p] = R.digest(ROOT / p)
    s, v, e, n = [parents[k] for k in stages]
    if (
        not np.array_equal(s["source_indices"], v["source_indices"])
        or not np.array_equal(s["source_indices"], e["source_indices"])
        or not np.array_equal(v["panel_indices"], e["application_indices"])
        or not np.array_equal(s["labels"], v["labels"])
        or set(v["panel_indices"]) & set(s["source_indices"])
    ):
        raise ValueError("Different source/application author families")
    for k in ("author_ids", "selected", "control"):
        if not np.array_equal(e[k], n[k]):
            raise ValueError("Native responses belong to different event addresses")
    for k in ("selected", "control"):
        if not np.array_equal(v[k], e[k]):
            raise ValueError("Changed removal addresses")
    x, c, basis = v["panel_vectors"] @ s["basis"], e["conditions"], s["basis"]
    err = float(np.max(np.abs(v["panel_vectors"] @ v["mapping"] - e["coordinates"])))
    if (
        x.shape != (24, 2, 2, 12, 15)
        or c.shape != (24, 2, 12)
        or err > cfg["comparison_tolerance"]
    ):
        raise ValueError("Changed event-vector-condition identities")
    accounts_path = ROOT / cfg["source_parent"] / "accounts.npz"
    level_path = ROOT / cfg["vector_parent"] / "reader_bundle.npz"
    w = arrays(accounts_path)["fields"][1, :, :, :, 6].mean((0, 1))
    level = arrays(level_path)["fields"][:, :, 0, 1].mean(0) @ basis
    if w.shape != (5, 15, 15) or level.shape != (5, 15):
        raise ValueError("Source reader scope changed")
    for p in (accounts_path, level_path):
        inputs[str(p.relative_to(ROOT))] = R.digest(p)
    support = np.zeros((24, 2, 3), bool)
    counts = np.zeros((24, 2, 3, 2), int)
    for u, view in np.ndindex(24, 2):
        keep_s = ~np.isin(np.arange(12), e["selected"][u, view])
        keep_c = ~np.isin(np.arange(12), e["control"][u, view])
        for k, mask in enumerate((np.ones(12, bool), keep_s, keep_c)):
            counts[u, view, k] = np.bincount(c[u, view, mask], minlength=2)
            support[u, view, k] = (counts[u, view, k] >= 2).all()
    complete = support.all(-1)
    actual = (
        int(complete.sum()),
        int(complete.any(1).sum()),
        int(complete.all(1).sum()),
    )
    expected = tuple(
        cfg[k]
        for k in (
            "expected_complete_views",
            "expected_authors_any",
            "expected_authors_both",
        )
    )
    if actual != expected:
        raise ValueError(f"Precontact availability changed: {actual} != {expected}")
    code.update({p: R.digest(ROOT / p) for p in SCIENCE})
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": code,
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        vectors=x,
        conditions=c,
        selected=e["selected"],
        control=e["control"],
        support=support,
        complete=complete,
        counts=counts,
        source_indices=s["source_indices"],
        panel_indices=v["panel_indices"],
    )
    R.archive(out / "source_readers.npz", relation=w, level=level, basis=basis)
    quality = {
        "authors_retained": 24,
        "complete_views": actual[0],
        "authors_any": actual[1],
        "authors_both": actual[2],
        "complete_per_view": complete.sum(0).tolist(),
        "coordinate_error": err,
        "source_application_overlap": 0,
        "native_answers_read_for_construction": False,
        "representation_historical_not_pristine_holdout": True,
        "new_neural_API_private_application_labels_training": 0,
    }
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "source_readers.npz", "quality.json"),
    )
    print(json.dumps(quality))


def construct(out):
    """Compute all source-defined actions and freeze them before opening endpoints."""
    cfg, z = R.check(out)["config"], R.prepared(out)
    readers, start = arrays(out / "source_readers.npz"), time.perf_counter()
    values = np.full((24, 2, 2, 3, 3, 5), np.nan)
    deletion = np.full((24, 2, 2, 2, 3, 5), np.nan)
    pieces = np.full_like(deletion, np.nan)
    action = np.full((24, 2, 2, 7, 5), np.nan)
    for u, arm, view in np.ndindex(24, 2, 2):
        x, c = z["vectors"][u, arm, view], z["conditions"][u, view]
        ds = (z["selected"][u, view], z["control"][u, view])
        subsets = (np.arange(12), *[np.setdiff1d(np.arange(12), d) for d in ds])
        for k, idx in enumerate(subsets):
            g = M.geometries(x[idx], c[idx])
            if g is not None:
                values[u, arm, view, k] = M.choice_weight(c[idx]) * np.einsum(
                    "qij,hij->hq", readers["relation"], g
                )
        for k, d in enumerate(ds):
            a = M.deletion_action(x, c, d, readers["relation"])
            if a is not None:
                deletion[u, arm, view, k], pieces[u, arm, view, k] = (
                    a["delta"],
                    a["pieces"],
                )
        if z["complete"][u, view]:
            level_delta = (x[subsets[1]].mean(0) - x[subsets[2]].mean(0)) @ readers[
                "level"
            ].T
            action[u, arm, view] = np.concatenate(
                (
                    level_delta[None],
                    values[u, arm, view, 1] - values[u, arm, view, 2],
                    pieces[u, arm, view, 0] - pieces[u, arm, view, 1],
                )
            )
    rows = []
    for u, arm, view, reading, q in np.ndindex(24, 2, 2, 7, 5):
        rows.append(
            {
                "author_row": u,
                "representation": cfg["representations"][arm],
                "view": view,
                "reading": M.READINGS[reading],
                "target": cfg["targets"][q],
                "complete_view": bool(z["complete"][u, view]),
                "action": number(action[u, arm, view, reading, q]),
            }
        )
    R.table(out / "all_actions.csv", rows)
    R.archive(
        out / "action_values.npz",
        values=values,
        deletion=deletion,
        pieces=pieces,
        action=action,
        complete=z["complete"],
    )
    receipt = {
        "seconds": time.perf_counter() - start,
        "native_answers_read": False,
        "all_targets_readings_representations_retained": True,
    }
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "constructed_seal.json",
        ("action_values.npz", "all_actions.csv", "construction_receipt.json"),
    )
    print(json.dumps(receipt))


def analyze(out):
    """Read old endpoints AFTER the unfit mathematical construction is sealed."""
    cfg = R.check(out, ("constructed_seal.json",))["config"]
    a = arrays(out / "action_values.npz")
    old = arrays(ROOT / cfg["event_parent"] / "action_accounts.npz")["native_effects"]
    logits = arrays(ROOT / cfg["native_parent"] / "composition_accounts.npz")[
        "native_logits"
    ]
    odds = logits[..., 1:] - logits[..., :1]
    endpoints = np.concatenate(
        (old[:, :, :1], odds[:, :, 1] - odds[:, :, 2], old[:, :, 1:3]), -1
    )
    if endpoints.shape != (24, 2, 12) or not np.isfinite(endpoints).all():
        raise ValueError("Changed historical native endpoint scope")
    names = [
        "natural_ordinal",
        *[f"natural_digit{k}_vs0" for k in range(1, 10)],
        "halfpatch_ordinal",
        "wholepatch_ordinal",
    ]
    native = M.author_mean(endpoints, a["complete"])
    mathematical = np.stack(
        [M.author_mean(a["action"][:, arm], a["complete"]) for arm in (0, 1)]
    )
    rows, covariance = [], []
    for arm, q, endpoint in np.ndindex(2, 5, 12):
        cov = []
        for reading in range(7):
            summary = M.contact(mathematical[arm, :, reading, q], native[:, endpoint])
            rows.append(
                {
                    "representation": cfg["representations"][arm],
                    "target": cfg["targets"][q],
                    "endpoint": names[endpoint],
                    "reading": M.READINGS[reading],
                    "permutation_upper_rank": None,
                    **summary,
                }
            )
            cov.append(summary["covariance"])
        covariance.append(
            {
                "representation": cfg["representations"][arm],
                "target": cfg["targets"][q],
                "endpoint": names[endpoint],
                **dict(zip(M.READINGS, cov, strict=True)),
                "M_minus_U_V": cov[1] - cov[2] - cov[3],
                "U_minus_three_pieces": cov[2] - sum(cov[4:]),
            }
        )
    primary = next(
        row
        for row in rows
        if row["representation"] == "cue_mask"
        and row["target"] == "N"
        and row["reading"] == "distinct_event"
        and row["endpoint"] == "natural_ordinal"
    )
    ids = np.where(a["complete"].any(1))[0]
    rng = np.random.default_rng(cfg["permutation_seed"])
    orders = np.stack(
        [rng.permutation(len(ids)) for _ in range(cfg["permutation_replicates"])]
    )
    x, y = mathematical[0, ids, 2, 1], native[ids, 0]
    reference = np.array([M.contact(x, y[order])["covariance"] for order in orders])
    primary["permutation_upper_rank"] = float(
        (1 + (reference >= primary["covariance"]).sum()) / (1 + len(reference))
    )
    availability = [
        {
            "author_row": u,
            "complete_view0": bool(a["complete"][u, 0]),
            "complete_view1": bool(a["complete"][u, 1]),
            "primary_included": bool(a["complete"][u].any()),
        }
        for u in range(24)
    ]
    R.table(out / "availability.csv", availability)
    R.table(out / "all_contacts.csv", rows)
    R.table(out / "covariance_accounts.csv", covariance)
    R.archive(
        out / "native_contact_accounts.npz",
        endpoints=endpoints,
        native_author_mean=native,
        mathematical_author_mean=mathematical,
        ids=ids,
        permutation_orders=orders,
        permutation_covariance=reference,
    )
    summary = {
        "primary": primary,
        "quality": R.read(out / "quality.json"),
        "natural_N_cue_contacts": [
            row
            for row in rows
            if row["representation"] == "cue_mask"
            and row["target"] == "N"
            and row["endpoint"] == "natural_ordinal"
        ],
        "not_human_personality_accuracy_causal_mediation_or_new_math": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "availability.csv",
            "all_contacts.csv",
            "covariance_accounts.csv",
            "native_contact_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One alternate covariance-form replay; do not improve decimal precision."""
    cfg, z = (
        R.check(out, ("constructed_seal.json", "analysis_seal.json"))["config"],
        R.prepared(out),
    )
    readers, a, contact = (
        arrays(out / "source_readers.npz"),
        arrays(out / "action_values.npz"),
        arrays(out / "native_contact_accounts.npz"),
    )
    errors = {
        "geometry": 0.0,
        "pair_pieces": 0.0,
        "M_U_V": 0.0,
        "native_author_mean": 0.0,
    }
    for u, arm, view in np.ndindex(24, 2, 2):
        x, c = z["vectors"][u, arm, view], z["conditions"][u, view]
        for k, d in enumerate(
            (np.array([], int), z["selected"][u, view], z["control"][u, view])
        ):
            keep = ~np.isin(np.arange(12), d)
            if not z["support"][u, view, k]:
                if not np.isnan(a["values"][u, arm, view, k]).all():
                    raise ValueError("Missing observation replaced with a score")
                continue
            row0, row1 = [x[keep][c[keep] == side] for side in (0, 1)]
            delta = row1.mean(0) - row0.mean(0)
            raw = np.outer(delta, delta)
            within = sum(
                (r - r.mean(0)).T @ (r - r.mean(0)) / (len(r) * (len(r) - 1))
                for r in (row0, row1)
            )
            gs = np.stack((raw, raw - within, within))
            expected = M.choice_weight(c[keep]) * np.einsum(
                "qij,hij->hq", readers["relation"], gs
            )
            errors["geometry"] = max(
                errors["geometry"],
                float(np.max(np.abs(expected - a["values"][u, arm, view, k]))),
            )
        if z["complete"][u, view]:
            delta = a["action"][u, arm, view]
            errors["M_U_V"] = max(
                errors["M_U_V"], float(np.max(np.abs(delta[1] - delta[2] - delta[3])))
            )
            errors["pair_pieces"] = max(
                errors["pair_pieces"],
                float(np.max(np.abs(delta[2] - delta[4:].sum(0)))),
            )
    for u in np.where(z["complete"].any(1))[0]:
        expected = contact["endpoints"][u, z["complete"][u]].mean(0)
        errors["native_author_mean"] = max(
            errors["native_author_mean"],
            float(np.max(np.abs(expected - contact["native_author_mean"][u]))),
        )
    if max(errors.values()) > cfg["comparison_tolerance"]:
        raise ValueError("One-shot action identity replay failed")
    receipt = {
        "errors": errors,
        "same_agent_not_independent_blind": True,
        "no_new_model_fit_or_rescue": True,
        "old_seals_unchanged": True,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "construct", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "construct": construct, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.out)
