#!/usr/bin/env python3
"""Compile one acquisition-aware source observer; never optimize native scores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_acquisition_law as M
from grain_research.construction import math_event_trait_action as E
from grain_research.construction import math_joint_evidence as J
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_acquisition_law_hm197.json"
DEFAULT = ROOT / "results/grain_math_response_v1/acquisition_observation_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ACQUISITION_LAW_PLAN_HM197.md",
    "grain_research/construction/math_acquisition_law.py",
    "scripts/run_grain_acquisition_law_hm197.py",
    "tests/test_grain_acquisition_law.py",
    "grain_research/construction/math_event_trait_action.py",
    "grain_research/construction/math_joint_evidence.py",
    "grain_research/construction/math_observed_family.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Read this experiment's declared numeric derivatives without pickle."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def clean(value):
    """Represent missing numerical support explicitly in immutable JSON."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def observation_cells(coordinates, conditions, cutoffs):
    """Flatten only batch axes for the inherited strictly2D cell encoder."""
    x, c = np.asarray(coordinates, float), np.asarray(conditions)
    if x.shape[:-1] != c.shape or x.shape[-1] != 15 or not np.isin(c, [0, 1]).all():
        raise ValueError("Matching batch/event conditions and15 coordinates required")
    return 4 * c + R.F.states(x.reshape(-1, 15), cutoffs).reshape(c.shape)


def prepare(out):
    """Bind SAME events, actual acquisition addresses and independent SOURCE labels."""
    cfg = R.read(ROOT / CONFIG)
    parents = {
        k: ROOT / cfg[k]
        for k in ("source_parent", "label_parent", "bank_parent", "event_parent")
    }
    R.check(parents["source_parent"], ("compiled_seal.json", "analysis_seal.json"))
    R.check(parents["label_parent"], ("scored_seal.json",))
    R.check(parents["bank_parent"], ("computed_seal.json",))
    R.check(parents["event_parent"], ("compiled_seal.json",))
    obs = arrays(parents["source_parent"] / "observations.npz")
    events = arrays(parents["event_parent"] / "observations.npz")
    bank = arrays(parents["bank_parent"] / "observations.npz")
    source, app = obs["source_indices"], obs["application_indices"]
    if (
        len(source) != cfg["source_authors"]
        or len(app) != cfg["authors"]
        or np.intersect1d(source, app).size
        or not np.array_equal(source, bank["source"])
        or np.bincount(bank["folds"]).tolist() != cfg["source_banks"]
    ):
        raise ValueError("Changed independent source/application author identity")
    old_mass, targets = J.validate_bundle(
        R.read(parents["label_parent"] / "source_bundle.json")
    )
    if (
        targets.shape != (386, 5)
        or not np.allclose(
            old_mass, obs["source_mass"], rtol=0, atol=cfg["audit_tolerance"]
        )
        or not np.array_equal(
            (targets @ (2 ** np.arange(5))).astype(int), bank["codes"]
        )
        or sorted(set(obs["splits"].tolist())) != ["discovery", "held_author"]
    ):
        raise ValueError("Changed five SOURCE target alignment or application roles")
    with np.load(ROOT / cfg["material"], allow_pickle=False) as z:
        raw = z["comments"]
        source_raw = raw[source, :24].reshape(386, 2, 12, 1024)
        application_raw = raw[app, :24].reshape(24, 2, 12, 1024)
    with np.load(ROOT / cfg["poles"], allow_pickle=False) as z:
        poles = z["query_poles"][2:4]
    with np.load(ROOT / cfg["directions"], allow_pickle=False) as z:
        directions = np.column_stack(
            [
                z[k]
                for k in (
                    "author_stability",
                    "response_stability",
                    "principal_components",
                )
            ]
        )
    inverse = np.argsort(events["original_units"], axis=1)
    coordinates = np.take_along_axis(events["coordinates"], inverse[:, :, None], axis=1)
    conditions = np.take_along_axis(events["conditions"], inverse, axis=1)
    law = R.read(parents["source_parent"] / "source_law.json")
    src_coord = coordinates[source, :24].reshape(386, 2, 12, 15)
    coord_error = float(
        np.max(np.abs(E.unit_vectors(source_raw) @ directions - src_coord))
    )
    if coord_error > cfg["audit_tolerance"] or not np.array_equal(
        coordinates[app, :24].reshape(24, 2, 12, 15), obs["coordinates"][:, 1]
    ):
        raise ValueError("Raw vector/event coordinate identity failed")
    source_ids, source_signed = M.select_top(source_raw, poles)
    actual_from_scores = np.sort(
        np.argsort(-np.abs(obs["signed_poles"]), axis=-1, kind="stable")[..., :4]
    )
    if not np.array_equal(actual_from_scores, obs["selected"]):
        raise ValueError(
            "Historical application selected addresses do not match input ranks"
        )
    raw_ids, _ = M.select_top(application_raw, poles)
    src_cond = conditions[source, :24].reshape(386, 2, 12)
    source_cells = observation_cells(src_coord, src_cond, law["coordinate_cutoffs"])
    app_cells = np.stack(
        [
            observation_cells(
                obs["coordinates"][:, arm], obs["conditions"], law["coordinate_cutoffs"]
            )
            for arm in (0, 1)
        ]
    )
    files = {CONFIG: R.digest(ROOT / CONFIG)}
    for key in ("material", "directions", "poles"):
        files[cfg[key]] = R.digest(ROOT / cfg[key])
    names = {
        "source_parent": (
            "manifest.json",
            "observations.npz",
            "source_law.json",
            "action_accounts.npz",
            "prepared_seal.json",
            "compiled_seal.json",
            "analysis_seal.json",
        ),
        "label_parent": (
            "manifest.json",
            "source_bundle.json",
            "prepared_seal.json",
            "scored_seal.json",
        ),
        "bank_parent": (
            "manifest.json",
            "observations.npz",
            "prepared_seal.json",
            "computed_seal.json",
        ),
        "event_parent": (
            "manifest.json",
            "observations.npz",
            "prepared_seal.json",
            "compiled_seal.json",
        ),
    }
    for key, items in names.items():
        for item in items:
            path = parents[key] / item
            files[str(path.relative_to(ROOT))] = R.digest(path)
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": files,
            "status": "source_acquisition_fields_and_new_native_links_not_computed_before_seal",
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        source_cells=source_cells,
        source_selected=M.selection_mask(source_ids),
        source_signed=source_signed,
        source_indices=source,
        targets=targets,
        folds=bank["folds"],
        application_cells=app_cells,
        application_selected=obs["selected"],
        application_control=obs["control"],
        application_indices=app,
        splits=obs["splits"],
        mask_changed=obs["mask_changed"],
        raw_selected=raw_ids,
    )
    quality = {
        "source_authors": len(source),
        "source_distinct_events": source_cells.size,
        "application_authors": len(app),
        "application_distinct_events": 24 * 2 * 12,
        "source_application_author_overlap": 0,
        "source_raw_coordinate_error": coord_error,
        "application_actual_cue_selected_identity_error": 0,
        "application_raw_vs_actual_selection_different_views": int(
            np.any(raw_ids != obs["selected"], axis=-1).sum()
        ),
        "application_mask_changed_views": int(obs["mask_changed"].sum()),
        "source_control_token_lengths_available": False,
        "control_protocol_not_calibrated": True,
        "source_representation": "raw",
        "native_representation": "cue_mask",
        "historical_representation_overlap_retained": True,
        "native_readouts_loaded_before_source_compile": False,
        "new_neural_API_private_raw_application_labels_training": 0,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def compile_source(out):
    """Freeze observation laws before inspecting any native outcome."""
    start = time.perf_counter()
    cfg = R.check(out)["config"]
    z = arrays(out / "observations.npz")
    c, s, y = z["source_cells"], z["source_selected"], z["targets"]
    pooled = M.compile_fields(c, s, y)
    banks = [
        [
            M.compile_fields(
                c[z["folds"] == b, v : v + 1],
                s[z["folds"] == b, v : v + 1],
                y[z["folds"] == b],
            )
            for v in (0, 1)
        ]
        for b in (0, 1)
    ]
    f = np.array([[r["fields"] for r in row] for row in banks])
    metrics = M.cross_transport(f, pooled["reference"][0])
    fields = pooled["fields"]
    delta = fields[1] - fields[2]
    bundle = {
        "version": "HM197_acquisition_observer_v1",
        "targets": cfg["targets"],
        "modes": list(M.MODES),
        "source": "equal author then equal mode-qualified event",
        "selector": "raw_E5_absolute_N_pole_top4_of12_stable_ties",
        "prior": pooled["prior"],
        "fields": fields,
        "support": pooled["support"],
        "reference": pooled["reference"],
        "selection_probability": pooled["selection_probability"],
        "control_protocol": "not_calibrated_missing_Qwen_token_lengths",
        "counterfactual_personality_change": False,
        "runtime_source_author_catalogue": False,
    }
    R.write(out / "source_bundle.json", clean(bundle))
    R.archive(
        out / "source_fields.npz",
        **pooled,
        banks=f,
        bank_reference=np.array([[r["reference"] for r in row] for row in banks]),
    )
    rows = []
    for a in range(8):
        for q, target in enumerate(cfg["targets"]):
            rows.append(
                {
                    "cell": a,
                    "target": target,
                    "selection_probability": float(pooled["selection_probability"][a]),
                    "full": fields[0, a, q],
                    "selected": fields[1, a, q],
                    "remainder": fields[2, a, q],
                    "conditional_difference": delta[a, q],
                }
            )
    R.table(out / "source_acquisition_fields.csv", clean(rows))
    summary = {
        "source_authors": len(y),
        "source_events": c.size,
        "conditional_difference_RMS": np.sqrt(
            (pooled["reference"][0, :, None] * delta**2).sum(0)
        ),
        "cross_author_cross_view": metrics,
        "mixture_identity_error": float(
            np.nanmax(np.abs(fields[0] - pooled["mixture"]))
        ),
        "unchanged_prior_error": float(
            np.max(np.abs(pooled["joint"].sum(1) - pooled["prior"]))
        ),
        "all_pooled_and_bank_mode_cells_supported": bool(
            np.isfinite(f).all() and pooled["support"].all()
        ),
        "elapsed_seconds": time.perf_counter() - start,
        "new_fit_to_native": False,
        "accuracy_promotion": False,
        "original_scope_complete": False,
    }
    R.write(out / "source_summary.json", clean(summary))
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_bundle.json",
            "source_fields.npz",
            "source_acquisition_fields.csv",
            "source_summary.json",
        ),
    )
    print(json.dumps(clean(summary)))


def analyze(out):
    """Apply fixed and protocol-qualified observers; never select a native winner."""
    start = time.perf_counter()
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    z, law = arrays(out / "observations.npz"), R.read(out / "source_bundle.json")
    f, pi = np.asarray(law["fields"], float), np.asarray(law["prior"], float)
    with np.load(
        ROOT / cfg["source_parent"] / "action_accounts.npz", allow_pickle=False
    ) as p:
        native = p["native_readouts"]
    if native.shape != (24, 2, 9) or not np.isfinite(native).all():
        raise ValueError("Incomplete historical nine-endpoint native family")
    values = np.empty((2, 24, 2, 2, 3, 5))
    rows, identity_error = [], 0.0
    for arm in range(2):
        for u in range(24):
            for v in (0, 1):
                accounts = M.evidence_accounts(
                    z["application_cells"][arm, u, v],
                    z["application_selected"][u, v],
                    f,
                    pi,
                )
                for m, r in enumerate(accounts):
                    for j, key in enumerate(
                        ("fixed_action", "protocol_drift", "total")
                    ):
                        values[arm, u, v, m, j] = r[key]
                    identity_error = max(
                        identity_error,
                        float(
                            np.max(
                                np.abs(
                                    r["total"] - r["fixed_action"] - r["protocol_drift"]
                                )
                            )
                        ),
                    )
                    native_delta = native[u, v, 3 if m == 0 else 1] - native[u, v, 0]
                    for q, target in enumerate(cfg["targets"]):
                        rows.append(
                            {
                                "arm": ("cue_mask", "raw")[arm],
                                "application_index": u,
                                "view": v,
                                "split": str(z["splits"][u]),
                                "mode": r["mode"],
                                "target": target,
                                "full_relative": r["full"][q],
                                "fixed_action": r["fixed_action"][q],
                                "protocol_drift": r["protocol_drift"][q],
                                "total": r["total"][q],
                                "mode_relative": r["value"][q],
                                "native_delta": native_delta,
                                "native_query": "N",
                                "matched_source_protocol": "source_raw_to_actual_cue_acquisition",
                            }
                        )
    contacts = []
    for arm in range(2):
        for m, mode in enumerate(("selected", "remainder")):
            nd = native[:, :, 3 if m == 0 else 1] - native[:, :, 0]
            for q, target in enumerate(cfg["targets"]):
                for j, account in enumerate(
                    ("fixed_action", "protocol_drift", "total")
                ):
                    for group in ("all", "discovery", "held_author"):
                        take = (
                            np.ones(24, bool)
                            if group == "all"
                            else z["splits"] == group
                        )
                        x, yy = values[arm, take, :, m, j, q].mean(1), nd[take].mean(1)
                        r = E.pearson(x, yy) if np.isfinite(x).all() else None
                        contacts.append(
                            {
                                "arm": ("cue_mask", "raw")[arm],
                                "mode": mode,
                                "target": target,
                                "account": account,
                                "group": group,
                                "authors": int(take.sum()),
                                "native_query": "N",
                                "descriptive_r_NOT_personality_accuracy": r,
                            }
                        )
    R.table(out / "same_event_protocol_accounts.csv", clean(rows))
    R.table(out / "native_contact_descriptive.csv", clean(contacts))
    R.archive(
        out / "application_accounts.npz",
        accounts=values,
        native_readouts=native,
        native_endpoint_names=np.asarray(
            (
                "full",
                "remove_selected",
                "remove_control",
                "selected_only",
                "control_only",
                "patch_remove_selected_0.5",
                "patch_remove_selected_1.0",
                "patch_remove_control_0.5",
                "patch_remove_control_1.0",
            )
        ),
    )
    primary = [
        r
        for r in contacts
        if r["arm"] == "cue_mask" and r["mode"] == "remainder" and r["target"] == "N"
    ]
    summary = {
        "authors": 24,
        "views": 48,
        "all_nine_native_endpoints_retained": True,
        "account_rows": len(rows),
        "contact_rows": len(contacts),
        "action_decomposition_identity_error": identity_error,
        "protocol_drift_RMS_by_mode_and_target": np.sqrt(
            np.mean(values[0, :, :, :, 1] ** 2, axis=(0, 1))
        ),
        "primary_descriptive_native_contacts": primary,
        "elapsed_seconds": time.perf_counter() - start,
        "no_native_fit_or_score_promotion": True,
        "no_new_neural_calls": True,
        "original_scope_complete": False,
    }
    R.write(out / "application_summary.json", clean(summary))
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "same_event_protocol_accounts.csv",
            "native_contact_descriptive.csv",
            "application_accounts.npz",
            "application_summary.json",
        ),
    )
    print(json.dumps(clean(summary)))


def verify(out):
    """One direct event-count/mean audit, different from the compile contractions."""
    cfg = R.check(out, ("compiled_seal.json", "analyzed_seal.json"))["config"]
    z, fields = arrays(out / "observations.npz"), arrays(out / "source_fields.npz")
    c, s, y = z["source_cells"], z["source_selected"], z["targets"]
    counts, sums = np.zeros((3, 8)), np.zeros((3, 8, 5))
    for u in range(len(c)):
        for v in (0, 1):
            for j in range(12):
                a = c[u, v, j]
                for mode in (0, 1 if s[u, v, j] else 2):
                    counts[mode, a] += 1
                    for q in range(5):
                        sums[mode, a, q] += y[u, q]
    direct = np.divide(
        sums,
        counts[:, :, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, :, None] > 0,
    )
    source_error = float(np.nanmax(np.abs(direct - fields["fields"])))
    with np.load(out / "application_accounts.npz", allow_pickle=False) as p:
        saved = p["accounts"]
    max_error = 0.0
    nchecks = 0
    for arm in range(2):
        for u in range(24):
            for v in (0, 1):
                row = z["application_cells"][arm, u, v]
                selected = set(z["application_selected"][u, v].tolist())
                for m in range(2):
                    ids = [j for j in range(12) if (j in selected) == (m == 0)]
                    for q in range(5):
                        full = sum(direct[0, a, q] for a in row) / 12
                        fixed = sum(direct[0, row[j], q] for j in ids) / len(ids)
                        changed = sum(direct[m + 1, row[j], q] for j in ids) / len(ids)
                        actual = np.array(
                            [fixed - full, changed - fixed, changed - full]
                        )
                        max_error = max(
                            max_error,
                            float(np.max(np.abs(actual - saved[arm, u, v, m, :, q]))),
                        )
                        nchecks += 3
    summary = R.read(out / "source_summary.json")
    passed = (
        max(
            source_error,
            max_error,
            summary["mixture_identity_error"],
            summary["unchanged_prior_error"],
        )
        <= cfg["audit_tolerance"]
    )
    R.write(
        out / "verification.json",
        {
            "status": "PASS" if passed else "FAIL",
            "source_event_count_audit": c.size,
            "application_scalar_checks": nchecks,
            "literal_source_field_error": source_error,
            "literal_application_action_error": max_error,
            "same_agent_not_blind_audit": True,
            "science_rerun": False,
            "original_scope_complete": False,
        },
    )
    print(json.dumps(R.read(out / "verification.json")))
    if not passed:
        raise ValueError("Preserve failed scientific audit;do not rescore")


def main():
    """Run an explicit finite stage; existing artifacts cannot be overwritten."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compile", "analyze", "verify"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    {
        "prepare": prepare,
        "compile": compile_source,
        "analyze": analyze,
        "verify": verify,
    }[a.stage](a.output)


if __name__ == "__main__":
    main()
