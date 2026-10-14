#!/usr/bin/env python3
"""Join independent source meaning to SAME historical natural-event actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_trait_action as E
from grain_research.construction import math_trait_anchor as A
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_trait_anchor_hm171 as T

CONFIG = "configs/grain_event_trait_action_hm173.json"
DEFAULT = ROOT / "results/grain_math_response_v1/event_trait_action_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EVENT_TRAIT_ACTION_PLAN_HM173.md",
    "grain_research/construction/math_event_trait_action.py",
    "scripts/run_grain_event_trait_action_hm173.py",
    "scripts/verify_grain_event_trait_action_hm173.py",
    "tests/test_grain_event_trait_action.py",
    "grain_research/construction/math_trait_anchor.py",
    "grain_research/construction/math_observed_family.py",
)
NATURAL = ("full", "remove_selected", "remove_control", "selected_only", "control_only")
PATCHED = tuple(
    f"patch_remove_{kind}_{alpha}"
    for kind in ("selected", "control")
    for alpha in ("0.5", "1.0")
)
EFFECTS = {
    "natural_removal_contrast": "removal_contrast",
    "patch_half_contrast": "removal_contrast",
    "patch_whole_contrast": "removal_contrast",
    "selected_removal_delta": "selected_removal_delta",
    "control_removal_delta": "control_removal_delta",
    "subset_contrast": "subset_contrast",
}


def load(out):
    """Read only this packet's prepared numerical observations."""
    with np.load(out / "observations.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check(out, stages=()):
    """Bind methods and existing numerical inputs; never restart old experiments."""
    m = R.read(out / "manifest.json")
    for kind in ("code", "inputs"):
        for path, digest in m[kind].items():
            if R.digest(ROOT / path) != digest:
                raise ValueError("Changed registered input or method: " + path)
    for stage in ("prepared_seal.json",) + tuple(stages):
        for path, digest in R.read(out / stage).items():
            if R.digest(out / path) != digest:
                raise ValueError("Changed sealed artifact: " + path)
    return m


def prepare(out):
    """Restore exact events and freeze the join before new alignment analysis."""
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["source_parent"]
    T.check(parent, T.STAGES)
    source_z = T.observations(parent)
    old_spec = R.read(parent / "manifest.json")["config"]
    d2 = ROOT / spec["intervention_parent"]
    manifest = R.read(d2 / "manifest.json")
    if R.read(d2 / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Historical intervention not complete")
    if R.digest(d2 / "private_profiles.npz") != manifest["profiles_sha256"]:
        raise ValueError("Historical pole identity mismatch")
    columns = [
        "author_id",
        "view",
        "task",
        "split",
        "selection_contrast",
        "selected_minus_control_length",
    ]
    frame = pd.read_csv(ROOT / spec["historical_table"], usecols=columns)
    frame = frame[frame.task == spec["task"]].sort_values(["author_id", "view"])
    if len(frame) != 48 or frame.duplicated(["author_id", "view"]).any():
        raise ValueError("Complete24-author/two-view Neuroticism panel required")
    ids = np.asarray(sorted(frame.author_id.unique()))
    if len(ids) != spec["authors"] or not np.array_equal(
        frame.view, np.tile([0, 1], 24)
    ):
        raise ValueError("Changed fixed author/view family")
    with np.load(ROOT / spec["material"], allow_pickle=False) as z:
        all_ids, all_raw = z["author_ids"], z["comments"]
    app = np.array([int(np.flatnonzero(all_ids == uid)[0]) for uid in ids])
    source = E.source_exclusion(source_z["source_train"], app, 624)
    if len(source) != spec["source_authors_after_exclusion"]:
        raise ValueError("Source/application author exclusion changed")
    with np.load(ROOT / spec["directions"], allow_pickle=False) as z:
        direction = np.column_stack(
            [
                z[k]
                for k in (
                    "author_stability",
                    "response_stability",
                    "principal_components",
                )
            ]
        )
    with np.load(d2 / "private_profiles.npz", allow_pickle=False) as z:
        poles = z["query_poles"][2:4]
    event_z = R.prepared(ROOT / spec["event_parent"])
    inverse = np.argsort(event_z["original_units"], axis=1)
    conditions = np.take_along_axis(event_z["conditions"], inverse, axis=1)
    old_coords = np.take_along_axis(event_z["coordinates"], inverse[:, :, None], axis=1)
    caches = {Path(p).stem: p for p in manifest["encoder_cache_sha256"]}
    input_files = {CONFIG: R.digest(ROOT / CONFIG)}
    for key in ("material", "directions", "historical_table"):
        input_files[spec[key]] = R.digest(ROOT / spec[key])
    for folder, names in (
        (parent, ("manifest.json", *T.STAGES, "observations.npz", "source_law.json")),
        (ROOT / spec["event_parent"], ("observations.npz", "source_law.json")),
        (d2, ("manifest.json", "run_status.json", "private_profiles.npz")),
    ):
        for name in names:
            path = str((folder / name).relative_to(ROOT))
            input_files[path] = R.digest(ROOT / path)
    input_files[old_spec["labels"]] = R.digest(ROOT / old_spec["labels"])
    coords, selected, control, signed, changed, split, lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    raw_error = coordinate_error = 0.0
    for i, uid in enumerate(ids):
        path = ROOT / caches[uid]
        if R.digest(path) != manifest["encoder_cache_sha256"][caches[uid]]:
            raise ValueError("Changed historical vector cache")
        meta = R.read(path.with_suffix(".json"))
        aliases = meta["feature_aliases"]
        with np.load(path, allow_pickle=False) as z:
            raw = z[aliases.get("raw_comment_vectors", "raw_comment_vectors")]
            cue = z[aliases.get("cue_mask_comment_vectors", "cue_mask_comment_vectors")]
        if raw.shape != (3, 12, 1024) or cue.shape != raw.shape:
            raise ValueError("Historical event layout changed")
        raw_error = max(
            raw_error, float(np.max(np.abs(raw.reshape(36, 1024) - all_raw[app[i]])))
        )
        xc, xr = E.unit_vectors(cue) @ direction, E.unit_vectors(raw) @ direction
        coordinate_error = max(
            coordinate_error,
            float(np.max(np.abs(xr.reshape(36, 15) - old_coords[app[i]]))),
        )
        coords.append(np.stack([xc[:2], xr[:2]]))
        sr, cr, pr, mr, lr = [], [], [], [], []
        for view in (0, 1):
            record = frame[(frame.author_id == uid) & (frame.view == view)].iloc[0]
            s, c, q = E.recover_controls(
                cue[view],
                poles,
                record.selection_contrast,
                tolerance=spec["control_recovery_tolerance"],
            )
            sr.append(s)
            cr.append(c)
            pr.append(q)
            mr.append(not np.array_equal(raw[view], cue[view]))
            lr.append(float(record.selected_minus_control_length))
            packet = d2 / "private_packets" / f"{uid}_{view}.json"
            path_key = str(packet.relative_to(ROOT))
            input_files[path_key] = R.digest(packet)
        selected.append(sr)
        control.append(cr)
        signed.append(pr)
        changed.append(mr)
        lengths.append(lr)
        labels = frame[frame.author_id == uid].split.unique()
        if len(labels) != 1:
            raise ValueError("Historical role differs across same author views")
        split.append(labels[0])
        input_files[caches[uid]] = R.digest(path)
        name = str(path.with_suffix(".json").relative_to(ROOT))
        input_files[name] = R.digest(ROOT / name)
    if raw_error != 0 or coordinate_error > 2e-12:
        raise ValueError("SAME event raw/coordinate identity failed")
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": input_files,
            "old_source_config": old_spec,
            "status": "post-outcome exploratory join;new alignments not calculated before seal",
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        author_ids=ids,
        application_indices=app,
        source_indices=source,
        source_mass=T.full_mass(source_z)[source],
        coordinates=np.asarray(coords),
        conditions=conditions[app, :24].reshape(24, 2, 12),
        selected=np.asarray(selected),
        control=np.asarray(control),
        signed_poles=np.asarray(signed),
        mask_changed=np.asarray(changed),
        splits=np.asarray(split),
        length_imbalance=np.asarray(lengths),
    )
    quality = {
        "authors": 24,
        "views": 48,
        "source_authors": len(source),
        "excluded_source_authors": 403 - len(source),
        "unique_control_restorations": 48,
        "raw_identity_error": raw_error,
        "coordinate_identity_error": coordinate_error,
        "mask_changed_views": int(np.sum(changed)),
        "new_model_calls": 0,
        "new_raw_text_or_questionnaire_reads": 0,
        "source_representation_historically_included_panel": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(quality))


def compile_source(out):
    """Calibrate386 nonpanel authors only; no model responses are loaded."""
    m, z = check(out), load(out)
    y = T.labels(m["old_source_config"])[z["source_indices"]]
    cuts = np.median(y, axis=0)
    binary = (y > cuts).astype(float)
    old = R.read(ROOT / m["config"]["source_parent"] / "source_law.json")
    coords = old["coordinate_cutoffs"]
    actual = A.compile_law(z["source_mass"], binary, coords, cuts)
    ix = np.random.default_rng(m["config"]["permutation_seed"]).permutation(len(y))
    permuted = A.compile_law(z["source_mass"], binary[ix], coords, cuts)
    for name, law in (("source_law.json", actual), ("permuted_law.json", permuted)):
        A.verify_law(law)
        if not np.isfinite(np.asarray(law["probabilities"], float)).all():
            raise ValueError("Missing source support;no smoothing rescue")
        R.write(out / name, law)
    R.archive(out / "source_permutation.npz", order=ix)
    R.seal(
        out,
        "compiled_seal.json",
        ("source_law.json", "permuted_law.json", "source_permutation.npz"),
    )
    print(
        json.dumps(
            {"calibrators": len(y), "targets": list(A.TRAITS), "native_fit": False}
        )
    )


def native(z, spec):
    """Read canonical historical nine endpoints and verify the numeric summary."""
    d2 = ROOT / spec["intervention_parent"]
    cols = ["author_id", "view", *NATURAL, *PATCHED]
    table = pd.read_csv(ROOT / spec["historical_table"], usecols=cols)
    values = np.empty((24, 2, 9))
    for u, uid in enumerate(z["author_ids"]):
        for v in (0, 1):
            p = R.read(d2 / "private_packets" / f"{uid}_{v}.json")
            if (p["author_id"], p["view"], p["task"], p["split"]) != (
                uid,
                v,
                spec["task"],
                z["splits"][u],
            ):
                raise ValueError("Historical endpoint-to-event identity mismatch")
            row = table[(table.author_id == uid) & (table.view == v)]
            if len(row) != 1 or set(p["readouts"]) != set(NATURAL + PATCHED):
                raise ValueError("Missing historical endpoints")
            values[u, v] = [
                p["readouts"][k]["conditional_score"] for k in NATURAL + PATCHED
            ]
            if not np.allclose(
                values[u, v],
                row.iloc[0][list(NATURAL + PATCHED)].to_numpy(float),
                rtol=0,
                atol=2e-12,
            ):
                raise ValueError("Canonical endpoint/summary disagreement")
    return values


def analyze(out):
    """Compare independent fields and model changes without response fitting."""
    start = time.perf_counter()
    m, z = check(out, ("compiled_seal.json",)), load(out)
    spec = m["config"]
    laws = [R.read(out / f) for f in ("source_law.json", "permuted_law.json")]
    readers = (
        list(A.TRAITS) + ["permuted_" + s for s in A.TRAITS] + ["query_pole_semantics"]
    )
    fields = [np.asarray(b["probabilities"]) - b["prior"] for b in laws]
    readouts = native(z, spec)
    shifts = np.stack(
        [
            readouts[:, :, 1] - readouts[:, :, 2],
            readouts[:, :, 5] - readouts[:, :, 7],
            readouts[:, :, 6] - readouts[:, :, 8],
            readouts[:, :, 1] - readouts[:, :, 0],
            readouts[:, :, 2] - readouts[:, :, 0],
            readouts[:, :, 3] - readouts[:, :, 4],
        ],
        axis=-1,
    )
    rows, max_error = [], 0.0
    scores = np.empty((2, 24, 2, 11, 6))
    variants = np.empty((2, 24, 2, 11, 5))
    for arm in range(2):
        for u, uid in enumerate(z["author_ids"]):
            for v in (0, 1):
                x = z["coordinates"][u, arm, v]
                cells = 4 * z["conditions"][u, v] + R.F.states(
                    x, laws[0]["coordinate_cutoffs"]
                )
                event_fields = np.column_stack(
                    [*(f[cells] for f in fields), z["signed_poles"][u, v]]
                )
                # Semantic comparator always describes the actual cue-mask query.
                accounts = E.deletion_accounts(
                    event_fields, z["selected"][u, v], z["control"][u, v]
                )
                max_error = max(max_error, float(accounts["identity_error"].max()))
                for j, reader in enumerate(readers):
                    variants[arm, u, v, j] = [accounts[k][j] for k in NATURAL]
                    scores[arm, u, v, j] = [accounts[k][j] for k in EFFECTS.values()]
                    row = {
                        "arm": ("cue_mask", "raw")[arm],
                        "author_id": str(uid),
                        "view": v,
                        "split": str(z["splits"][u]),
                        "reader": reader,
                        "mask_changed": bool(z["mask_changed"][u, v]),
                        "length_imbalance": float(z["length_imbalance"][u, v]),
                    }
                    row.update({k: float(accounts[k][j]) for k in NATURAL})
                    for k, name in enumerate(EFFECTS):
                        row[name] = float(scores[arm, u, v, j, k])
                        row["native_" + name] = float(shifts[u, v, k])
                    rows.append(row)
    metrics = []
    groups = ["author_mean", "view0", "view1"] + [
        "split_" + s for s in sorted(set(z["splits"].tolist()))
    ]
    for arm in range(2):
        for j, reader in enumerate(readers):
            for k, effect in enumerate(EFFECTS):
                for group in groups:
                    if group.startswith("view"):
                        v = int(group[-1])
                        x, y = scores[arm, :, v, j, k], shifts[:, v, k]
                    else:
                        take = (
                            np.ones(24, bool)
                            if group == "author_mean"
                            else z["splits"] == group[6:]
                        )
                        x, y = (
                            scores[arm, take, :, j, k].mean(1),
                            shifts[take, :, k].mean(1),
                        )
                    metric = E.alignment(
                        x, y, spec["bootstrap_draws"], spec["bootstrap_seed"]
                    )
                    ci = metric.pop("paired_author_bootstrap_ci")
                    metrics.append(
                        {
                            "arm": ("cue_mask", "raw")[arm],
                            "reader": reader,
                            "effect": effect,
                            "group": group,
                            **metric,
                            "ci_lower": None if ci is None else ci[0],
                            "ci_upper": None if ci is None else ci[1],
                        }
                    )
    R.table(out / "event_action_accounts.csv", rows)
    R.table(out / "alignment_metrics.csv", metrics)
    R.archive(
        out / "action_accounts.npz",
        mathematical=scores,
        variants=variants,
        native_readouts=readouts,
        native_effects=shifts,
        readers=np.asarray(readers),
    )
    main = [
        r
        for r in metrics
        if r["arm"] == "cue_mask"
        and r["group"] == "author_mean"
        and r["reader"]
        in ("Neuroticism", "permuted_Neuroticism", "query_pole_semantics")
    ]
    summary = {
        "authors": 24,
        "views": 48,
        "source_authors": 386,
        "account_rows": len(rows),
        "alignment_rows": len(metrics),
        "main": main,
        "normalization_identity_error": max_error,
        "elapsed_seconds": time.perf_counter() - start,
        "new_neural_calls": 0,
        "fit_to_native": False,
        "accuracy_promotion": False,
        "original_scope_complete": False,
        "historical_reuse": True,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "event_action_accounts.csv",
            "alignment_metrics.csv",
            "action_accounts.npz",
            "summary.json",
        ),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "main"}))


def main():
    """Execute one explicitly requested stage, never an automatic retry queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compile": compile_source, "analyze": analyze}[args.stage](
        args.out
    )


if __name__ == "__main__":
    main()
