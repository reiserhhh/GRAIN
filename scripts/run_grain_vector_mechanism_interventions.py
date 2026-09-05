#!/usr/bin/env python3
"""VM01 functional follow-up: patch known facts, NOT personality directions.

Prespecified locations/ranks are not selected from test effects. A changed
unrelated setting answer is a specificity failure, not successful inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.controls import (
    SCENARIOS,
    SYSTEM,
    control_content_anchor,
    control_user_message,
    material_hash,
)
from grain_research.vector_mechanisms.interventions import (
    contrast_basis,
    fit_margin_threshold,
    matched_random_effect,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json

FAMILIES = ("ownership", "voluntary_choice")
SITES = ("block_05", "block_16")
VIEWS = ("content", "decision")
RANKS = (2, 4)
ALPHAS = (0.5, 1.0)


def archive_digest(paths):
    """Bind the parent numeric cache to the intervention run."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def summarize_effects(frame):
    """Compare candidate and random effects within targets, then cluster stories."""
    keys = ["family", "site", "view", "rank", "alpha"]
    rows = []
    for setting, sub in frame.groupby(keys):
        candidate = (
            sub[sub.kind == "candidate"]
            .set_index(["group_id", "target_id"])
            .oriented_delta
        )
        random = (
            sub[sub.kind == "random"]
            .groupby(["group_id", "target_id"])
            .oriented_delta.mean()
        )
        paired = (candidate - random).dropna()
        stats = grouped_bootstrap(
            paired.to_numpy(), paired.index.get_level_values("group_id")
        )
        rows.append(
            {
                **dict(zip(keys, setting)),
                "candidate_mean": float(candidate.mean()),
                "random_mean": float(random.mean()),
                "paired_difference": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "scenario_n": stats["clusters"],
            }
        )
    return pd.DataFrame(rows)


def main():
    """Run a bounded intervention battery with identity-bound per-target resumes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=4096)
    args = parser.parse_args()
    parent, output = args.parent_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "private_cache"
    cache.mkdir(exist_ok=True)
    pm = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or not status["instrument_pass"]:
        raise ValueError("Parent instrument is not complete/passed")
    config = pm["config"]
    metadata = {
        r["case_id"]: r
        for r in json.loads((parent / "contrast_manifest.json").read_text())
    }
    records, states = {}, {}
    for key in metadata:
        records[key] = json.loads(
            (parent / "private_cache" / f"{key}.json").read_text()
        )
        with np.load(parent / "private_cache" / f"{key}.npz") as data:
            states[key] = {k: data[k].copy() for k in data.files}
    targets = [
        r
        for r in metadata.values()
        if r["split"] == "heldout_scenario"
        and r["family"] in FAMILIES
        and r["variant"] in (0, 1)
    ]
    if args.limit_pairs:
        targets = targets[: args.limit_pairs]
    manifest = {
        "protocol": "GRAIN_VM01_FACT_PATCH_V1",
        "parent_identity": pm["identity"],
        "parent_cache_sha256": archive_digest((parent / "private_cache").glob("*")),
        "code_sha256": code_fingerprint(ROOT),
        "target_ids": [r["case_id"] for r in targets],
        "families": FAMILIES,
        "sites": SITES,
        "views": VIEWS,
        "ranks": RANKS,
        "alphas": ALPHAS,
        "random_replicates": 3,
        "max_calls": args.max_calls,
        "scope": "synthetic_known_fact_function; not human personality",
    }
    manifest["identity"] = material_hash(manifest)
    if (output / "manifest.json").exists() and json.loads(
        (output / "manifest.json").read_text()
    )["identity"] != manifest["identity"]:
        raise ValueError("Intervention identity changed; use a new output directory")
    write_json(output / "manifest.json", manifest)
    archive_source(ROOT, output / "source_snapshot.zip")
    basis = {}
    for family in FAMILIES:
        for site in SITES:
            for view in VIEWS:
                differences = []
                for row in metadata.values():
                    if (
                        row["split"] != "discovery"
                        or row["family"] != family
                        or row["variant"] not in (0, 2)
                    ):
                        continue
                    other = f"{row['group_id']}__{family}__{row['variant'] + 1}"
                    key = f"{site}__{view}"
                    differences.append(states[row["case_id"]][key] - states[other][key])
                for rank in RANKS:
                    basis[(family, site, view, rank)] = contrast_basis(
                        np.stack(differences), rank
                    )
    # This calibration is secondary analysis of already saved output, not part of patch selection.
    calibration = []
    from sklearn.metrics import balanced_accuracy_score

    for family in sorted({r["family"] for r in metadata.values()}):
        tr = [
            r
            for r in records.values()
            if r["family"] == family and r["split"] == "discovery"
        ]
        te = [
            r
            for r in records.values()
            if r["family"] == family and r["split"] != "discovery"
        ]
        threshold = fit_margin_threshold(
            [r["margin"] for r in tr], [r["expected_yes"] for r in tr]
        )
        calibration.append(
            {
                "family": family,
                "discovery_threshold": threshold,
                "heldout_balanced_accuracy": float(
                    balanced_accuracy_score(
                        [r["expected_yes"] for r in te],
                        [r["margin"] >= threshold for r in te],
                    )
                ),
            }
        )
    pd.DataFrame(calibration).to_csv(
        output / "discovery_only_calibration.csv", index=False
    )
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        config["model"],
        revision=config["revision"],
        device=config["device"],
        dtype=config["dtype"],
        max_tokens=config["max_tokens"],
        local_files_only=config["local_files_only"],
        pad_to_length=config.get("pad_to_length"),
    )
    old_model = json.loads((parent / "model_manifest.json").read_text())
    model = observer.manifest()
    for field in [
        "revision",
        "dtype",
        "template_sha256",
        "torch",
        "transformers",
        "pad_to_length",
    ]:
        if model[field] != old_model[field]:
            raise ValueError(f"Parent model mismatch: {field}")
    write_json(output / "model_manifest.json", model)
    started = time.monotonic()
    all_effects, all_controls, all_collateral = [], [], []
    for row in targets:
        path = cache / f"{row['case_id']}.json"
        if path.exists():
            done = json.loads(path.read_text())
            all_effects.extend(done["effects"])
            all_controls.extend(done["controls"])
            all_collateral.extend(done["collateral"])
            continue
        if observer.calls + 220 > args.max_calls or time.monotonic() - started > 21600:
            break
        target_id = row["case_id"]
        donor_id = f"{row['group_id']}__{row['family']}__{1 - row['variant']}"
        ids = observer.prompt_ids(SYSTEM, control_user_message(row))
        if records[target_id].get("prompt_token_sha256") != material_hash(ids):
            raise ValueError(
                "Actual token prefix differs from parent; regenerate parent with token hashes"
            )
        positions = {
            "content": observer.content_position(
                SYSTEM, control_user_message(row), control_content_anchor(row)
            ),
            "decision": len(ids) - 1,
        }
        base = observer.margin(ids)
        if abs(base - records[target_id]["margin"]) > 5e-4:
            raise ValueError("Baseline margin no longer replays parent")
        donor_margin = records[donor_id]["margin"]
        orientation = float(np.sign(donor_margin - base))
        effects, controls, collateral = [], [], []
        base_fields = {
            "target_id": target_id,
            "group_id": row["group_id"],
            "family": row["family"],
            "baseline_margin": base,
            "donor_margin": donor_margin,
        }
        for site in SITES:
            for view in VIEWS:
                key, pos = f"{site}__{view}", positions[view]
                current, donor = states[target_id][key], states[donor_id][key]
                delta = donor - current
                for mode, vector in [("self", current), ("full_donor", donor)]:
                    margin = observer.margin(ids, patch=Patch(site, pos, vector))
                    controls.append(
                        {
                            **base_fields,
                            "site": site,
                            "view": view,
                            "kind": mode,
                            "delta": margin - base,
                        }
                    )
                    if mode == "self" and abs(margin - base) > 5e-4:
                        raise ValueError(
                            f"Self-donor changes output: {target_id}/{site}/{view}, delta={margin - base}"
                        )
                for rank in RANKS:
                    b = basis[(row["family"], site, view, rank)]
                    candidate = b @ (b.T @ delta)
                    vectors = [("candidate", -1, candidate)]
                    vectors += [
                        (
                            "random",
                            rep,
                            matched_random_effect(
                                delta,
                                candidate,
                                rank,
                                f"{row['family']}::{site}::{view}::{rank}::{rep}",
                            ),
                        )
                        for rep in range(3)
                    ]
                    for alpha in ALPHAS:
                        for kind, rep, effect in vectors:
                            margin = observer.margin(
                                ids,
                                patch=Patch(site, pos, current + effect, alpha=alpha),
                            )
                            effects.append(
                                {
                                    **base_fields,
                                    "site": site,
                                    "view": view,
                                    "rank": rank,
                                    "alpha": alpha,
                                    "kind": kind,
                                    "random_rep": rep,
                                    "effect_norm": float(
                                        np.linalg.norm(alpha * effect)
                                    ),
                                    "delta": margin - base,
                                    "oriented_delta": orientation * (margin - base),
                                }
                            )
                # Apply the SAME additive vector to an unrelated setting question.
                b = basis[(row["family"], site, view, 2)]
                candidate = b @ (b.T @ delta)
                scene = SCENARIOS[int(row["group_id"].split("_")[1])][0]
                for expected, asked in [(1, scene), (0, "airport")]:
                    other_row = {
                        **row,
                        "question": f"Is the stated setting of this account a {asked}?",
                    }
                    other_ids = observer.prompt_ids(
                        SYSTEM, control_user_message(other_row)
                    )
                    other_pos = (
                        observer.content_position(
                            SYSTEM,
                            control_user_message(other_row),
                            control_content_anchor(other_row),
                        )
                        if view == "content"
                        else len(other_ids) - 1
                    )
                    other_states, _ = observer.capture(other_ids, positions=[other_pos])
                    other_base = observer.margin(other_ids)
                    for kind, effect in [
                        ("candidate", candidate),
                        (
                            "random",
                            matched_random_effect(
                                delta,
                                candidate,
                                2,
                                f"{row['family']}::{site}::{view}::2::0",
                            ),
                        ),
                    ]:
                        changed = observer.margin(
                            other_ids,
                            patch=Patch(
                                site, other_pos, other_states[site][0] + effect
                            ),
                        )
                        collateral.append(
                            {
                                **base_fields,
                                "site": site,
                                "view": view,
                                "kind": kind,
                                "setting_expected_yes": expected,
                                "setting_base": other_base,
                                "setting_changed": changed,
                                "absolute_delta": abs(changed - other_base),
                                "prediction_flipped": int(
                                    (changed >= 0) != (other_base >= 0)
                                ),
                                "pre_question_state_delta": float(
                                    np.linalg.norm(other_states[site][0] - current)
                                )
                                if view == "content"
                                else None,
                            }
                        )
        restored = observer.margin(ids)
        if abs(restored - base) > 5e-4:
            raise ValueError("Hooks did not restore baseline")
        controls.append({**base_fields, "kind": "restore", "delta": restored - base})
        write_json(
            path, {"effects": effects, "controls": controls, "collateral": collateral}
        )
        all_effects.extend(effects)
        all_controls.extend(controls)
        all_collateral.extend(collateral)
        print(
            f"PATCHED {target_id} calls={observer.calls} elapsed={time.monotonic() - started:.1f}",
            flush=True,
        )
    if not all_effects:
        raise RuntimeError("No completed intervention target")
    frame = pd.DataFrame(all_effects)
    frame.to_csv(output / "intervention_effects.csv", index=False)
    summarize_effects(frame).to_csv(
        output / "paired_intervention_summary.csv", index=False
    )
    pd.DataFrame(all_controls).to_csv(output / "restoration_controls.csv", index=False)
    pd.DataFrame(all_collateral).to_csv(output / "collateral_effects.csv", index=False)
    run_status = {
        "status": "COMPLETE"
        if frame.target_id.nunique() == len(targets)
        else "PARTIAL",
        "targets_complete": int(frame.target_id.nunique()),
        "targets_planned": len(targets),
        "calls_this_invocation": observer.calls,
        "processed_tokens_this_invocation": observer.processed_tokens,
        "seconds": time.monotonic() - started,
        "scope": "known semantic facts; causal output intervention, not human personality",
    }
    write_json(output / "run_status.json", run_status)
    print(json.dumps(run_status, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--output-dir" in sys.argv:
            folder = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error_log.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
