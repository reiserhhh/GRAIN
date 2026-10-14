#!/usr/bin/env python3
"""Run the bounded OpenAI-math-inspired GRAIN paired-response research packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_response as M

DEFAULT_OUT = ROOT / "results/grain_math_response_v1/r1"
MODELS = {
    "e5": (
        "intfloat/e5-large-v2",
        "f169b11e22de13617baa190a028a32f3493550b6",
        "passage: ",
    ),
    "bge": ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09", ""),
}


def sha(path):
    """Hash source/artifact bytes without displaying their contents."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Write strict JSON; reject nonfinite metrics rather than hiding them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    """Write a rectangular metric table with structured values encoded as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def checkpoint(out, packet, status, evidence):
    """Update only this new bounded loop's progress, preserving old ledgers."""
    path = out / "execution_progress.json"
    progress = (
        json.loads(path.read_text())
        if path.exists()
        else {"scope": "MATH-GRAIN-00..05", "packets": {}}
    )
    progress["packets"][packet] = {
        "status": status,
        "evidence": evidence,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(path, progress)


def prepare(out):
    """Freeze the input/reader/material contracts before obtaining vector scores."""
    contract_path = out / "00/input_contract.json"
    if contract_path.exists():
        return
    families = json.loads(
        (ROOT / "configs/grain_math_material_families_v1.json").read_text()
    )
    assignment = np.random.default_rng(107).permutation(32)
    role = {
        int(f): ("source" if i < 16 else "validation" if i < 24 else "test")
        for i, f in enumerate(assignment)
    }
    materials = []
    for f, (name, setup, positive, negative) in enumerate(families):
        for bits in M.product((-1, 1), repeat=4):
            p, r, o, c = bits
            condition = (
                "There was enough time to arrange the work."
                if c == -1
                else "An unexpected deadline left very little time to arrange the work."
            )
            predicate = positive if p == 1 else negative
            event = f"I {predicate}."
            account = (
                f"This is my own account: {event} I am describing my own response, not another person's."
                if o == 1
                else f"A collaborator gave this account: '{event}' This describes the collaborator's response. My own response is not reported."
            )
            text = (
                f"Record of the event. {setup} {condition} {account} End of the recorded observations."
                if r == 1
                else f"Here is what happened. {setup} {condition} {account} That is the story."
            )
            materials.append(
                {
                    "material_id": f"f{f:02d}_"
                    + "".join(str((b + 1) // 2) for b in bits),
                    "family": f,
                    "content_name": name,
                    "role": role[f],
                    "bits": list(bits),
                    "text": text,
                    "event_id": f"synthetic_f{f:02d}",
                    "human_author_id": None,
                }
            )
    bank = out / "00/material_bank.jsonl"
    bank.parent.mkdir(parents=True, exist_ok=True)
    bank.write_text("".join(json.dumps(x) + "\n" for x in materials))
    write_csv(
        out / "00/family_split.csv", [{"family": f, "role": role[f]} for f in range(32)]
    )
    write_csv(
        out / "00/operation_declarations.csv",
        [
            {
                "operation": "pole",
                "meaning": "initiated coordination vs waited",
                "role": "target content",
            },
            {
                "operation": "register",
                "meaning": "record vs conversational frame",
                "role": "declared non-target for these units",
            },
            {
                "operation": "ownership",
                "meaning": "own account vs explicit other account",
                "role": "evidence qualification",
            },
            {
                "operation": "condition",
                "meaning": "ordinary time vs sudden deadline",
                "role": "recorded context, retained",
            },
        ],
    )
    registry = {}
    for reader, (name, revision, prefix) in MODELS.items():
        cache = (
            Path.home()
            / ".cache/huggingface/hub"
            / ("models--" + name.replace("/", "--"))
            / "snapshots"
            / revision
        )
        if not (cache / "model.safetensors").exists():
            raise FileNotFoundError(f"offline reader missing: {reader} {cache}")
        registry[reader] = {
            "model_id": name,
            "revision": revision,
            "snapshot": str(cache),
            "prefix": prefix,
            "normalize": True,
            "weights_sha256": sha(cache / "model.safetensors"),
            "pooling": json.loads((cache / "1_Pooling/config.json").read_text()),
            "max_length": 512,
            "local_files_only": True,
        }
    write_json(out / "00/reader_registry.json", registry)
    originals = [
        "C1/hidden_map_r4/source_application_contract.json",
        "C1/hidden_map_r4/frozen_tolerances.json",
        "C2/evidence_unit_r1/protocol_before_run.json",
        "C2/evidence_unit_r1/observation_law.json",
        "C4/transport_r1/protocol_before_run.json",
        "C4/transport_r1/decision.md",
    ]
    inputs = {p: sha(ROOT / "results/grain_construction_v1" / p) for p in originals}
    for reader in ("reader_recovery_r1", "nli_reader_r1"):
        p = (
            ROOT
            / "results/grain_gptpro_foundation_v3/F08"
            / reader
            / "reader_scores.csv"
        )
        inputs[str(p.relative_to(ROOT))] = sha(p)
    write_json(
        contract_path,
        {
            "historical_inputs": inputs,
            "materials": {"count": 512, "families": 32, "sha256": sha(bank)},
            "source_roles": [16, 8, 8],
            "worlds": list(M.WORLDS),
            "seeds": list(M.SEEDS),
            "world_roles": M.ROLES,
            "target_rho": [0, 0.6],
            "dimension": 64,
            "reader_truth_boundary": "paired controls allowed; hidden directions/evaluator parameters forbidden",
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "torch", "sentence-transformers")
            },
        },
    )
    write_json(
        out / "00/target_contracts.json",
        {
            "T1": "qualified own signed material expression",
            "T2": "condition-tagged qualified expression; not a single-text personal slope",
            "unit": "signed content code +/-1; own-vs-other gates contribution, not observed neutral authors",
            "test_policy": "fit/source and select/validation; score all test families once, then analyze frozen scores",
        },
    )
    (out / "00/resource_budget.md").write_text(
        "# Budget\n\n24 finite configurations; two frozen encoders, 512 texts each, batch 8.\nNo API, neural training, private MEPS, remote writes or old lockbox.\nSource-only linear fits and 1000 family bootstrap draws. MPS preferred, CPU fallback explicit.\n"
    )
    checkpoint(out, "00", "complete", [str(contract_path)])


def run_finite(out):
    """Run and preserve finite mechanism outcomes, including negative ones."""
    dest = out / "01A"
    if (dest / "comparisons.csv").exists():
        return
    write_json(
        dest / "world_contract.json",
        {
            "worlds": list(M.WORLDS),
            "roles": M.ROLES,
            "seeds": list(M.SEEDS),
            "noise_sd": 0.15,
            "source_covariance_rank": 2,
            "rho": [0, 0.6],
            "resolution_rule": "3 times aggregate source paired-column standard error",
            "truncated_constraint_tolerance": "2*source_resolution/spectral_norm(M); exact solver default unchanged",
            "map_source": "historical C1 world_maps; new matched-control observation protocol",
            "coverage": "conditional fixed-loading noise plus source linearized calibration uncertainty, not exact psychological posterior",
        },
    )
    rows, witnesses, arrays = M.finite_experiments()
    write_csv(dest / "comparisons.csv", rows)
    write_json(dest / "identifiability_witnesses.json", witnesses)
    np.savez_compressed(dest / "response_loadings.npz", **arrays)
    summaries = []
    for world, arm, rho in sorted({(r["world"], r["arm"], r["rho"]) for r in rows}):
        rr = [
            r
            for r in rows
            if r["world"] == world and r["arm"] == arm and r["rho"] == rho
        ]
        ok = [r for r in rr if r["status"] == "identified"]
        summaries.append(
            {
                "world": world,
                "arm": arm,
                "rho": rho,
                "identified_seeds": len(ok),
                **{
                    k: float(np.mean([r[k] for r in ok])) if ok else None
                    for k in ("unit_gain", "style_leak", "mse", "coverage")
                },
            }
        )
    write_csv(dest / "summary.csv", summaries)
    checkpoint(
        out,
        "01A",
        "complete",
        [str(dest / "summary.csv"), str(dest / "identifiability_witnesses.json")],
    )


def run_boundary(out):
    """Diagnose old frozen C2 channels; do not generate new model requests."""
    dest = out / "02"
    if (dest / "boundary_decision.json").exists():
        return
    rows, channels = [], {}
    for reader, folder in (("lexical", "reader_recovery_r1"), ("nli", "nli_reader_r1")):
        with (
            ROOT
            / "results/grain_gptpro_foundation_v3/F08"
            / folder
            / "reader_scores.csv"
        ).open() as stream:
            old = list(csv.DictReader(stream))
        for row in old:
            p, q = float(row["pole_score"]), float(row["qualification_score"])
            for axis, score, true in (
                ("pole", p, int(row["pole"])),
                ("qualification", q, int(row["qualification"])),
            ):
                rows.append(
                    {
                        "reader": reader,
                        "role": row["role"],
                        "family": row["family"],
                        "status": row["subject_status"],
                        "material_id": row["material_id"],
                        "axis": axis,
                        "score": score,
                        "truth": true,
                        "wrong": int((score >= 0.5) != bool(true)),
                        "margin": abs(score - 0.5),
                        "confident_wrong": int(
                            abs(score - 0.5) >= 0.25 and (score >= 0.5) != bool(true)
                        ),
                        "near_boundary": int(abs(score - 0.5) <= 0.1),
                    }
                )
            key = (reader, row["role"], int(row["pole"]), row["subject_status"])
            channels.setdefault(key, np.zeros(4))[2 * (p >= 0.5) + (q >= 0.5)] += 1
    write_csv(dest / "margin_by_family.csv", rows)
    summaries, shifts = [], []
    for reader, role, status, axis in sorted(
        {(r["reader"], r["role"], r["status"], r["axis"]) for r in rows}
    ):
        rr = [
            r
            for r in rows
            if (r["reader"], r["role"], r["status"], r["axis"])
            == (reader, role, status, axis)
        ]
        summaries.append(
            {
                "reader": reader,
                "role": role,
                "status": status,
                "axis": axis,
                "n": len(rr),
                **{
                    k: sum(r[k] for r in rr)
                    for k in ("wrong", "confident_wrong", "near_boundary")
                },
            }
        )
    for reader, role, pole, status in sorted(channels):
        if role == "source":
            continue
        a, b = (
            channels[reader, "source", pole, status],
            channels[reader, role, pole, status],
        )
        shifts.append(
            {
                "reader": reader,
                "role": role,
                "pole": pole,
                "status": status,
                "TV": float(np.sum(np.abs(a / a.sum() - b / b.sum())) / 2),
            }
        )
    write_csv(dest / "error_summary.csv", summaries)
    write_csv(dest / "channel_shift_decomposition.csv", shifts)
    decision = {}
    for reader in ("lexical", "nli"):
        rr = [r for r in rows if r["reader"] == reader and r["role"] != "source"]
        wrong = sum(r["wrong"] for r in rr)
        confident = sum(r["confident_wrong"] for r in rr)
        near_wrong = sum(r["wrong"] * r["near_boundary"] for r in rr)
        decision[reader] = {
            "errors": wrong,
            "confident_errors": confident,
            "near_boundary_errors": near_wrong,
            "soft_boundary_candidate": bool(wrong and near_wrong > wrong / 2),
            "recommendation": "new-family channel calibration, not temperature rescue"
            if confident
            else "no confident error mechanism observed",
        }
    write_json(dest / "boundary_decision.json", decision)
    checkpoint(out, "02", "complete", [str(dest / "boundary_decision.json")])


def encode(out, reader, device):
    """Offline frozen encoding with block-level restart and content-hash checks."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    bank = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    reg = json.loads((out / "00/reader_registry.json").read_text())[reader]
    dest = out / "01B/embeddings"
    dest.mkdir(parents=True, exist_ok=True)
    signature = {"bank_sha256": sha(out / "00/material_bank.jsonl"), "model": reg}
    sig_path = dest / f"{reader}_input_contract.json"
    manifest_path = dest / f"{reader}_manifest.json"
    if sig_path.exists() and json.loads(sig_path.read_text()) != signature:
        raise ValueError("encoding resume rejected: changed materials or reader")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["bank_sha256"] != signature["bank_sha256"]
            or manifest["model"] != reg
        ):
            raise ValueError("completed cache belongs to another input/reader")
        if sha(dest / f"{reader}.npy") != manifest["sha256"]:
            raise ValueError("completed embedding cache hash mismatch")
        write_json(sig_path, signature)
        print(f"{reader} verified complete cache; no encoder calls", flush=True)
        return
    write_json(sig_path, signature)
    model = SentenceTransformer(reg["snapshot"], device=device, local_files_only=True)
    model.max_seq_length = 512
    start = time.monotonic()
    parts = []
    for i in range(0, len(bank), 32):
        path = dest / f"{reader}_{i:04d}.npy"
        if path.exists():
            part = np.load(path)
        else:
            part = model.encode(
                [reg["prefix"] + row["text"] for row in bank[i : i + 32]],
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            np.save(path, part)
        if part.shape != (min(32, len(bank) - i), 1024) or not np.all(
            np.isfinite(part)
        ):
            raise ValueError(f"bad embedding block {path}")
        parts.append(part)
        print(
            f"{reader} {i + len(part)}/{len(bank)} elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )
    x = np.concatenate(parts)
    np.save(dest / f"{reader}.npy", x)
    write_json(
        dest / f"{reader}_manifest.json",
        {
            "bank_sha256": sha(out / "00/material_bank.jsonl"),
            "shape": list(x.shape),
            "device": device,
            "model": reg,
            "elapsed": time.monotonic() - start,
            "max_norm_error": float(np.max(np.abs(np.linalg.norm(x, axis=1) - 1))),
            "sha256": sha(dest / f"{reader}.npy"),
        },
    )


def run_text(out):
    """Freeze models/tolerances before a single held-family scoring pass."""
    dest = out / "01B"
    if (dest / "held_metrics.json").exists():
        return
    bank = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    bits, groups, roles = (
        np.array([r["bits"] for r in bank]),
        np.array([r["family"] for r in bank]),
        np.array([r["role"] for r in bank]),
    )
    models_by_reader, tolerance, info = {}, {}, {}
    for reader in MODELS:
        x = np.load(dest / f"embeddings/{reader}.npy").astype(float)
        models, info[reader] = M.fit_text_models(x, bits, groups, roles)
        models_by_reader[reader] = models
        # Source LOFO supplies a calibration-resolution bound, without test readout.
        lofo_errors = []
        for f in np.unique(groups[roles == "source"]):
            folded = roles.copy()
            folded[(roles == "source") & (groups == f)] = "validation"
            folded[roles == "validation"] = "unused"
            fitted, _ = M.fit_text_models(x, bits, groups, folded)
            mask = groups == f
            lofo_errors.append(
                M.score_bundle(fitted["paired_dual"], x[mask])
                - M.text_targets(bits[mask])
            )
        error = np.concatenate(lofo_errors)
        tolerance[reader] = {
            "unit_rmse_resolution": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "method": "source leave-family-out RMSE; a resolution bound, not an a priori clinical equivalence margin",
        }
        for arm, model in models.items():
            path = dest / f"reader_bundles/{reader}_{arm}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **model)
    write_json(dest / "tolerances.json", tolerance)
    write_json(dest / "source_fits.json", info)
    write_json(
        dest / "model_freeze.json",
        {
            "phase": "before_test_scoring",
            "families": {
                r: sorted(set(groups[roles == r].tolist()))
                for r in ("source", "validation", "test")
            },
            "bundles": {
                p.name: sha(p)
                for p in sorted((dest / "reader_bundles").glob("*.npz"))
                if not p.name.startswith("._")
            },
        },
    )
    metrics, response_rows, score_rows = {}, [], []
    test = roles == "test"
    for reader, models in models_by_reader.items():
        x = np.load(dest / f"embeddings/{reader}.npy").astype(float)
        metrics[reader] = {}
        for arm, model in models.items():
            scores = M.score_bundle(model, x)
            np.save(dest / f"scores_{reader}_{arm}.npy", scores)
            metrics[reader][arm] = []
            for target in range(2):
                effects = M.family_effects(
                    scores[test], bits[test], groups[test], target
                )
                gain = np.array([r["target_gain"] for r in effects])
                leak = np.array([r["style_rms"] for r in effects])
                other = np.array([r["other_target_gain"] for r in effects])
                mse = np.array(
                    [
                        np.mean(
                            (
                                scores[test & (groups == f), target]
                                - M.text_targets(bits[test & (groups == f)])[:, target]
                            )
                            ** 2
                        )
                        for f in np.unique(groups[test])
                    ]
                )
                metric = {
                    "target": target,
                    "target_gain": float(gain.mean()),
                    "gain_ci": M.bootstrap_interval(gain),
                    "positive_families": int((gain > 0).sum()),
                    "style_rms": float(leak.mean()),
                    "style_ci": M.bootstrap_interval(leak),
                    "other_gain": float(other.mean()),
                    "unit_rmse": float(np.sqrt(mse.mean())),
                    "mse_ci": M.bootstrap_interval(mse),
                    "source_resolution": tolerance[reader]["unit_rmse_resolution"][
                        target
                    ],
                }
                metrics[reader][arm].append(metric)
                response_rows.extend(
                    {"reader": reader, "arm": arm, "target": target, **r}
                    for r in effects
                )
            for k, row in enumerate(bank):
                score_rows.append(
                    {
                        "reader": reader,
                        "arm": arm,
                        "material_id": row["material_id"],
                        "family": row["family"],
                        "role": row["role"],
                        "T1": float(scores[k, 0]),
                        "T2": float(scores[k, 1]),
                    }
                )
    write_csv(dest / "paired_response_scores.csv", response_rows)
    write_csv(dest / "predictions.csv", score_rows)
    write_json(dest / "held_metrics.json", metrics)
    checkpoint(
        out,
        "01B",
        "complete",
        [str(dest / "held_metrics.json"), str(dest / "model_freeze.json")],
    )


def run_joint(out):
    """Test joint actions and held-combination score/response transport from cache."""
    dest = out / "03"
    if (dest / "joint_summary.json").exists():
        return
    bank = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    bits, groups, roles = (
        np.array([r["bits"] for r in bank]),
        np.array([r["family"] for r in bank]),
        np.array([r["role"] for r in bank]),
    )
    source, test = roles == "source", roles == "test"
    score = {
        reader: np.load(out / f"01B/scores_{reader}_paired_dual.npy")
        for reader in MODELS
    }
    summaries, interactions, cards = {}, [], []
    for reader in MODELS:
        full = score[reader]
        additive = np.load(out / f"01B/scores_{reader}_additive_space.npy")
        effects = M.family_effects(full[test], bits[test], groups[test], 0)
        joint = np.array([r["joint_pole_owner"] for r in effects])
        style = np.array([r["joint_pole_style"] for r in effects])
        improvement = []
        for f in np.unique(groups[test]):
            mask = test & (groups == f)
            y = M.text_targets(bits[mask])[:, 0]
            improvement.append(
                float(np.mean((additive[mask, 0] - y) ** 2 - (full[mask, 0] - y) ** 2))
            )
        summaries[reader] = {
            "joint_pole_owner": float(joint.mean()),
            "joint_ci": M.bootstrap_interval(joint),
            "pole_style_joint": float(style.mean()),
            "pole_style_ci": M.bootstrap_interval(style),
            "full_minus_additive_mse_gain": float(np.mean(improvement)),
            "increment_ci": M.bootstrap_interval(np.array(improvement)),
        }
        interactions.extend({"reader": reader, **r} for r in effects)
        for r in effects:
            cards.append(
                {
                    "reader": reader,
                    "family": r["family"],
                    "operations": ["pole", "ownership"],
                    "checked_proper_subsets": [[], ["pole"], ["ownership"]],
                    "joint_contrast": r["joint_pole_owner"],
                    "meaning": "qualification changes whether content counts as own evidence",
                    "scope": "this frozen four-corner operation library; not global token minimality",
                }
            )
    # A maps two declared score coordinates, not an unrestricted embedding space.
    withheld = (bits[:, 2] == -1) & (bits[:, 0] == 1)
    fit = source & ~withheld
    design = np.column_stack([np.ones(fit.sum()), score["e5"][fit]])
    a = np.linalg.lstsq(design, score["bge"][fit], rcond=None)[0]
    residual = np.column_stack([np.ones(len(bank)), score["e5"]]) @ a - score["bge"]
    transport = []
    for f in np.unique(groups[test]):
        mask = groups == f
        e, z = residual[mask], bits[mask]
        level = np.sqrt(np.mean(e**2, axis=0))
        first = np.mean(e * z[:, 0, None], axis=0) * 2
        joint = np.mean(e * (z[:, 0] * z[:, 2])[:, None], axis=0) * 4
        for target in range(2):
            transport.append(
                {
                    "family": int(f),
                    "target": target,
                    "level_rmse": float(level[target]),
                    "pole_defect": float(first[target]),
                    "pole_owner_defect": float(joint[target]),
                    "held_combination_rmse": float(
                        np.sqrt(np.mean(e[withheld[mask], target] ** 2))
                    ),
                }
            )
    write_csv(dest / "interaction_by_family.csv", interactions)
    write_csv(dest / "transport_defects.csv", transport)
    write_json(
        dest / "transport_map.json",
        {
            "source_map": a.tolist(),
            "fit_rows": int(fit.sum()),
            "withheld_combination": "other + positive",
            "held_families": np.unique(groups[test]).tolist(),
        },
    )
    write_json(dest / "witness_cards.json", cards)
    write_json(dest / "joint_summary.json", summaries)
    # A duplicate is not a second event. This is an algebraic ledger check, not
    # empirical author reliability inferred from the single-event text bank.
    event_ids = ["event-A", "event-A", "event-B"]
    write_json(
        dest / "event_count_check.json",
        {
            "raw_rows": len(event_ids),
            "event_ids": event_ids,
            "independent_event_count": len(set(event_ids)),
            "scope": "finite duplicate accounting only; no new author stability claim",
        },
    )
    checkpoint(
        out,
        "03",
        "complete",
        [str(dest / "joint_summary.json"), str(dest / "transport_defects.csv")],
    )


def run_dynamics(out):
    """Complete the bounded coupled-state prerequisite; no natural-path claim."""
    rows, fits = M.dynamics_probe()
    write_csv(out / "04/transient_responses.csv", rows)
    write_json(out / "04/source_fits.json", fits)
    checkpoint(
        out, "04", "finite_precheck_complete", [str(out / "04/transient_responses.csv")]
    )


def main():
    """Execute one finite packet, supporting safe encoding resumes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "finite",
            "boundary",
            "encode",
            "text",
            "joint",
            "dynamics",
            "all",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reader", choices=tuple(MODELS))
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()
    stages = {
        "prepare": prepare,
        "finite": run_finite,
        "boundary": run_boundary,
        "text": run_text,
        "joint": run_joint,
        "dynamics": run_dynamics,
    }
    start = time.monotonic()
    if args.stage == "all":
        prepare(args.out)
        run_finite(args.out)
        run_boundary(args.out)
        for reader in MODELS:
            encode(args.out, reader, args.device)
        run_text(args.out)
        run_joint(args.out)
        run_dynamics(args.out)
        followup = ROOT / "scripts/run_grain_math_binding_followup.py"
        subprocess.run(
            [sys.executable, str(followup), "prepare", "--out", str(args.out)],
            check=True,
        )
        for reader in MODELS:
            encode(args.out / "06", reader, args.device)
        subprocess.run(
            [sys.executable, str(followup), "run", "--out", str(args.out)], check=True
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_grain_math_response.py"),
                "--out",
                str(args.out),
            ],
            check=True,
        )
    elif args.stage == "encode":
        if not args.reader:
            parser.error("encode requires --reader")
        encode(args.out, args.reader, args.device)
    else:
        stages[args.stage](args.out)
    print(
        f"stage={args.stage} completed elapsed={time.monotonic() - start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
