#!/usr/bin/env python3
"""Second-formula replay and estimand/unit-transport diagnosis, no new forwards."""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from run_grain_math_evidence import DEFAULT_OUT, OLD, ROOT, load_bundle
from run_grain_math_response import sha, write_csv, write_json

from grain_research.construction.math_evidence import replay_qualified
from grain_research.construction.math_response import score_bundle


def population_readout_operator(bank, scores, families):
    """Build the expectation map from controlled p/c cells, not author truths.

    E[T1 | a,b,c] = alpha_c + beta_c(a+bc). These units concern the
    stipulated binary-event world only, not an unobserved human-personality law.
    """
    alpha, beta = [], []
    for c in (-1, 1):
        means = []
        for p in (-1, 1):
            take = [
                i
                for i, row in enumerate(bank)
                if row["family"] in families
                and row["bits"][2] == 1
                and row["bits"][3] == c
                and row["bits"][0] == p
            ]
            if not take:
                raise ValueError("Controlled endpoint action missing")
            means.append(float(np.mean(scores[take])))
        alpha.append((means[0] + means[1]) / 2)
        beta.append((means[1] - means[0]) / 2)
    offset = np.array([(alpha[0] + alpha[1]) / 2, (alpha[1] - alpha[0]) / 2])
    matrix = np.array(
        [
            [(beta[0] + beta[1]) / 2, (beta[1] - beta[0]) / 2],
            [(beta[1] - beta[0]) / 2, (beta[0] + beta[1]) / 2],
        ]
    )
    return offset, matrix


def verify(out: Path, old: Path):
    """Recompute cold scores, posterior precision and target-unit boundaries."""
    errors, rows, operators = {}, [], {}
    bank = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    oldbank = [
        json.loads(s)
        for s in (old / "06/00/material_bank.jsonl").read_text().splitlines()
    ]
    models = json.loads((out / "source_working_models.json").read_text())
    schedule = [
        (b, c, e, r)
        for b in range(4)
        for c in (-1, 1)
        for e in range(2)
        for r in range(2)
    ]
    design = np.array([[1, c] for _, c, _, _ in schedule], float)
    block = np.array([b for b, _, _, _ in schedule])
    event = np.array([(b, c, e) for b, c, e, _ in schedule])
    source_families = sorted({row["family"] for row in oldbank})[:8]
    app_families = sorted({row["family"] for row in oldbank})[8:]
    for reader in ("e5", "bge"):
        bundle = load_bundle(old, reader)
        x = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        saved = np.load(out / f"scores_{reader}.npy")
        errors[f"{reader}_cold"] = float(np.max(abs(score_bundle(bundle, x) - saved)))
        artifact = json.loads(
            (out / f"01B/embeddings/{reader}_manifest.json").read_text()
        )
        if sha(out / f"01B/embeddings/{reader}.npy") != artifact["sha256"]:
            raise ValueError("Embedding hash mismatch")
        if (
            sha(old / f"06/reader_bundles/{reader}_joint_binding.npz")
            != json.loads((out / "00/protocol_before_run.json").read_text())[
                "reader_bundles"
            ][reader]
        ):
            raise ValueError("Frozen program changed")
        if not np.isfinite(saved).all() or len(saved) != len(bank):
            raise ValueError("Nonfinite or unaligned scores")
        oldx = np.load(old / f"06/01B/embeddings/{reader}.npy").astype(float)
        old_scores = score_bundle(bundle, oldx)[:, 0]
        source_offset, source_map = population_readout_operator(
            oldbank, old_scores, source_families
        )
        app_offset, app_map = population_readout_operator(
            oldbank, old_scores, app_families
        )
        operators[reader] = {
            "source_offset": source_offset.tolist(),
            "source_map": source_map.tolist(),
            "application_offset_evaluator_only": app_offset.tolist(),
            "application_map_evaluator_only": app_map.tolist(),
            "source_inverse_resolution": float(np.linalg.svd(source_map)[1][-1]),
            "source_inverse_error_offset": np.linalg.solve(
                source_map, app_offset - source_offset
            ).tolist(),
            "source_inverse_error_matrix": np.linalg.solve(
                source_map, app_map
            ).tolist(),
        }
        for seed in (71, 72, 73):
            model = models[f"{reader}:{seed}"]
            data = np.load(out / f"panel_{reader}_{seed}.npz")
            comp = np.array(model["components"])
            noise = (
                comp[2] * (block[:, None] == block[None, :])
                + comp[3] * np.all(event[:, None, :] == event[None, :, :], axis=2)
                + comp[4] * np.eye(len(schedule))
            )
            inverse_x = np.linalg.solve(noise, design)
            precision = np.diag(1 / comp[:2]) + design.T @ inverse_x
            cov = np.linalg.inv(precision)
            mean = (
                data["values"] @ inverse_x + np.array(model["mean"]) / comp[:2]
            ) @ cov
            errors[f"{reader}_{seed}_precision_mean"] = float(
                np.max(abs(mean - data["mean"]))
            )
            errors[f"{reader}_{seed}_precision_covariance"] = float(
                np.max(abs(cov - data["covariance"]))
            )
            latent = data["truth"]
            expected_readout = latent @ app_map.T + app_offset
            corrected = np.linalg.solve(source_map, (mean - source_offset).T).T
            inverse = np.linalg.inv(source_map)
            corrected_cov = inverse @ cov @ inverse.T
            for label, target, prediction, working_cov in (
                ("ideal_event_units_raw", latent, mean, cov),
                ("application_readout_units", expected_readout, mean, cov),
                (
                    "source_only_affine_unit_correction",
                    latent,
                    corrected,
                    corrected_cov,
                ),
            ):
                error = prediction - target
                sd = np.sqrt(np.diag(working_cov))
                rows.append(
                    {
                        "reader": reader,
                        "seed": seed,
                        "estimand": label,
                        "level_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
                        "response_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
                        "level_coverage": float(
                            np.mean(abs(error[:, 0]) <= 1.96 * sd[0])
                        ),
                        "response_coverage": float(
                            np.mean(abs(error[:, 1]) <= 1.96 * sd[1])
                        ),
                        "level_bias": float(np.mean(error[:, 0])),
                        "response_bias": float(np.mean(error[:, 1])),
                    }
                )
    aliases = pd.read_csv(out / "duplicate_controls.csv")
    if (
        aliases[["ledger_level_change", "ledger_response_change"]]
        .abs()
        .to_numpy()
        .max()
        > 1e-12
    ):
        raise ValueError("Aliases changed event profiles")
    checks = pd.read_csv(out / "ledger_checks.csv")
    if (
        not checks.revision_removes_parent.all()
        or not checks.withdrawal_keeps_other_report.all()
        or not checks.excluded_rows.eq(2).all()
        or not checks.dequalified_revision_removes_old_parent.all()
        or not checks.dequalified_revision_not_scored.all()
    ):
        raise ValueError("Evidence version/qualification failure")
    if max(errors.values()) > 1e-9:
        raise ValueError(f"Arithmetic disagreement: {max(errors.values())}")
    old_code = out.parent / "evidence_r1/followup_r1/code_snapshot/math_evidence.py"
    if old_code.exists():
        fixture = [
            {
                "record_id": "old",
                "author_id": "u",
                "event_id": "e",
                "block_id": "b",
                "kind": "new",
                "qualification": "own_actual",
                "condition": 1,
                "value": 0.8,
            },
            {
                "record_id": "correction",
                "author_id": "u",
                "event_id": "e",
                "block_id": "b",
                "kind": "revise",
                "parent_id": "old",
                "qualification": "other_actual",
                "condition": 1,
                "value": None,
            },
        ]
        prior = runpy.run_path(str(old_code))["replay_qualified"](fixture)
        repaired = replay_qualified(fixture)
        if len(prior["records"]) != 1 or repaired["records"]:
            raise ValueError("Qualification revision repair not reproduced")
        write_json(
            out / "qualification_revision_repair.json",
            {
                "initial_code_sha256": sha(old_code),
                "initial_qualified_records_after_correction": len(prior["records"]),
                "repaired_qualified_records_after_correction": len(repaired["records"]),
                "mechanism": "version replay precedes qualification routing",
            },
        )
    write_json(
        out / "verifier_code_manifest.json",
        {
            "verifier_sha256": sha(Path(__file__)),
            "evidence_module_sha256": sha(
                ROOT / "grain_research/construction/math_evidence.py"
            ),
        },
    )
    write_csv(out / "estimand_audit.csv", rows)
    write_json(
        out / "unit_transport_operators.json",
        {
            "result_guided_diagnostic": True,
            "source_families": source_families,
            "application_families": app_families,
            "operators": operators,
            "correction_scope": "source controlled-cell means only, never latent authors; diagnostic reuse of old panels, not a new independent confirmation",
            "application_map_usage": "evaluation/transport diagnosis only, never used to correct predictions",
        },
    )
    write_json(
        out / "verification.json",
        {
            "status": "PASS",
            "errors": errors,
            "max_absolute_error": max(errors.values()),
            "independent_human_audit": False,
            "scope": "same-agent second-formula precision/cold replay, version checks, no new model forward",
        },
    )
    outputs = {
        str(p.relative_to(out)): sha(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    }
    write_json(out / "artifact_hashes.json", outputs)
    progress = json.loads((out / "execution_progress.json").read_text())
    progress.update(
        status="finite_packet_executed_and_numerically_verified",
        verification="verification.json",
        natural_unit_discovery="target and eligibility shifts measured, not natural personality accuracy",
    )
    write_json(out / "execution_progress.json", progress)
    # Include the final progress bytes rather than the pre-review version.
    outputs["execution_progress.json"] = sha(out / "execution_progress.json")
    write_json(out / "artifact_hashes.json", outputs)
    print(
        json.dumps(
            {
                "status": "PASS",
                "max_error": max(errors.values()),
                "artifacts": len(outputs),
            }
        )
    )


def main():
    """Audit this new finite packet, without changing frozen scientific scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old", type=Path, default=OLD)
    args = parser.parse_args()
    verify(args.out, args.old)


if __name__ == "__main__":
    main()
