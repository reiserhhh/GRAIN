#!/usr/bin/env python3
"""Result-after HM71 interpretation: retain means, separate moment origin effects."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_cold_relational as M
from grain_research.construction.math_ownership_composition import STAGES
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_cold_relational import OUT, check, load_chart
from scripts.run_grain_math_common_observables import SOURCES, write, write_csv

CODE = (
    "docs/GRAIN_RELATIONAL_ORIGIN_DIAGNOSTIC_HM71.md",
    "scripts/diagnose_grain_cold_moment_origin.py",
    "tests/test_grain_cold_moment_origin.py",
)


def covariance_pair(p, z):
    """Preserve location and compute within-profile covariance by two formulas."""
    observation = M.moments(p, z)
    covariance = observation["second"] - observation["independent_second"]
    direct = np.array(
        [
            [
                [
                    math.fsum(
                        float(p[u, e])
                        * float(z[e, i] - observation["means"][u, i])
                        * float(z[e, j] - observation["means"][u, j])
                        for e in range(len(z))
                    )
                    for j in range(z.shape[1])
                ]
                for i in range(z.shape[1])
            ]
            for u in range(2)
        ]
    )
    return observation, covariance, float(np.max(abs(covariance - direct)))


def run(out):
    """Save separate derived accounts without altering or recalibrating primary data."""
    check(out)
    if (out / "04_origin_diagnostic").exists():
        raise FileExistsError("Preserve the existing result-after companion")
    for name, sha in read_json(out / "03/output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Primary output drift")
    if read_json(out / "verification.json")["actual_calls"] != 2880:
        raise ValueError("Complete actual primary readings required")
    destination = out / "04_origin_diagnostic"
    destination.mkdir()
    code = {}
    for name in CODE:
        target = destination / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code[name] = digest(target)
    write(
        destination / "manifest.json",
        {
            "result_after_primary": True,
            "new_calls_fits": [0, 0],
            "code": code,
            "primary_seal_sha256": digest(out / "03/output_seal.json"),
            "profile_sha256": digest(SOURCES["HM69"] / "01/profiles.npz"),
            "not_replacement_score_or_cold_recalibration": True,
        },
    )
    with np.load(SOURCES["HM69"] / "01/profiles.npz", allow_pickle=False) as data:
        profiles = data["probabilities"].reshape(30, 8, 2, 8)
    rows, tensors, maximum_error, scalar_checks = [], {}, 0.0, 0
    with np.load(out / "03/observations_and_moments.npz", allow_pickle=False) as data:
        for form, stage, pair_id in itertools.product(M.FORMS, STAGES, range(30)):
            key = f"{form}_{stage}_pair{pair_id:02d}"
            cold, source = data[key + "__z"], load_chart(stage, pair_id)["basis"]
            b = cold.mean(axis=0)
            for case_id in (6, 7):
                p = profiles[pair_id, case_id]
                _src, src_cov, error0 = covariance_pair(p, source)
                new, cold_cov, error1 = covariance_pair(p, cold)
                shifted, shifted_cov, error2 = covariance_pair(p, cold - b)
                _, wrong_cov, error3 = covariance_pair(p, np.roll(cold, 1, axis=0))
                ds, dn = src_cov[1] - src_cov[0], cold_cov[1] - cold_cov[0]
                raw = new["second"][1] - new["second"][0]
                independent = (
                    new["independent_second"][1] - new["independent_second"][0]
                )
                dm = new["means"][1] - new["means"][0]
                translation = np.outer(b, dm) + np.outer(dm, b)
                changed = raw - (shifted["second"][1] - shifted["second"][0])
                errors = (
                    error0,
                    error1,
                    error2,
                    error3,
                    np.max(abs(cold_cov - shifted_cov)),
                    np.max(abs(changed - translation)),
                )
                maximum_error = max(maximum_error, *map(float, errors))
                scalar_checks += 4 * cold_cov.size + cold_cov.size + raw.size
                if max(errors) > 1e-7 * max(1.0, np.max(abs(new["second"]))):
                    raise ValueError("Centered covariance/translation identity failed")
                tensors[f"{key}__case{case_id}"] = np.stack(
                    (ds, raw, dn, independent, wrong_cov[1] - wrong_cov[0], translation)
                )
                rows.append(
                    {
                        "form": form,
                        "stage": stage,
                        "pair_id": pair_id,
                        "case_id": case_id,
                        "source_covariance_norm": float(np.linalg.norm(ds)),
                        "cold_second_norm": float(np.linalg.norm(raw)),
                        "cold_independent_norm": float(np.linalg.norm(independent)),
                        "cold_covariance_norm": float(np.linalg.norm(dn)),
                        "cold_covariance_cosine": M.cosine(ds, dn),
                        "wrong_covariance_cosine": M.cosine(
                            ds, wrong_cov[1] - wrong_cov[0]
                        ),
                        "origin_translation_norm": float(np.linalg.norm(translation)),
                        "translation_invariance_max_error": float(
                            np.max(abs(cold_cov - shifted_cov))
                        ),
                    }
                )
    summaries = []
    for form, stage in itertools.product(M.FORMS, STAGES):
        chosen = [r for r in rows if r["form"] == form and r["stage"] == stage]
        summaries.append(
            {
                "form": form,
                "stage": stage,
                "rows": len(chosen),
                "covariance_direction_mean": float(
                    np.mean([r["cold_covariance_cosine"] for r in chosen])
                ),
                "wrong_covariance_direction_mean": float(
                    np.mean([r["wrong_covariance_cosine"] for r in chosen])
                ),
                "covariance_RMS_norm": float(
                    np.sqrt(np.mean([r["cold_covariance_norm"] ** 2 for r in chosen]))
                ),
                "independent_RMS_norm": float(
                    np.sqrt(np.mean([r["cold_independent_norm"] ** 2 for r in chosen]))
                ),
                "translation_term_RMS_norm": float(
                    np.sqrt(
                        np.mean([r["origin_translation_norm"] ** 2 for r in chosen])
                    )
                ),
            }
        )
    write_csv(destination / "rows.csv", rows)
    np.savez_compressed(destination / "tensors.npz", **tensors)
    write(
        destination / "summary.json",
        {
            "result_after_primary_not_preregistered": True,
            "rows": len(rows),
            "stages": summaries,
            "scalar_checks": scalar_checks,
            "max_arithmetic_error": maximum_error,
            "new_calls_fits": [0, 0],
            "primary_not_changed": True,
            "same_agent_alternate_arithmetic_not_third_party": True,
        },
    )
    files = [
        destination / n
        for n in ("manifest.json", "rows.csv", "tensors.npz", "summary.json")
    ]
    write(destination / "output_seal.json", {p.name: digest(p) for p in files})
    print(
        f"HM71 origin companion:720 rows,{scalar_checks} scalars,maxerror={maximum_error:.3g}"
    )


def verify(out):
    """Verify companion seals and the unchanged primary/cold source provenance."""
    check(out)
    destination = out / "04_origin_diagnostic"
    for name, sha in read_json(destination / "output_seal.json").items():
        if digest(destination / name) != sha:
            raise ValueError("Derived output drift")
    manifest = read_json(destination / "manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(destination / "code" / name) != sha:
            raise ValueError("Derived code drift")
    if digest(out / "03/output_seal.json") != manifest["primary_seal_sha256"]:
        raise ValueError("Primary changed")
    if digest(SOURCES["HM69"] / "01/profiles.npz") != manifest["profile_sha256"]:
        raise ValueError("Profiles changed")
    print("HM71 origin companion seals verified; primary intact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    args = parser.parse_args()
    {"run": run, "verify": verify}[args.action](OUT)
