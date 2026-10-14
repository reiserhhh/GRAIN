#!/usr/bin/env python3
"""Focused tests for public calibrated F16 bundles and the sealed cold scorer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_serialize as pub
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1_source
from grain_research.gptpro_foundation_v3.f12_transport import fit_transport
from scripts.audit_grain_gptpro_f16_serialize import (
    independent_dynamic,
    independent_t1,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results/grain_gptpro_foundation_v3/F16/serialize_smoke_r1/bundles"
CANON = (
    ROOT
    / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1/scores/20260915"
)


def _spd(rng, n, scale):
    a = rng.normal(size=(n, n)) * scale
    return (a @ a.T + np.eye(n) * scale).tolist()


def _calibration(rng, z, authors, draws, mode="global"):
    old = rng.normal(size=(authors, len(z)))
    new = 1 + 1.5 * old + rng.normal(scale=0.05, size=old.shape)
    tiled = np.tile(z, authors)
    plugin = fit_transport(old.ravel(), new.ravel(), tiled, mode)
    indices = rng.integers(0, authors, (draws, authors))
    fits = [
        fit_transport(old[ix].ravel(), new[ix].ravel(), tiled, mode) for ix in indices
    ]
    as_json = json.loads(
        json.dumps({"plugin": plugin, "draws": fits}, default=lambda x: x.tolist())
    )
    return {
        "kind": "numeric_affine_pilot",
        "mode": mode,
        "status": "READY",
        "plugin": as_json["plugin"],
        "draws": as_json["draws"],
        "draw_count": draws,
        "pilot_fit_pairs": {
            "old": old.tolist(),
            "new": new.tolist(),
            "conditions": z.tolist(),
        },
        "bootstrap_indices": indices.tolist(),
        "data_provenance": {"role": "pilot"},
    }


def _t1_bundle(estimator="mixture", seed=3):
    rng = np.random.default_rng(seed)
    models = [
        {"mu": rng.normal(size=4).tolist(), "Sigma": _spd(rng, 4, 0.3), "sigma2": 0.4}
        for _ in range(4)
    ]
    engine = {
        "scorer_code_hash": t1_source.code_hash(),
        "formula_version": "T1_four_cell_source_mixture_v1",
        "target_id": "T1",
        "world_contract": "F04_L_four_condition",
        "reader_hash": t1_source.READER_HASH,
        "source_bootstraps": 4,
        "source_models": models,
        "reference_mu": 0.2,
        "reference_sigma": 0.9,
    }
    z = np.tile(np.array(t1_source.CONDITIONS), 2)
    return pub.build_bundle(
        route="numeric_t1",
        estimator=estimator,
        seed=seed,
        target="T1",
        unit="test unit",
        observation={
            "route": "numeric",
            "definition": {"world": "global"},
            "cache_sha256": None,
        },
        source={"components": 4},
        calibration=_calibration(rng, z, 10, 5),
        reference={
            "kind": "INHERITED_COMMON_UNIT",
            "mu": 0.2,
            "sigma": 0.9,
            "application_derived": False,
            "scope": "test",
        },
        engine=engine,
        known_validation={},
    ), z


def _t1_requests(bundle, z, n=3, seed=9):
    rng = np.random.default_rng(seed)
    return [
        pub.make_request(
            bundle,
            {
                "author_id": f"a{i}",
                "records": [
                    {"t": t, "z": float(z[t]), "y": float(1 + 1.5 * rng.normal())}
                    for t in range(len(z))
                ],
            },
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("estimator", ["mixture", "plugin"])
def test_numeric_t1_matches_independent_event_level_formula(estimator):
    bundle, z = _t1_bundle(estimator)
    requests = _t1_requests(bundle, z)
    rows = pub.score(bundle, requests)
    assert [r["status"] for r in rows] == ["SCORED"] * 3
    new = np.array([[x["y"] for x in q["data"]["records"]] for q in requests])
    score, variance = independent_t1(bundle, new, z)
    assert np.max(np.abs(score - [r["score"] for r in rows])) < 1e-10
    assert np.max(np.abs(variance - [r["variance"] for r in rows])) < 1e-10
    for row in rows:
        assert row["explanation"]["identity_error"] < 1e-12
        assert row["reference"]["policy"] == "INHERITED_COMMON_UNIT"
        parts = row["components"]
        assert abs(sum(parts.values()) - row["variance"]) < 1e-12
    if estimator == "plugin":
        assert all(r["components"]["pilot"] == 0 for r in rows)


def test_numeric_request_rejections_are_row_level():
    bundle, z = _t1_bundle()
    good = _t1_requests(bundle, z, n=1)[0]
    cases = []
    for field, value, code in (
        ("reader_id", "0" * 64, "WRONG_READER"),
        ("target_id", "T3", "WRONG_TARGET"),
        ("bundle_id", "1" * 64, "WRONG_VERSION"),
        ("target_time", 47, "WRONG_CLOCK"),
    ):
        bad = copy.deepcopy(good)
        bad["contract"][field] = value
        cases.append((bad, code))
    hidden = copy.deepcopy(good)
    hidden["data"]["truth"] = 1.0
    cases.append((hidden, "HIDDEN_OR_UNDECLARED_FIELD"))
    partial = copy.deepcopy(good)
    partial["data"]["records"] = [
        r for r in partial["data"]["records"] if r["z"] != 1.0
    ]
    cases.append((partial, "INCOMPLETE_CONDITION_SUPPORT"))
    off = copy.deepcopy(good)
    off["data"]["records"][0]["z"] = 0.5
    cases.append((off, "UNSUPPORTED_CONDITION"))
    late = copy.deepcopy(good)
    late["data"]["records"][0]["t"] = 32
    cases.append((late, "WRONG_CLOCK"))
    rows = pub.score(bundle, [good] + [c[0] for c in cases])
    assert rows[0]["status"] == "SCORED"
    assert [r["code"] for r in rows[1:]] == [c[1] for c in cases]


def test_bundle_tamper_and_construction_refusals():
    bundle, _ = _t1_bundle()
    tampered = copy.deepcopy(bundle)
    tampered["calibration"]["draws"][0]["coefficients"][0] += 1e-6
    with pytest.raises(pub.Refusal) as caught:
        pub.verify(tampered)
    assert caught.value.code == "BUNDLE_HASH_MISMATCH"
    tampered["bundle_id"] = pub.body_id(tampered)
    with pytest.raises(pub.Refusal) as caught:
        pub.verify(tampered)
    assert caught.value.code == "CALIBRATION_DRAWS_NOT_REPRODUCIBLE"
    norm = copy.deepcopy(bundle)
    norm["reference_policy"]["application_derived"] = True
    norm["bundle_id"] = pub.body_id(norm)
    with pytest.raises(pub.Refusal) as caught:
        pub.verify(norm)
    assert caught.value.code == "WRONG_REFERENCE_POLICY"
    base = {
        "route": "numeric_t1",
        "estimator": "mixture",
        "seed": 1,
        "target": "T1",
        "unit": "u",
        "observation": {"route": "numeric", "definition": {}, "cache_sha256": None},
        "source": {},
        "engine": {},
        "known_validation": {},
        "reference": {"kind": "INHERITED_COMMON_UNIT", "application_derived": False},
    }
    for calibration, code in (
        (
            {"status": "MISSING_ANCHOR", "data_provenance": {"role": "pilot"}},
            "MISSING_ANCHOR",
        ),
        (
            {
                "status": "READY",
                "plugin": {"status": "NONMONOTONE_INVERSE"},
                "data_provenance": {"role": "pilot"},
            },
            "NONMONOTONE_INVERSE",
        ),
        (
            {"status": "READY", "data_provenance": {"role": "application"}},
            "APPLICATION_CALIBRATION_NOT_OPERATIONAL",
        ),
        (
            {
                "status": "READY",
                "plugin": {"status": "READY"},
                "draws": [{"status": "MISSING_ANCHOR"}],
                "data_provenance": {"role": "pilot"},
            },
            "INCOMPLETE_CALIBRATION_DRAWS",
        ),
    ):
        with pytest.raises(pub.Refusal) as caught:
            pub.build_bundle(calibration=calibration, **base)
        assert caught.value.code == code
    with pytest.raises(pub.Refusal) as caught:
        pub.build_bundle(
            calibration={
                "status": "READY",
                "plugin": {"status": "READY"},
                "draws": [],
                "data_provenance": {"role": "pilot"},
            },
            **dict(
                base,
                estimator="plugin",
                reference={
                    "kind": "INHERITED_COMMON_UNIT",
                    "application_derived": True,
                },
            ),
        )
    assert caught.value.code == "APPLICATION_REFERENCE_FORBIDDEN"


def _dynamic_bundle(seed=5):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(2):
        models.append(
            {
                "prior": {
                    "mu": [1.0 + 0.1 * rng.normal(), 0.2, 0.5],
                    "Sigma": [[1.0, 0.05, 0.0], [0.05, 0.5, 0.0], [0.0, 0.0, 0.1]],
                },
                "log_sigma": [np.log(0.5), np.log(0.5)],
            }
        )
    z = np.where((np.arange(48) // 4) % 2 == 0, -1.0, 1.0)
    engine = {
        "formula": "test",
        "eta_nodes": 64,
        "source_components": 2,
        "source_models": models,
    }
    bundle = pub.build_bundle(
        route="numeric_dynamic",
        estimator="mixture",
        seed=seed,
        target="T3",
        unit="u",
        observation={
            "route": "numeric",
            "definition": {"world": "global"},
            "cache_sha256": None,
        },
        source={"components": 2},
        calibration=_calibration(rng, z, 6, 3),
        reference={
            "kind": "FRESH_SOURCE_SCORED",
            "mu": 0.0,
            "sigma": 1.0,
            "application_derived": False,
            "scope": "test",
        },
        engine=engine,
        known_validation={},
    )
    return bundle, z


def test_numeric_dynamic_matches_direct_conditioning_and_attribution_identity():
    bundle, z = _dynamic_bundle()
    rng = np.random.default_rng(11)
    new = 1 + 1.5 * (1 + rng.normal(size=(2, 48)))
    requests = [
        pub.make_request(
            bundle,
            {
                "author_id": f"d{i}",
                "records": [
                    {"t": t, "z": float(z[t]), "y": float(new[i, t])} for t in range(48)
                ],
            },
        )
        for i in range(2)
    ]
    gap = copy.deepcopy(requests[0])
    gap["data"]["records"] = gap["data"]["records"][:-1]
    rows = pub.score(bundle, [*requests, gap])
    assert rows[2]["code"] == "INCOMPLETE_CLOCK_SUPPORT"
    mean, cov = independent_dynamic(bundle, new, z)
    for i in range(2):
        assert abs(mean[i, 1] - rows[i]["score"]) < 1e-8
        assert abs(cov[i, 1, 1] - rows[i]["variance"]) < 1e-8
        assert abs(mean[i, 0] - rows[i]["joint"]["mean_T2_T3"][0]) < 1e-8
        assert rows[i]["explanation"]["identity_error"] < 1e-10


@pytest.mark.skipif(not SMOKE.exists(), reason="serialize smoke artifacts absent")
def test_text_bundle_reconciles_legacy_fields_and_replays_archive():
    bundle = json.loads((SMOKE / "text_dynamic_20260915_mixture_T3.json").read_text())
    pub.verify(bundle)
    legacy = bundle["legacy_reconciliation"]
    check = legacy["inherited_inner_reader_hash_check"]
    assert check["identifies_old_reader"] and not check["identifies_changed_reader"]
    assert (
        "reader_hash" in legacy["moved_out"] and "reader_hash" not in bundle["engine"]
    )
    stale = copy.deepcopy(bundle)
    stale["engine"]["reader_hash"] = check["legacy_inner_reader_hash"]
    stale["bundle_id"] = pub.body_id(stale)
    swap = copy.deepcopy(bundle)
    for key in ("hybrid_joint", "nli_joint"):
        d = swap["engine"]["channel_draws"][key]
        d[0], d[1] = d[1], d[0]
    swap["bundle_id"] = pub.body_id(swap)
    for obj, code in (
        (stale, "STALE_LEGACY_FIELD_PRESENT"),
        (swap, "CALIBRATION_DRAWS_NOT_REPRODUCIBLE"),
    ):
        with pytest.raises(pub.Refusal) as caught:
            pub.verify(obj)
        assert caught.value.code == code
    raw = [
        json.loads(x)
        for x in (CANON / "dynamic_inputs.jsonl").read_text().splitlines()[:2]
    ]
    rows = pub.score(bundle, [pub.make_request(bundle, r) for r in raw])
    canon = json.loads((CANON / "dynamic_changed_pilot_mixture.json").read_text())[:2]
    for c, r in zip(canon, rows, strict=True):
        assert c["targets"]["T3"]["score"] == r["score"]
        assert c["joint_covariance"] == r["engine_output"]["joint_covariance"]
        assert r["reference"]["policy"] == "FRESH_SOURCE_SCORED"


def test_sealed_process_refuses_undeclared_reads(tmp_path):
    bundle, z = _t1_bundle()
    path = tmp_path / "bundle.json"
    path.write_bytes(pub.json_bytes(bundle))
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(pub.json_bytes(r).decode() for r in _t1_requests(bundle, z)) + "\n"
    )
    secret = tmp_path / "application_truth.json"
    secret.write_text("{}")
    output = tmp_path / "scores.jsonl"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "grain_research.gptpro_foundation_v3.f16_serialize",
            "score",
            "--bundle",
            str(path),
            "--sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--input",
            str(requests),
            "--output",
            str(output),
            "--probe",
            str(secret),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["probe"]["refused"] is True
    assert manifest["scored"] == 3
    assert str(secret.resolve()) not in [
        p for p in manifest["files_opened"] if p != manifest["probe"]["path"]
    ] or manifest["refused_opens"] == [str(secret.resolve())]
    rows = [json.loads(x) for x in output.read_text().splitlines()]
    direct = pub.score(bundle, _t1_requests(bundle, z))
    assert [r["score"] for r in rows] == [r["score"] for r in direct]
    wrong = subprocess.run(
        [
            sys.executable,
            "-m",
            "grain_research.gptpro_foundation_v3.f16_serialize",
            "score",
            "--bundle",
            str(path),
            "--sha256",
            "0" * 64,
            "--input",
            str(requests),
            "--output",
            str(tmp_path / "x.jsonl"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong.returncode != 0 and "BUNDLE_HASH_MISMATCH" in wrong.stderr
