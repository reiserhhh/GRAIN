#!/usr/bin/env python3
"""Standalone T3 tests. Run this file directly; do not discover repository tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_grain_gptpro_f16_independent_t3.py"
SPEC = importlib.util.spec_from_file_location("independent_t3_runner_for_tests", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
t3 = runner.t3


class T3IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brief, cls.manifest = runner.synthetic_fixture()
        cls.bundle, cls.checkpoint = t3.build_instrument(cls.brief, cls.manifest)
        cls.value = {
            "world": "core",
            "reader_hash": t3.READER_HASH,
            "prefix": cls.manifest["source_prefixes"]["synthetic_0"],
        }

    def test_stationary_gap_filter_matches_dense_gaussian(self):
        observations = [(0, -1.0, 0.4), (1, 1.0, -0.2), (4, 1.0, 0.9), (2, -1.0, -0.7)]
        times = np.cumsum([row[0] for row in observations])
        noise = np.array([-0.4, -0.9])
        q, r = np.exp(2 * noise)
        for rho in (0.0, -0.72, 0.6, 0.98):
            psi = np.array([0.2, -0.3, math.atanh(rho / 0.99)])
            mean, variance, loglik = t3.filter_parameters(psi, observations, noise)
            state = q / (1 - rho**2) * rho ** np.abs(times[:, None] - times[None, :])
            covariance = state + np.eye(len(times)) * r
            residual = np.array([y - psi[0] - psi[1] * z for _, z, y in observations])
            solve = np.linalg.solve(covariance, residual)
            expected_ll = -0.5 * (
                len(times) * math.log(2 * math.pi)
                + np.linalg.slogdet(covariance)[1]
                + residual @ solve
            )
            self.assertAlmostEqual(float(loglik[0]), float(expected_ll), places=10)
            self.assertAlmostEqual(float(mean[0]), float(state[-1] @ solve), places=10)
            expected_p = state[-1, -1] - state[-1] @ np.linalg.solve(
                covariance, state[:, -1]
            )
            self.assertAlmostEqual(float(variance[0]), float(expected_p), places=10)

    def test_map_gradient_matches_finite_difference(self):
        observations = [(0, -1.0, 0.5), (3, 0.2, -0.4), (1, 1.0, 1.2), (5, -1.0, -0.6)]
        mu = np.array([0.1, 0.2, -0.3])
        precision = np.array([[2, 0.2, 0.1], [0.2, 3, -0.1], [0.1, -0.1, 1]])
        for eta in (0.0, -0.9, 0.8):
            point = np.array([0.4, -0.2, eta])
            value, gradient = t3.objective_and_gradient(
                point, observations, mu, precision, [-0.6, -0.8]
            )
            _, _, loglik = t3.filter_parameters(point, observations, [-0.6, -0.8])
            delta = point - mu
            self.assertAlmostEqual(
                value, float(-loglik[0] + 0.5 * delta @ precision @ delta), places=11
            )
            numeric = []
            for axis in np.eye(3) * 1e-5:
                plus = t3.objective_and_gradient(
                    point + axis, observations, mu, precision, [-0.6, -0.8]
                )[0]
                minus = t3.objective_and_gradient(
                    point - axis, observations, mu, precision, [-0.6, -0.8]
                )[0]
                numeric.append((plus - minus) / 2e-5)
            np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-7)

    def test_central_hessian_quadratic(self):
        matrix = np.array([[2, 0.4, 0.1], [0.4, 3, -0.2], [0.1, -0.2, 4]])
        actual = t3.central_hessian(
            lambda x: 0.5 * x @ matrix @ x, np.array([0.2, -0.1, 0.3])
        )
        np.testing.assert_allclose(actual, matrix, rtol=1e-7, atol=1e-7)

    def test_full_covariance_and_scalar_agree_with_loading_matrix(self):
        for rho in (-0.98, -0.4, 0.0, 0.75, 0.989):
            parameters = [[0.3, 7.2, math.atanh(rho / 0.99)]]
            q, r, p, m = 0.7, 0.3, 0.9, 0.8
            score, variance, components = t3.predictive_moments(
                parameters, np.array([m]), np.array([p]), np.log([q, r]) / 2
            )
            state_load = rho ** np.arange(1, 5)
            innovation_load = np.zeros((4, 4))
            for k in range(4):
                for j in range(k + 1):
                    innovation_load[k, j] = rho ** (k - j)
            covariance = (
                p * np.outer(state_load, state_load)
                + q * innovation_load @ innovation_load.T
                + r * np.eye(4)
            )
            self.assertAlmostEqual(
                float(variance[0]),
                float(np.ones(4) @ covariance @ np.ones(4) / 16),
                places=12,
            )
            self.assertAlmostEqual(
                float(score[0]), 0.3 + m * float(np.mean(state_load)), places=12
            )
            self.assertLess(components["covariance_max_abs_error"], 1e-12)
            if rho == 0.75:
                self.assertGreater(
                    abs(float(variance[0]) - float(np.trace(covariance) / 16)), 0.1
                )

    def test_source_reference_exact_same_scorer_and_ddof_one(self):
        scores = [row["score"]["raw_score"] for row in self.checkpoint["records"]]
        self.assertEqual(self.bundle["construction_status"], "SOURCE_REFERENCE_FROZEN")
        self.assertEqual(self.bundle["reference_mu"], float(np.mean(scores)))
        self.assertEqual(self.bundle["reference_sigma"], float(np.std(scores, ddof=1)))
        direct = t3.cold_score(self.value, self.bundle)
        self.assertEqual(direct["raw_score"], scores[0])
        self.assertEqual(direct["calibration_status"], "NOT_EVALUATED")

    def test_uncertainty_decomposition_and_standardization(self):
        score = t3.cold_score(self.value, self.bundle)
        components = score["variance_components"]
        within = np.mean(
            [
                row["event_state"] + row["personal_parameter"]
                for row in score["per_source"]
            ]
        )
        between = np.var([row["mean"] for row in score["per_source"]], ddof=0)
        self.assertAlmostEqual(components["total"], within + between, places=13)
        self.assertAlmostEqual(
            components["event_state"],
            sum(
                components[key]
                for key in ("terminal_state", "future_innovation", "future_observation")
            ),
            places=13,
        )
        self.assertAlmostEqual(
            score["source_standardized_variance"],
            components["total"] / self.bundle["reference_sigma"] ** 2,
        )
        self.assertAlmostEqual(
            score["source_standardized_score"],
            (score["raw_score"] - self.bundle["reference_mu"])
            / self.bundle["reference_sigma"],
        )
        self.assertAlmostEqual(
            score["interval"][1] - score["raw_score"],
            1.96 * math.sqrt(components["total"]),
        )

    def test_batch_order_and_author_rename_invariance(self):
        inputs = [
            {"world": "core", "reader_hash": t3.READER_HASH, "prefix": rows}
            for rows in self.manifest["source_prefixes"].values()
        ]
        forward = [t3.cold_score(value, self.bundle) for value in inputs]
        reverse = [t3.cold_score(value, self.bundle) for value in reversed(inputs)]
        self.assertEqual(forward, list(reversed(reverse)))
        renamed = copy.deepcopy(self.value)
        for index, row in enumerate(renamed["prefix"]):
            row["author_id"] = "unseen_author"
            row["episode_id"] = "unseen_episode"
            row["event_id"] = f"new_event_{index}"
        self.assertEqual(t3.cold_score(renamed, self.bundle), forward[0])

    def test_prior_only_is_explicit_and_stationary(self):
        score = t3.cold_score(
            {"world": "core", "reader_hash": t3.READER_HASH, "prefix": []}, self.bundle
        )
        self.assertTrue(score["supported"])
        self.assertTrue(score["prior_only"])
        self.assertEqual(score["observed_count"], 0)
        model = self.bundle["ensemble"][0]
        draws, diagnostics = t3.personal_draws([], model, self.bundle["seed"], 200)
        xi = np.random.default_rng(
            np.random.SeedSequence([20260915, 16, 3, 0])
        ).standard_normal((200, 3))
        expected = (
            np.asarray(model["fit"]["prior"]["mu"])
            + xi @ np.linalg.cholesky(model["fit"]["prior"]["Sigma"]).T
        )
        np.testing.assert_array_equal(draws, expected)
        self.assertTrue(diagnostics["prior_only"])
        mean, variance, _ = t3.filter_parameters(draws, [], model["fit"]["log_sigma"])
        np.testing.assert_array_equal(mean, np.zeros(200))
        np.testing.assert_allclose(
            variance,
            np.exp(2 * model["fit"]["log_sigma"][0])
            / (1 - (0.99 * np.tanh(draws[:, 2])) ** 2),
        )

    def test_wrong_world_and_reader_rejected(self):
        for key in ("world", "reader_hash"):
            value = copy.deepcopy(self.value)
            value[key] = "wrong"
            with self.assertRaises(t3.ContractError):
                t3.cold_score(value, self.bundle)

    def test_duplicate_mixed_and_nonfinite_input_rejected(self):
        cases = [
            ("event_id", self.value["prefix"][0]["event_id"]),
            ("author_id", "other"),
            ("world", "other"),
            ("role", "application"),
            ("episode_id", "other"),
            ("t", 0),
            ("t", 1.0),
            ("t", True),
            ("y", float("nan")),
            ("z", float("inf")),
            ("y", True),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.value)
                data["prefix"][1][key] = value
                with self.assertRaises(t3.ContractError):
                    t3.cold_score(data, self.bundle)

    def test_no_future_leakage_input_surface(self):
        for extra in ("truth", "future", "future_conditions", "reference_mu"):
            value = copy.deepcopy(self.value)
            value[extra] = [9] * 4
            with self.assertRaises(t3.ContractError):
                t3.cold_score(value, self.bundle)
        value = copy.deepcopy(self.value)
        value["prefix"][0]["future_y"] = 999
        with self.assertRaises(t3.ContractError):
            t3.cold_score(value, self.bundle)
        value = copy.deepcopy(self.value)
        value["prefix"].append(
            {**value["prefix"][-1], "t": 8, "event_id": "future_event"}
        )
        with self.assertRaises(t3.ContractError):
            t3.cold_score(value, self.bundle)

    def test_no_source_files_or_author_score_lookup_in_cold(self):
        original = Path.read_bytes

        def restricted(path):
            if path.resolve() != runner.MODULE_PATH:
                raise AssertionError(f"cold read outside own implementation: {path}")
            return original(path)

        with (
            patch.object(Path, "read_bytes", restricted),
            patch.object(
                Path, "read_text", side_effect=AssertionError("unexpected file read")
            ),
        ):
            result = t3.cold_score(self.value, self.bundle)
        self.assertTrue(result["supported"])
        self.assertNotIn("source_prefixes", self.bundle)
        self.assertNotIn("records", self.bundle)

    def test_failed_bootstrap_retained_not_renormalized(self):
        bundle = copy.deepcopy(self.bundle)
        bundle["ensemble"][0]["status"] = "FAILED"
        bundle["ensemble_hash"] = t3.object_hash(bundle["ensemble"])
        score = t3.cold_score(self.value, bundle)
        self.assertFalse(score["supported"])
        self.assertIsNone(score["raw_score"])
        self.assertIsNone(score["variance_components"])
        self.assertEqual(len(score["per_source"]), 2)
        self.assertEqual(score["unsupported_count"], 1)
        self.assertTrue(score["per_source"][1]["supported"])

    def test_no_converged_map_retained_unsupported(self):
        class Failed:
            success = False
            fun = 1.0
            x = np.zeros(3)
            status = 1
            message = "injected iteration limit"
            nit = 100

        with patch.object(t3, "minimize", return_value=Failed()):
            score = t3.cold_score(self.value, self.bundle)
        self.assertFalse(score["supported"])
        self.assertEqual(score["unsupported_count"], 2)
        self.assertEqual(score["per_source"][0]["reason"], "NO_CONVERGED_PERSONAL_MAP")
        self.assertEqual(len(score["per_source"][0]["starts"]), 2)
        self.assertIsNone(score["raw_score"])

    def test_nonpositive_hessian_not_repaired(self):
        with patch.object(
            t3, "central_hessian", return_value=np.diag([1.0, -0.1, 1.0])
        ):
            score = t3.cold_score(self.value, self.bundle)
        self.assertFalse(score["supported"])
        self.assertEqual(score["per_source"][0]["reason"], "NONPOSITIVE_HESSIAN")

    def test_unsupported_source_stops_reference(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["ensemble"][1]["status"] = "FAILED"
        bundle, checkpoint = t3.build_instrument(self.brief, manifest)
        self.assertEqual(bundle["construction_status"], "UNSUPPORTED_SOURCE_REFERENCE")
        self.assertIsNone(bundle["reference_mu"])
        self.assertEqual(checkpoint["completed_authors"], 1)
        self.assertEqual(len(checkpoint["records"][0]["score"]["per_source"]), 2)
        with self.assertRaises(t3.ContractError):
            t3.cold_score(self.value, bundle)

    def test_source_identity_and_source_hash_rejection(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["ensemble"][0]["source_ids"][0] = "not_a_source_author"
        with self.assertRaisesRegex(t3.ContractError, "identity"):
            t3.build_instrument(self.brief, manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["source_hash"] = "0" * 64
        with self.assertRaisesRegex(t3.ContractError, "hash"):
            t3.build_instrument(self.brief, manifest)
        bundle = copy.deepcopy(self.bundle)
        bundle["source_hash"] = "0" * 64
        with self.assertRaisesRegex(t3.ContractError, "hash"):
            t3.cold_score(self.value, bundle)

    def test_checkpoint_resume_is_exact_and_hash_bound(self):
        partial, checkpoint = t3.build_instrument(
            self.brief, self.manifest, max_new_authors=1
        )
        self.assertEqual(partial["construction_status"], "CHECKPOINT_ONLY")
        with self.assertRaises(t3.ContractError):
            t3.cold_score(self.value, partial)
        resumed, final_checkpoint = t3.build_instrument(
            self.brief, self.manifest, checkpoint=checkpoint
        )
        self.assertEqual(resumed, self.bundle)
        self.assertEqual(final_checkpoint, self.checkpoint)
        corrupt = copy.deepcopy(checkpoint)
        corrupt["records"][0]["score"]["raw_score"] += 1
        with self.assertRaisesRegex(t3.ContractError, "hash"):
            t3.build_instrument(self.brief, self.manifest, checkpoint=corrupt)
        corrupt = copy.deepcopy(checkpoint)
        corrupt["identity"]["scorer_code_hash"] = "0" * 64
        with self.assertRaisesRegex(t3.ContractError, "identity"):
            t3.build_instrument(self.brief, self.manifest, checkpoint=corrupt)

    def test_ten_author_checkpoint_cadence(self):
        brief, manifest = copy.deepcopy((self.brief, self.manifest))
        brief["source_authors"] = 21
        records = {}
        for index in range(21):
            author = f"checkpoint_author_{index:02d}"
            rows = copy.deepcopy(self.value["prefix"])
            for row in rows:
                row["author_id"] = author
            records[author] = rows
        manifest["source_prefixes"] = records
        for model in manifest["ensemble"]:
            model["source_ids"] = sorted(records)
        calls = []
        counter = iter(range(21))
        with patch.object(
            t3,
            "_score",
            side_effect=lambda value, bundle: {
                "supported": True,
                "raw_score": next(counter),
            },
        ):
            t3.build_instrument(
                brief,
                manifest,
                checkpoint_callback=lambda snapshot: calls.append(
                    snapshot["completed_authors"]
                ),
            )
        self.assertEqual(calls, [10, 20, 21])

    def test_strict_json_and_external_bundle_hash(self):
        directory = runner.OUTPUT_ROOT / "unit_tests"
        path = directory / "bundle.json"
        digest = runner.write_json(path, self.bundle)
        self.assertEqual(
            t3.load_bundle(path, digest, self.bundle["source_hash"]), self.bundle
        )
        with self.assertRaisesRegex(t3.ContractError, "SHA256"):
            t3.load_bundle(path, "0" * 64)
        with self.assertRaisesRegex(t3.ContractError, "source hash"):
            t3.load_bundle(path, digest, "0" * 64)
        with self.assertRaises(ValueError):
            t3.strict_json_bytes({"invalid": float("nan")})
        corrupt = copy.deepcopy(self.bundle)
        corrupt["ensemble"][0]["fit"]["prior"]["mu"][0] += 1
        with self.assertRaisesRegex(t3.ContractError, "ensemble hash"):
            t3.cold_score(self.value, corrupt)

    def test_actual_warm_and_cold_cli_equal(self):
        warm = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "synthetic-warm"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        metadata = json.loads(warm.stdout)
        directory = runner.OUTPUT_ROOT / "synthetic_smoke"
        command = [
            sys.executable,
            "-B",
            str(RUNNER),
            "cold",
            "--bundle",
            str(directory / "bundle.json"),
            "--expected-sha256",
            metadata["bundle_sha256"],
            "--expected-source-hash",
            metadata["source_hash"],
            "--input",
            str(directory / "input.json"),
            "--output",
            str(directory / "cold_score.json"),
        ]
        cold = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=True
        )
        self.assertEqual(
            (directory / "warm_score.json").read_bytes(),
            (directory / "cold_score.json").read_bytes(),
        )
        self.assertEqual(json.loads(cold.stdout)["raw_score"], metadata["raw_score"])
        rejected = command.copy()
        rejected[rejected.index("--expected-sha256") + 1] = "0" * 64
        failure = subprocess.run(
            rejected, cwd=ROOT, capture_output=True, text=True, check=False
        )
        self.assertEqual(failure.returncode, 2)
        self.assertIn("SHA256 mismatch", failure.stderr)
        rejected_source = command.copy()
        rejected_source[rejected_source.index("--expected-source-hash") + 1] = "0" * 64
        source_failure = subprocess.run(
            rejected_source, cwd=ROOT, capture_output=True, text=True, check=False
        )
        self.assertEqual(source_failure.returncode, 2)
        self.assertIn("source hash mismatch", source_failure.stderr)
        runner.write_json(
            directory / "cli_evidence.json",
            {
                "warm_command": warm.args,
                "warm_stdout": warm.stdout,
                "cold_command": command,
                "cold_stdout": cold.stdout,
                "equal_bytes": True,
                "warm_score_sha256": hashlib.sha256(
                    (directory / "warm_score.json").read_bytes()
                ).hexdigest(),
                "bad_hash_exit": failure.returncode,
                "bad_hash_stderr": failure.stderr,
                "bad_source_hash_exit": source_failure.returncode,
                "bad_source_hash_stderr": source_failure.stderr,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
