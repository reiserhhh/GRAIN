#!/usr/bin/env python3
"""Alternate-formula verification of HM10B2 without new observer calls."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import hadamard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/target_orbits_r2"


def table(path):
    """Read exact primary CSV cells without importing the primary algebra."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def probability(logits):
    """Use direct stable exponentiation, rather than the primary logsumexp."""
    z = np.asarray(logits, float)
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def verify(source, out):
    """Replay native/affine units, full response laws and cold source actions."""
    if out.exists():
        raise FileExistsError("preserve previous verification")
    m = read_json(source / "00/manifest.json")
    summary = read_json(source / "03/summary.json")
    checks, errors = defaultdict(int), defaultdict(float)

    def check(name, actual, expected):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid shape or finite values: {name}")
        checks[name] += a.size
        errors[name] = max(errors[name], float(np.max(abs(a - b))))

    for name, expected in read_json(source / "03/artifact_hashes.json").items():
        if digest(source / name) != expected:
            raise ValueError(f"primary output changed: {name}")
        checks["artifact_hashes"] += 1
    for name, expected in m["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"primary source changed: {name}")
        checks["source_hashes"] += 1
    if read_json(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("actual forward collection incomplete")
    values = np.asarray(m["alphabet"]["values"], float)
    docs = read_json(source / "00/documents.json")
    design = read_json(source / "00/design.json")
    if len(docs) != 800 or len({d["text"] for d in docs}) != 656 or len(design) != 1968:
        raise ValueError("declared material coverage mismatch")
    for name, expected in {
        "02/readouts.csv": 2400,
        "02/characters.csv": 1536,
        "02/binding.csv": 96,
        "02/conditional_link_split.csv": 768,
        "02/same_word_edges.csv": 2304,
        "02/exact_raw_nulls.csv": 432,
        "03/predictions.csv": 4464,
        "03/completion_summary.csv": 36,
        "03/source_application_uncertainty.csv": 3,
        "03/binding_summary.csv": 6,
        "03/evidence_status_summary.csv": 5,
    }.items():
        if len(table(source / name)) != expected:
            raise ValueError(f"declared output coverage mismatch: {name}")
        checks["output_coverage_rows"] += expected
    with np.load(source / "01/head.npz", allow_pickle=False) as arrays:
        head = arrays["head"].astype(float)
    reference = ROOT / m["spec"]["reference"]
    units = {}
    with np.load(reference / "reference.npz", allow_pickle=False) as functions:
        for d in design:
            path = source / "01/units" / f"unit_{d['unit']:04d}.json"
            meta = read_json(path)
            if any(meta[k] != d[k] for k in ("unit", "text_id", "query")):
                raise ValueError("forward unit identity mismatch")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                z = arrays["logits"].astype(float)
                p = probability(z)
                check("head_float32_replay", head @ arrays["final_normalized"], z)
                check("native_probabilities", p, arrays["probabilities"])
                check("native_scores", p @ values, meta["native"]["conditional_score"])
                mass = np.exp(z - meta["native"]["full_vocab_log_normalizer"]).sum()
                check("candidate_mass", mass, meta["native"]["candidate_mass"])
                q = d["query"]
                mean = functions[f"{q}__G"].sum(axis=0) @ arrays["mean_delta"]
                config = float(arrays["config_token_terms"].sum())
                final = functions[f"{q}__g"] @ (arrays["Q"] - functions[f"{q}__h"])
                affine = float(functions[f"{q}__b"] + mean + config + final)
                check(
                    "affine_projection_replay",
                    [mean, config, final, affine],
                    [
                        meta["affine"][k]
                        for k in ("mean", "configuration", "final", "score")
                    ],
                )
                check("affine_direct_pre_save", meta["affine_direct_error"], 0)
            key = d["text_id"], d["query"]
            if key in units:
                raise ValueError("duplicate actual forward address")
            units[key] = {"z": z, "p": p, "y": float(p @ values), "affine": affine}

    document_lookup = {d["material_id"]: d for d in docs}
    cubes = defaultdict(dict)
    for d in docs:
        expected_role = (
            "source"
            if d["family"] < 8
            else "validation"
            if d["family"] < 12
            else "application"
        )
        if d["role"] != expected_role:
            raise ValueError("family source/application allocation changed")
        if d["kind"] == "full":
            for q in m["spec"]["queries"]:
                cubes[d["family"], q][d["mask"]] = (d, units[d["text_id"], q])
    for r in table(source / "02/readouts.csv"):
        d = document_lookup[r["material_id"]]
        q = r["query"]
        u = units[d["text_id"], q]
        check(
            "readout_scores",
            [float(r["native"]), float(r["affine"])],
            [u["y"], u["affine"]],
        )
        expected = "unobserved_direct_self_evidence"
        if (
            q == "adapted_na8"
            and d["anxiety_polarity"] is not None
            and d["anxiety_owner"] == 0
        ):
            expected = "direct_expression_subset"
        elif (
            q == "pre_trust_positive"
            and d["trust_polarity"] is not None
            and d["trust_owner"] == 0
        ):
            expected = "direct_general_view_subset"
        if r["evidence_status"] != expected or d["evidence_status"][q] != expected:
            raise ValueError("declared evidence identity changed")
        if expected.startswith("direct_"):
            check("qualified_readouts", float(r["qualified_native"]), u["y"])
        elif r["qualified_native"] != "":
            raise ValueError("unobserved evidence became a numeric personal score")
        checks["evidence_statuses"] += 1
    basis = hadamard(32).astype(float)
    check("character_orthogonality", basis.T @ basis, 32 * np.eye(32))
    laws = {}
    for key, cells in cubes.items():
        if set(cells) != set(range(32)):
            raise ValueError("full crossed design incomplete")
        # Log probabilities choose the same zero-sum gauge without raw-logit subtraction.
        theta = np.array([np.log(cells[g][1]["p"]) for g in range(32)])
        theta -= theta.mean(axis=1, keepdims=True)
        y = np.array([cells[g][1]["y"] for g in range(32)])
        affine = np.array([cells[g][1]["affine"] for g in range(32)])
        laws[key] = {"theta": theta, "y": y, "affine": affine}
        for name, f in (("natural", theta), ("native", y), ("affine", affine)):
            c = basis @ f / 32
            energy = c**2 if c.ndim == 1 else np.sum(c**2, axis=1)
            laws[key][name] = c
            laws[key][f"{name}_energy"] = energy
            check("character_reconstruction", basis @ c, f)
            centered = f - f.mean(axis=0)
            expected = (
                np.mean(centered**2)
                if f.ndim == 1
                else np.mean(np.sum(centered**2, axis=1))
            )
            check("parseval_closure", energy[1:].sum(), expected)
    for r in table(source / "02/characters.csv"):
        law = laws[int(r["family"]), r["query"]]
        term = int(r["term"])
        check(
            "natural_coefficients",
            law["natural"][term],
            json.loads(r["natural_coefficient"]),
        )
        for name in ("native", "affine"):
            check(
                "scalar_coefficients", law[name][term], float(r[f"{name}_coefficient"])
            )
        check(
            "character_energies",
            [law[f"{n}_energy"][term] for n in ("natural", "native", "affine")],
            [float(r[f"{n}_energy"]) for n in ("natural", "native", "affine")],
        )
    for r in table(source / "02/binding.csv"):
        law = laws[int(r["family"]), r["query"]]
        pbit, obit, term = (1, 4, 5) if r["axis"] == "anxiety" else (2, 8, 10)
        y = law["y"]
        delta = []
        for owner in (0, 1):
            backgrounds = [g for g in range(32) if g & (pbit | obit) == 0]
            delta.append(
                np.mean(
                    [
                        y[g | (obit if owner else 0) | pbit]
                        - y[g | (obit if owner else 0)]
                        for g in backgrounds
                    ]
                )
            )
        check(
            "binding_contrasts",
            [*delta, delta[0] - delta[1]],
            [
                float(r[k])
                for k in (
                    "self_polarity_delta",
                    "other_polarity_delta",
                    "binding_contrast",
                )
            ],
        )
        check(
            "binding_character_identity", delta[0] - delta[1], -4 * law["native"][term]
        )
        check("affine_binding", -4 * law["affine"][term], float(r["affine_binding"]))
        check(
            "natural_binding_norm",
            4 * np.linalg.norm(law["natural"][term]),
            float(r["natural_binding_norm"]),
        )
    for r in table(source / "02/conditional_link_split.csv"):
        law = laws[int(r["family"]), r["query"]]
        pbit, obit = (1, 4) if r["axis"] == "anxiety" else (2, 8)
        g = int(r["anchor_mask"])
        z = law["theta"][[g, g | pbit, g | obit, g | pbit | obit]]
        y = law["y"][[g, g | pbit, g | obit, g | pbit | obit]]
        origin = z[1] + z[2] - z[0]
        direction = z[3] - origin
        scalar = y[1] + y[2] - y[0]
        link = probability(origin) @ values
        check(
            "conditional_link_split",
            [scalar - y[3], scalar - link, link - y[3]],
            [
                float(r[k])
                for k in ("binding_contrast", "link_binding", "representation_binding")
            ],
        )

        def integrand(t, base=origin, d=direction):
            p = probability(base + t * d)
            return float(p @ (values * d) - (p @ values) * (p @ d))

        integral = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
        check(
            "adaptive_gradient_integral", -integral, float(r["representation_binding"])
        )
        check(
            "conditional_natural_norm",
            np.linalg.norm(direction),
            float(r["natural_interaction_norm"]),
        )
    for r in table(source / "02/same_word_edges.csv"):
        family, q, g, bit = (
            int(r["family"]),
            r["query"],
            int(r["mask"]),
            int(r["generator"]),
        )
        first, second = cubes[family, q][g], cubes[family, q][g ^ bit]
        bags = [
            Counter(re.findall(r"\b\w+\b", cell[0]["text"].casefold()))
            for cell in (first, second)
        ]
        if bags[0] != bags[1]:
            raise ValueError("same-word control changed content")
        check(
            "same_word_edges",
            [second[1]["y"] - first[1]["y"], second[1]["affine"] - first[1]["affine"]],
            [float(r["native_delta"]), float(r["affine_delta"])],
        )
    for r in table(source / "02/exact_raw_nulls.csv"):
        a, b = document_lookup[r["left"]], document_lookup[r["right"]]
        if a["text"] != b["text"] or a["text_id"] != b["text_id"]:
            raise ValueError("raw null is not the same actual text")
        check("exact_raw_nulls", float(r["native_delta"]), 0)

    for r in summary["binding_summary"]:
        q = r["query"]
        pbit, obit, term = (1, 4, 5) if r["axis"] == "anxiety" else (2, 8, 10)
        expected = []
        contrasts = []
        for f in range(16):
            law = laws[f, q]
            backgrounds = [g for g in range(32) if g & (pbit | obit) == 0]
            self_delta = np.mean(
                [law["y"][g | pbit] - law["y"][g] for g in backgrounds]
            )
            other_delta = np.mean(
                [law["y"][g | pbit | obit] - law["y"][g | obit] for g in backgrounds]
            )
            contrast = self_delta - other_delta
            contrasts.append(contrast)
            expected.append(
                [
                    self_delta,
                    other_delta,
                    contrast,
                    -4 * law["affine"][term],
                    4 * np.linalg.norm(law["natural"][term]),
                ]
            )
        check(
            "binding_summary",
            np.mean(expected, axis=0),
            [
                r[k]
                for k in (
                    "self_polarity_delta",
                    "other_polarity_delta",
                    "binding_contrast",
                    "affine_binding",
                    "natural_binding_norm",
                )
            ],
        )
        if r["families"] != 16 or r["positive_binding_families"] != sum(
            c > 0 for c in contrasts
        ):
            raise ValueError("binding family count changed")
    for r in summary["evidence_status_summary"]:
        q = r["query"]
        selected = [d for d in docs if d["evidence_status"][q] == r["evidence_status"]]
        y = np.array([units[d["text_id"], q]["y"] for d in selected])
        check(
            "evidence_summary",
            [y.min(), y.max(), y.mean()],
            [r[k] for k in ("native_min", "native_max", "native_mean")],
        )
        if r["records"] != len(selected) or r["unique_texts"] != len(
            {d["text_id"] for d in selected}
        ):
            raise ValueError("evidence summary coverage mismatch")

    source_means = {
        q: {
            n: np.mean([laws[f, q][n] for f in range(8)], axis=0)
            for n in ("natural", "native")
        }
        for q in m["spec"]["queries"]
    }
    masks = {
        "logit_main": [1, 2, 4, 8, 16],
        "logit_owner_binding": [1, 2, 4, 8, 16, 5, 10],
        "logit_full_law": list(range(1, 32)),
        "logit_wrong_binding": [1, 2, 4, 8, 16, 5, 10],
        "scalar_owner_binding": [1, 2, 4, 8, 16, 5, 10],
    }
    recomputed = defaultdict(list)
    for r in table(source / "03/predictions.csv"):
        f, q, g, arm = int(r["family"]), r["query"], int(r["mask"]), r["arm"]
        if not 8 <= f < 16 or not 1 <= g < 32:
            raise ValueError("cold cell reused as source or prediction anchor")
        law = laws[f, q]
        if arm == "constant_anchor":
            predicted = law["y"][0]
        else:
            scalar = arm == "scalar_owner_binding"
            c = source_means[q]["native" if scalar else "natural"]
            shift = sum(
                (basis[g, j] - 1)
                * c[j]
                * (-1 if arm == "logit_wrong_binding" and j in (5, 10) else 1)
                for j in masks[arm]
            )
            predicted = (
                law["y"][0] + shift
                if scalar
                else probability(law["theta"][0] + shift) @ values
            )
        loss = (predicted - law["y"][g]) ** 2
        check(
            "cold_predictions_and_losses",
            [predicted, law["y"][g], loss],
            [float(r[k]) for k in ("prediction", "actual", "squared_error")],
        )
        recomputed[q, r["role"], arm].append(loss)
    for r in table(source / "03/completion_summary.csv"):
        loss = recomputed[r["query"], r["role"], r["arm"]]
        check("completion_rmse", np.sqrt(np.mean(loss)), float(r["rmse"]))
        if len(loss) != int(r["cells"]) or len(loss) != 124:
            raise ValueError("cold completion coverage changed")
    for r in summary["application_completion"]:
        check(
            "application_summary",
            np.sqrt(np.mean(recomputed[r["query"], "application", r["arm"]])),
            r["rmse"],
        )
    for qi, q in enumerate(m["spec"]["queries"]):
        rng = np.random.default_rng(m["spec"]["seed"] + qi)
        src = np.array([laws[f, q]["natural"] for f in range(8)])
        anchors = np.array([laws[f, q]["theta"][0] for f in range(12, 16)])
        actual = np.array([laws[f, q]["y"][1:] for f in range(12, 16)])
        samples = []
        for _ in range(m["spec"]["bootstrap_draws"]):
            w = np.bincount(rng.integers(0, 8, 8), minlength=8) / 8
            c = np.einsum("f,fjd->jd", w, src)
            losses = []
            for arm in ("logit_main", "logit_owner_binding"):
                terms = masks[arm]
                shift = (basis[1:, terms] - 1) @ c[terms]
                predicted = probability(anchors[:, None, :] + shift) @ values
                losses.append(np.mean((predicted - actual) ** 2, axis=1))
            app_w = np.bincount(rng.integers(0, 4, 4), minlength=4) / 4
            samples.append(float(app_w @ (losses[0] - losses[1])))
        r = next(r for r in summary["source_application_intervals"] if r["query"] == q)
        check(
            "source_application_ci",
            np.quantile(samples, [0.025, 0.975]),
            [r["ci_low"], r["ci_high"]],
        )
    algebra_max = max(v for k, v in errors.items() if k != "head_float32_replay")
    passed = algebra_max < 1e-10 and errors["head_float32_replay"] < 3e-4
    out.mkdir(parents=True)
    report = {
        "packet": "HM-10B2",
        "status": "PASS" if passed else "FAIL",
        "checks": dict(checks),
        "max_errors": dict(errors),
        "max_algebra_error": algebra_max,
        "head_float32_tolerance": 3e-4,
        "new_model_calls": 0,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
        "qualification_is_declared_not_independently_human_coded": True,
    }
    write_json(out / "verification.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise ValueError("target response verification failed")


def main():
    """Verify once, preserving both primary and any previous verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/target_orbits_verification_r1",
    )
    args = parser.parse_args()
    base = (ROOT / "results/grain_math_response_v1").resolve()
    if not all(p.resolve().is_relative_to(base) for p in (args.source, args.out)):
        parser.error("scoped local research paths required")
    verify(args.source, args.out)


if __name__ == "__main__":
    main()
