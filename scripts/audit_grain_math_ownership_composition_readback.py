#!/usr/bin/env python3
"""Result-after HM68 target/summary/toy readback without production effect helpers."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
STAGES = (
    "score",
    "log_odds",
    "centered_logits",
    "body_endpoint",
    "score_link_control",
    "log_odds_link_control",
)


def read(path):
    """Read only local protocol/result metadata."""
    return json.loads(path.read_text())


def rows(name):
    """Load a saved complete result table without interpreting a headline."""
    with (OUT / "03" / name).open() as stream:
        return list(csv.DictReader(stream))


def tensor_effects(a):
    """Use independent direct tensor indexing for all eight matched contrasts."""
    t = a.reshape(2, 2, 2, *a.shape[1:])
    other = t[1, 1, 0] - t[1, 0, 0] - t[0, 1, 0] + t[0, 0, 0]
    own = t[1, 1, 1] - t[1, 0, 1] - t[0, 1, 1] + t[0, 0, 1]
    return {"own": own, "other": other, "gate": own - other, "edges": t[1] - t[0]}


def main():
    """Check every target action, stage summary and generated numeric world."""
    from hashlib import sha256

    def digest(path):
        return sha256(path.read_bytes()).hexdigest()

    if read(OUT / "verification.json")["status"] != "VERIFIED_FINITE_HM68_NOT_GLOBAL":
        raise ValueError("Primary verified HM68 required")
    for name, value in read(OUT / "03/output_seal.json").items():
        if digest(OUT / name) != value:
            raise ValueError("Frozen primary result changed")
    for name, value in read(OUT / "01/raw_seal.json").items():
        if digest(OUT / name) != value:
            raise ValueError("Frozen acquisition changed")
    with np.load(OUT / "03/stages_and_worlds.npz", allow_pickle=False) as archive:
        a = {key: archive[key].copy() for key in archive.files}
    errors, count = [], 0

    def compare(actual, expected):
        nonlocal count
        x, y = np.asarray(actual, float), np.asarray(expected, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Readback shape/nonfinite mismatch")
        errors.append(float(np.max(np.abs(x - y))))
        count += x.size

    effects, spectra, profiles = {}, {}, []
    coordinates = np.asarray(list(itertools.product((-1, 1), repeat=3)), float)
    masks = np.asarray(list(itertools.product((0, 1), repeat=3)), int)
    characters = np.prod(np.where(masks[None] == 1, coordinates[:, None], 1), axis=2)
    # The parity perturbation preserves every moment of order at most two.
    weights_plus = (1 + 0.5 * characters[:, 7]) / 8
    weights_minus = (1 - 0.5 * characters[:, 7]) / 8
    compare(weights_plus @ characters[:, :7], weights_minus @ characters[:, :7])
    compare(weights_plus.sum(), 1.0)
    compare(weights_minus.sum(), 1.0)
    for family in (0, 1):
        for j in TARGETS:
            for k in (*[t for t in TARGETS if t != j], "neutral"):
                for stage in STAGES:
                    key = f"f{family}_{j}_{k}_{stage}"
                    effects[family, j, k, stage] = tensor_effects(a[key])
                    values = a[key]
                    spectrum = characters.T @ values.reshape(8, -1) / 8
                    spectra[key] = spectrum.reshape(values.shape)
                    compare((characters @ spectrum).reshape(values.shape), values)
                    compare(np.square(spectrum).sum(), np.square(values).sum() / 8)
                    plus = weights_plus @ values.reshape(8, -1)
                    minus = weights_minus @ values.reshape(8, -1)
                    compare(plus - minus, spectrum[7])
                    compare(
                        plus - minus,
                        effects[family, j, k, stage]["gate"].reshape(-1) / 8,
                    )
                    profiles.append(
                        {
                            "family": family,
                            "focal": j,
                            "companion": k,
                            "stage": stage,
                            "delta": 0.5,
                            "profile_difference_norm": float(
                                np.linalg.norm(plus - minus)
                            ),
                            "gate_norm_over_eight": float(
                                np.linalg.norm(effects[family, j, k, stage]["gate"]) / 8
                            ),
                            "first_second_moment_max_error": float(
                                np.max(
                                    np.abs(
                                        weights_plus @ characters[:, :7]
                                        - weights_minus @ characters[:, :7]
                                    )
                                )
                            ),
                        }
                    )
                prefix = f"f{family}_{j}_{k}_"
                z = a[prefix + "centered_logits"]
                xy = characters[:, 6, None, None]
                xyg = characters[:, 7, None, None]
                clean = z - xy * (z * xy).mean(0) - xyg * (z * xyg).mean(0)
                compare(
                    a[prefix + "score_link_control"],
                    softmax(clean, axis=-1) @ (np.arange(10) / 9),
                )
                compare(
                    a[prefix + "log_odds_link_control"],
                    logsumexp(clean[..., 5:], -1) - logsumexp(clean[..., :5], -1),
                )

    actions = rows("target_actions.csv")
    if len(actions) != 1728:
        raise ValueError("All72 groups x4 scalar stages x6 queries required")
    for row in actions:
        f, j, k, stage = (
            int(row["family"]),
            row["focal"],
            row["companion"],
            row["stage"],
        )
        e = effects[f, j, k, stage]
        neutral = effects[f, j, "neutral", stage]["gate"]
        q = TARGETS.index(row["query"])
        edge = e["edges"][:, :, q]
        expected = {
            "own_focal_low_companion": edge[0, 1],
            "own_focal_high_companion": edge[1, 1],
            "other_focal_low_companion": edge[0, 0],
            "other_focal_high_companion": edge[1, 0],
            "own_mixed": e["own"][q],
            "other_mixed": e["other"][q],
            "gate": e["gate"][q],
            "neutral_gate": neutral[q],
            "neutral_adjusted": e["gate"][q] - neutral[q],
        }
        for name, value in expected.items():
            compare(float(row[name]), value)

    summaries = rows("stage_summary.csv")
    if len(summaries) != 12:
        raise ValueError("Two families x six stages required")
    for row in summaries:
        f, stage = int(row["family"]), row["stage"]
        pairs = list(itertools.permutations(TARGETS, 2))
        values = [effects[f, j, k, stage] for j, k in pairs]
        neutral = np.asarray(
            [effects[f, j, "neutral", stage]["gate"] for j, _ in pairs]
        )
        g = np.asarray([e["gate"] for e in values])
        expected = {"directed_pairs": 30}
        for name, v in (
            ("own", [e["own"] for e in values]),
            ("other", [e["other"] for e in values]),
            ("gate", g),
            ("neutral", neutral),
            ("neutral_adjusted", g - neutral),
        ):
            flat = np.asarray(v).ravel()
            expected[name + "_rms"] = math.sqrt(
                math.fsum(float(t) ** 2 for t in flat) / len(flat)
            )
        if stage in (
            "score",
            "log_odds",
            "score_link_control",
            "log_odds_link_control",
        ):
            for name in ("own", "other", "gate"):
                for role, index in (("focal", 0), ("companion", 1)):
                    expected[name + "_" + role + "_query_mean"] = (
                        math.fsum(
                            float(e[name][TARGETS.index(pair[index])])
                            for pair, e in zip(pairs, values, strict=True)
                        )
                        / 30
                    )
        for name, value in expected.items():
            compare(float(row[name]), value)

    for row in read(OUT / "00/rows.json"):
        z = np.asarray(
            read(OUT / f"01/units/unit_{row['unit']:04d}.json")["native"][
                "candidate_logits"
            ],
            float,
        )
        prefix = f"f{row['family']}_{row['focal']}_{row['companion']}_"
        c, q = row["corner"], TARGETS.index(row["query"])
        compare(a[prefix + "score"][c, q], softmax(z) @ (np.arange(10) / 9))
        compare(a[prefix + "log_odds"][c, q], logsumexp(z[5:]) - logsumexp(z[:5]))
        compare(a[prefix + "centered_logits"][c, q], z - z.mean())
    x, y, g = np.asarray(list(itertools.product((-1, 1), repeat=3)), float).T
    x, y, g = (v[:, None, None] for v in (x, y, g))
    for world, seed in itertools.product(
        ("qualified", "generic", "role_marker", "link_only"),
        (20261009, 20261010, 20261011),
    ):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(6, 10))
        dx, dy, vector = rng.normal(size=(3, 6, 10)) * 0.35
        main_z, neutral_z = base + x * dx + y * dy, base + x * dx + 0.1 * y * dy
        if world == "qualified":
            main_z += 0.4 * x * y * (g + 1) / 2 * vector
        elif world == "generic":
            main_z += 0.4 * x * y * vector
        elif world == "role_marker":
            main_z += 0.4 * x * y * g * vector
            neutral_z += 0.4 * x * y * g * vector
        else:
            main_z += 0.3 * y * g * vector
            neutral_z += 0.1 * y * g * vector
        for branch, z in (("main", main_z), ("neutral", neutral_z)):
            centered = z - z.mean(-1, keepdims=True)
            prefix = f"toy_{world}_{seed}_{branch}_"
            compare(a[prefix + "centered_logits"], centered)
            compare(
                a[prefix + "score"], softmax(centered, axis=-1) @ (np.arange(10) / 9)
            )
            compare(
                a[prefix + "log_odds"],
                logsumexp(centered[..., 5:], -1) - logsumexp(centered[..., :5], -1),
            )
    if max(errors) > 1e-9:
        raise ValueError("Alternate readback failed")
    destination = OUT / "04_summary_readback"
    if destination.exists():
        raise FileExistsError("Preserve existing readback")
    destination.mkdir()
    np.savez_compressed(destination / "complete_stage_spectra.npz", **spectra)
    with (destination / "joint_profile_witnesses.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    with (destination / "joint_profile_weights.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "g", "plus", "minus"))
        writer.writerows(
            (*point, float(plus), float(minus))
            for point, plus, minus in zip(
                coordinates, weights_plus, weights_minus, strict=True
            )
        )
    report = {
        "status": "VERIFIED_RESULT_AFTER_HM68_TARGET_SUMMARY_TOYS",
        "actual_comparisons": count,
        "maximum_alternate_error": max(errors),
        "target_rows_checked": len(actions),
        "stage_rows_checked": len(summaries),
        "numeric_worlds_seeds": [4, 3],
        "complete_stage_spectra_saved": len(spectra),
        "spectra_export_is_result_after_packaging_completion": True,
        "joint_profile_witnesses_result_after": len(profiles),
        "joint_profile_delta_fixed": 0.5,
        "new_neural_calls": 0,
        "same_agent_not_independent_replication": True,
    }
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "result_after": True,
                "script_sha256": digest(Path(__file__)),
                "primary_seal_sha256": digest(OUT / "03/output_seal.json"),
                "spectra_sha256": digest(destination / "complete_stage_spectra.npz"),
                "profile_witnesses_sha256": digest(
                    destination / "joint_profile_witnesses.csv"
                ),
                "profile_weights_sha256": digest(
                    destination / "joint_profile_weights.csv"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
