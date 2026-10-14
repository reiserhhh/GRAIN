#!/usr/bin/env python3
"""Independent scalar-cube audit of the finite operation, not its optimizer."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R


def audit(out):
    """Enumerate supported cubes and check the source-only action and bounds."""
    manifest = R.check(
        out, ("compiled_seal.json", "scored_seal.json", "analysis_seal.json")
    )
    spec = manifest["config"]
    for name, expected in spec["failed_artifact_hashes"].items():
        if R.digest(ROOT / spec["failed_parent"] / name) != expected:
            raise ValueError("Failed parent provenance no longer matches")
    pairs = list(itertools.combinations(range(4), 2))
    bits = list(itertools.product(range(2), repeat=4))
    errors = {
        "operation": 0.0,
        "components": 0.0,
        "target_isolation": 0.0,
        "bound": 0.0,
    }
    counts, gains, lowers, maximum_incidence = [], [], [], 0
    with np.load(out / "source_trade_components.npz", allow_pickle=False) as stored:
        source_components = stored["changes"]
    for held in (-1, 0, 1):
        name = "source_bundle.json" if held == -1 else f"crossfit_bundle_{held}.json"
        bundle = R.read(out / name)
        profiles = np.asarray(bundle["profiles"], int)
        lookup = {tuple(row): i for i, row in enumerate(profiles)}
        tables = np.asarray(bundle["profile_tables"], float)
        for v in range(2):
            h, f = tables[v]
            direct = np.zeros((5, *f.shape))
            incidence = np.zeros(f.shape, int)
            for z, receipt in zip(
                itertools.product(range(2), repeat=3),
                bundle["solver_receipts"][v],
                strict=True,
            ):
                active, product = np.zeros(5, int), 0.0
                for q in range(5):
                    for g0, profile in enumerate(profiles):
                        if profile[q]:
                            continue
                        other = profile.copy()
                        other[q] = 1
                        if tuple(other) not in lookup:
                            continue
                        group = (g0, lookup[tuple(other)])
                        for pa, pb, pc in itertools.product(pairs, repeat=3):
                            corners = [
                                (
                                    group[b[0]],
                                    pa[b[1]] + 4 * z[0],
                                    pb[b[2]] + 4 * z[1],
                                    pc[b[3]] + 4 * z[2],
                                )
                                for b in bits
                            ]
                            values = np.array([f[index] for index in corners])
                            if np.any(values <= 0):
                                continue
                            signs = np.array(
                                [1 if sum(b) % 2 == 0 else -1 for b in bits]
                            )
                            ell = float(
                                sum(
                                    float(sign) * np.log(float(value))
                                    for sign, value in zip(signs, values, strict=True)
                                )
                            )
                            tau = float(values.min()) * ell / (32 + 2 * abs(ell))
                            for index, sign in zip(corners, signs, strict=True):
                                direct[(q, *index)] += tau * sign / 135
                                incidence[index] += 1
                            active[q] += 1
                            product += tau * ell
                key = (slice(None), *(slice(4 * r, 4 * r + 4) for r in z))
                positive_h, positive_f = h[key] > 0, f[key] > 0
                gain = float(
                    -np.sum(h[key][positive_h] * np.log(h[key][positive_h]))
                    + np.sum(f[key][positive_f] * np.log(f[key][positive_f]))
                )
                lower = product / 270
                errors["bound"] = max(
                    errors["bound"],
                    abs(lower - receipt["entropy_gain_lower_bound"]),
                    abs(gain - receipt["entropy_gain"]),
                )
                if (
                    active.tolist() != receipt["supported_cubes_by_target"]
                    or gain + 1e-14 < lower
                    or int(incidence[key].max()) != receipt["observed_max_incidence"]
                    or receipt["ipf_used"]
                ):
                    raise ValueError("Supported cubes/entropy certificate changed")
                counts.append(active.tolist())
                gains.append(gain)
                lowers.append(lower)
            maximum_incidence = max(maximum_incidence, int(incidence.max()))
            errors["operation"] = max(
                errors["operation"], float(np.max(np.abs(h - f + direct.sum(0))))
            )
            if held == -1:
                errors["components"] = max(
                    errors["components"],
                    float(np.max(np.abs(direct - source_components[v]))),
                )
            for q, other in itertools.product(range(5), repeat=2):
                if q != other:
                    errors["target_isolation"] = max(
                        errors["target_isolation"],
                        float(
                            np.max(np.abs(direct[q, profiles[:, other] == 1].sum(0)))
                        ),
                    )
            if (
                np.any(h < f / 2 - 1e-15)
                or np.any(h[f == 0])
                or maximum_incidence > 135
            ):
                raise ValueError("Support/positivity/incidence certificate failed")
    if max(errors.values()) > 2e-12:
        raise ValueError("Finite algebraic operation arithmetic disagrees")
    result = {
        "status": "PASS",
        "view_laws": 6,
        "condition_laws": 48,
        "supported_cubes_by_target": np.sum(counts, axis=0).tolist(),
        "source_entropy_gains": gains[:16],
        "source_entropy_lower_bounds": lowers[:16],
        "maximum_incidence": maximum_incidence,
        "errors": errors,
        "same_agent_not_blind": True,
        "maximum_entropy_claim": False,
        "complete_markov_basis_claim": False,
        "failed_parent_hashes_unchanged": True,
    }
    R.write(out / "trade_verification.json", result)
    R.seal(out, "trade_verification_seal.json", ("trade_verification.json",))
    print(json.dumps(result))


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
