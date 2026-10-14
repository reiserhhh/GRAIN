#!/usr/bin/env python3
"""Offline alternate-formula finite action audit; no neural or answer calls."""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(out):
    """Rebuild all paired/actor contrasts and every finite family matching."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    pairs = H.load(out / "00/pairs.json")
    with np.load(out / "00/interventions.npz", allow_pickle=False) as z:
        mass, visible, scores, arms, scale = [
            z[k] for k in ("mass", "visible", "scores", "arms", "read_scale")
        ]
    reconstructed = np.empty_like(arms)
    reconstructed[:, 0] = mass
    for u in range(288):
        ix = np.flatnonzero(visible[u])
        for q, align, sign_ix in np.ndindex(6, 2, 2):
            s = scores[u, q, ix]
            if align:
                s = np.roll(s, len(ix) // 2)
            weight = mass[u, ix] * np.exp((2 * sign_ix - 1) * m["spec"]["epsilon"] * s)
            arm = 1 + q * 4 + align * 2 + sign_ix
            reconstructed[u, arm] = mass[u]
            reconstructed[u, arm, ix] = weight * mass[u, ix].sum() / weight.sum()
    errors = {
        "mass_reconstruction": float(abs(reconstructed - arms).max()),
        "total_mass": float(abs(arms.sum(-1) - mass.sum(-1)[:, None]).max()),
    }
    with np.load(out / "02/responses.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    r = data["readings"]
    odd = np.zeros_like(data["actor_odd"])
    even = np.zeros_like(data["actor_even"])
    lookup = {tuple(p["key"]): p for p in pairs}
    for v, w, f, c, q, align, mode, readq in np.ndindex(3, 2, 6, 2, 6, 2, 2, 6):
        values = []
        lo, hi = 1 + q * 4 + align * 2, 2 + q * 4 + align * 2
        for owner in (-1, 1):
            p = lookup[(v, w, f, 2 * c - 1, owner)]

            def action(u, hi=hi, lo=lo, mode=mode, readq=readq):
                """Finite sign difference then focus difference in one cell."""
                return (
                    (r[u, hi, 0, mode, readq] - r[u, lo, 0, mode, readq])
                    - (r[u, hi, 1, mode, readq] - r[u, lo, 1, mode, readq])
                ) / 2

            values.append((action(p["high"]) - action(p["low"])) / 2)
        odd[v, w, f, c, q, align, mode, readq] = (values[1] - values[0]) / 2
        even[v, w, f, c, q, align, mode, readq] = sum(values) / 2
    errors["actor_odd"] = float(abs(odd - data["actor_odd"]).max())
    errors["actor_even"] = float(abs(even - data["actor_even"]).max())
    result = H.load(out / "02/summary.json")
    match_error = 0.0
    for row in result["comparisons"]:
        if row["status"] == "UNAVAILABLE_DIRECTION":
            continue
        field = {"actor_odd": odd, "actor_even": even}[row["component"]] / scale
        s = field[0, 0, :, :, :, row["alignment"], row["mode"]].reshape(6, 2, -1)
        t = field[
            row["variant"], row["wording"], :, :, :, row["alignment"], row["mode"]
        ].reshape(6, 2, -1)
        actual = []
        for perm in itertools.permutations(range(6)):
            cells = [
                float(
                    np.dot(s[f, c], t[perm[f], c])
                    / (np.linalg.norm(s[f, c]) * np.linalg.norm(t[perm[f], c]))
                )
                for f, c in np.ndindex(6, 2)
            ]
            actual.append(sum(cells) / 12)
        match_error = max(
            match_error,
            float(abs(np.asarray(actual) - row["permutation_values"]).max()),
        )
    errors["all_matchings"] = match_error
    primary_error = 0.0
    for row in result["primary"]:
        means = []
        for align in (0, 1):
            values = []
            for f, c in np.ndindex(6, 2):
                x = odd[2, row["wording"], f, c, f, align, 1] / scale[1]
                values.append(
                    abs(x[f]) - sum(abs(x[q]) for q in range(6) if q != f) / 5
                )
            means.append(sum(values) / 12)
        primary_error = max(
            primary_error,
            abs(means[0] - row["correct_selectivity"]),
            abs(means[1] - row["wrong_selectivity"]),
            abs(means[0] - means[1] - row["gain"]),
        )
    errors["primary"] = primary_error
    if max(errors.values()) > 1e-9:
        raise AssertionError(errors)
    H.write(
        out / "02/readback.json",
        {
            "status": "NUMERIC_READBACK_PASSED_NOT_GLOBAL",
            "errors": errors,
            "same_agent_not_third_party": True,
            "comparisons": len(result["comparisons"]),
            "permutations_each": 720,
            "audit_code_sha256": H.sha(Path(__file__)),
        },
    )
    H.write(out / "02/seal.json", H.seals(out, out / "02"))
    print(errors, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    audit(p.parse_args().out.resolve())
