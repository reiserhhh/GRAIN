#!/usr/bin/env python3
"""Read-only closed-form audit; preserve the frozen verifier's serialization fault."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

OUT = ROOT / "results/grain_math_response_v1/author_belief_r1/runtime_io_recovery_r1"
AUDIT = OUT / "readonly_verification_r1"


def main():
    """Compare beta-integral likelihoods, all endpoints, source counts and cold output."""
    if AUDIT.exists():
        raise FileExistsError("One read-only audit only; no scientific rerun")
    manifest = R.check(
        OUT, ("prepared_seal.json", "computed_seal.json", "analyzed_seal.json")
    )
    z, a = load_bundle(OUT / "observations.npz"), load_bundle(OUT / "trajectories.npz")
    started = time.monotonic()
    errors = {
        k: 0.0
        for k in (
            "source_counts",
            "source_codes",
            "likelihood",
            "profiles",
            "marginals",
            "prediction_sums",
            "shares",
            "cold",
        )
    }
    checks = 0
    for r, ref in enumerate(manifest["config"]["references"]):
        b = load_bundle(OUT / f"source_{ref}.npz")
        mask = z["bank"] == r if r < 2 else np.ones(386, bool)
        paths = z["source_paths"][mask]
        c0 = np.zeros_like(b["count0"])
        c = np.zeros_like(b["count"])
        for u, row in enumerate(paths):
            for episode in row:
                c0[u, episode[0]] += 1
                for left, right in zip(episode[:-1], episode[1:], strict=True):
                    c[u, left, right] += 1
        errors["source_counts"] = max(
            errors["source_counts"],
            float(abs(c0 - b["count0"]).max()),
            float(abs(c - b["count"]).max()),
        )
        expected_codes = z["targets"][:, mask].astype(int) @ (1 << np.arange(5))
        errors["source_codes"] = max(
            errors["source_codes"], float(abs(expected_codes - b["codes"]).max())
        )
        bits = (np.arange(32)[:, None] >> np.arange(5)) & 1
        for u in range(120):
            for mode, kernel in ((1, False), (3, True)):
                cases = [("fresh", w, [z["held_paths"][u, w]], []) for w in range(2)]
                cases.append(
                    (
                        "carry",
                        int(kernel),
                        list(z["held_paths"][u]),
                        [z["held_paths"][u, 0]],
                    )
                )
                donor = int(z["donors"][u])
                if donor >= 0:
                    early = [z["held_paths"][donor, 0]]
                    cases.append(
                        (
                            "stranger",
                            int(kernel),
                            early + [z["held_paths"][u, 1]],
                            early,
                        )
                    )
                for kind, w, episodes, previous in cases:
                    ell = M.component_log_likelihood(b, episodes, kernel)
                    total = float(logsumexp(ell) - np.log(len(ell)))
                    if previous:
                        old = M.component_log_likelihood(b, previous, kernel)
                        total -= float(logsumexp(old) - np.log(len(old)))
                    base = f"{ref}/{kind}"
                    logp = (
                        a[f"{base}/logp"][u, mode, w]
                        if kind == "fresh"
                        else a[f"{base}/logp"][u, w]
                    )
                    expected = M.profile(np.exp(ell - logsumexp(ell)), b["codes"])
                    profile = (
                        a[f"{base}/profiles"][u, mode, w, :, -1]
                        if kind == "fresh"
                        else a[f"{base}/profiles"][u, w, :, -1]
                    )
                    marginal = (
                        a[f"{base}/marginals"][u, mode, w, :, -1]
                        if kind == "fresh"
                        else a[f"{base}/marginals"][u, w, :, -1]
                    )
                    errors["likelihood"] = max(
                        errors["likelihood"], abs(total - float(logp.sum()))
                    )
                    errors["profiles"] = max(
                        errors["profiles"], float(abs(profile - expected).max())
                    )
                    errors["marginals"] = max(
                        errors["marginals"],
                        float(abs(marginal - expected @ bits).max()),
                    )
                    checks += 1
        for key, values in a.items():
            if not key.startswith(ref + "/"):
                continue
            if "/stranger/" in key:
                active = z["donors"] >= 0
                if (
                    not np.isfinite(values[active]).all()
                    or not np.isnan(values[~active]).all()
                ):
                    raise ArithmeticError("Stranger missing mask is not preserved")
                values = values[active]
            if not np.isfinite(values).all():
                raise ArithmeticError("Unexpected nonfinite supported output")
            if key.endswith("predictions"):
                errors["prediction_sums"] = max(
                    errors["prediction_sums"], float(abs(values.sum(-1) - 1).max())
                )
            if key.endswith("probability_shares_source_prior_application"):
                errors["shares"] = max(
                    errors["shares"], float(abs(values.sum(-1) - 1).max())
                )
    cold = R.read(OUT / "cold_readback.json")
    for w, episode in enumerate(cold["episodes"]):
        for key in ("profiles", "marginals", "predictions", "logp"):
            expected = (
                a[f"pooled/fresh/{key}"][0, 3, 0]
                if w == 0
                else a[f"pooled/carry/{key}"][0, 1]
            )
            errors["cold"] = max(
                errors["cold"], float(abs(np.array(episode[key]) - expected).max())
            )
    report = {
        "PASS": bool(max(errors.values()) < 1e-10),
        "errors": errors,
        "shared_complete_path_endpoints": checks,
        "matched_stranger_authors": int((z["donors"] >= 0).sum()),
        "seconds": time.monotonic() - started,
        "same_agent_not_blind": True,
        "new_neural_scientific_reruns": 0,
        "old_verifier_failure": "NumPy boolean PASS serialization after completed arithmetic and cold export; partial verification.json retained",
    }
    if not report["PASS"]:
        raise ArithmeticError(str(report))
    AUDIT.mkdir()
    inputs = (
        "manifest.json",
        "computed_seal.json",
        "analyzed_seal.json",
        "observations.npz",
        "trajectories.npz",
        "cold_readback.json",
        "verification.json",
    ) + tuple(f"source_{r}.npz" for r in manifest["config"]["references"])
    R.write(
        AUDIT / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(Path(__file__))},
            "inputs": {
                str((OUT / p).relative_to(ROOT)): R.digest(OUT / p) for p in inputs
            },
            "not_repaired_scientific_output": True,
        },
    )
    R.write(AUDIT / "verification.json", report)
    R.seal(AUDIT, "verification_seal.json", ("manifest.json", "verification.json"))
    print(report)


if __name__ == "__main__":
    main()
