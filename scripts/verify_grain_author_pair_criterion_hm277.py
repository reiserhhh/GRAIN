#!/usr/bin/env python3
"""Different-form checks of cross-fit identity, paired sums and author association."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_relational_diagnostic_update import BITS
from scripts import run_grain_author_pair_criterion_hm277 as X
from scripts import run_grain_observed_family_hm170 as R


def literal_profile(logs, codes, relink):
    """Positive kernel sums, without logsumexp/group code of the production scorer."""
    a = np.exp(logs[0, :, 0] - np.max(logs[0, :, 0]))
    b = np.exp(logs[0, :, 1] - np.max(logs[0, :, 1]))
    result = np.empty((4, 32))
    for g in range(32):
        ix = np.flatnonzero(codes == g)
        result[0, g] = sum(np.mean(a[j] * b[j]) for j in ix)
        result[1, g] = sum(np.mean(a[j]) * np.mean(b[j]) for j in ix)
        result[2, g] = np.sum(a[ix]) * np.sum(b[ix]) / (len(ix) * a.shape[-1] ** 2)
        result[3, g] = sum(np.mean(a[j] * b[j, relink[j]]) for j in ix)
    return result / np.sum(result, axis=1, keepdims=True)


def verify(out):
    """Verify the changed scientific question without an instrument-precision loop."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    data = X.load(ROOT / cfg["source_parent"] / "source_pair_bank.npz")
    value, inference = X.load(out / "readings.npz"), X.load(out / "inference.npz")
    profiles, weights = value["pair_profiles"], value["source_weights"]
    bank = data["folds"]
    own = max(
        float(np.max(abs(weights[:, u, :, :, bank == bank[u]]))) for u in range(386)
    )
    if own != 0:
        raise ValueError("Self/own-bank source exposure")
    for source_bank in (0, 1):
        for gp in inference["profile_permutations"]:
            np.testing.assert_array_equal(
                np.sort(gp[bank == source_bank]),
                np.sort(data["codes"][bank == source_bank]),
            )
    profile_error = 0.0
    for sp in (0, 1):
        z = data["prototypes"] if sp == 0 else data["prototypes"] @ data["projection"]
        for source_bank in (0, 1):
            src = bank == source_bank
            app = np.flatnonzero(~src)
            for u in (int(app[0]), int(app[-1])):
                for edge in (0, 11, 21):
                    x = z[u, edge]
                    distance = np.sum((z[src] - x[None, None]) ** 2, axis=-1)
                    logs = (-distance / (2 * cfg["variance"])).transpose(0, 2, 1)[None]
                    direct = literal_profile(
                        logs, data["codes"][src], data["relink"][src]
                    )
                    profile_error = max(
                        profile_error,
                        float(np.max(abs(direct - profiles[sp, u, edge]))),
                    )
    half = np.stack(
        (profiles[:, :, :11].mean(2), profiles[:, :, 11:].mean(2), profiles.mean(2)),
        axis=2,
    )
    aggregate_error = float(np.max(abs(half - value["aggregate_profiles"])))
    label = BITS[data["codes"]]
    d = weights.mean(2)[:, :, 0] - weights.mean(2)[:, :, 1]
    reading_error = float(np.max(abs(d @ label - value["series"][:, :, 2, 4])))
    covariances, vx, vy = np.zeros((2, 5)), np.zeros((2, 5)), np.zeros(5)
    for b in (0, 1):
        ix = bank == b
        yy = label[ix]
        vy += np.sum(yy * yy, 0) - len(yy) * yy.mean(0) ** 2
        for sp in (0, 1):
            xx = value["series"][sp, ix, 2, 4]
            covariances[sp] += np.sum(xx * yy, 0) - len(xx) * xx.mean(0) * yy.mean(0)
            vx[sp] += np.sum(xx * xx, 0) - len(xx) * xx.mean(0) ** 2
    direct_r = covariances / np.sqrt(vx * vy)
    corr_error = float(np.max(abs(direct_r - inference["observed"])))
    match = value["matching"]
    if np.any(match == np.arange(386)) or np.any(bank[match] != bank):
        raise ValueError("Invalid stranger control")
    errors = {
        "positive_profile_sums": profile_error,
        "aggregate_means": aggregate_error,
        "weight_target_readings": reading_error,
        "literal_covariance": corr_error,
        "own_bank_source_weights": own,
    }
    if max(errors.values()) > 1e-10:
        raise ValueError(str(errors))
    result = {
        "packet": "HM277",
        "alternate_max_errors": errors,
        "explicit_pair_checks": 24,
        "all386_authors_crossfit": True,
        "all199_whole_profile_nulls_preserved": True,
        "same_agent_not_blind_replication": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(result, flush=True)


def main():
    """Run immutable read-only numerical verification and seal its result once."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.output)


if __name__ == "__main__":
    main()
