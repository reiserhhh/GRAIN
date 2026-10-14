#!/usr/bin/env python3
"""Recompute official coordinates and author actions using alternate formulae."""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_observer_parity_hm96 import (
    check,
    load,
    seal,
    settings,
    verify,
    write,
)


def run():
    """Audit finite interface results, not psychological validity or novelty."""
    spec, source, out = settings()
    check(out)
    verify(out, load(out / "01_seal.json"))
    verify(out, load(out / "02_seal.json"))
    dest = out / "03"
    dest.mkdir()
    errors, checks = [], 0

    def compare(actual, expected):
        """Count substantive scalar checks and fail on any unexplained drift."""
        nonlocal checks
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Shape or finite-value failure in companion")
        delta = float(abs(a - b).max())
        if delta > 1e-9:
            raise ValueError(f"Alternate formula disagreement {delta}")
        errors.append(delta)
        checks += a.size

    z = np.load(out / "01/coordinates.npz")
    vectors, scores = z["vectors"], z["factors"]
    bundle = load(ROOT / spec["bundle"])
    if bundle["feature_names"] != [f"mean::svd_{i:03d}" for i in range(24)]:
        raise ValueError("Companion requires the declared mean-only bundle")
    x = (vectors - np.array(bundle["feature_center"])) / np.array(
        bundle["feature_scale"]
    )
    w = np.array(bundle["factor_loadings"])
    psi = np.maximum(bundle["noise_variance"], 1e-12)
    covariance = w.T @ w + np.diag(psi)
    posterior = (x - np.array(bundle["fa_mean"])) @ np.linalg.solve(covariance, w.T)
    alternate = (posterior * bundle["factor_signs"] - bundle["norm_mean"]) / bundle[
        "norm_scale"
    ]
    compare(scores, alternate)
    units = load(out / "01/units.json")
    runtime = joblib.load(ROOT / spec["runtime"])
    sparse = load_npz(out / "01/tfidf.npz")
    replay = runtime.vectorizer.transform([r["text"] for r in units])
    diff = replay - sparse
    if diff.nnz and abs(diff.data).max() > 1e-14:
        raise ValueError("Sparse transform replay differs")
    compare(vectors, sparse @ runtime.svd.components_.T)
    index = {(r["unit"], r["view"], r["slice"]): i for i, r in enumerate(units)}
    rows = load(source / "00/rows.json")
    lookup = {
        (r["lexicon"], r["frame"], r["family"], r["previous"], r["current"]): r["unit"]
        for r in rows
        if r["kind"] == "pair"
    }
    groups = list(itertools.product(range(2), spec["frames"], spec["targets"]))
    address = load(out / "02/addresses.json")
    with np.load(out / "02/kernels.npz") as data:
        kernels = {k: data[k] for k in data.files}
    rebuilt = {}
    for arm, group_ids in address.items():
        if arm == "target":
            with np.load(source / "02/kernels.npz") as data:
                compare(kernels[arm], data["kernel"].reshape(24, 3, 3, 18))
            continue
        for p, g in enumerate(group_ids):
            k = np.zeros_like(kernels[arm][p])
            for i, j in itertools.product(range(3), repeat=2):
                uid = lookup[*groups[g], i, j]
                if arm == "whole":
                    value = alternate[index[uid, "whole", 0]]
                else:
                    view = "event" if arm.startswith("event") else "quartet"
                    n = 3 if view == "event" else 4
                    path = np.array([alternate[index[uid, view, s]] for s in range(n)])
                    if arm.endswith("mean"):
                        value = path.sum(0) / n
                    elif arm == "event_step":
                        value = np.diff(path[:2], axis=0)[0]
                    elif arm == "event_area":
                        wedge = (
                            sum(
                                np.outer(path[t], path[(t + 1) % 3])
                                - np.outer(path[(t + 1) % 3], path[t])
                                for t in range(3)
                            )
                            / 2
                        )
                        value = wedge[np.triu_indices(4, 1)]
                    elif arm == "quartet_dispersion":
                        value = np.sqrt(np.sum((path - path.mean(0)) ** 2, axis=0) / n)
                    else:
                        var = path.var(0)
                        sd = np.sqrt(var)
                        expected = 2 * n * var / (n - 1)
                        rough = np.mean(np.diff(path, axis=0) ** 2, axis=0)
                        excess = np.array(
                            [
                                r / e - 1 if e > 1e-20 else 0
                                for r, e in zip(rough, expected, strict=True)
                            ]
                        )
                        standardized = (path - path.mean(0)) / np.where(
                            sd > 1e-10, sd, 1
                        )
                        wedge = sum(
                            np.outer(standardized[t], standardized[t + 1])
                            - np.outer(standardized[t + 1], standardized[t])
                            for t in range(n - 1)
                        ) / (n - 1)
                        value = np.r_[
                            (path[-1] - path[0]) / (n - 1),
                            excess,
                            wedge[np.triu_indices(4, 1)],
                        ]
                k[i, j] = value
            compare(k, kernels[arm][p])
            rebuilt[arm, g] = k
    law_delta = np.zeros((3, 3))
    for i in range(3):
        law_delta[i, (i + 1) % 3] = 6 / 45
        law_delta[i, (i - 1) % 3] = -6 / 45
    parity = pd.read_csv(out / "02/parity.csv")
    cache = {}
    for arm, group_ids in address.items():
        for p, g in enumerate(group_ids):
            k = kernels[arm][p]
            odd = (k - np.einsum("ijc->jic", k)) / 2
            expected = np.einsum("ij,ijc->c", law_delta, k)
            saved = parity[parity.arm.eq(arm) & parity.group.eq(g)].sort_values(
                "channel"
            )
            compare(saved.cycle_contrast.to_numpy(), expected)
            compare(saved.odd_rms.to_numpy(), np.sqrt((odd**2).sum((0, 1)) / 9))
            cache[arm, g] = k
    paths = {
        r["author"]: r["states"] for r in load(source / "00/application_paths.json")
    }
    application = pd.read_csv(out / "02/application.csv")
    for (author, arm, group), saved in application.groupby(
        ["author", "arm", "group"], sort=False
    ):
        path = paths[author]
        counts = Counter(zip(path, path[1:] + path[:1], strict=True))
        expected = sum(
            count * cache[arm, group][i, j] for (i, j), count in counts.items()
        ) / len(path)
        compare(saved.sort_values("channel").value.to_numpy(), expected)
    truth = {r["author"]: r for r in load(source / "00/evaluation_truth.json")}
    application["direction"] = application.author.map(lambda a: truth[a]["direction"])
    application["seed"] = application.author.map(lambda a: truth[a]["seed"])
    application["replicate"] = application.author.map(lambda a: truth[a]["replicate"])
    for r in pd.read_csv(out / "02/contrasts.csv").itertuples(index=False):
        saved = application[
            application.group.eq(r.group)
            & application.arm.eq(r.arm)
            & application.channel.eq(r.channel)
            & application.seed.eq(r.seed)
            & application.replicate.eq(r.replicate)
        ]
        values = dict(zip(saved.direction, saved.value, strict=True))
        compare(
            np.array([r.plus_minus, r.midpoint_error]),
            np.array(
                [values[1] - values[-1], values[0] - (values[1] + values[-1]) / 2]
            ),
        )
        compare(
            r.plus_minus,
            np.einsum("ij,ij->", law_delta, cache[r.arm, r.group][..., r.channel]),
        )
    payload = {
        "status": "VERIFIED_FINITE_HM96_NOT_GLOBAL",
        "alternate_scalar_checks": checks,
        "maximum_absolute_error": max(errors),
        "sparse_replay_nnz": int(diff.nnz),
        "no_new_neural_calls_or_fits": True,
        "same_agent_not_third_party": True,
        "psychological_validity_claim": False,
        "scope_complete": False,
    }
    write(dest / "verification.json", payload)
    write(out / "03_seal.json", seal(out, dest))
    write(out / "verification.json", payload)
    print(payload, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
