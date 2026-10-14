#!/usr/bin/env python3
"""Exact checks for the F15 recovery theory certificates."""

from __future__ import annotations

import numpy as np

from grain_research.gptpro_foundation_v3 import f02
from grain_research.gptpro_foundation_v3 import f15_theory as th


def test_c2_law_matches_f02_definition():
    order, p_one = th.markov_world("c2_xor")
    assert order == 2
    for a in (0, 1):
        for b in (0, 1):
            assert p_one((a, b)) == f02.c2_next_probability(a, b, 1, "xor")


def test_sequence_probabilities_normalize_and_are_stationary():
    for world in ("c2_xor", "hidden_order3", "c2_iid"):
        for n in (1, 3, 5):
            total = sum(th.sequence_probability(s, world) for s in th.strings(n, n))
            assert abs(total - 1) < 1e-12
        # stationarity: P(s) = sum_y P(y s)
        for s in th.strings(3, 3):
            left = sum(th.sequence_probability((y, *s), world) for y in (0, 1))
            assert abs(left - th.sequence_probability(s, world)) < 1e-12


def test_hankel_ranks_and_psr_closure():
    assert th.numerical_rank(th.hankel("c2_iid", 3, 3)[2])[0] == 1
    assert th.numerical_rank(th.hankel("c2_xor", 3, 3)[2])[0] == 4
    assert th.numerical_rank(th.hankel("hidden_order3", 2, 3)[2])[0] == 4
    assert th.numerical_rank(th.hankel("hidden_order3", 3, 3)[2])[0] == 8
    good = th.fit_linear_psr("c2_xor", 2, 3)
    assert th.closure_error(good, "c2_xor", 5, 3)[0] < 1e-10
    short = th.fit_linear_psr("hidden_order3", 2, 3)
    assert th.closure_error(short, "hidden_order3", 5, 3)[0] > 0.1
    long = th.fit_linear_psr("hidden_order3", 3, 3)
    assert th.closure_error(long, "hidden_order3", 5, 3)[0] < 1e-10


def test_memory_term_zero_for_markov_state_and_nonzero_when_state_too_short():
    assert th.memory_term("c2_xor", 2, 4)["weighted_variance"] < 1e-20
    m = th.memory_term("hidden_order3", 2, 3)
    assert abs(m["weighted_variance"] - 0.09) < 1e-12
    assert abs(m["weighted_mean"]) < 1e-12


def test_blackwell_order_on_f04_binary_analog():
    spec = th.f04_binary_channel()
    res = th.blackwell_certificates(spec, {"world": spec["hypotheses"]}, run_lp=True)
    assert all(v["status"] == "EXACT_KERNEL" for v in res["explicit_kernels"].values())
    assert res["lp"]["full_from_mean"]["status"] == "INFEASIBLE_CERTIFIED"
    mi = res["mutual_information_nats"]
    assert abs(mi["world:full"] - mi["world:minimal"]) < 1e-12
    assert mi["world:mean"] < mi["world:full"]


def test_f09_mean_statistic_has_no_family_information():
    spec = th.f09_family_channel(reps=1)
    a = spec["channel"]
    _, k = th.deterministic_garble(spec["outcomes"], sum)
    b = a @ k
    fam = np.array([h[0] for h in spec["hypotheses"]])
    merged = np.array([b[fam == c].mean(0) for c in range(3)])
    assert np.abs(merged - merged[0]).max() < 1e-12


def test_auxiliary_rank_certificates():
    assert th.auxiliary_rank(th.CONDITIONS)["rank"] == 4
    assert th.auxiliary_rank(th.CONDITIONS[:4])["rank"] == 3
    assert th.auxiliary_rank([th.CONDITIONS[0]] * 5)["rank"] == 0
    assert th.variability_rank(th.CONDITIONS, [0.3, -1.2]) == 4
    s = np.random.default_rng(1).standard_normal((100, 2))
    assert np.abs(th.unmix(th.mix(s)) - s).max() < 1e-12


def test_density_ratio_is_normalized_mixture_identity():
    rng = np.random.default_rng(3)
    s = rng.standard_normal((200, 2))
    x = th.mix(s)
    # sum_u p(u) p(x|u)/p(x) = 1 for every x
    ratios = np.stack(
        [np.exp(th.log_density_ratio(np.full(200, k), x)) for k in range(5)]
    )
    assert np.abs(ratios.mean(0) - 1).max() < 1e-12


def test_contrastive_null_world_is_near_chance():
    rows = th.contrastive_run(
        20260915,
        "null_world",
        sizes=(("source", 1000), ("validation", 500), ("application", 500)),
    )
    app = next(r for r in rows if r["role"] == "application")
    assert abs(app["accuracy"] - 0.5) < 0.06


def test_garbling_certificate_positive_and_farkas_negative():
    from grain_research.gptpro_foundation_v3.f15 import garbling_certificate

    informative = np.array([[0.9, 0.1], [0.2, 0.8]])
    blurred = informative @ np.array([[0.7, 0.3], [0.3, 0.7]])
    ok = garbling_certificate(informative, blurred)
    assert ok["status"] == "FEASIBLE"
    kernel = np.array(ok["kernel"])
    assert np.abs(informative @ kernel - blurred).max() < 1e-9
    bad = garbling_certificate(blurred, informative)
    assert bad["status"] == "INFEASIBLE_CERTIFIED"
    y = np.array(bad["farkas_y"])
    n, m = blurred.shape[1], informative.shape[1]
    e = np.vstack((np.kron(blurred, np.eye(m)), np.kron(np.eye(n), np.ones((1, m)))))
    d = np.concatenate((informative.ravel(), np.ones(n)))
    assert (e.T @ y).min() >= -1e-9 and d @ y < 0


def test_process_subworld_equivalence_ordering():
    res = th.process_blackwell(reps=1)
    mi = res["mutual_information_nats"]
    assert mi["process:E1"] < 1e-12
    assert mi["process:E3"] > mi["process:E2"] > mi["process:E1"]
    assert res["certificates"]["E1_from_E2_explicit"]["status"] == "EXACT_KERNEL"
    assert res["certificates"]["E1_from_E3_explicit"]["status"] == "EXACT_KERNEL"
    assert res["corner_law_max_gap"] < 1e-12


def test_f10_recompute_replays_archived_metrics():
    import csv
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "results/grain_gptpro_foundation_v3/F10/geometry_r1/metrics.csv"
    )
    if not path.exists():
        return
    rows, _ = th.f10_geometry_recompute()
    with path.open() as f:
        archived = list(csv.DictReader(f))
    for a in archived[:4]:
        vals = [
            r["se"]
            for r in rows
            if (r["world"], r["role"], r["method"])
            == (a["world"], a["role"], a["method"])
        ]
        assert abs(np.mean(vals) / 2 - float(a["full_mse"])) < 1e-12
