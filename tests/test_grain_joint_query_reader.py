#!/usr/bin/env python3
"""Paired target identities, coherent moving-common derivatives and inventory."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms import joint_query_reader as f
from grain_research.vector_mechanisms import teaching_composition as old_account


def toy():
    """Finite source12, paired observations and fixed GRAIN feature units."""
    rng = np.random.default_rng(890)
    x = rng.normal(size=(12, 63))
    source = {
        "x": x,
        "v": np.zeros_like(x),
        "kinds": np.zeros(12, bool),
        "weights": np.ones(12) * 3,
        "regularizer": np.asarray(0.02),
        "units": np.arange(1.0, 6.0),
    }
    baseline = source | {
        "m": rng.normal(size=9),
        "d0": rng.normal(size=9),
        "beta": rng.normal(size=(12, 9)),
        "scale": np.ones(6),
    }
    cur, nat = rng.normal(size=(2, 12, 2, 5, 9))
    readers = rng.normal(size=(12, 2, 9))
    t = f.teaching(cur, nat, readers, baseline, source["units"])
    return rng, source, baseline, t, (cur, nat, readers)


def test_paired_teacher_recovers_native_query_values_after_lift():
    rng, source, b, t, (cur, _, _) = toy()
    for i in range(12):
        for q in range(2):
            d = f.coord.lift_value(
                rng.normal(size=9),
                t["frame"][i, q],
                t["compatible_coordinates"][i, q],
                f.origin(b)[q],
            )
            np.testing.assert_allclose(
                cur[i, q] @ d, t["target_values"][i, q], atol=1e-10
            )
    np.testing.assert_allclose(
        (t["compatible_coordinates"] - t["native_coordinates"]) * source["units"],
        t["target_values"] - np.einsum("nqkh,nqh->nqk", cur, toy()[-1][2]),
        atol=1e-10,
    )


def test_joint_source_solve_preserves_units_and_ten_output_linearity():
    _, s, _, t, _ = toy()
    m = f.fit(s, s["x"], t["compatible_coordinates"])
    assert m["coefficients"] == 120
    for k in ("x", "v", "kinds", "weights", "regularizer", "units"):
        np.testing.assert_array_equal(m[k], s[k])
    np.testing.assert_allclose(
        f.fit(s, s["x"], 2 * t["compatible_coordinates"])["beta"],
        2 * m["beta"],
        atol=1e-10,
    )
    assert m["normal_residual_relative"] < 1e-12


def test_independent_lift_derivative_includes_the_changing_common_reader():
    rng, s, b, t, _ = toy()
    m = f.fit(s, s["x"], t["compatible_coordinates"])
    # Test a supported point, rather than a distant Gaussian near-zero field.
    blocks = np.split(s["x"][0] * np.sqrt(6), [18, 27, 36, 45, 54])
    tangents = [rng.normal(size=(4, len(a))) for a in blocks]
    r = rng.normal(size=(2, 5, 9))
    dr = rng.normal(size=(2, 4, 5, 9))
    rv, rd, _, _, checks = f.readers(b, m, blocks, tangents, r, dr)
    np.testing.assert_allclose(rv, f.value_readers(b, m, blocks, r), atol=1e-12)
    assert max(checks.ravel()) < 1e-10
    eps = 1e-5
    for j in range(4):
        values = [
            f.value_readers(
                b,
                m,
                [x + sgn * eps * dx[j] for x, dx in zip(blocks, tangents, strict=True)],
                r + sgn * eps * dr[:, j],
            )
            for sgn in (-1, 1)
        ]
        np.testing.assert_allclose(
            rd[:, j], (values[1] - values[0]) / (2 * eps), atol=1e-6
        )
    assert np.linalg.norm(rd.mean(axis=0)) > 1e-3


def test_projection_shared_mode_requires_nonzero_mean_derivative():
    rng = np.random.default_rng(891)
    p = rng.normal(size=(2, 4, 9))
    v = rng.normal(size=(2, 4, 3, 4, 9))
    v[:, :, 2] = v[:, :, 0] + v[:, :, 1]
    r = rng.normal(size=(2, 9))
    dr = rng.normal(size=(2, 4, 9))
    eps = 1e-5
    result = f.projections(p, v, r, dr)
    for j in range(4):
        values = [
            np.einsum(
                "qah,qmh->qam",
                p + s * eps * v[:, :, 2, j],
                f.modes(r + s * eps * dr[:, j]),
            )
            for s in (-1, 1)
        ]
        np.testing.assert_allclose(
            result["diagonal"][:, :, :, j],
            (values[1] - values[0]) / (2 * eps),
            atol=1e-8,
        )
    assert np.max(abs(result["correction"][:, :, 1])) > 1e-3


def test_variable_common_signed_decomposition_closes_and_reduces_to_old_case():
    rng, _, b, _, (cur, nat, nr) = toy()
    r = rng.normal(size=nr.shape)
    acc = f.decompose(cur, nat, r, nr)
    energy = f.energies(acc)
    assert energy["closure_max"] < 1e-10
    d = r[:, 0] - r[:, 1]
    constant = np.stack((b["m"] + d / 2, b["m"] - d / 2), axis=1)
    before = old_account.decompose(cur, nat, constant, nr, b["m"])
    after = f.decompose(cur, nat, constant, nr)
    np.testing.assert_allclose(before["pieces"], after["pieces"], atol=1e-12)


def test_rank_units_and_partial_source_rejected():
    _, s, b, t, args = toy()
    with pytest.raises(ValueError):
        f.fit(s, s["x"][:8], t["compatible_coordinates"][:8])
    with pytest.raises(ValueError):
        f.teaching(*args, b, np.zeros(5))
    with pytest.raises(ValueError):
        f.teaching(np.ones_like(args[0]), args[1], args[2], b, s["units"])


def test_full_summary_keeps_old_prefix_and_all_query_views():
    from scripts.analyze_grain_joint_query_reader import summarize
    from scripts.run_grain_joint_query_functional_reader import COUNTS

    rng, s, b, t, _ = toy()
    m = f.fit(s, s["x"], t["compatible_coordinates"])
    rows, programs, prior_rows = [], [], []
    for _ in range(8):
        p = rng.normal(size=(2, 4, 9))
        v = rng.normal(size=(2, 4, 3, 4, 9))
        rr = rng.normal(size=(2, 9))
        dr = rng.normal(size=(2, 4, 9))
        np_ = rng.normal(size=(2, 9))
        nv = rng.normal(size=(2, 3, 4, 9))
        nr = rng.normal(size=(2, 9))
        target = f.projections(np_[:, None], nv[:, None], nr, np.zeros_like(dr))
        row = {k: z[None] for k, z in f.projections(p, v, rr, dr).items()} | {
            "readers": rr[None],
            "native_r": nr,
            "native_primal": np_,
            "native_vectors": nv,
            "target_scalar": target["scalar"][:, 0],
            "target_actions": target["actions"][:, 0],
            "frame": f.frames(p, v, s["units"]),
            "coordinate_values": rng.normal(size=(1, 2, 5)),
        }
        rows.append(row)
        programs.append({"primal": p, "vectors": v})
        prior_rows.append(
            {k: np.repeat(row[k], 2, axis=0) for k in ("scalar", "actions", "readers")}
        )
    prior = [
        pd.DataFrame({"variant": ["old"] * n, "authors": [8] * n})
        for n in (528, 1584, 6336, 286, 22, 2112)
    ]
    errors = pd.DataFrame({"variant": ["old"] * 32, "authors": [8] * 32})
    tables = summarize(rows, m, b, t, prior, programs, prior_rows, errors)
    assert [len(a) for a in tables] == COUNTS
    for a, z in zip(prior, tables, strict=False):
        pd.testing.assert_frame_equal(
            a, z.iloc[: len(a)][list(a)].reset_index(drop=True)
        )
    assert set(tables[7].target) == {"native_coordinates", "compatible_coordinates"}
    assert set(tables[9].observable) == {"scalar", "key", "value", "total"}
