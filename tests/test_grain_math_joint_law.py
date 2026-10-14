#!/usr/bin/env python3
"""Finite-law tests use synthetic cells, never private author text."""

import numpy as np
import pytest

from grain_research.construction import math_joint_law as J


def test_mobius_roundtrip_vector():
    f = np.random.default_rng(10).normal(size=(8, 7))
    np.testing.assert_allclose(J.zeta(J.mobius(f)), f, atol=1e-14)


def test_joint_difference_and_three_way_contrast():
    coefficients = np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8])
    f = J.zeta(coefficients)
    assert f[3] - f[1] - f[2] + f[0] == pytest.approx(0.4)
    assert f[7] - f[5] - f[6] + f[4] - (f[3] - f[1] - f[2] + f[0]) == pytest.approx(0.8)


@pytest.mark.parametrize("p", [[0.5, 0.5, 0.5], [0.2, 0.7, 0.35]])
def test_product_basis_closure(p):
    result = J.product_fourier(np.arange(8.0) ** 2, p)
    assert result["gram_error"] < 1e-14
    assert result["reconstruction_error"] < 1e-12
    assert result["variance_closure_error"] < 1e-10


def test_anchor_and_pure_interaction_are_distinct():
    f = np.array([0.0, 0.0, 0.0, 1.0])
    assert J.mobius(f)[3] == 1
    assert J.product_fourier(f, [0.5, 0.5])["coefficients"][3] == pytest.approx(0.25)


def test_reference_changes_decomposition_not_cell_values():
    f = np.array([0.0, 0.0, 0.0, 1.0])
    a = J.product_fourier(f, [0.5, 0.5])
    b = J.product_fourier(f, [0.2, 0.7])
    assert not np.allclose(a["energies"], b["energies"])
    assert a["reconstruction_error"] < 1e-14
    assert b["reconstruction_error"] < 1e-14


def test_missing_joint_not_resolved_by_more_singletons():
    w = J.support_witness()
    assert w["singleton_rank"] == w["duplicate_singleton_rank"] == 3
    assert w["full_joint_rank"] == 4
    assert w["source_null_error"] == 0
    values = [v["cells"] for v in w["worlds"].values()]
    np.testing.assert_equal(np.array(values)[:, :3], np.tile(values[0][:3], (3, 1)))
    assert len({v[3] for v in values}) == 3


def test_source_only_interaction_fit():
    rows = [
        {
            "role": "source",
            "frame": 0,
            "actor": "self",
            "incidence": 0,
            "family": i,
            "interaction": v,
        }
        for i, v in enumerate([0.2, 0.4])
    ]
    base = J.fit_source_interactions(rows)
    mixed = J.fit_source_interactions(
        rows + [{**rows[0], "role": "application", "interaction": 999}]
    )
    assert mixed == base
    assert base["global"] == pytest.approx(0.3)


def test_duplicate_source_family_rejected():
    row = {
        "role": "source",
        "frame": 0,
        "actor": "self",
        "incidence": 0,
        "family": 0,
        "interaction": 0.2,
    }
    with pytest.raises(ValueError, match="duplicated"):
        J.fit_source_interactions([row, row])


@pytest.mark.parametrize("bad", [[1, 2, 3], [0, 1, 2, np.nan]])
def test_missing_or_nonfinite_cells_rejected(bad):
    with pytest.raises(ValueError):
        J.mobius(bad)


def test_source_support_and_probability_validation():
    with pytest.raises(ValueError):
        J.fit_source_interactions([])
    with pytest.raises(ValueError):
        J.product_fourier(np.ones(4), [0, 0.5])
    with pytest.raises(ValueError):
        J.design_matrix(np.array([4]), 2)


def test_family_bootstrap_not_duplicated_row_count():
    a = J.paired_family_interval([0, 0, 1, 1], [1, 1, 3, 3])
    b = J.paired_family_interval([0, 1], [1, 3])
    assert a == b


def test_coordinate_transport_preserves_joint_dual_action():
    rng = np.random.default_rng(110)
    h, w = rng.normal(size=(8, 5)), rng.normal(size=5)
    transform = rng.normal(size=(5, 5)) + 4 * np.eye(5)
    shift = rng.normal(size=5)
    transformed_h = h @ transform + shift
    transformed_w = np.linalg.solve(transform, w)
    for mask in range(1, 8):
        assert J.mobius(h)[mask] @ w == pytest.approx(
            J.mobius(transformed_h)[mask] @ transformed_w, abs=1e-12
        )


def test_factor_permutation_is_not_text_token_permutation():
    values = np.arange(8.0) ** 3

    def address(mask):
        return ((mask & 1) << 1) | ((mask & 2) >> 1) | (mask & 4)

    transported = values[[address(m) for m in range(8)]]
    original = J.mobius(values)
    np.testing.assert_allclose(
        J.mobius(transported), original[[address(m) for m in range(8)]]
    )


def runner_modules():
    """Load isolated public runner helpers without executing an experiment."""
    import importlib
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_grain_math_joint_law"), importlib.import_module(
        "diagnose_grain_math_joint_law"
    )


def test_materials_retain_addresses_wordbags_and_disjoint_families():
    import re
    from collections import Counter, defaultdict

    runner, _ = runner_modules()
    singles = []
    for target in runner.S.TARGETS:
        for ai, action in enumerate(
            [
                "organize",
                "coordinate",
                "assemble",
                "brief",
                "photograph",
                "sketch",
                "describe",
                "observe",
            ]
        ):
            singles.append(
                {
                    "target": target,
                    "role": "source",
                    "bits": [1 if ai < 4 else -1, 1, 1, 1],
                    "action_index": ai,
                    "object_index": 0,
                    "action": action,
                    "object": "the workers",
                }
            )
    spec = {
        "frames": [
            ["uneasy", "relieved"],
            ["restless", "settled"],
            ["tense", "calm"],
            ["uncertain", "confident"],
        ],
        "trigger_actors": ["self", "other"],
        "new_materials": 1024,
    }
    bank = runner.material_bank(singles, spec)
    groups, roles = defaultdict(dict), defaultdict(set)
    for row in bank:
        groups[row["cube_id"]][row["mask"]] = row
        roles[row["role"]].add(row["family"])
        for report in row["expected_reports"]:
            assert row["text"][slice(*report["root_span"])] == "feel"
            assert report["experiencer"] == "self"
            for trigger in report["triggers"]:
                assert row["text"][slice(*trigger["object_span"])] == trigger["object"]
                assert trigger["actor"] == row["actor"]
    assert [len(roles[k]) for k in ("source", "validation", "application")] == [8, 4, 4]
    assert not roles["source"] & (roles["validation"] | roles["application"])
    for cells in groups.values():
        assert set(cells) == set(range(8))
        for mask in range(4):
            bag = lambda text: Counter(re.findall(r"\w+", text.lower()))
            assert bag(cells[mask]["text"]) == bag(cells[mask + 4]["text"])


def test_dual_bootstrap_resamples_whole_families():
    _, diagnostic = runner_modules()
    keys = [
        (frame, actor, incidence)
        for frame in range(4)
        for actor in ("other", "self")
        for incidence in range(2)
    ]
    source, application = np.ones((8, 16)), np.ones((4, 16))
    result = diagnostic.independent_family_bootstrap(
        source, application, keys, draws=100
    )
    for row in result:
        assert row["source_and_application_family_ci"] == [1, 1]
        assert row["source_and_application_rmse_ci"] == [0, 0]


def test_joint_uncertainty_refuses_missing_or_duplicate_cells():
    _, diagnostic = runner_modules()
    a = {
        "role": "source",
        "family": "0",
        "frame": "0",
        "actor": "self",
        "incidence": "0",
        "interaction": "0.3",
    }
    with pytest.raises(ValueError, match="duplicated"):
        diagnostic.matched_arrays([a, a], "source")
    b = {**a, "family": "1", "incidence": "1"}
    with pytest.raises(ValueError, match="missing"):
        diagnostic.matched_arrays([a, b], "source")
