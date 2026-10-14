#!/usr/bin/env python3
"""Finite structural-transfer tests, not personality-validity evidence."""

import numpy as np
import pytest

from grain_research.construction import math_relational_cells as M


def test_complete_fixed_design_and_label_free_intervals():
    rows, truth = M.material_bank()
    assert len(rows) == 84
    assert sum(r["role"] == "source" for r in rows) == 24
    assert sum(r["role"] == "cold_fact" for r in rows) == 12
    assert sum(r["role"] == "cold_behaviour" for r in rows) == 48
    for row, target in zip(rows, truth, strict=True):
        assert set(row) == {"id", "role", "body"}
        assert len(M.record_intervals(row["body"])) == 6
        assert sorted(target["roles"]) == [[i, j] for i in range(3) for j in range(2)]


def test_source_export_and_cold_labels_unnecessary():
    rng = np.random.default_rng(129)
    x = rng.normal(size=(80, 12))
    y = x[:, :2] @ np.array([[1, 2], [2, -1]])
    program = M.fit_program(x[:60], y[:60])
    cold = M.predict(program, x[60:])
    assert cold.shape == (20, 2)
    assert np.isfinite(cold).all()
    with pytest.raises(TypeError):
        M.predict(program, x[60:], labels=y[60:])


def test_empty_cell_is_unobserved():
    mean, count = M.cell_ledger(np.array([[0, 0], [0, 0]]), np.array([[2.0], [4.0]]))
    assert count[0, 0] == 2 and mean[0, 0, 0] == 3
    assert np.isnan(mean[2, 1, 0])
    with pytest.raises(ValueError):
        M.cell_ledger(np.array([[3, 0]]), np.array([[1.0]]))


def test_owner_swap_changes_cells_not_event_values():
    roles = np.array([[i, j] for i in range(3) for j in range(2)])
    values = np.arange(12).reshape(6, 2)
    a, _ = M.cell_ledger(roles, values)
    swapped = roles.copy()
    swapped[:, 0] = np.array([1, 0, 2])[roles[:, 0]]
    b, _ = M.cell_ledger(swapped, values)
    np.testing.assert_array_equal(b, a[[1, 0, 2]])


def test_value_formula_matches_independent_head_loop():
    rng = np.random.default_rng(10)
    v, d = rng.normal(size=(6, 2, 8)), rng.normal(size=(6, 12, 8))
    ref = np.zeros((6, 6))
    for e in range(6):
        for q in range(6):
            ref[e, q] = sum(v[e, h // 6] @ d[q, h] for h in range(12))
    np.testing.assert_allclose(M.target_values(v, d), ref, atol=1e-12)


def test_cold_cli_needs_only_numeric_inputs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    rng = np.random.default_rng(12)
    source = rng.normal(size=(30, 8))
    fitted = M.fit_program(source, source[:, :2])
    program = {
        f"{arm}_{k}": v for arm in ("real", "permuted") for k, v in fitted.items()
    }
    program["position"] = np.array([[i, j] for i in range(3) for j in range(2)])
    program["directions"] = rng.normal(size=(6, 12, 4))
    np.savez(tmp_path / "program.npz", **program)
    np.savez(
        tmp_path / "numeric.npz",
        ids=[34, 35],
        states=rng.normal(size=(2, 6, 8)),
        values=rng.normal(size=(2, 6, 2, 4)),
    )
    entry = (
        Path(__file__).resolve().parents[1]
        / "scripts/score_grain_relational_cells_cold.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(entry),
            str(tmp_path / "program.npz"),
            str(tmp_path / "numeric.npz"),
            str(tmp_path / "out.npz"),
        ],
        check=True,
    )
    with np.load(tmp_path / "out.npz") as out:
        assert out["real_roles"].shape == (2, 6, 2)
        assert out["position_counts"].sum() == 12
        assert not {"gold", "body", "labels", "logits"}.intersection(out.files)
