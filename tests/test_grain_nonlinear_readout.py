#!/usr/bin/env python3
"""Small numerical witnesses for the M22 reader, not new human-data evidence."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.nonlinear_readout import (
    Geometry,
    fit_heads,
    fit_readers,
    grouped_folds,
    head_prediction,
    mixed_difference,
    paired_mse_interval,
    select_readers,
)


def require_cpu_lightgbm():
    """Keep full-suite neural imports from mixing incompatible macOS OpenMP libs."""
    import sys

    if sys.platform == "darwin" and "torch" in sys.modules:
        pytest.skip("Run this test file separately in the documented CPU-only process")
    return pytest.importorskip("lightgbm")


def test_group_splits_follow_historical_rule():
    """Repeated views stay together and permuting row order changes no assignment."""
    names = np.repeat([f"author{i}" for i in range(12)], 3)
    folds = grouped_folds(names)
    assert all(len(set(folds[names == name])) == 1 for name in set(names))
    assert np.array_equal(grouped_folds(names[::-1]), folds[::-1])
    with pytest.raises(ValueError):
        grouped_folds(["one"], 5)


def test_geometry_is_training_only_and_batch_invariant():
    """An extreme held-out sample cannot move the frozen training center."""
    rng = np.random.default_rng(4)
    train, test = rng.normal(size=(60, 8)), rng.normal(size=(10, 8))
    geometry = Geometry.fit(train, rank=4)
    assert np.allclose(geometry.scaler.mean_, train.mean(0))
    a = geometry.transform(test[:1])
    b = geometry.transform(np.vstack([test[:1], np.full((1, 8), 1e8)]))[:1]
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        geometry.transform(np.full((1, 8), np.nan))


def test_mixed_difference_linear_additive_and_interaction():
    """Mixed differences separate a joint term from affine or marginal curvature."""
    x = np.zeros((4, 2))
    d, e = np.tile([2.0, 0.0], (4, 1)), np.tile([0.0, 3.0], (4, 1))
    assert np.allclose(mixed_difference(lambda z: z @ [3, 4] + 2, x, d, e), 0)
    assert np.allclose(mixed_difference(lambda z: np.sum(z**2, axis=1), x, d, e), 0)
    assert np.allclose(mixed_difference(lambda z: z[:, 0] * z[:, 1], x, d, e), 6)
    with pytest.raises(ValueError):
        mixed_difference(lambda z: z[:, 0], x, d[:2], e)


def test_stumps_additive_in_fitted_coordinate_basis():
    """Depth-one boosted trees have no cross-coordinate mixed difference."""
    require_cpu_lightgbm()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 3))
    y = x[:, 0] ** 2 + np.sin(x[:, 1])
    heads = fit_heads(x, y, "stumps", {"trees": 50})
    d, e = np.zeros((20, 3)), np.zeros((20, 3))
    d[:, 0], e[:, 1] = 0.4, 0.6
    assert (
        np.max(
            np.abs(mixed_difference(lambda z: head_prediction(z, heads), x[:20], d, e))
        )
        < 1e-10
    )


def test_joint_signal_is_readable_beyond_stumps():
    """A synthetic interaction witness exercises the actual LightGBM implementation."""
    require_cpu_lightgbm()
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, size=(1400, 2))
    y = 0.2 * x[:, 0] + 3 * x[:, 0] * x[:, 1]
    stump = fit_heads(x[:1000], y[:1000], "stumps", {"trees": 300})
    tree = fit_heads(x[:1000], y[:1000], "trees", {"trees": 300})
    mse = lambda h: np.mean((head_prediction(x[1000:], h) - y[1000:]) ** 2)
    assert mse(tree) < mse(stump) * 0.4


def test_reader_parts_and_no_test_target_interface():
    """A residual reader predicts as the sum of two train-fitted components."""
    require_cpu_lightgbm()
    rng = np.random.default_rng(6)
    x = rng.normal(size=(60, 6))
    y = x[:, 0] + x[:, 1] * x[:, 2]
    names = np.array([f"u{i}" for i in range(len(x))])
    specs = select_readers(x, y, names, "test_inner", rank=4, alphas=(1.0,), trees=(5,))
    readers = fit_readers(x, y, specs, rank=4)
    for reader in readers.values():
        a, b = reader.parts(x[:5])
        assert np.allclose(a + b, reader.predict(x[:5]))
        assert np.isfinite(reader.predict(x[:5])).all()
    assert np.allclose(
        readers["ridge"].predict(x[:5]), readers["ridge_trees"].parts(x[:5])[0]
    )


def test_paired_gain_direction_and_finiteness():
    """Positive paired gain means lower teacher MSE, not a sign-flipped loss."""
    y = np.arange(20, dtype=float)
    ci = paired_mse_interval(y, y, y + 1, draws=50)
    assert ci["teacher_mse_gain"] == 1
    assert ci["teacher_mse_gain_lower"] == 1
    with pytest.raises(ValueError):
        paired_mse_interval(y, y[:-1], y)
    with pytest.raises(ValueError):
        paired_mse_interval(y, np.full(20, np.nan), y)


def test_bad_teacher_rejected():
    """Invalid teacher values fail explicitly without dropping authors."""
    x = np.ones((12, 4))
    with pytest.raises(ValueError):
        fit_heads(x, np.full(12, np.nan), "ridge", {"alpha": 1})
    with pytest.raises(ValueError):
        fit_heads(x, np.ones(12), "unknown", {})


def test_prefix_loader_ignores_appledouble_and_joins_query_column(
    tmp_path, monkeypatch
):
    """Synthetic cached data catches dataframe-name and macOS sidecar hazards."""
    import pandas as pd

    from scripts import run_grain_nonlinear_readout as runner

    prefix = tmp_path / "prefix"
    packets = prefix / "private_packets"
    packets.mkdir(parents=True)
    source = tmp_path / "source_remainder_s9_r1"
    source.mkdir()
    monkeypatch.setattr(runner, "PREFIX", prefix)
    monkeypatch.setattr(runner, "BASE", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    material = source / "private_material.json"
    runner.write(material, [{"author_id": "u0", "query": "TF_cont"}])
    runner.write(
        prefix / "manifest.json", {"source_material_sha256": runner.sha(material)}
    )
    runner.write(prefix / "run_status.json", {"status": "COMPLETE"})
    scores = prefix / "private_scores_before_outcomes.csv"
    pd.DataFrame(
        [{"author_id": "u0", "query": "TF_cont", "native": 0.5, "PQ": 0.4}]
    ).to_csv(scores, index=False)
    runner.write(prefix / "scores_frozen.json", {"sha256": runner.sha(scores)})
    arrays = packets / "author_000.npz"
    np.savez(arrays, P=np.ones((3, 2, 4)), Q=np.ones((3, 4)))
    runner.write(
        arrays.with_suffix(".json"),
        {"author_id": "u0", "query": "TF_cont", "arrays_sha256": runner.sha(arrays)},
    )
    (packets / "._author_000.json").write_bytes(b"\xff\x00")
    targets, vectors, hashes = runner.prefix_features(np.array(["u0"]))
    assert len(targets) == len(vectors) == len(hashes) == 1
    assert vectors["u0", "TF_cont"].shape == (8,)
    with pytest.raises(ValueError):
        runner.prefix_features(np.array(["different_author"]))


def test_natural_interaction_separates_feature_map_from_reader():
    """An affine reader can inherit nonadditivity from natural text aggregation."""
    from scripts.analyze_grain_nonlinear_edits import natural_interaction_parts

    parts = natural_interaction_parts(
        np.array([1]), np.array([2]), np.array([3]), np.array([7]), np.array([4])
    )
    assert parts["reader"] == 0
    assert parts["feature"] == parts["natural"] == 3
    rng = np.random.default_rng(18)
    parts = natural_interaction_parts(*rng.normal(size=(5, 30)))
    assert np.allclose(parts["natural"], parts["reader"] + parts["feature"])


def test_cpu_reader_import_does_not_load_torch():
    """Avoid the macOS PyTorch/Homebrew OpenMP collision in the fit process."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts import run_grain_nonlinear_readout; import sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_pooling_matches_explicit_query_weight_formula():
    """Keep the same frozen mathematical feature map after runtime isolation."""
    from grain_research.vector_mechanisms.readout_cache_cpu import pooled_features

    rng = np.random.default_rng(23)
    comments = rng.normal(size=(3, 12, 8))
    poles = rng.normal(size=(2, 2, 8))
    result = pooled_features(comments, 1, poles, "query")
    assert np.allclose(result[:, :8], comments.mean(1))
    p = poles[1] / np.linalg.norm(poles[1], axis=1, keepdims=True)
    for i, v in enumerate(comments):
        score = np.abs((v / np.linalg.norm(v, axis=1, keepdims=True)) @ (p[1] - p[0]))
        weight = np.exp(20 * (score - score.max()))
        assert np.allclose(result[i, 8:], (weight / weight.sum()) @ v)
