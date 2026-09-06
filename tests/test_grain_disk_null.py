#!/usr/bin/env python3
"""Exact disk/in-memory nested-null equivalence and yielded-array lifetime tests."""

import gc
import weakref

import numpy as np
import pytest

from grain_research.vector_mechanisms.external_null import (
    nested_predict_cached,
    prepare_nested_folds,
)
from grain_research.vector_mechanisms.external_null_disk import prepare_disk_folds


@pytest.mark.parametrize(
    "kind,task", [("dense", "EI_cont"), ("dense", "big5"), ("word_char", "EI_cont")]
)
def test_disk_is_exact_for_observed_and_permuted_labels(tmp_path, kind, task):
    rng = np.random.default_rng(11)
    ids = np.asarray([str(i) for i in range(60)])
    folds = np.arange(60) % 5
    x = rng.normal(size=(60, 6))
    if kind == "word_char":
        x = np.asarray(
            [f"word sample common day {i % 7} {i % 13} quiet calm" for i in range(60)]
        )
    y = (
        rng.normal(size=(60, 5))
        if task == "big5"
        else (np.arange(60) % 3 == 0).astype(int)
    )
    expected = prepare_nested_folds(x, ids, folds, kind, task)
    actual = prepare_disk_folds(
        tmp_path, x, ids, folds, kind, task, parent_identity="synthetic"
    )
    for labels in (y, y[rng.permutation(60)]):
        ep, es = nested_predict_cached(expected, labels, task)
        ap, aselect = nested_predict_cached(actual, labels, task)
        np.testing.assert_allclose(ap, ep, rtol=0, atol=1e-12)
        assert aselect == es
    replay = prepare_disk_folds(
        tmp_path, x, ids, folds, kind, task, parent_identity="synthetic"
    )
    assert replay.manifest["identity"] == actual.manifest["identity"]
    changed = x.copy()
    if kind == "word_char":
        changed[0] = "different corpus"
    else:
        changed[0, 0] += 1
    with pytest.raises(ValueError, match="Changed disk fold plan"):
        prepare_disk_folds(
            tmp_path, changed, ids, folds, kind, task, parent_identity="synthetic"
        )


def test_yielded_arrays_released_and_mutations_detected(tmp_path):
    ids = np.array([str(i) for i in range(60)])
    x = np.random.default_rng(3).normal(size=(60, 4))
    folds = np.arange(60) % 5
    disk = prepare_disk_folds(
        tmp_path, x, ids, folds, "dense", "EI_cont", parent_identity="synthetic"
    )
    it = iter(disk)
    old = next(it)
    alias = old["inner"][-1]
    refs = [weakref.ref(old["xtrain"]), weakref.ref(alias["xfit"])]
    new = next(it)
    gc.collect()
    assert old == {} and alias == {} and all(r() is None for r in refs)
    assert new["outer"] == 1
    it.close()
    assert new == {}
    row = disk.manifest["arrays"][0]
    with (tmp_path / row["path"]).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        disk.validate_files()
    with pytest.raises(ValueError):
        next(iter(disk))
