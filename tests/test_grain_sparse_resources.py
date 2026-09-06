"""Guard matrix lifetime without changing nested selection mathematics."""

import weakref

import numpy as np
from sklearn.pipeline import Pipeline

from grain_research.vector_mechanisms.external_heads import select_head
from scripts.run_grain_vector_mechanism_sparse_baselines import cases


def test_inner_design_matrices_are_released_before_final_fit(monkeypatch):
    refs = []
    original_getitem, original_fit = Pipeline.__getitem__, Pipeline.fit

    class WatchedTransform:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def fit_transform(self, x):
            result = self.pipeline.fit_transform(x)
            refs.append(weakref.ref(result))
            return result

        def transform(self, x):
            result = self.pipeline.transform(x)
            refs.append(weakref.ref(result))
            return result

    def watched_getitem(pipe, index):
        result = original_getitem(pipe, index)
        return WatchedTransform(result) if index == slice(None, -1) else result

    def watched_fit(pipe, x, y, **kwargs):
        if len(pipe.steps) > 1 and len(x) == 60:
            assert len(refs) == 6
            assert all(ref() is None for ref in refs)
        return original_fit(pipe, x, y, **kwargs)

    monkeypatch.setattr(Pipeline, "__getitem__", watched_getitem)
    monkeypatch.setattr(Pipeline, "fit", watched_fit)
    x = np.random.default_rng(7).normal(size=(60, 4))
    y = (x[:, 0] > 0).astype(int)
    model, search, parameter = select_head(
        x, y, np.asarray([str(i) for i in range(60)]), "dense", "EI_cont", pca=False
    )
    assert len(search) == 4 and parameter in (0.01, 0.1, 1, 10)
    assert np.mean(model.predict(x) == y) > 0.8


def test_remaining_axes_do_not_repeat_completed_ei():
    assert len(cases()) == 12
    remaining = cases(["SN_cont", "TF_cont", "JP_cont"])
    assert len(remaining) == 9
    assert all(row["task"] != "EI_cont" for row in remaining)
