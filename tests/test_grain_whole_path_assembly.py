#!/usr/bin/env python3
"""Full composed derivatives, source baselines, pairing and restricted-data tests."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.selection_endpoint import endpoint
from grain_research.vector_mechanisms.source_state_transport import apply_map
from grain_research.vector_mechanisms.whole_path_assembly import (
    error_account,
    paired_queries,
    reverse_response,
    score_metrics,
)
from tests.test_grain_complete_transport import tiny_model


def test_actual_suffix_after_affine_map_matches_composition_derivative():
    model = tiny_model()
    rng = np.random.default_rng(55)
    x, d = rng.normal(size=(9, 16)), rng.normal(size=(4, 9, 16))
    m = {
        "mode": np.asarray("pointwise"),
        "mean": np.zeros(16),
        "basis": np.eye(16, 3),
        "center": np.zeros(6),
        "scale": np.ones(6),
        "coef": rng.normal(size=(6, 16)) * 0.05,
    }
    mapped, tangent = apply_map(x, d, m)[:2]
    t = torch.tensor(mapped[None], dtype=torch.float32)
    td = torch.tensor(tangent, dtype=torch.float32)
    kw = cached_suffix_kwargs(model, t)
    ids, c = torch.tensor([1, 2, 3]), torch.tensor([-0.5, 0.0, 0.5])
    value, gradient, tau = reverse_response(model, t, td, kw, ids, c, block=1)
    assert gradient.shape == (9, 16) and tau.shape == (4,)
    _, jvp = torch.func.jvp(
        lambda z: endpoint(model, z, kw, ids, c, block=1), (t,), (td[:1],)
    )
    np.testing.assert_allclose(float(jvp), tau[0], rtol=1e-4, atol=1e-7)
    eps = 0.001
    scores = []
    for sign in (1, -1):
        plus = apply_map(x + sign * eps * d[0], d, m)[0]
        with torch.no_grad():
            scores.append(
                float(
                    endpoint(
                        model,
                        torch.tensor(plus[None], dtype=torch.float32),
                        kw,
                        ids,
                        c,
                        block=1,
                    )
                )
            )
    np.testing.assert_allclose(
        (scores[0] - scores[1]) / (2 * eps), tau[0], rtol=0.003, atol=1e-4
    )
    assert np.isfinite(value)


def test_query_pairs_and_error_origin_do_not_disappear_in_high_correlation():
    y = np.arange(1, 6, dtype=float)
    r = score_metrics(y, y + 10, np.full(5, 3.0))
    assert (
        r["score_r"] == pytest.approx(1) and r["score_mse_over_source_constant"] == 50
    )
    assert r["error_variance"] == 0 and r["mean_error"] == 10
    a = error_account(y, y + 10, y + 5)
    assert a["relative_error_reduction"] == 0.75 and a["identity_error"] == 0
    df = pd.DataFrame(
        [
            {
                "id": i,
                "query_name": q,
                "actual": float(i + qi),
                "predicted": float(i + 2 * qi),
            }
            for i in range(4)
            for qi, q in enumerate(("neuroticism", "confirmation_seeking"))
        ]
    ).sample(frac=1, random_state=3)
    paired = paired_queries(df, ["id"], ["actual", "predicted"])
    np.testing.assert_array_equal(paired.actual, -1)
    np.testing.assert_array_equal(paired.predicted, -2)
    with pytest.raises(ValueError):
        paired_queries(df.iloc[:-1], ["id"], ["actual", "predicted"])
    with pytest.raises(ValueError):
        paired_queries(pd.concat([df, df.iloc[:1]]), ["id"], ["actual", "predicted"])


def test_guard_allows_only_declared_candidate_data(tmp_path):
    allowed, target = tmp_path / "input.npz", tmp_path / "native_target.npz"
    np.savez(allowed, x=np.ones(2))
    np.savez(target, score=np.ones(1))
    code = """
import sys
from pathlib import Path
import numpy as np
from grain_research.vector_mechanisms.stage_file_guard import install
p,q=map(Path,sys.argv[1:])
d=install([p],[],[q],[Path('grain_research')])
with np.load(p,allow_pickle=False) as z:
 assert z['x'].sum()==2
try:
 np.load(q,allow_pickle=False)
except PermissionError:
 assert len(d)==2
else:
 raise AssertionError('Native target was readable')
"""
    subprocess.run(
        [sys.executable, "-c", code, str(allowed), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
