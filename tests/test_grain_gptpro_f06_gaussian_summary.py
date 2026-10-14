"""Reporting controls for paired design comparisons and descriptive frontiers."""

import numpy as np
import pandas as pd

from scripts.summarize_grain_gptpro_f06_gaussian_design import (
    paired_interval,
    pareto_rows,
)


def test_pareto_retains_tradeoffs_and_exact_ties():
    frame = pd.DataFrame(
        {
            "arm": ["low_cost", "low_error", "dominated", "tied"],
            "MSE": [0.3, 0.1, 0.4, 0.1],
            "sum_squared_perturbation": [0.0, 1.0, 2.0, 1.0],
            "effective_actions": [0, 2, 3, 2],
        }
    )
    result = {r["arm"]: r for r in pareto_rows(frame)}
    assert result["low_cost"]["point_nondominated"]
    assert result["low_error"]["point_nondominated"]
    assert result["tied"]["point_nondominated"]
    assert not result["dominated"]["point_nondominated"]


def test_paired_bootstrap_keeps_author_pair_and_seed_strata():
    rows = []
    for seed, offset in ((1, 1.0), (2, 3.0)):
        for i, baseline in enumerate([0.0, 2.0, 8.0]):
            for arm, value in (("random", baseline), ("new", baseline + offset)):
                rows.append(
                    {
                        "author_id": f"s{seed}:{i}",
                        "seed": seed,
                        "arm": arm,
                        "MSE": value,
                    }
                )
    result, strata = paired_interval(pd.DataFrame(rows), "new", "random", "MSE")
    np.testing.assert_allclose(
        [result["delta"], result["ci_low"], result["ci_high"]], [2, 2, 2], atol=1e-14
    )
    assert [row["authors"] for row in strata] == [3, 3]
