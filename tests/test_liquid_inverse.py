#!/usr/bin/env python3
import numpy as np
from grain_core.liquid_inverse import bounds


def test_indistinguishable_sources_and_strengthened_joint_constraints():
    a = np.array([[5, 5, 6], [8, 9, 8]])
    truth = [0.2, 0.3, 0.5]
    first = bounds(a[:1], (a @ truth)[:1], 0)
    assert np.allclose(first["bounds"], [[0, 0.5], [0, 0.5], [0.5, 0.5]])
    both = bounds(a, a @ truth, 0)
    assert np.allclose(both["bounds"], np.repeat(np.array(truth)[:, None], 2, axis=1))
    assert bounds(a, [5, 12], 0.01)["status"] == "outside_declared_source_bank"
    wider = bounds(a, a @ truth, 0.02)
    assert np.min(np.ptp(wider["bounds"], axis=1)) > 0.02
