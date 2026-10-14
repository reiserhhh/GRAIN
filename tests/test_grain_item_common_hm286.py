#!/usr/bin/env python3
"""Known key/common orthogonality,not a new psychometric theory."""

import numpy as np
import pytest

from grain_research.construction import math_behavior_predictive as B
from grain_research.construction import math_item_configuration as M
from scripts.describe_grain_item_common_hm286 import parts4
from tests.test_grain_item_configuration import bank


def test_common_direction_replays_old_unequal_pole_account():
    items = bank()
    raw = np.arange(150.0).reshape(3, 50)
    z, v, ids = M.item_frame(raw, items, 0)
    sign = np.array([items[i]["pole"] for i in ids])
    parts = parts4(z, v, sign)
    assert parts[0] == pytest.approx(parts[1:].sum(0))
    assert z @ (v * sign) == pytest.approx(
        B.key_accounts(raw, items)["common_support"][:, 0]
    )
    assert parts[3] @ v == pytest.approx(np.zeros(3), abs=1e-13)
    assert parts[3] @ (v * sign) == pytest.approx(np.zeros(3), abs=1e-13)


def test_uniform_endorsement_is_common_not_within_pole_pattern():
    items = bank()
    z, v, ids = M.item_frame(np.ones((3, 50)) * 4, items, 0)
    parts = parts4(z, v, np.array([items[i]["pole"] for i in ids]))
    assert parts[1] == pytest.approx(np.zeros_like(z), abs=1e-14)
    assert parts[2] == pytest.approx(z)
    assert parts[3] == pytest.approx(np.zeros_like(z), abs=1e-14)
