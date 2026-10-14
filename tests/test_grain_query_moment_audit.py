#!/usr/bin/env python3
"""External linkage must use the frozen author namespace, not raw names."""

import pandas as pd
import pytest

from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.data_views import stable_key
from scripts.audit_grain_query_moments import label_namespace


def test_existing_pseudonym_links_without_mutating_source():
    """Scores and labels meet through existing IDs without any text-dependent fit."""
    frame = pd.DataFrame({"Neuroticism": [0.3]}, index=["synthetic-user"])
    result = label_namespace(frame)
    key = stable_key("synthetic-user", ID_SEED)[:24]
    assert result.loc[key, "Neuroticism"] == 0.3
    assert frame.index.tolist() == ["synthetic-user"]


def test_duplicate_label_ids_fail():
    """A join with duplicate authors cannot silently multiply evidence."""
    with pytest.raises(ValueError, match="Duplicate"):
        label_namespace(pd.DataFrame({"score": [0.2, 0.4]}, index=["x", "x"]))
