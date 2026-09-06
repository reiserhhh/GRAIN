#!/usr/bin/env python3
"""Independent semantic arithmetic and mutation rejection on all declared cases."""

import copy

import numpy as np
import pytest

from grain_research.vector_mechanisms.direct_semantics import panel, summarize
from grain_research.vector_mechanisms.direct_traits import BIG5, native_readout
from scripts.audit_grain_direct_semantics import compare_tables, reconstruct


def synthetic_packet():
    """Use invented logits only; the collector supplies the independent reference."""
    rng = np.random.default_rng(197)
    rows, pre = [], []
    for c in panel():
        size = 10 if c["task"] in BIG5 else 2
        values = np.linspace(0, 1, size)
        if c["task"] in BIG5 and c["variant"] == "counter":
            values = 1 - values
        alphabet = {
            "token_ids": list(range(size)),
            "labels": [str(i) for i in range(size)],
            "values": values.tolist(),
        }
        p = c | {
            "prompt_tokens": 20,
            "prompt_sha256": c["case_id"],
            "alphabet": alphabet,
        }
        pre.append(copy.deepcopy(p))
        rows.append(p | native_readout(rng.normal(size=20), alphabet))
    return rows, pre


def test_all_cases_and_tables_independently_reconstruct():
    rows, pre = synthetic_packet()
    cs, ts, err = reconstruct(rows, pre)
    wanted_c, wanted_t = summarize(rows)
    assert len(cs) == 36 and len(ts) == 9 and err < 1e-12
    assert compare_tables(wanted_c, cs) < 1e-12
    assert compare_tables(wanted_t, ts) < 1e-12


def test_mutations_missing_cases_and_counter_code_rejected():
    rows, pre = synthetic_packet()
    with pytest.raises(ValueError, match="complete"):
        reconstruct(rows[:-1], pre)
    bad = copy.deepcopy(rows)
    bad[0]["conditional_score"] += 0.1
    with pytest.raises(ValueError, match="arithmetic"):
        reconstruct(bad, pre)
    bad = copy.deepcopy(rows)
    bad[5]["alphabet"]["values"] = list(reversed(bad[5]["alphabet"]["values"]))
    with pytest.raises(ValueError, match="alphabet"):
        reconstruct(bad, pre)
    cs, ts, _ = reconstruct(rows, pre)
    bad = copy.deepcopy(cs)
    bad[0]["high_minus_low"] += 0.01
    with pytest.raises(ValueError, match="value"):
        compare_tables(bad, cs)
    bad = copy.deepcopy(ts)
    bad[0]["task"] = "changed"
    with pytest.raises(ValueError, match="category"):
        compare_tables(bad, ts)
