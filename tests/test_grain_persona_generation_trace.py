#!/usr/bin/env python3
"""Material, audit and statistics tests; no model or API call."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import persona_trace as pt


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_aligned_pairs_and_factorial_counts():
    tok = FakeTokenizer()
    for axis in pt.AXES:
        for family in pt.FAMILIES:
            high, low, n = pt.choose_aligned(tok, axis, family)
            assert len(tok.encode(high)) == len(tok.encode(low)) == n
    rows, aligned = pt.build_materials(tok, "pilot")
    assert len(rows) == 88
    assert sum(r["condition"] == "neutral" for r in rows) == 8
    assert sum(r["condition"].startswith("nonpersona") for r in rows) == 16
    assert all(r["system"].endswith(pt.SHARED_SUFFIX) for r in rows)
    assert len({r["text_id"] for r in rows}) == 88
    rows_x, _ = pt.build_materials(tok, "expanded")
    assert len(rows_x) == 6 * 3 * (2 * 3 * 2 + 1 + 2)


def test_split_roles_disjoint():
    r = pt.split_roles("pilot")
    assert set(r["discovery"]) & set(r["evaluation"]) == set()
    assert r["evaluation"] == ["advice"]


def test_audit_and_leak_strip():
    a = pt.audit_text("I went to the community event last weekend and I loved it. My personality is extraverted.", "diary", "eos")
    assert a["leak_hits"] == 2 and a["topic_hits"] >= 2 and not a["refusal"]
    assert "[X]" in pt.strip_leak("an extroverted, neurotic personality") and "extro" not in pt.strip_leak("extroverted")


def test_auc_and_signflip_null():
    scores = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.4])
    labels = np.array([1, -1, 1, -1, 1, -1])
    blocks = np.array(["a", "a", "b", "b", "c", "c"])
    assert pt.auc(scores, labels) == 1.0
    null = pt.paired_signflip_null(scores, labels, blocks, n=200, seed=1)
    assert null["observed"] == 1.0 and 0.3 < null["null_mean"] < 0.7
    assert pt.balanced_accuracy(np.array([1, -1, 1, 1, 1, -1]), labels) == pytest.approx(5 / 6)


def test_displacement_summary_uses_shrinkage_metric():
    rng = np.random.default_rng(0)
    rows = []
    feats = {}
    for sit in ("diary", "email"):
        for seed in (11, 23):
            for pole, name in ((1, "high"), (-1, "low")):
                tid = f"extraversion_{name}|named|{sit}|s{seed}"
                rows.append({"text_id": tid, "axis": "extraversion", "pole": pole, "family": "named", "situation": sit, "seed": seed, "condition": f"extraversion_{name}"})
                feats[tid] = rng.normal(0, 0.1, 4) + pole * np.array([1.0, 0, 0, 0])
    cov, shrink, n = pt.within_condition_covariance(feats, rows)
    summary = pt.displacement_summary(feats, rows, "extraversion")
    assert summary["pairs"] == 4 and summary["D2_mean_displacement"] > summary["D2_within_seed_median"]
    assert 0 <= shrink <= 1 and n == 8


def test_head_prehook_replaces_only_selected_head_slice_during_prefill():
    import torch

    from scripts.run_grain_persona_generation_trace import make_head_prehook

    donor = torch.arange(5 * 8, dtype=torch.float32).reshape(1, 5, 8) + 100.0
    current = torch.zeros(1, 5, 8)
    span, head, dim = [1, 2], 1, 4
    hook = make_head_prehook(donor, span, head, dim)
    (patched,) = hook(None, (current,))
    assert torch.equal(patched[0, span, dim:2 * dim], donor[0, span, dim:2 * dim])
    assert torch.equal(patched[0, span, :dim], torch.zeros(2, dim))
    assert torch.equal(patched[0, [0, 3, 4]], torch.zeros(3, 8))
    decode_step = torch.zeros(1, 1, 8)
    (unchanged,) = hook(None, (decode_step,))
    assert torch.equal(unchanged, decode_step)


def test_kv_hook_replaces_span_rows_and_head_slice():
    import importlib.util
    import sys

    import torch

    spec = importlib.util.spec_from_file_location("run_gpgt", "scripts/run_grain_persona_generation_trace.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_gpgt"] = mod
    spec.loader.exec_module(mod)
    out = torch.arange(6 * 8, dtype=torch.float32).reshape(1, 6, 8)
    donor = -torch.ones(2, 8)
    span = [1, 2]
    full = mod.make_kv_hook(donor, span)(None, None, out)
    assert torch.equal(full[0, span], donor)
    assert torch.equal(full[0, [0, 3, 4, 5]], out[0, [0, 3, 4, 5]])
    head1 = mod.make_kv_hook(donor, span, kv_head=1, dim=4)(None, None, out)
    assert torch.equal(head1[0, span, 4:8], donor[:, 4:8])
    assert torch.equal(head1[0, span, 0:4], out[0, span, 0:4])
    assert torch.equal(out[0, span], torch.arange(6 * 8, dtype=torch.float32).reshape(1, 6, 8)[0, span])  # input untouched


def test_row_replace_hook_touches_only_downstream_rows_and_head_slice():
    import importlib.util
    import sys

    import torch

    spec = importlib.util.spec_from_file_location("run_gpgt2", "scripts/run_grain_persona_generation_trace.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_gpgt2"] = mod
    spec.loader.exec_module(mod)
    x = torch.arange(7 * 8, dtype=torch.float32).reshape(1, 7, 8)
    donor = -torch.ones(1, 7, 8)
    span = [2, 3]
    downstream = [p for p in range(7) if p not in span]
    (full,) = mod.make_row_replace(donor, downstream)(None, (x,))
    assert torch.equal(full[0, span], x[0, span])  # span rows untouched
    assert torch.equal(full[0, downstream], donor[0, downstream])
    (head,) = mod.make_row_replace(donor, downstream, head=1, dim=4)(None, (x,))
    assert torch.equal(head[0, downstream, 4:8], donor[0, downstream, 4:8])
    assert torch.equal(head[0, downstream, 0:4], x[0, downstream, 0:4])
    assert torch.equal(head[0, span], x[0, span])
