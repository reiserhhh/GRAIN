"""R15 channel utilization: frozen inputs, source-only channel, text as unit, independent recomputation, replay rows."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_channel_utilization_r1"
RES = json.loads((OUT / "results.json").read_text())
CH = json.loads((OUT / "channel_source.json").read_text())
ROWS = [json.loads(l) for l in (IN / "readings.jsonl").read_text().splitlines()]
Q5 = ["own_commitment", "uncertainty", "deference", "reported_other", "no_commitment"]


def texts():
    t = defaultdict(dict)
    for r in ROWS:
        d = t[r["text_id"]]
        d.update({k: r[k] for k in ("core", "split", "form", "owner", "frame", "wording")})
        d[f"{r['role']}:ent"] = r["p_entailment"]; d[f"{r['role']}:con"] = r["p_contradiction"]
    return t


def test_inputs_frozen_and_protocol_pinned():
    assert hashlib.sha256((IN / "readings.jsonl").read_bytes()).hexdigest() == RES["readings_sha256"]
    assert hashlib.sha256((ROOT / RES["protocol"]).read_bytes()).hexdigest() == RES["protocol_sha256"]
    assert RES["neural_calls"] == 0 and RES["unit"].startswith("text")
    assert (ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_channel_utilization_r1_attempt1_generating_target/results.json").exists()


def test_channel_uses_source_texts_only_independent_recomputation():
    t = texts(); feats = CH["variants"]["V1"]["features"]
    cells = [tuple(c) for c in CH["cells"]]
    for ci, c in enumerate(cells):
        src = [d for d in t.values() if d["split"] == "source" and (d["form"], d["owner"], d["frame"]) == c]
        assert len(src) == 4 and {d["core"] for d in src} == {1, 2}
        for j, f in enumerate(feats):
            assert abs(sum(d[f] for d in src) / 4 - CH["variants"]["V1"]["mu_source"][ci][j]) < 1e-6


def test_tolerance_recomputed_without_numpy():
    t = texts(); feats = CH["variants"]["V1"]["features"]; cells = [tuple(c) for c in CH["cells"]]
    for j, f in enumerate(feats):
        sq, res2 = [], 0.0
        for c in cells:
            by = {k: [d[f] for d in t.values() if d["split"] == "source" and d["core"] == k and (d["form"], d["owner"], d["frame"]) == c] for k in (1, 2)}
            sq.append((sum(by[1]) / 2 - sum(by[2]) / 2) ** 2)
            allv = by[1] + by[2]; m = sum(allv) / 4
            res2 += sum((v - m) ** 2 for v in allv)
        S = res2 / (64 - 16)
        d_inf = 2 * math.sqrt(0.75 * sum(sq) / 16)
        d100 = 2 * math.sqrt(0.75 * sum(sq) / 16 + S / 100)
        assert abs(d_inf - CH["variants"]["V1"]["delta_inf"][j]) < 1e-5
        assert abs(d100 - CH["variants"]["V1"]["delta_N100"][j]) < 1e-5


def test_raw_rows_reaggregate_summary_and_text_unit():
    agg = defaultdict(list)
    with (OUT / "utilization_rows.csv").open() as h:
        for r in csv.DictReader(h):
            real = [float(r[f"real_{f}"]) for f in ("bare", "epistemic", "deferential", "noncommittal")]
            assert abs(sum(real) - 1) < 1e-4 and all(abs(x * 100 - round(x * 100)) < 1e-3 for x in real)  # N = 100 whole texts
            if r["context"] == "test_pooled_source_channel":
                em = float(r["est_merged"]); tm = real[1] + real[3]
                agg[(r["scenario"], r["estimator"])].append(abs(em - tm))
    summ = {(s["scenario"], s["estimator"]): s for s in RES["utilization_summary"] if s["context"] == "test_pooled_source_channel"}
    for k, v in agg.items():
        assert abs(sum(v) / len(v) - summ[k]["mae_merged"]) < 1e-3  # summary rounded to 4 dp, rows to 5 dp


def test_compatible_sets_and_support_accounting():
    cs = [r for r in RES["compatible_set_summary"] if r["context"] == "test_pooled_source_channel" and r["variant"] == "V1"]
    merged = [r for r in cs if r["functional"] == "pi_epi_plus_non"]
    assert len(merged) == 4 and all(r["coverage"] == 1.0 and r["mean_width"] < .15 for r in merged)
    assert all(r["mean_width"] > .2 for r in cs if r["functional"] in ("pi_bare", "pi_epistemic", "pi_deferential", "pi_noncommittal"))
    assert RES["hard_code"]["test_unseen_texts"] == 4 and RES["hard_code"]["source_dictionary_size"] == 19
    k3 = [r for r in RES["identifiability"] if r["variant"] == "V1" and r["scenario"] == "K3"]
    assert all(r["rank"] == 11 and r["null_dim"] == 5 for r in k3)
