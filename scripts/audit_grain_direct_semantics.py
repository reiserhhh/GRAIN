#!/usr/bin/env python3
"""Independently reconstruct frozen semantic-panel arithmetic and complete tables."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_semantics import panel
from grain_research.vector_mechanisms.direct_traits import BIG5
from grain_research.vector_mechanisms.history_collection import observation_signature
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json


def reconstruct(rows, preflight):
    """Use shifted exponentials, not native_readout/summarize, for all90 saved cases."""
    if len(rows) != 90 or len(preflight) != 90:
        raise ValueError("Require the complete90-case semantic grid")
    errors, by_key = [], {}
    for c, r, pre in zip(panel(), rows, preflight, strict=True):
        if any(r.get(k) != v for k, v in pre.items()) or any(
            r.get(k) != v for k, v in c.items()
        ):
            raise ValueError("Frozen semantic case/template/alphabet differs")
        a = r["alphabet"]
        z, v = np.asarray(r["candidate_logits"]), np.asarray(a["values"])
        normalizer = r["full_vocab_log_normalizer"]
        if (
            z.shape != v.shape
            or not np.isfinite(z).all()
            or not np.isfinite(normalizer)
        ):
            raise ValueError("Invalid candidate logits/normalizer")
        w = np.exp(z - max(z))
        q = w / sum(w)
        log_mass = float(max(z) + np.log(sum(w)) - normalizer)
        if log_mass > 1e-10:
            raise ValueError("Candidate mass exceeds vocabulary mass")
        score = float(sum(q * v))
        expected = {
            "conditional_score": score,
            "conditional_probabilities": q,
            "conditional_entropy": float(-sum(q[q > 0] * np.log(q[q > 0]))),
            "candidate_log_mass": log_mass,
            "candidate_mass": float(np.exp(log_mass)),
        }
        for k, x in expected.items():
            y = np.asarray(r[k])
            if y.shape != np.asarray(x).shape or not np.isfinite(y).all():
                raise ValueError("Invalid saved semantic arithmetic")
            err = float(np.max(np.abs(y - x)))
            if err > 1e-10:
                raise ValueError("Saved semantic arithmetic does not reproduce")
            errors.append(err)
        top = int(np.argmax(z))
        greedy = r["greedy_first_token_id"]
        if (
            r["candidate_argmax_label"] != a["labels"][top]
            or r["candidate_argmax_value"] != v[top]
            or r["greedy_first_token_is_candidate"] != (greedy in a["token_ids"])
            or (greedy in a["token_ids"] and z[a["token_ids"].index(greedy)] != z[top])
            or r["complete_response_parse_rate_measured"] is not False
        ):
            raise ValueError("Saved semantic compliance/argmax differs")
        by_key[r["task"], r["variant"], r["pair"], r["polarity"]] = score
    contrasts, tasks = [], []
    for task in dict.fromkeys(c["task"] for c in panel()):
        result = {
            "task": task,
            "counter_type": "numeric_scale_inversion"
            if task in BIG5
            else "answer_listing_reversal",
        }
        for variant in ("primary", "counter"):
            ds = [
                by_key[task, variant, j, 1] - by_key[task, variant, j, 0]
                for j in range(2)
            ]
            for j, d in enumerate(ds):
                contrasts.append(
                    {
                        "task": task,
                        "variant": variant,
                        "pair": j,
                        "high_minus_low": d,
                        "expected_sign": 1,
                        "positive": bool(d > 1e-6),
                    }
                )
            result[variant + "_positive_pairs"] = sum(d > 1e-6 for d in ds)
            result[variant + "_no_evidence_score"] = by_key[task, variant, -1, -1]
        result["mean_absolute_recoded_counter_change"] = (
            sum(
                abs(by_key[task, "primary", j, p] - by_key[task, "counter", j, p])
                for j in range(2)
                for p in range(2)
            )
            / 4
        )
        tasks.append(result)
    return contrasts, tasks, max(errors)


def compare_tables(actual, expected):
    """Check every row, numeric value and categorical field without rounding."""
    if len(actual) != len(expected):
        raise ValueError("Semantic summary rows missing")
    errors = []
    for a, e in zip(actual, expected, strict=True):
        if set(a) != set(e):
            raise ValueError("Semantic summary fields differ")
        for k, v in e.items():
            if isinstance(v, (float, np.floating)):
                err = abs(a[k] - v)
                if not np.isfinite(err) or err > 1e-10:
                    raise ValueError("Semantic summary value differs")
                errors.append(err)
            elif a[k] != v:
                raise ValueError("Semantic summary category differs")
    return max(errors, default=0.0)


def run(a):
    """Bind native/token-only parents, then save aggregate independent audit evidence."""
    m, s = verified_packet(a.packet)
    pm, ps = verified_packet(a.preflight)
    qm, qs = verified_packet(a.qualification)
    if (
        m["protocol"] != "VM05_DIRECT_SEMANTICS_V1"
        or m["preflight_only"]
        or not pm["preflight_only"]
        or s["cases"] != 90
        or s["forwards"] != 180
        or not s["real_neural_collection"]
        or ps["forwards"] != 0
        or ps["cases"] != 90
        or m["panel_sha256"] != material_hash(panel())
        or m["material_identity"] != pm["material_identity"]
        or m["qualification_identity"] != qm["identity"]
        or m["qualified_model_signature"] != qs["model_signature"]
        or m["source_snapshot_sha256"] != digest(a.packet / "source_snapshot.zip")
        or any(m[k] or s[k] for k in ("labels_read", "bridge_outcomes_read"))
    ):
        raise ValueError("Semantic panel is not a complete qualified real observation")
    for n, h in m["code_sha256"].items():
        if digest(ROOT / n) != h:
            raise ValueError("Frozen direct code differs")
    actual = json.loads((a.packet / "actual_observer.json").read_text(encoding="utf-8"))
    if observation_signature(actual) != qs["model_signature"]:
        raise ValueError("Semantic observer differs from qualified model")
    rows = json.loads((a.packet / "observations.json").read_text(encoding="utf-8"))
    pre = json.loads((a.preflight / "observations.json").read_text(encoding="utf-8"))
    contrasts, tasks, err = reconstruct(rows, pre)
    for n, computed in (("semantic_contrasts", contrasts), ("task_sensitivity", tasks)):
        saved = json.loads((a.packet / (n + ".json")).read_text(encoding="utf-8"))
        err = max(err, compare_tables(saved, computed))
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Audit artifacts belong under results")
    out.mkdir(parents=True, exist_ok=False)
    for n, table in (("semantic_contrasts", contrasts), ("task_sensitivity", tasks)):
        pd.DataFrame(table).to_csv(out / (n + ".csv"), index=False)
    summary = {
        "status": "COMPLETE",
        "packet_identity": m["identity"],
        "preflight_identity": pm["identity"],
        "code_sha256": digest(Path(__file__)),
        "cases": 90,
        "max_arithmetic_error": err,
        "candidate_mass_min": min(r["candidate_mass"] for r in rows),
        "candidate_mass_median": float(np.median([r["candidate_mass"] for r in rows])),
        "reported_first_token_compliance": sum(
            r["greedy_first_token_is_candidate"] for r in rows
        )
        / 90,
        "full_vocab_argmax_independently_available": False,
        "native_repeat_checked_by_original_collector": True,
        "labels_read": False,
        "neural_forwards": 0,
        "scope": "Complete invented-grid arithmetic; no population significance or human validity",
        "outputs_sha256": {
            n: digest(out / n)
            for n in ("semantic_contrasts.csv", "task_sensitivity.csv")
        },
    }
    summary["identity"] = material_hash(summary)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("packet", "preflight", "qualification", "output-dir"):
        p.add_argument("--" + n, type=Path, required=True)
    run(p.parse_args())
