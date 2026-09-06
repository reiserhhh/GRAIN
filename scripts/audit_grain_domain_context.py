#!/usr/bin/env python3
"""Replay domain context lineage, token controls, readouts and case bootstraps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.domain_context import user_parts
from grain_research.vector_mechanisms.response_order import permute_prefix
from scripts.run_grain_domain_context import CONDITIONS
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
)


def replay_summary(frame):
    """Independent endpoint formula and explicit bootstrap draw loop."""
    rows = []
    for (corpus, language), g in frame.groupby(["corpus", "language"], sort=True):
        for score in ("tail_logp", "mean_logp"):
            p = g.pivot(
                index="case_code", columns="condition", values=score
            ).sort_index()
            natural = np.asarray(p["actual"]) - np.asarray(p["matched"])
            altered = np.asarray(p["actual_block8"]) - np.asarray(p["matched_block8"])
            signals = {
                "actual_minus_matched": natural,
                "actual_minus_random": np.asarray(p["actual"])
                - np.asarray(p["random"]),
                "actual_minus_none": np.asarray(p["actual"]) - np.asarray(p["none"]),
                "block8_actual_minus_matched": altered,
                "order_interaction": natural - altered,
                "common_block8_penalty": (
                    np.asarray(p["actual"])
                    + np.asarray(p["matched"])
                    - np.asarray(p["actual_block8"])
                    - np.asarray(p["matched_block8"])
                )
                * 0.5,
            }
            for endpoint, s in signals.items():
                rng = np.random.default_rng(731)
                boot = []
                for _ in range(1000):
                    indices = rng.integers(len(s), size=len(s))
                    boot.append(float(np.sum(s[indices]) / len(s)))
                low, high = np.quantile(boot, [0.025, 0.975])
                rows.append(
                    {
                        "corpus": corpus,
                        "language": language,
                        "score": score,
                        "endpoint": endpoint,
                        "mean": float(s.mean()),
                        "lower": float(low),
                        "upper": float(high),
                        "positive_count": int(np.sum(s > 0)),
                        "participants": len(s),
                    }
                )
    return pd.DataFrame(rows)


def main():
    """Audit saved evidence, not a second independent neural-model extraction."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private audit must stay in ignored results")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh audit directory")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer

    read = lambda p: json.loads(p.read_text())
    manifest = read(a.run_dir / "manifest.json")
    summary = read(a.run_dir / "summary.json")
    material = read(a.material_dir / "manifest.json")
    cases = read(a.material_dir / "private_cases.json")
    for m in (manifest, material):
        if m["identity"] != material_hash(
            {k: v for k, v in m.items() if k != "identity"}
        ):
            raise ValueError("Manifest identity mismatch")
    if (
        summary["identity"] != manifest["identity"]
        or summary["status"] != "COMPLETE_DOMAIN_CONTEXT_FUNCTION"
        or manifest["material_identity"] != material["identity"]
        or material_hash(cases) != material["case_material_sha256"]
    ):
        raise ValueError("Material and model packet mismatch")
    if (
        summary["conditions"] != 6 * len(cases)
        or summary["cases"] != len(cases)
        or manifest["conditions"] != list(CONDITIONS)
    ):
        raise ValueError("Condition coverage incomplete")
    for name, sha in summary["outputs_sha256"].items():
        if file_hash(a.run_dir / name) != sha:
            raise ValueError("Output artifact checksum mismatch")
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        if file_hash(a.model_dir / name) != EXPECTED_FILES[name]:
            raise ValueError("Frozen tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model_dir), local_files_only=True, trust_remote_code=False
    )
    observations = pd.read_csv(a.run_dir / "private_conditions.csv")
    if observations.duplicated(["case_code", "condition"]).any() or len(
        observations
    ) != 6 * len(cases):
        raise ValueError("Duplicate or missing observation key")
    lookup = {c["case_code"]: c for c in cases}
    for cell in {c["stratum"] for c in cases}:
        selected = [c for c in cases if c["stratum"] == cell]
        ids = {c["case_code"] for c in selected}
        if len(ids) < 4 or len({c["author_id"] for c in selected}) != len(ids):
            raise ValueError("Invalid author cell")
        for key in ("matched_case_code", "random_case_code"):
            if {c[key] for c in selected} != ids or any(
                c[key] == c["case_code"] for c in selected
            ):
                raise ValueError("Broken donor bijection")
    replayed = []
    error64, error32, geometry_error = 0.0, 0.0, 0.0
    shuffled_conditions = 0
    for case in cases:
        code = case["case_code"]
        cache = np.load(a.run_dir / "private_cases" / f"{code}.npz", allow_pickle=False)
        meta = read(a.run_dir / "private_cases" / f"{code}.json")
        _, alone, _, _ = user_parts(tokenizer, case["current"])
        np.testing.assert_array_equal(cache["body"], alone)
        for condition in CONDITIONS:
            donor = (
                case
                if condition.startswith("actual")
                else lookup[
                    case["random_case_code"]
                    if condition == "random"
                    else case["matched_case_code"]
                ]
            )
            context = None if condition == "none" else donor["context"]
            prefix, body, ix, _ = user_parts(tokenizer, case["current"], context)
            if condition.endswith("block8"):
                native = list(prefix)
                prefix, record = permute_prefix(
                    prefix, ix, donor["case_code"], "block8", 731
                )
                assert sorted(prefix) == sorted(native)
                assert all(
                    prefix[j] == native[j]
                    for j in range(len(prefix))
                    if j not in set(ix)
                )
                for key, val in record.items():
                    if meta[condition][key] != val:
                        raise ValueError("Order instrumentation metadata mismatch")
                shuffled_conditions += 1
            np.testing.assert_array_equal(prefix, cache[f"{condition}__prefix"])
            np.testing.assert_array_equal(body, cache["body"])
            if material_hash(body) != meta[condition]["body_sha256"]:
                raise ValueError("Current body hash changed")
            logp = cache[f"{condition}__logp"]
            if logp.shape != (len(body),) or not np.all(np.isfinite(logp)):
                raise ValueError("Invalid token likelihood vector")
            source = observations[
                (observations.case_code == code) & (observations.condition == condition)
            ].iloc[0]
            row = {
                "case_code": code,
                "corpus": case["corpus"],
                "language": case["language"],
                "condition": condition,
                "mean_logp": float(logp.mean()),
                "tail_logp": float(logp[1:].mean()),
            }
            for name, array in (("mean_logp", logp), ("tail_logp", logp[1:])):
                error32 = max(error32, abs(row[name] - source[name]))
                error64 = max(
                    error64, abs(float(array.astype(float).mean()) - source[name])
                )
            for site in ("input_embedding", "block_10", "block_16", "final_norm"):
                actual = cache[f"actual__{site}"]
                other = cache[f"{condition}__{site}"]
                if (
                    actual.shape != (1536,)
                    or other.shape != (1536,)
                    or not np.isfinite(other).all()
                ):
                    raise ValueError("Saved mean-state shape/finiteness mismatch")
                aa, bb = actual.astype(float), other.astype(float)
                cosine = np.clip(
                    np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)), -1, 1
                )
                norm = float(np.linalg.norm(actual - other))
                geometry_error = max(
                    geometry_error,
                    abs(cosine - source[f"{site}_cosine_to_actual"]),
                    abs(norm - source[f"{site}_norm_delta_to_actual"]),
                )
            np.testing.assert_array_equal(
                cache["actual__input_embedding"], cache[f"{condition}__input_embedding"]
            )
            replayed.append(row)
    if error32 > 1e-12 or geometry_error > 1e-10:
        raise ValueError("Stored point statistic does not replay")
    estimated = replay_summary(pd.DataFrame(replayed))
    published = pd.read_csv(a.run_dir / "domain_sensitivity.csv")
    keys = ["corpus", "language", "score", "endpoint"]
    joined = estimated.merge(
        published, on=keys, suffixes=("_replay", "_saved"), validate="one_to_one"
    )
    if len(joined) != len(published) or len(estimated) != len(published):
        raise ValueError("Missing summary endpoints")
    summary_error = max(
        float(abs(joined[k + "_replay"] - joined[k + "_saved"]).max())
        for k in ("mean", "lower", "upper", "positive_count", "participants")
    )
    if summary_error > 1e-12:
        raise ValueError("Independent case-bootstrap replay failed")
    estimated.to_csv(out / "replayed_summary.csv", index=False)
    result = {
        "status": "PASS_DOMAIN_CONTEXT_SAVED_PACKET_AUDIT",
        "parent_identity": manifest["identity"],
        "cases": len(cases),
        "conditions": len(replayed),
        "shuffled_conditions": shuffled_conditions,
        "bootstrap_endpoints": len(estimated),
        "bootstrap_draws": 1000 * len(estimated),
        "max_float32_point_error": error32,
        "max_float64_mean_difference": error64,
        "max_geometry_error": geometry_error,
        "max_summary_error": summary_error,
        "new_model_forwards": 0,
        "psychological_labels_read": False,
        "prepared_text_read": True,
        "audit_code_sha256": file_hash(Path(__file__)),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
