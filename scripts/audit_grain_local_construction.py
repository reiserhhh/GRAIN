#!/usr/bin/env python3
"""Independent construction arithmetic, isolated scorer replay and length follow-up."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, pearsonr, t
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.local_measure import TOKEN_PATTERN
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, load_texts


def run(args):
    """Recalculate using native matrices, then replay cold CLI without fit libraries."""
    out = args.output
    summary = json.loads((out / "summary.json").read_text())
    for name, sha in summary["files"].items():
        assert digest(out / name) == sha, name
    materials = load_texts()
    freeze = json.loads((out / "construction_freeze.json").read_text())
    assert not freeze["criterion_arrays_opened"]
    audit_dir = out / "audit_replay"
    audit_dir.mkdir(exist_ok=True)
    audit_dir = Path(tempfile.mkdtemp(prefix="cold_", dir=audit_dir))
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    labels = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = labels.loc[materials["essays"]["ids"], list(TRAITS)].to_numpy()
    assoc = pd.read_csv(out / "criterion_associations.csv")
    np.testing.assert_allclose(
        assoc.q_bh_100, false_discovery_control(assoc.p.to_numpy()), atol=1e-12
    )
    intervals = pd.read_csv(out / "private_material_intervals.csv")
    checks, partial_rows, all_scores = [], [], {}
    for key, sha in freeze["scorers"].items():
        path = out / f"private_{key}_scorer.json"
        assert digest(path) == sha
        b = json.loads(path.read_text())
        d = materials[b["domain"]]
        source, held = d["train"], d["held"]
        assert b["source_ids"] == [d["ids"][i] for i in source]
        assert not set(b["source_ids"]) & {d["ids"][i] for i in held}
        vectorizer = CountVectorizer(
            vocabulary=b["vocabulary"], token_pattern=TOKEN_PATTERN
        )
        counts = vectorizer.transform(d["documents"]).toarray().astype(float)
        totals = np.array(
            [len(re.findall(TOKEN_PATTERN, text.lower())) for text in d["documents"]]
        )
        rates = counts / totals[:, None]
        np.testing.assert_allclose(rates[source].mean(0), b["rate_mean"], atol=1e-12)
        np.testing.assert_allclose(
            rates[source].std(0, ddof=1), b["rate_scale"], atol=1e-12
        )
        raw = ((rates - b["rate_mean"]) / b["rate_scale"]) @ np.asarray(
            b["raw_weights"]
        ) + b["raw_offset"]
        np.testing.assert_allclose(raw[source].mean(0), b["score_center"], atol=1e-12)
        np.testing.assert_allclose(
            raw[source].std(0, ddof=1), b["score_scale"], atol=1e-12
        )
        z = (raw - b["score_center"]) / b["score_scale"]
        np.testing.assert_allclose(
            np.sort(z[source], axis=0), b["reference_sorted"], atol=1e-12
        )
        stored = pd.read_csv(out / f"private_{key}_held_scores.csv")
        assert stored.author_id.tolist() == [d["ids"][i] for i in held]
        np.testing.assert_allclose(stored[b["factor_ids"]], z[held], atol=1e-12)
        all_scores[key] = z[held]
        fixture = audit_dir / f"private_{key}_input.jsonl"
        result = audit_dir / f"private_{key}_output.jsonl"
        if not fixture.exists():
            with fixture.open("x") as f:
                for i in held:
                    f.write(
                        json.dumps({"id": d["ids"][i], "text": d["documents"][i]})
                        + "\n"
                    )
        if not result.exists():
            blocked = """
import importlib.abc, runpy, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','sklearn','pandas','joblib','scipy'}:
            raise ImportError('Cold scorer cannot import ' + fullname)
sys.meta_path.insert(0, BlockModels())
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
"""
            script = ROOT / "scripts/score_grain_local_measure.py"
            command = [
                sys.executable,
                "-c",
                blocked,
                str(script),
                "--bundle",
                str(path),
                "--input",
                str(fixture),
                "--output",
                str(result),
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=60, check=False
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr)
        replay = [json.loads(line) for line in result.open()]
        assert [r["id"] for r in replay] == stored.author_id.tolist()
        error = float(np.max(np.abs(np.array([r["score"] for r in replay]) - z[held])))
        assert error < 1e-12
        explained = json.loads((out / f"private_{key}_explanation.json").read_text())
        contributions = np.zeros(len(b["factor_ids"]))
        for hit in explained["hits"]:
            assert (
                explained["document"][hit["start"] : hit["end"]].lower() == hit["term"]
            )
            contributions += hit["contribution"]
        np.testing.assert_allclose(
            contributions + explained["intercept"], explained["score"], atol=1e-12
        )
        bound_table = intervals[
            (intervals.domain == b["domain"]) & (intervals.method == b["method"])
        ].set_index(["author_id", "factor"])
        bound_error = 0.0
        for i in held:
            units = d["units"][i]
            if len(units) < 2:
                assert bound_table.loc[d["ids"][i], "lower"].isna().all()
                continue
            uc = vectorizer.transform(units).toarray().astype(float)
            ut = np.array([len(re.findall(TOKEN_PATTERN, s.lower())) for s in units])
            ix = np.random.default_rng(923 + int(i)).integers(
                len(units), size=(200, len(units))
            )
            denominators = ut[ix].sum(1)
            sampled_rates = uc[ix].sum(1) / np.maximum(denominators[:, None], 1)
            values = ((sampled_rates - b["rate_mean"]) / b["rate_scale"]) @ np.asarray(
                b["raw_weights"]
            ) + b["raw_offset"]
            values = (values - b["score_center"]) / b["score_scale"]
            expected = np.quantile(values[denominators > 0], [0.025, 0.975], axis=0).T
            actual = (
                bound_table.loc[d["ids"][i]]
                .loc[b["factor_ids"], ["lower", "upper"]]
                .to_numpy()
            )
            np.testing.assert_allclose(actual, expected, atol=1e-11)
            bound_error = max(bound_error, float(np.max(np.abs(actual - expected))))
        y = ys[b["domain"]][held]
        observed = assoc[
            (assoc.domain == b["domain"]) & (assoc.method == b["method"])
        ].set_index(["factor", "criterion"])
        length_design = np.column_stack([np.ones(len(held)), np.log1p(totals[held])])
        zx = (
            z[held]
            - length_design @ np.linalg.lstsq(length_design, z[held], rcond=None)[0]
        )
        yy = y - length_design @ np.linalg.lstsq(length_design, y, rcond=None)[0]
        for j, factor in enumerate(b["factor_ids"]):
            for k, trait in enumerate(TRAITS):
                original = float(np.corrcoef(z[held, j], y[:, k])[0, 1])
                assert abs(original - observed.loc[(factor, trait), "r"]) < 1e-12
                r = float(pearsonr(zx[:, j], yy[:, k]).statistic)
                dof = len(held) - 3
                p = float(2 * t.sf(abs(r) * np.sqrt(dof / (1 - r * r)), dof))
                partial_rows.append(
                    {
                        "domain": b["domain"],
                        "method": b["method"],
                        "factor": factor,
                        "criterion": trait,
                        "n": len(held),
                        "raw_r": original,
                        "partial_r_loglength": r,
                        "p": p,
                        "status": "POSTHOC_DESCRIPTIVE_NOT_SCORE_REFIT",
                    }
                )
        checks.append(
            {
                "key": key,
                "source": len(source),
                "held": len(held),
                "cold_error": error,
                "material_interval_error": bound_error,
                "explanation_terms": len(explained["hits"]),
                "cold_input_sha": digest(fixture),
                "cold_output_sha": digest(result),
                "cold_output_path": str(result.relative_to(out)),
            }
        )
    partial = pd.DataFrame(partial_rows)
    partial["q_bh_100"] = false_discovery_control(partial.p.to_numpy())
    partial.to_csv(out / "posthoc_length_adjusted_associations.csv", index=False)
    # Same-domain factor alignment is descriptive, not five extra independent effects.
    alignment = []
    for domain in materials:
        a, b = all_scores[domain + "_common_factor"], all_scores[domain + "_pca"]
        for j in range(5):
            for k in range(5):
                alignment.append(
                    {
                        "domain": domain,
                        "common_factor": j + 1,
                        "pca": k + 1,
                        "held_score_r": float(np.corrcoef(a[:, j], b[:, k])[0, 1]),
                    }
                )
    pd.DataFrame(alignment).to_csv(
        out / "same_domain_score_relationships.csv", index=False
    )
    result = {
        "status": "PASS",
        "scorers": checks,
        "associations_checked": len(assoc),
        "material_interval_rows": len(intervals),
        "new_fits": 0,
        "new_neural_calls": 0,
        "posthoc_length_q_below_0_05": int((partial.q_bh_100 < 0.05).sum()),
        "summary_sha256": digest(out / "summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
        "scorer_code_sha256": digest(ROOT / "grain_research/local_measure.py"),
        "scorer_cli_sha256": digest(ROOT / "scripts/score_grain_local_measure.py"),
        "posthoc_sha256": digest(out / "posthoc_length_adjusted_associations.csv"),
        "alignment_sha256": digest(out / "same_domain_score_relationships.csv"),
    }
    write(out / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/local_construction_r1",
    )
    run(p.parse_args())
