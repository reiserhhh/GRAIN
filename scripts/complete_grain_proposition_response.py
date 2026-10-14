#!/usr/bin/env python3
"""Resume frozen proposition results after a pandas/SciPy indexing failure.

No parser or NLI inference is repeated. This companion changes only Series-mask
conversion to NumPy in the declared source calibration and replays saved inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_explanation_chain import sha, write
from scripts.run_grain_proposition_response import BASE, simulate


def source_calibrate(main, probabilities):
    """Fit only source-wording condition vocabulary and the fixed content units."""
    main = main.copy()
    source = main.part.eq("source")
    available = source & main.condition_text.ne("")
    if not available.any() or main.loc[available, "condition"].nunique() != 2:
        raise ValueError("Both source conditions required")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(main.loc[available, "condition_text"])
    centroids = np.stack(
        [
            np.asarray(
                x[main.loc[available, "condition"].eq(c).to_numpy()].mean(axis=0)
            ).ravel()
            for c in [0, 1]
        ]
    )
    sim = np.asarray(vectorizer.transform(main.condition_text) @ centroids.T)
    main["estimated_condition"] = np.where(
        main.condition_text.ne("") & (sim.max(axis=1) > 0), sim.argmax(axis=1), np.nan
    )
    main["content"] = main.id.map(probabilities.set_index("audit_id").p_entailment)
    truth_event = (
        main.sign.to_numpy()
        * np.where(main["mode"].eq("past_action_report"), 1, -1)
        * main.actor.eq("respondent").to_numpy()
    )
    model = Ridge(alpha=0.001).fit(
        main.loc[source, ["content"]], truth_event[source.to_numpy()]
    )
    main["content_target"] = model.predict(main[["content"]])
    main["oracle_target"] = truth_event
    calibration = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "centroids": centroids.tolist(),
        "content_intercept": float(model.intercept_),
        "content_coefficient": model.coef_.tolist(),
        "alpha": 0.001,
        "source_ids": main.loc[source, "id"].tolist(),
    }
    return main, calibration


def complete(out):
    """Write completion artifacts alongside, never over the frozen raw reads."""
    dst = out / "completion"
    dst.mkdir(exist_ok=False)
    frozen = json.loads((out / "freeze.json").read_text())
    for path, digest in frozen["files"].items():
        assert sha(ROOT / path) == digest, path
    files = [
        "private_graphs.json",
        "private_inputs.json",
        "private_parses.json",
        "nli_readings.csv",
        "nli_requests.csv",
        "config.json",
    ]
    write(
        dst / "resume_contract.json",
        {
            "files": {f: sha(out / f) for f in files},
            "script_sha256": sha(__file__),
            "change": "SciPy sparse indexing requires NumPy boolean mask, not pandas Series",
            "new_neural_calls": 0,
            "original_terminal": "AttributeError: Series object has no attribute nonzero",
        },
    )
    config = json.loads((out / "config.json").read_text())
    raw = json.loads((out / "private_inputs.json").read_text())
    graphs = {x["id"]: x for x in json.loads((out / "private_graphs.json").read_text())}
    rows = []
    for record in raw:
        observed = graphs[record["id"]]
        rows.append(
            {k: v for k, v in record.items() if k not in ["partner", "reply"]}
            | {
                "status": observed["binding"]["status"],
                "condition_text": observed["condition_text"],
                "graph_target": observed["graph_target"],
                "frames": len(observed["partner_frames"])
                + len(observed["reply_frames"]),
                "supported": observed["graph_target"] is not None,
            }
        )
    table = pd.DataFrame(rows)
    probabilities = pd.read_csv(out / "nli_readings.csv")
    main, calibration = source_calibrate(
        table.loc[table.part.isin(["source", "held"])], probabilities
    )
    main.to_csv(dst / "controlled_readings.csv", index=False)
    table.to_csv(dst / "private_interface_summary.csv", index=False)
    write(dst / "source_calibration.json", calibration)
    metrics = simulate(main, config, dst)
    checks = []
    for part, group in main.groupby("part"):
        valid = group.dropna(subset=["graph_target"])
        checks.append(
            {
                "part": part,
                "rows": len(group),
                "scored": len(valid),
                "target_exact": int(
                    np.isclose(valid.graph_target, valid.oracle_target).sum()
                ),
                "condition_correct": int(
                    group.estimated_condition.eq(group.condition).sum()
                ),
                "unknown_conditions": int(group.estimated_condition.isna().sum()),
            }
        )
    write(
        dst / "summary.json",
        {
            "status": "complete_after_indexing_repair",
            "public_records": 40,
            "natural_records": 12,
            "unique_parsed_texts": sum(
                bool(x["text"])
                for x in json.loads((out / "private_parses.json").read_text())
            ),
            "nli_pairs": len(probabilities),
            "original_runtime_seconds": None,
            "runtime_note": "Original process ended before timing summary; not reconstructed from file mtime",
            "binding_checks": checks,
            "simulation": metrics,
            "natural_status_counts": table.loc[table.part.eq("natural"), "status"]
            .value_counts()
            .to_dict(),
            "new_human_validity": False,
            "new_neural_calls_in_completion": 0,
        },
    )
    write(
        dst / "manifest.json",
        {
            p.name: sha(p)
            for p in dst.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "proposition_response_r1")
    complete(parser.parse_args().out)
